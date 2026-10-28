"""Exercise real dispatch, coordinate isolation, model routing and journal replay."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tests/grounding_skill')]
from test_skill import episode


class AdaptiveTests(unittest.TestCase):
    def test_independent_tools_skill_visual_route_review_and_replay(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                l=importlib.import_module(ds+'.loop');m=importlib.import_module(ds+'.tools.model')
                g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                j=importlib.import_module(ds+'.tools.jev_evidence')
                calls=[];requests=[];step=[0]
                def backend(name):
                    def call(payload):
                        calls.append(name)
                        return dict(backend=name,coordinate_extent=[1000,1000],
                            candidates=[dict(candidate_id='0',bbox_2d=[400,400,600,600],score=.8)])
                    tool=g.Grounder(transport=call);tool.backend=name;return tool
                grounders={n:backend(n) for n in ('dino','sam3','locate_anything')}
                def jev_call(payload):
                    criteria=payload['questions']['visual_fact']['criteria']
                    return dict(model='test-jev',answers=dict(visual_fact=dict(choice='unknown',
                        probabilities={c:float(c=='unknown') for c in criteria})))
                def transport(url,model,key,messages,**kw):
                    ctx=json.loads(messages[1]['content'][0]['text']);mode=ctx['mode']
                    requests.append((mode,model,copy.deepcopy(ctx)))
                    self.assertEqual(model,'qwen3.8-flash' if mode=='visual_evidence' else 'qwen3.5-flash')
                    if mode=='grounding_fusion':
                        self.assertNotIn('instruction_raw',ctx)
                        self.assertEqual([x['backend'] for x in ctx['sources']],['dino','sam3'])
                        answer=dict(candidates=[dict(bbox=[400,400,600,600],source_ids=['dino/0','sam3/0'],
                            reason='visible car',uncertainty=[])],rejected=[],unresolved=[])
                    elif mode=='visual_evidence':
                        self.assertNotIn('instruction_raw',ctx)
                        self.assertNotIn('initialization',ctx)
                        self.assertEqual(ctx['views'][0]['source'],'zoom')
                        answer=dict(observations=[dict(region_id='A',support=['car beside road'],contradictions=[],unobservable=[])],
                            comparison='Only one region supplied',uncertainty=[],suggested_check='none',
                            proposals=[dict(id='p1',bbox=[250,250,750,750],category='car',evidence='visible car beside road',uncertainty=[])])
                    elif mode=='review':
                        self.assertIn('review_evidence',ctx)
                        answer=dict(action='review',verdict='confirm',view_id=ctx['views'][0]['view_id'],
                            bbox=[400,400,600,600],reason='support: visible relation; contradiction: none; unobservable: none')
                    else:
                        self.assertEqual(set(ctx['available_grounding_tools']),set(grounders))
                        self.assertEqual(ctx['available_skills'],['grounding'])
                        self.assertIn('skill',ctx['allowed_next_actions'])
                        step[0]+=1;n=step[0];view=ctx['views'][0]['view_id']
                        if n==1:answer=dict(action='tool',tool='dino',view_id=view,query='car',reason='candidate map')
                        elif n==2:answer=dict(action='skill',skill='grounding',view_id=view,query='car',backends=['sam3','dino'],reason='consolidate')
                        elif n==3:answer=dict(action='tool',tool='sam3',view_id=view,query='car',reason='inspect raw cached source')
                        elif n==4:answer=dict(action='tool',tool='locate_anything',view_id=view,query='small car',reason='independent referring phrase')
                        elif n==5:answer=dict(action='tool',tool='jev',view_id=view,questions=[dict(id='q',kind='color',object='car',bbox=[400,400,600,600])],reason='appearance')
                        elif n==6:answer=dict(action='tool',tool='zoom',view_id=view,bbox=[300,300,700,700],reason='relation detail')
                        elif n==7:
                            zoom=next(x['view_id'] for x in ctx['views'] if x['source']=='zoom')
                            answer=dict(action='tool',tool='visual_question',view_id=zoom,question='What relation is visible?',
                                regions=[dict(id='A',bbox=[250,250,750,750])],reason='Jev unknown; inspect relation')
                        else:answer=dict(action='locate',target='final',view_id=view,bbox=[400,400,600,600],reason='distinct relation')
                    return dict(status=200,text=json.dumps(answer),usage={})
                def run():
                    env=v.Environment(ep['environment'],root/'views')
                    model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False,
                        tool_protocol='structured_v2',visual_model='qwen3.8-flash')
                    agent=l.Agent(ep,env,model,grounders['sam3'],root,max_rounds=2,max_tools=8,
                        grounders=grounders,grounding_mode='adaptive',max_observation_tools=8,
                        jev=j.JevEvidence(None,model='test-jev',transport=jev_call))
                    try:return agent.run(),agent
                    finally:env.close()
                with patch.object(m,'call_model',side_effect=transport):result,agent=run()
                self.assertEqual(result['termination'],'confirmed_final')
                self.assertEqual(calls,['dino','sam3','locate_anything'])
                self.assertEqual(result['model_requests'],11)
                self.assertEqual(sum(e['kind']=='skill' for e in agent.events),1)
                self.assertTrue(next(e['output']['cache_hit'] for e in agent.events
                    if e['kind']=='tool' and e['input']['tool']=='sam3'))
                self.assertEqual(len(agent.trajectory),1)
                visual=next(e['output'] for e in agent.events if e['kind']=='tool' and e['input']['tool']=='visual_question')
                self.assertIn('unverified',visual['proposals'][0]['proposal_status'])
                self.assertEqual(visual['proposals'][0]['view_id'],'r00-zoom6')
                with patch.object(m,'call_model',side_effect=AssertionError('Replay made a paid request')):
                    replay,replayed=run()
                self.assertEqual(replay,result)
                self.assertEqual(replayed.memory,agent.memory)
                self.assertEqual(calls,['dino','sam3','locate_anything'])

    def test_rejected_review_requests_new_source_then_resumes_normal_decision(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                l=importlib.import_module(ds+'.loop');m=importlib.import_module(ds+'.tools.model')
                g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                env=v.Environment(ep['environment'],root/'views')
                model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False,
                    tool_protocol='structured_v2',visual_model='qwen3.8-flash')
                decisions=[];reviews=[];helpers=[]
                def transport(url,name,key,messages,**kwargs):
                    ctx=json.loads(messages[1]['content'][0]['text']);view=ctx['views'][0]['view_id']
                    self.assertEqual(name,'qwen3.8-flash' if ctx['mode']=='visual_evidence' else 'qwen3.5-flash')
                    if ctx['mode']=='review':
                        reviews.append(ctx)
                        self.assertNotIn('previous_review',ctx['review_evidence'])
                        self.assertNotIn('history',ctx)
                        response=dict(action='review',verdict='reject' if len(reviews)==1 else 'confirm',
                            view_id=view,bbox=[400,400,600,600],reason='relation contradicted' if len(reviews)==1 else 'new relation evidence')
                    elif ctx['mode']=='visual_evidence':
                        helpers.append(ctx)
                        response=dict(observations=[],comparison='New visible relation supports central object',
                            uncertainty=[],suggested_check='compare relation')
                    else:
                        decisions.append(ctx);n=len(decisions)
                        if n in (2,3):
                            self.assertTrue(ctx['evidence_focus'])
                            self.assertNotIn('locate',ctx['allowed_next_actions'])
                            self.assertNotIn('jev',ctx['tools_callable_now'])
                            self.assertIn('Select NEW visual evidence',messages[0]['content'])
                        else:self.assertFalse(ctx['evidence_focus'])
                        if n==3:response=dict(action='tool',tool='visual_question',view_id=view,
                            question='Which visible relation distinguishes the central object from alternatives?',regions=[],reason='resolve contradiction')
                        else:response=dict(action='locate',target='final',view_id=view,bbox=[400,400,600,600],reason='central relation')
                    return dict(status=200,text=json.dumps(response),usage={})
                try:
                    agent=l.Agent(ep,env,model,g.SAM3(),root,grounders={},grounding_mode='adaptive',max_rounds=3)
                    with patch.object(m,'call_model',side_effect=transport):result=agent.run()
                    self.assertEqual(result['termination'],'confirmed_final')
                    self.assertEqual((len(decisions),len(reviews),len(helpers)),(4,2,1))
                    self.assertEqual(len([e for e in agent.events if e['kind']=='invalid_action']),1)
                    self.assertEqual([c['backend'] for c in result['tool_calls']],['visual_question'])
                finally:env.close()

    def test_cached_calls_focus_clears_on_fresh_evidence_and_respects_limits(self):
        for ds in ('AVDN','CityNav'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                l=importlib.import_module(ds+'.loop');m=importlib.import_module(ds+'.tools.model')
                g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                env=v.Environment(ep['environment'],root/'views')
                model=m.Model(root/'requests','qwen3.5-flash','test',max_requests=8,visual_model='qwen3.8-flash')
                try:
                    agent=l.Agent(ep,env,model,g.SAM3(),root,grounders={},grounding_mode='adaptive')
                    agent.begin_loop(env.main([50,50],0,'main'))
                    agent.active_loop['tools']=[dict(tool='jev',cache_hit=True)]*2
                    self.assertEqual(agent.evidence_focus(1)['preferred_tool'],'visual_question')
                    agent.active_loop['tools'].append(dict(tool='zoom',cache_hit=False))
                    self.assertIsNone(agent.evidence_focus(1))
                    agent.active_loop['tools']=[dict(tool='jev',cache_hit=True)]*2
                    model.cursor=5
                    self.assertNotIn('visual_question',agent.evidence_focus(1)['allowed_tools'])
                    model.cursor=6;self.assertIsNone(agent.evidence_focus(1))
                    model.cursor=0;self.assertIsNone(agent.evidence_focus(0))
                    agent.observation_executions[agent.observation_id]=agent.max_observation_tools
                    self.assertIsNone(agent.evidence_focus(1))
                finally:env.close()

    def test_late_visual_request_cannot_spend_the_final_decision_and_review_budget(self):
        for ds in ('AVDN','CityNav'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                l=importlib.import_module(ds+'.loop');m=importlib.import_module(ds+'.tools.model')
                g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                env=v.Environment(ep['environment'],root/'views')
                model=m.Model(root/'requests','qwen3.5-flash','test',max_requests=3,visual_model='qwen3.8-flash')
                calls=[]
                def transport(url,name,key,messages,**kwargs):
                    ctx=json.loads(messages[1]['content'][0]['text']);calls.append(ctx['mode'])
                    self.assertEqual(name,'qwen3.5-flash')
                    view=ctx['views'][0]['view_id']
                    if len(calls)==1:
                        self.assertNotIn('visual_question',ctx['tools_callable_now'])
                        self.assertEqual(ctx['skills_callable_now'],[])
                        answer=dict(action='tool',tool='visual_question',view_id=view,question='Visible alternatives?',regions=[],reason='inspect')
                    elif ctx['mode']=='decision':
                        self.assertEqual(ctx['tools_callable_now'],[])
                        answer=dict(action='locate',target='final',view_id=view,bbox=[400,400,600,600],reason='existing RGB supports relation')
                    else:answer=dict(action='review',verdict='confirm',view_id=view,bbox=[400,400,600,600],reason='support: relation; contradiction: none; unobservable: none')
                    return dict(status=200,text=json.dumps(answer),usage={})
                try:
                    with patch.object(m,'call_model',side_effect=transport):
                        result=l.Agent(ep,env,model,g.SAM3(),root,grounders={},grounding_mode='adaptive').run()
                    self.assertEqual(result['termination'],'confirmed_final')
                    self.assertEqual(calls,['decision','decision','review'])
                    self.assertEqual(result['tool_calls'],[])
                finally:env.close()

    def test_visual_validation_scope_and_unknown_are_not_confirmation(self):
        for ds in ('AVDN','CityNav'):
            m=importlib.import_module(ds+'.tools.visual_evidence')
            l=importlib.import_module(ds+'.loop')
            regions=[dict(id='A',bbox=[10,10,30,30])]
            good=dict(observations=[dict(region_id='A',support=[],contradictions=[],unobservable=['tiny'])],
                comparison='unknown',uncertainty=['resolution'],suggested_check='zoom')
            self.assertEqual(m.validate_answer(good,regions),good)
            for bad in (dict(good,verdict='confirm'),dict(good,observations=[])):
                with self.assertRaises(ValueError):m.validate_answer(bad,regions)
            with self.assertRaises(ValueError):m.validate_request('question',[dict(id='A',bbox=[0,0,1001,10])])
            with self.assertRaises(ValueError):l.validate_action(dict(action='skill',skill='grounding',view_id='v',
                query='car',backends=['dino','dino'],reason='x'))

    def test_skill_can_select_one_or_three_backends_without_changing_raw_tools(self):
        for ds in ('AVDN','CityNav'):
            for selected in (['dino'],['dino','sam3','locate_anything']):
                with self.subTest(dataset=ds,selected=selected),tempfile.TemporaryDirectory() as td:
                    root=Path(td);ep=episode(root,ds)
                    g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                    m=importlib.import_module(ds+'.tools.model')
                    env=v.Environment(ep['environment'],root/'views');view=env.main([50,50],0,'main')
                    raw={n:dict(backend=n,status='ok',candidates=[]) for n in selected}
                    model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False)
                    def transport(url,model,key,messages,**kwargs):
                        ctx=json.loads(messages[1]['content'][0]['text'])
                        self.assertEqual([s['backend'] for s in ctx['sources']],selected)
                        self.assertEqual(len(ctx['views']),len(selected)+1)
                        return dict(status=200,text=json.dumps(dict(candidates=[],rejected=[],unresolved=['none visible'])),usage={})
                    try:
                        with patch.object(m,'call_model',side_effect=transport):
                            result,overlay,_=g.GroundingSkill({},model,selected).run(view,'car',root/'skill',raw_results=raw)
                        self.assertEqual(result['status'],'ok');self.assertIsNone(overlay)
                    finally:env.close()

    def test_visual_call_uses_global_budget_and_invalid_response_cannot_confirm(self):
        for ds in ('AVDN','CityNav'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                m=importlib.import_module(ds+'.tools.model');v=importlib.import_module(ds+'.tools.views')
                e=importlib.import_module(ds+'.tools.visual_evidence')
                env=v.Environment(ep['environment'],root/'views');view=env.main([50,50],0,'main')
                model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False,max_requests=1,visual_model='qwen3.8-flash')
                try:
                    with patch.object(m,'call_model',return_value=dict(status=200,text='{"verdict":"confirm"}',usage={})) as call:
                        result=e.VisualEvidence(model).inspect(view,'Visible layout?',[],root/'evidence.json')
                        self.assertEqual(result['status'],'error')
                        with self.assertRaises(m.Halt):model.request(dict(mode='decision'),[view])
                        self.assertEqual(call.call_count,1)
                finally:env.close()

    def test_partial_padding_is_evidence_and_fusion_does_not_receive_navigation_geometry(self):
        for ds in ('AVDN','CityNav'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                l=importlib.import_module(ds+'.loop');m=importlib.import_module(ds+'.tools.model')
                g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                env=v.Environment(ep['environment'],root/'views')
                model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False)
                tool=g.GroundingDINO(transport=lambda _:dict(backend='dino',coordinate_extent=[1000,1000],
                    candidates=[dict(candidate_id='0',bbox_2d=[0,0,600,600],score=.8)]))
                agent=l.Agent(ep,env,model,g.SAM3(),root,grounders={'dino':tool},grounding_mode='adaptive')
                view=env.main([10,10],0,'edge');agent.begin_loop(view)
                try:
                    result,_=agent.execute_tool(dict(action='tool',tool='dino',view_id='edge',query='building',reason='map'),view,1)
                    self.assertEqual(len(result['candidates']),1)
                    self.assertTrue(result['candidates'][0]['quality']['touches_padding'])
                    with self.assertRaises(ValueError):env.locate(view,[0,0,600,600])
                    raw=dict(dino=dict(status='ok',candidates=[dict(result['candidates'][0],public_geometry={'secret_target_hint':'must be removed'})]))
                    sources,_=g.prepare_sources(view,'building',raw,root/'source-check',backends=['dino'])
                    self.assertNotIn('public_geometry',sources[0]['candidates'][0])
                    self.assertNotIn('object_ref',sources[0]['candidates'][0])
                finally:env.close()

    def test_compact_context_preserves_every_candidate_and_probability_once(self):
        for ds in ('AVDN','CityNav'):
            m=importlib.import_module(ds+'.tools.memory')
            candidates=[dict(candidate_id=str(i),bbox=[i,0,i+1,1],public_geometry=dict(
                distance_from_initial_m=i,note='repeated frame explanation')) for i in range(24)]
            result=dict(backend='dino',status='ok',view_id='v',evidence_id='raw',candidates=candidates)
            probabilities={'gray':.4,'blue':.35,'unknown':.25}
            answer=dict(evidence_id='answer',probabilities=probabilities)
            context=dict(mode='decision',grounding_mode='adaptive',optional_tools_remaining=3,
                capabilities={'zoom':True},available_evidence_tools=['jev'],available_grounding_tools=['dino'],
                available_skills=['grounding'],memory={},visual_evidence=[answer],
                feedback=[dict(tool_result=result),dict(error='bad category',rejected_action={'action':'tool','query':'gray van','reason':'old opinion'})]*3,
                history=dict(previous_loop=None,recent_loops=[],latest_review=None,current_observation=dict(evidence=[
                    dict(evidence_id='raw',action={'reason':'old opinion'},result=result),
                    dict(evidence_id='jev',action={},result=dict(answers=[answer]))])))
            original=copy.deepcopy(context);out=m.presentation_adaptive(context)
            self.assertEqual(context,original)
            evidence=out['history']['current_observation']['evidence']
            self.assertEqual(len(evidence[0]['result']['candidates']),24)
            self.assertEqual(evidence[1]['result']['answers'][0]['probabilities'],probabilities)
            self.assertEqual(out['visual_evidence'],[])
            self.assertNotIn('tool_result',json.dumps(out['feedback']))
            self.assertEqual(out['latest_feedback'][0]['repeated_count'],3)
            self.assertNotIn('reason',out['latest_feedback'][0]['rejected_action'])
            self.assertEqual(evidence[0]['result']['candidates'][23]['public_geometry'],{'distance_from_initial_m':23})

    def test_visual_rgb_proposals_are_checked_and_remain_unverified_evidence(self):
        for ds in ('AVDN','CityNav'):
            e=importlib.import_module(ds+'.tools.visual_evidence')
            answer=dict(observations=[],comparison='two visible objects',uncertainty=[],suggested_check='compare relation',
                proposals=[dict(id='p',bbox=[20,30,40,50],category='car',evidence='visible small vehicle',uncertainty=[])])
            self.assertEqual(e.validate_answer(answer,[]),answer)
            for p in (dict(answer['proposals'][0],bbox=[20,30,1001,50]),dict(answer['proposals'][0],verdict='confirm')):
                with self.assertRaises(ValueError):e.validate_answer(dict(answer,proposals=[p]),[])

    def test_invalid_category_does_not_consume_an_unexecuted_tool_allowance(self):
        for ds in ('AVDN','CityNav'):
            with tempfile.TemporaryDirectory() as td:
                root=Path(td);ep=episode(root,ds)
                l=importlib.import_module(ds+'.loop');m=importlib.import_module(ds+'.tools.model')
                g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                env=v.Environment(ep['environment'],root/'views');model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False)
                tool=g.GroundingDINO(transport=lambda _:dict(backend='dino',candidates=[]));calls=[]
                def transport(url,name,key,messages,**kwargs):
                    ctx=json.loads(messages[1]['content'][0]['text']);calls.append(ctx)
                    if len(calls)==1:response=dict(action='tool',tool='dino',view_id='r00-main',query='red car',reason='appearance')
                    elif len(calls)==2:
                        self.assertEqual(ctx['optional_tools_remaining'],1)
                        self.assertTrue(ctx['latest_feedback'])
                        response=dict(action='tool',tool='dino',view_id='r00-main',query='car',reason='category')
                    else:response=dict(action='give_up',reason='no supported identity')
                    return dict(status=200,text=json.dumps(response),usage={})
                try:
                    with patch.object(m,'call_model',side_effect=transport):
                        result=l.Agent(ep,env,model,g.SAM3(),root,grounders={'dino':tool},grounding_mode='adaptive',max_tools=1).run()
                    self.assertEqual(result['termination'],'give_up')
                    self.assertEqual(len(result['tool_calls']),1)
                finally:env.close()


if __name__=='__main__':unittest.main()
