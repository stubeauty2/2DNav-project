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
sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
from fixtures import episode


class VisualContextTests(unittest.TestCase):
    def test_visual_questions_receive_public_original_frame_and_keep_old_view_cache(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ep=episode(root,ds)
                ep['agent_input']['initial_pose']['heading_deg']=287.
                ep['initialization']['direction']=167.
                ep['agent_input']['instruction_raw']='Find the brown dirt lot at eight oclock from the starting pose.'
                loop=importlib.import_module(ds+'.loop');m=importlib.import_module(ds+'.tools.model')
                g=importlib.import_module(ds+'.tools.grounding_skill');v=importlib.import_module(ds+'.tools.views')
                questions=[];decisions=[];failures=[]
                action=dict(action='tool',tool='visual_question',view_id='r00-main',
                    question='Eight oclock from current heading 167 means bottom left; compare this region.',
                    regions=[dict(id='A',bbox=[450,100,550,200])],reason='Check direction premise')
                def transport(url,name,key,messages,**kw):
                    try:
                        ctx=json.loads(messages[1]['content'][0]['text'])
                        self.assertNotIn('HIDDEN_TARGET_SENTINEL',json.dumps(messages))
                        if ctx['mode']=='visual_evidence':
                            self.assertEqual(name,'qwen3.8-flash');questions.append(ctx)
                            if ds=='AVDN':
                                self.assertEqual(len(ctx['region_details']),1)
                                self.assertEqual(ctx['region_details'][0]['region_id'],'A')
                                self.assertEqual(len(ctx['views']),2)
                                self.assertEqual(ctx['source_view_id'],'r00-main')
                                self.assertIn('ORIGINAL',ctx['detail_coordinate_note'])
                            self.assertEqual(ctx['question'],action['question'])
                            ref=ctx['public_reference']
                            self.assertEqual(ref['original_pose'],ep['agent_input']['initial_pose'])
                            self.assertEqual(ref['instruction_raw'],ep['agent_input']['instruction_raw'])
                            geo=ref['region_geometry'][0]
                            self.assertEqual(geo['region_id'],'A');self.assertAlmostEqual(geo['bearing_from_initial_deg'],167.,delta=.2)
                            self.assertEqual(geo['clock_from_initial_pose'],'8:00')
                            self.assertNotIn('distance_from_current_m',geo)
                            answer=dict(observations=[dict(region_id='A',support=['Visible ground'],contradictions=[],unobservable=[])],
                                comparison='Original-pose direction differs from image-relative direction.',uncertainty=[],suggested_check='none',proposals=[])
                        else:
                            self.assertEqual(name,'qwen3.5-flash');decisions.append(ctx)
                            answer=action if len(decisions)==1 else dict(action='give_up',reason='Fixture complete')
                        return dict(status=200,text=json.dumps(answer),usage={})
                    except Exception as exc:failures.append(repr(exc));raise
                env=v.Environment(ep['environment'],root/'views')
                model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False,visual_model='qwen3.8-flash')
                try:
                    agent=loop.Agent(ep,env,model,g.SAM3(),root,grounding_mode='adaptive')
                    with patch.object(m,'call_model',side_effect=transport):result=agent.run()
                    self.assertFalse(failures,str(failures));self.assertEqual(result['termination'],'give_up')
                    self.assertEqual(len(questions),1)
                    # Reusing an old view after moving does not rebind its original reference.
                    agent.position=[55.,60.];agent.heading=0.
                    cached,_=agent.execute_tool(action,env.views['r00-main'],9)
                    self.assertTrue(cached['cache_hit']);self.assertEqual(len(questions),1)
                finally:env.close()


    def test_visual_reference_uses_zoom_projection_and_preserves_unknown_heading(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ep=episode(root,ds)
                e=importlib.import_module(ds+'.tools.visual_evidence')
                ep['private_target']='HIDDEN_TARGET_SENTINEL'
                # Two view coordinate systems describing the same public point [60,60].
                main=dict(view_id='main',image_size=[101,101],pixel_to_world=[[1,0,0],[0,1,0],[0,0,1]])
                zoom=dict(view_id='zoom',image_size=[101,101],pixel_to_world=[[.2,0,50],[0,.2,50],[0,0,1]])
                a=e.reference_context(main,[dict(id='A',bbox=[550,550,650,650])],ep)
                b=e.reference_context(zoom,[dict(id='A',bbox=[450,450,550,550])],ep)
                self.assertEqual(a['region_geometry'],b['region_geometry'])
                self.assertNotIn('HIDDEN_TARGET_SENTINEL',json.dumps(a))
                self.assertEqual(a['region_geometry'][0]['clock_from_initial_pose'],'1:30')
                ep['agent_input']['initial_pose']['heading_deg']=None
                c=e.reference_context(zoom,[dict(id='A',bbox=[450,450,550,550])],ep)
                self.assertIsNone(c['region_geometry'][0]['clock_from_initial_pose'])
                self.assertIsNone(c['original_pose']['heading_deg'])


    def test_fusion_rejection_regions_remain_inspectable_in_decision_and_review(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ep=episode(root,ds)
                loop=importlib.import_module(ds+'.loop');g=importlib.import_module(ds+'.tools.grounding_skill')
                m=importlib.import_module(ds+'.tools.model');v=importlib.import_module(ds+'.tools.views')
                memory=importlib.import_module(ds+'.tools.memory')
                def detect(payload):
                    return dict(backend='dino',coordinate_extent=[1000,1000],
                        candidates=[dict(candidate_id='0',bbox_2d=[400,400,600,600],score=.8)])
                tool=g.GroundingDINO(transport=detect);decisions=[]
                def transport(url,name,key,messages,**kwargs):
                    ctx=json.loads(messages[1]['content'][0]['text']);mode=ctx['mode']
                    if mode=='grounding_fusion':
                        answer=dict(candidates=[],rejected=[dict(source_ids=['dino/0'],reason='Possibly ground texture')],unresolved=[])
                    else:
                        if mode=='decision':decisions.append(ctx)
                        if mode=='review' or len(decisions)>1:
                            history=ctx['review_evidence'] if mode=='review' else ctx['history']
                            evidence=history['current_observation']['evidence'][0]['result']
                            rejected=evidence['rejected_candidates'][0]
                            self.assertEqual(rejected['reason'],'Possibly ground texture')
                            self.assertEqual(rejected['source_regions'],[dict(source_id='dino/0',view_id='r00-main',bbox=[400.,400.,600.,600.])])
                            self.assertEqual(evidence['candidates'],[]);self.assertEqual(evidence['unassessed_sources'],[])
                            self.assertEqual(memory.evidence_summary(evidence),evidence)
                        if mode=='review':
                            answer=dict(action='review',verdict='confirm',view_id=ctx['views'][0]['view_id'],bbox=[400,400,600,600],reason='support: visible object; contradiction: none; unobservable: none')
                        elif len(decisions)==1:
                            answer=dict(action='skill',skill='grounding',query='car',backends=['dino'],view_id='r00-main',reason='Inspect category')
                        elif len(decisions)==2:
                            region=rejected['source_regions'][0]
                            answer=dict(action='tool',tool='zoom',view_id=region['view_id'],bbox=region['bbox'],reason='Check the fallible rejection in RGB')
                        else:
                            answer=dict(action='locate',target='final',view_id='r00-main',bbox=[400,400,600,600],reason='Proposal after inspection')
                    return dict(status=200,text=json.dumps(answer),usage={})
                env=v.Environment(ep['environment'],root/'views')
                model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False)
                failures=[]
                def checked_transport(*args,**kwargs):
                    try:return transport(*args,**kwargs)
                    except Exception as exc:
                        failures.append(repr(exc));raise
                try:
                    agent=loop.Agent(ep,env,model,g.SAM3(),root,grounders={'dino':tool},grounding_mode='adaptive')
                    with patch.object(m,'call_model',side_effect=checked_transport):result=agent.run()
                    self.assertEqual(result['termination'],'confirmed_final',str(failures));self.assertEqual(len(decisions),3)
                    stored=next(e['output'] for e in agent.events if e['kind']=='skill')
                    self.assertNotIn('source_regions',stored['rejected_candidates'][0])
                    self.assertEqual(stored['candidates'],[])
                finally:env.close()


    def test_fused_component_regions_survive_decision_and_independent_review(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);ep=episode(root,ds)
                loop=importlib.import_module(ds+'.loop');g=importlib.import_module(ds+'.tools.grounding_skill')
                m=importlib.import_module(ds+'.tools.model');v=importlib.import_module(ds+'.tools.views')
                memory=importlib.import_module(ds+'.tools.memory')
                component=[400.,400.,600.,600.];whole=[300.,300.,700.,700.]
                def detect(payload):
                    return dict(backend='dino',coordinate_extent=[1000,1000],candidates=[
                        dict(candidate_id=str(i),bbox_2d=box,score=.8) for i,box in enumerate([component,whole])])
                tool=g.GroundingDINO(transport=detect);decisions=[];failures=[]
                def transport(url,name,key,messages,**kw):
                    try:
                        ctx=json.loads(messages[1]['content'][0]['text']);mode=ctx['mode']
                        if mode=='grounding_fusion':
                            answer=dict(candidates=[dict(bbox=whole,source_ids=['dino/0','dino/1'],reason='One connected complex',uncertainty=['Part boundaries unclear'])],rejected=[],unresolved=[])
                        else:
                            if mode=='decision':decisions.append(ctx)
                            if mode=='review' or len(decisions)>1:
                                history=ctx['review_evidence'] if mode=='review' else ctx['history']
                                evidence=history['current_observation']['evidence'][0]['result']
                                fused=evidence['candidates'][0]
                                self.assertEqual(fused['bbox'],whole)
                                self.assertEqual(fused['source_regions'],[
                                    dict(source_id='dino/'+str(i),view_id='r00-main',bbox=box)
                                    for i,box in enumerate([component,whole])])
                                self.assertEqual(evidence['unassessed_sources'],[])
                                self.assertEqual(evidence['sources'][0]['candidate_count'],2)
                                self.assertIn('Original detector extents, not independent confirmation',evidence['source_regions_note'])
                                self.assertEqual(memory.evidence_summary(evidence),evidence)
                            if mode=='review':
                                answer=dict(action='review',verdict='confirm',view_id='r00-main',bbox=component,reason='support: component visible; contradiction: none; unobservable: none')
                            elif len(decisions)==1:
                                answer=dict(action='skill',skill='grounding',query='building',backends=['dino'],view_id='r00-main',reason='Find category regions')
                            else:
                                region=fused['source_regions'][0]
                                answer=dict(action='locate',target='final',view_id=region['view_id'],bbox=region['bbox'],reason='Requested component rather than whole complex')
                        return dict(status=200,text=json.dumps(answer),usage={})
                    except Exception as exc:failures.append(repr(exc));raise
                env=v.Environment(ep['environment'],root/'views')
                model=m.Model(root/'requests','qwen3.5-flash','test',enable_thinking=False)
                try:
                    agent=loop.Agent(ep,env,model,g.SAM3(),root,grounders={'dino':tool},grounding_mode='adaptive')
                    with patch.object(m,'call_model',side_effect=transport):result=agent.run()
                    self.assertFalse(failures,str(failures));self.assertEqual(result['termination'],'confirmed_final')
                    self.assertEqual(result['final']['bbox'],component)
                    stored=next(e['output'] for e in agent.events if e['kind']=='skill')
                    self.assertNotIn('source_regions',stored['candidates'][0])
                    self.assertEqual(stored['candidates'][0]['bbox'],whole)
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
