"""Offline regressions run against BOTH independent dataset packages."""
import importlib
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

class ScriptModel:
    def __init__(self, actions, module):
        self.actions=iter(actions);self.cursor=0;self.successful_views=set();self.contexts=[]
        self.module=module
    def request(self, context, views):
        self.cursor+=1;self.successful_views.update(v['view_id'] for v in views)
        self.contexts.append(context)
        a=next(self.actions)
        if isinstance(a,Exception):raise a
        return dict(status='ok',response=dict(text=json.dumps(a)))

class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        im=Image.new('RGB',(400,400),(110,140,80));im.save(self.root/'map.png')
    def tearDown(self):self.temp.cleanup()
    def modules(self,d):
        names = dict(engine='loop', environment='tools.views', geometry='tools.geometry',
                     model='tools.model', contracts='loop', tools='tools.grounding', io='tools.runtime')
        modules = {n:importlib.import_module(d+'.'+target) for n,target in names.items()}
        modules['evaluate'] = importlib.import_module('nav_offline.'+d.lower()+'.evaluate')
        return modules
    def fixture(self,d,mod):
        folder=self.root/d;folder.mkdir(exist_ok=True)
        agent=dict(dataset=d,setting='visual' if d=='AVDN' else 'landmark_assisted',
          instruction_raw='Find a fictional blue plaza.',turns=[dict(turn_id='t0',role='INS',text='Find the plaza')],
          initial_pose=dict(xy_m=[200.,200.],heading_deg=0.),coordinate_frame='east/north meters',
          named_landmarks=[] if d=='AVDN' else [dict(id='lm-1',name='fictional museum',type='Building',xy_m=[260.,200.],bbox_xy_m=None)],
          views=[])
        spec=dict(image_path=str(self.root/'map.png'),image_size=[400,400],
                  world_to_raster=[[1,0,0],[0,-1,400],[0,0,1]],main_width_m=100.,main_height_m=100.)
        episode=dict(episode_id=d+'-synthetic',agent_input=agent,
                     initialization=dict(direction=0.,final_target='blue plaza',auxiliary=[]),environment=spec)
        return episode,mod['environment'].Environment(spec,folder/'views'),folder
    def test_candidate_public_geometry_uses_candidate_not_agent(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m)
            context=importlib.import_module(d+'.tools.context').candidate_context
            ep['initialization']['direction']=350
            geometry=context([200,300],ep,[300,300],90)
            self.assertEqual(geometry['bearing_from_initial_deg'],0)
            self.assertEqual(geometry['turn_from_initial_prior_deg'],10)
            self.assertEqual(geometry['turn_from_current_heading_deg'],-180)
            self.assertEqual(geometry['distance_from_current_m'],100)
            at_start=context([200,200],ep,[200,200],90)
            self.assertIsNone(at_start['bearing_from_initial_deg'])
            if d=='CityNav':
                ep['agent_input']['named_landmarks']=[
                    dict(id='a',name='fictional road',xy_m=[100,300],bbox_xy_m=[90,295,110,305]),
                    dict(id='b',name='fictional road',xy_m=[200,310],bbox_xy_m=[190,305,210,315])]
                geometry=context([200,300],ep,[300,300],90)
                refs=geometry['public_reference_distances']
                self.assertEqual(len(refs),1)
                self.assertEqual(refs[0]['nearest_center']['landmark_id'],'b')
                self.assertEqual(refs[0]['nearest_center']['distance_to_center_m'],10)
                self.assertEqual(refs[0]['nearest_extent']['distance_to_extent_m'],5)
            # A rotated tool view projects the same physical candidate.
            center=[200,210]
            for heading in (0,90,230):
                view=env.main([200,200],heading,str(heading))
                pixel=m['geometry'].normalized(view,[center])[0]
                located=env.locate(view,[pixel[0]-10,pixel[1]-10,pixel[0]+10,pixel[1]+10])
                self.assertEqual(context(located['center_xy_m'],ep,[300,300],90)['distance_from_initial_m'],10)
            self.assertNotIn('goal_xy_m',json.dumps(geometry))
            env.close()

    def test_geometry_views_and_bounds(self):
        for d in ('AVDN','CityNav'):
            with self.subTest(dataset=d):
                m=self.modules(d);ep,env,out=self.fixture(d,m);g=m['geometry']
                v=env.main([200,200],45,'main')
                self.assertAlmostEqual(v['image_up_heading_deg'],45)
                np.testing.assert_allclose(g.normalized(v,g.world(v,[[123,456],[789,321]])),[[123,456],[789,321]],atol=1e-8)
                pos,h=g.offset([1,2],0,90,2,100);np.testing.assert_allclose(pos,[201,2]);self.assertEqual(h,90)
                candidate=env.locate(v,[400,400,500,500]);review=env.review(candidate,v,'detail')
                self.assertAlmostEqual(review['width_m'],25)
                mapped=g.normalized(review,candidate['polygon_xy_m'])
                np.testing.assert_allclose(g.world(review,mapped),candidate['polygon_xy_m'],atol=1e-8)
                edge=env.main([10,10],0,'edge')
                self.assertTrue(0<edge['valid_fraction']<1)
                with self.assertRaises(ValueError):env.check_point(edge,[0,500])
                with self.assertRaises(ValueError):env.check_position([-1,30])
                with self.assertRaises(ValueError):env.view('unseen',{'main'})
                with self.assertRaises(ValueError):env.locate(v,[10,20,10,30])
                env.close()
    def execute(self,d,actions,rounds=3):
        m=self.modules(d);ep,env,out=self.fixture(d,m)
        model=ScriptModel(actions,m)
        agent=m['engine'].Agent(ep,env,model,m['tools'].SAM3(),out,max_rounds=rounds)
        result=agent.run();trace=m['io'].load_json(out/'trace.json');env.close()
        return result,trace,model,m,ep,out
    def test_review_marker_preserves_frame_and_unmarked_rgb(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m)
            main=env.main([200,200],35,'main')
            candidate=env.locate(main,[100,250,300,600])
            raw=env.review(candidate,main,'review')
            original=Path(raw['image_path']).read_bytes()
            marked=env.mark_review(raw,candidate)
            self.assertEqual(marked['view_id'],raw['view_id'])
            self.assertEqual(Path(marked['unmarked_image_path']).read_bytes(),original)
            self.assertNotEqual(Path(marked['image_path']).read_bytes(),original)
            np.testing.assert_allclose(marked['pixel_to_world'],raw['pixel_to_world'])
            result=env.locate(marked,marked['proposal_bbox'])
            np.testing.assert_allclose(result['center_xy_m'],candidate['center_xy_m'],atol=1e-8)
            self.assertNotIn('unmarked_image_path',m['model'].safe_view(marked))
            env.close()

    def test_exploration_no_review(self):
        for d in ('AVDN','CityNav'):
            result,trace,*_=self.execute(d,[
                dict(action='explore',mode='offset',turn_deg=90,step_scale=.5,reason='new region'),
                dict(action='give_up',reason='no evidence')])
            self.assertEqual(len(trace['trajectory']),2)
            self.assertFalse(any(e['kind']=='review_created' for e in trace['events']))
            np.testing.assert_allclose(result['actual_stop_xy_m'],[250,200])
    def test_final_stops_without_move_and_aux_uses_refined_center(self):
        for d in ('AVDN','CityNav'):
            actions=[
                dict(action='locate',target='auxiliary',view_id='r00-main',bbox=[350,350,550,550],reason='reference'),
                dict(action='review',verdict='confirm',view_id='r00-review',bbox=[550,550,650,650],reason='refined object'),
                dict(action='locate',target='final',view_id='r01-main',bbox=[300,300,600,600],reason='plaza'),
                dict(action='review',verdict='confirm',view_id='r01-review',bbox=[300,300,500,500],reason='confirmed')]
            result,trace,model,m,ep,out=self.execute(d,actions)
            self.assertEqual(result['termination'],'confirmed_final');self.assertEqual(len(trace['trajectory']),2)
            review=[e for e in trace['events'] if e['kind']=='review_result'][0]
            np.testing.assert_allclose(trace['trajectory'][1]['xy_m'],review['refined']['center_xy_m'])
            self.assertNotEqual(trace['trajectory'][1]['xy_m'],review['candidate']['center_xy_m'])
            self.assertNotEqual(result['actual_stop_xy_m'],result['final']['center_xy_m'])
            self.assertEqual(result['model_requests'],4)
    def test_rejected_review_no_fallback(self):
        for d in ('AVDN','CityNav'):
            result,trace,*_=self.execute(d,[
                dict(action='locate',target='final',view_id='r00-main',bbox=[350,350,550,550],reason='candidate'),
                dict(action='review',verdict='reject',reason='shadow not roof'),
                dict(action='give_up',reason='no other evidence')])
            self.assertIsNone(result['final']);self.assertEqual(len(trace['trajectory']),1)
            self.assertEqual(trace['memory']['candidates'][0]['status'],'reject')

    def test_explicit_xyxy_metadata_preserves_box_and_mandatory_review(self):
        for d in ('AVDN','CityNav'):
            with self.subTest(dataset=d):
                proposed=dict(action='locate',target='final',view_id='r00-main',
                    bbox=[350,350,550,550],bbox_format='xyxy',reason='visible candidate')
                confirm=dict(action='review',verdict='confirm',view_id='r00-review',
                    bbox=[300,300,600,600],bbox_format='xyxy',reason='refined after review')
                result,trace,model,m,*_=self.execute(d,[proposed,confirm])
                self.assertEqual(result['termination'],'confirmed_final')
                self.assertEqual(model.cursor,2)
                self.assertEqual(result['final']['bbox'],confirm['bbox'])
                self.assertEqual(len(trace['trajectory']),1)
                self.assertEqual(sum(e['kind']=='format_normalization' for e in trace['events']),2)
                self.assertTrue(any(e['kind']=='review_created' for e in trace['events']))
                validate=m['contracts'].validate_action
                for value in ('xywh','cxcywh',None):
                    with self.assertRaises(ValueError):validate(dict(proposed,bbox_format=value))
                with self.assertRaises(ValueError):validate(confirm,review=False)
                with self.assertRaises(ValueError):validate(dict(proposed,unexpected_field=True))
    def test_invalid_move_reselected_without_clipping(self):
        for d in ('AVDN','CityNav'):
            result,trace,*_=self.execute(d,[
                dict(action='explore',mode='offset',turn_deg=0,step_scale=100,reason='too far 1'),
                dict(action='explore',mode='offset',turn_deg=0,step_scale=100,reason='too far 2'),
                dict(action='explore',mode='offset',turn_deg=0,step_scale=100,reason='too far 3'),
                dict(action='explore',mode='offset',turn_deg=0,step_scale=100,reason='too far'),
                dict(action='explore',mode='point',view_id='r00-main',point=[500,250],reason='valid new point'),
                dict(action='give_up',reason='done')])
            self.assertEqual(len(trace['trajectory']),2)
            self.assertTrue(any(e['kind']=='invalid_action' for e in trace['events']))
            np.testing.assert_allclose(result['actual_stop_xy_m'],[200,225])
    def test_limits_and_tools(self):
        for d in ('AVDN','CityNav'):
            tool=dict(action='tool',tool='sam3',view_id='r00-main',query='roof',reason='need segmentation')
            result,trace,model,*_=self.execute(d,[tool,tool,tool,dict(action='give_up',reason='tool unavailable')])
            outputs=[e['output'] for e in trace['events'] if e['kind']=='tool']
            self.assertEqual(len(outputs),2);self.assertTrue(all(x['status']=='unavailable' for x in outputs))
            self.assertTrue(any('limit' in e.get('error','') for e in trace['events']))
            # Unique directory for the second run fixture.
            for p in (self.root/d).glob('result.json'):p.unlink()
            result,trace,*_=self.execute(d,[dict(action='explore',mode='offset',turn_deg=0,step_scale=.1,reason='scan')],rounds=1)
            self.assertEqual(result['termination'],'step_limit');self.assertIsNone(result['final'])
    def test_sam_mock_mask_and_score(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m);v=env.main([200,200],0,'main')
            mask=Image.new('L',(768,768),0);mask.paste(255,(200,200,300,300));mask.save(out/'mask.png')
            sam=m['tools'].SAM3(transport=lambda payload:dict(candidates=[
                dict(view_id='main',bbox_2d=[200,200,300,300],score=.9,mask_path=str(out/'mask.png'))]))
            result=sam.candidates(v,'roof',out/'tools'/'mock.json')
            self.assertEqual(result['status'],'ok');self.assertEqual(result['candidates'][0]['score'],.9)
            overlay=env.tool_overlay(v,result['candidates'],'overlay')
            self.assertEqual(overlay['pixel_to_world'],v['pixel_to_world'])
            with Image.open(overlay['image_path']) as displayed, Image.open(v['image_path']) as raw:
                # The filled mask contains this interior point. Proposal display
                # must not turn its actual roof color into a detector color.
                self.assertEqual(displayed.getpixel((250,250)),raw.getpixel((250,250)))
                self.assertNotEqual(displayed.tobytes(),raw.tobytes())
            self.assertTrue(Path(result['candidates'][0]['mask_path']).exists());env.close()

    def test_grounder_dispatch_requires_review_and_replays_without_inference(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m)
            backend=importlib.import_module(d+'.tools.grounding')
            calls=[]
            def detect(payload):
                calls.append(payload)
                return dict(coordinate_extent=[768,768],candidates=[dict(view_id='r00-main',
                    bbox_2d=[192,192,384,384],score=.85,candidate_id='dino-1')])
            grounder=backend.GroundingDINO(transport=detect)
            actions=[dict(action='tool',tool='dino',view_id='r00-main',query='building',reason='compare candidates'),
                dict(action='locate',target='final',view_id='r00-main',bbox=[250,250,500,500],reason='match'),
                dict(action='review',verdict='reject',reason='wrong relation'),
                dict(action='give_up',reason='no supported candidate')]
            model=ScriptModel(actions,m)
            result=m['engine'].Agent(ep,env,model,m['tools'].SAM3(),out,grounders={'dino':grounder}).run()
            self.assertIsNone(result['final'])
            self.assertEqual(model.contexts[0]['capabilities']['dino'],'available')
            self.assertEqual(model.contexts[0]['capabilities']['locate_anything'],'unavailable')
            tool=[e for e in m['io'].load_json(out/'trace.json')['events'] if e['kind']=='tool'][0]
            self.assertEqual(tool['output']['candidates'][0]['bbox'],[250,250,500,500])
            self.assertIn('r00-dino1',model.successful_views)
            grounder.candidates(env.views['r00-main'],'building',out/'tools/r00-1.json')
            self.assertEqual(len(calls),1)
            with self.assertRaises(ValueError):
                grounder.candidates(env.views['r00-main'],'car',out/'tools/r00-1.json')
            env.close()

    def test_grounder_invalid_cross_view_and_empty_are_distinct(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m);v=env.main([200,200],0,'main')
            backend=importlib.import_module(d+'.tools.grounding')
            responses=[dict(candidates=[]),dict(error='inference failed'),dict(candidates=[
                dict(view_id='other',bbox_2d=[20,20,30,30])]),dict(candidates=[dict(bbox_2d=[20,20,float('nan'),30])])]
            statuses=[]
            for i,raw in enumerate(responses):
                tool=backend.GroundingDINO(transport=lambda payload,r=raw:r)
                statuses.append(tool.candidates(v,'building',out/f'tools/check{i}.json')['status'])
            self.assertEqual(statuses,['ok','error','error','error'])
            env.close()

    def test_observation_grounder_is_bounded_and_does_not_auto_confirm(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m)
            backend=importlib.import_module(d+'.tools.grounding')
            detector=backend.GroundingDINO(transport=lambda payload:dict(candidates=[
                dict(view_id=payload['views'][0]['view_id'],bbox_2d=[200,200,400,400])]))
            model=ScriptModel([dict(action='give_up',reason='candidate lacks required relation')],m)
            result=m['engine'].Agent(ep,env,model,m['tools'].SAM3(),out,
                grounders={'dino':detector},observation_grounder='dino').run()
            self.assertIsNone(result['final']);self.assertEqual(model.cursor,1)
            self.assertEqual(model.contexts[0]['optional_tools_remaining'],1)
            self.assertEqual(model.contexts[0]['feedback'][0]['tool_result']['backend'],'dino')
            self.assertIn('public_geometry',model.contexts[0]['feedback'][0]['tool_result']['candidates'][0])
            self.assertEqual(len(result['tool_calls']),1)
            env.close()

    def test_category_proposals_keep_full_selection_context(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m)
            ep['initialization']['final_target']='A white car near a fictional library'
            sent=[]
            def detect(payload):
                sent.append(payload)
                return dict(candidates=[])
            model=ScriptModel([dict(action='give_up',reason='No supported identification')],m)
            tool=m['tools'].SAM3(transport=detect)
            result=m['engine'].Agent(ep,env,model,tool,out,
                observation_grounder='sam3',observation_query_mode='category').run()
            self.assertEqual(sent[0]['concepts'][0]['text'],'car')
            self.assertEqual(model.contexts[0]['initialization']['final_target'],ep['initialization']['final_target'])
            self.assertEqual(model.contexts[0]['optional_tools_remaining'],1)
            self.assertIsNone(result['final'])
            self.assertEqual(result['observation_query_mode'],'category')
            env.close()

    def test_locate_anything_normalized_boxes_and_malformed_output(self):
        for d in ('AVDN','CityNav'):
            parser=importlib.import_module(d+'.tools.locate_anything_service').boxes_from_answer
            self.assertEqual(parser('<box><10><20><300><400></box>'),[[10,20,300,400]])
            self.assertEqual(parser('None'),[])
            self.assertEqual(parser('<ref>roof</ref>None<|im_end|>'),[])
            self.assertEqual(parser('<ref>roof</ref><box>None</box><|im_end|>'),[])
            with self.assertRaises(ValueError):parser('I cannot parse this image')
            with self.assertRaises(ValueError):parser('<box><10><20><1001><400></box>')
            with self.assertRaises(ValueError):parser('<box><300><20><10><400></box>')

    def test_review_public_landmarks_use_displayed_parent_not_current_position_view(self):
        m=self.modules('CityNav');ep,env,out=self.fixture('CityNav',m)
        agent=m['engine'].Agent(ep,env,ScriptModel([],m),m['tools'].SAM3(),out)
        old=env.main([260,200],90,'old-main')
        agent.current_main=env.main([200,200],0,'new-main')
        context=agent.context('review',[],0,displayed_main=old)
        self.assertEqual(context['navigation']['current_main_view_id'],'old-main')
        self.assertAlmostEqual(context['navigation']['main_image_up_heading_deg'],90)
        np.testing.assert_allclose(context['navigation']['landmarks'][0]['center_image_point'],[500,500])
        self.assertEqual(context['position_xy_m'],[200,200])
        env.close()

    def test_model_never_receives_internal_pixel_coordinates(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m);mod=m['model']
            view=env.main([200,200],0,'main')
            candidate=env.locate(view,[300,300,600,600])

            response=dict(status=200,text='{"action":"give_up","reason":"done"}',usage=dict(input_tokens=100,output_tokens=10))
            context=dict(mode='review',candidate=candidate,memory=dict(candidates=[candidate]))
            with patch.object(mod,'call_model',return_value=response) as api:
                model=mod.Model(out/'req','qwen3.5-flash','test')
                record=model.request(context,[view])
                transmitted=api.call_args.args[3][1]['content'][0]['text']
                self.assertNotIn('bbox_pixels',transmitted)
                self.assertNotIn('pixel_to_world',transmitted)
                self.assertEqual(record['context']['candidate']['bbox'],[300,300,600,600])
                self.assertIn('bbox_pixels',candidate)  # Internal geometry is retained.
            env.close()
    def test_each_image_is_labelled_and_review_box_is_in_its_frame(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m);mod=m['model']
            main=env.main([200,200],90,'main')
            candidate=env.locate(main,[200,300,500,600])
            detail=env.review(candidate,main,'detail')
            points=np.asarray(m['geometry'].normalized(detail,candidate['polygon_xy_m']))
            box=[*points.min(axis=0).tolist(),*points.max(axis=0).tolist()]
            candidate['projection_in_review']=dict(view_id='detail',bbox=box)
            response=dict(status=200,text='{"action":"review","verdict":"uncertain","reason":"ambiguous"}',usage=dict(input_tokens=100,output_tokens=10))

            with patch.object(mod,'call_model',return_value=response) as api:
                record=mod.Model(out/'requests','qwen3.5-flash','test').request(
                    dict(mode='review',candidate=candidate),[main,detail])
                content=api.call_args.args[3][1]['content']
                self.assertEqual([x['type'] for x in content],['text','text','image_url','text','image_url'])
                labels=[json.loads(content[i]['text'].removeprefix('The NEXT image is ')) for i in (1,3)]
                self.assertEqual([x['view_id'] for x in labels],['main','detail'])
                for label in labels:self.assertAlmostEqual(label['image_up_heading_deg'],90)
                self.assertEqual(labels[0]['current_unconfirmed_proposal_bbox'],candidate['bbox'])
                np.testing.assert_allclose(labels[1]['current_unconfirmed_proposal_bbox'],box)
                self.assertEqual(api.call_args.kwargs['temperature'],0)
                self.assertEqual(record['temperature'],0)
                self.assertEqual(api.call_args.args[3][0]['content'],mod.REVIEW_SYSTEM)
                self.assertEqual(record['system_prompt'],mod.REVIEW_SYSTEM)
                self.assertNotIn('reservation_cny',record)
                self.assertEqual(record['response']['usage'], response['usage'])
            env.close()

    def test_native_temperature_and_image_labels_reach_wire(self):
        for d in ('AVDN','CityNav'):
            client=importlib.import_module(d+'.tools.model')
            messages=[dict(role='user',content=[dict(type='text',text='view=main'),
                dict(type='image_url',image_url=dict(url='data:image/png;base64,abc')),
                dict(type='text',text='view=review'),
                dict(type='image_url',image_url=dict(url='data:image/png;base64,def'))])]
            with patch.object(client.requests,'post') as post:
                response=post.return_value.__enter__.return_value
                response.status_code=200;response.headers={'Content-Type':'application/json'}
                response.json.return_value=dict(output=dict(choices=[dict(message=dict(content=[dict(text='ok')]))]))
                self.assertEqual(client.call_model(client.DEFAULT_URL,'test','test',messages,
                    timeout=10,enable_thinking=False,temperature=0,response_format={'type':'json_object'})['text'],'ok')
                payload=post.call_args.kwargs['json']
                self.assertEqual(payload['parameters']['temperature'],0)
                self.assertEqual(payload['parameters']['response_format'],{'type':'json_object'})
                parts=payload['input']['messages'][0]['content']
                self.assertEqual([list(x)[0] for x in parts],['text','image','text','image'])

    def test_gt_isolation_and_thresholds(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m)
            ep['goal_xy_m']=[1,2]
            with self.assertRaises(ValueError):m['contracts'].public_episode(ep)
            ep.pop('goal_xy_m');ep['agent_input']['target_id']='secret'
            with self.assertRaises(ValueError):m['contracts'].public_episode(ep)
            self.assertAlmostEqual(m['evaluate'].iou([[0,0],[2,0],[2,2],[0,2]],[[0,0],[2,0],[2,2],[0,2]]),1)
            c=dict(center_xy_m=[20,0])
            self.assertTrue(m['evaluate'].candidate_correct(c,'CityNav',dict(goal_xy_m=[0,0]),{}))
            c['center_xy_m']=[20.001,0]
            self.assertFalse(m['evaluate'].candidate_correct(c,'CityNav',dict(goal_xy_m=[0,0]),{}));env.close()
    def test_cached_resume_request_limit_and_pending_crash(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m);mod=m['model']
            context=dict(mode='decision')
            response=dict(status=200,text='{"action":"give_up","reason":"done"}',usage=dict(input_tokens=100,output_tokens=10))
            with patch.object(mod,'call_model',return_value=response) as call:
                one=mod.Model(out/'req','unlisted-model','test',max_requests=1)
                self.assertEqual(one.request(context,[])['status'],'ok')
                with self.assertRaisesRegex(mod.Halt,'model_request_limit'):one.request(context,[])
                two=mod.Model(out/'req','unlisted-model','test')
                two.request(context,[])
                self.assertEqual(call.call_count,1)
                record=m['io'].load_json(out/'req/000.json');record['status']='pending'
                m['io'].save_json(out/'req/000.json',record)
                three=mod.Model(out/'req','unlisted-model','test')
                with self.assertRaisesRegex(mod.Halt,'uncertain_request'):three.request(context,[])
                self.assertEqual(call.call_count,1)
                self.assertFalse((out/'ledger.json').exists())
            env.close()

    def test_api_failures_remain_cached_without_cost_accounting(self):
        for dataset in ('AVDN','CityNav'):
            m=self.modules(dataset);mod=m['model']
            for name, response, error in [('http', dict(status=403,text=None,usage=None), None),
                                          ('timeout', None, TimeoutError('offline timeout'))]:
                directory=self.root/dataset/name
                with patch.object(mod,'call_model',return_value=response,side_effect=error) as api:
                    first=mod.Model(directory,'unlisted-model','test').request({'mode':'decision'},[])
                    self.assertEqual(first['status'],'api_error')
                    again=mod.Model(directory,'unlisted-model','test').request({'mode':'decision'},[])
                    self.assertEqual(first,again)
                    self.assertEqual(api.call_count,1)
                    self.assertNotIn('reservation',first)

    def test_engine_interrupt_replays_cached_requests(self):
        for d in ('AVDN','CityNav'):
            m=self.modules(d);ep,env,out=self.fixture(d,m);mod=m['model']

            script=[dict(action='explore',mode='offset',turn_deg=90,step_scale=.2,reason='observe east'),
                    dict(action='give_up',reason='no further evidence')]
            def response(*args,**kwargs):
                return dict(status=200,text=json.dumps(script.pop(0)),usage=dict(input_tokens=100,output_tokens=50))
            original=mod.Model.request
            calls=[0]
            def interrupt_request(*args,**kwargs):
                calls[0]+=1
                if calls[0]>1:raise mod.Halt('synthetic_interrupt')
                return original(*args,**kwargs)
            with patch.object(mod,'call_model',side_effect=response) as api,patch.object(mod.Model,'request',new=interrupt_request):
                model=mod.Model(out/'requests','qwen3.5-flash','test')
                first=m['engine'].Agent(ep,env,model,m['tools'].SAM3(),out).run()
                self.assertEqual(first['termination'],'synthetic_interrupt')
                self.assertFalse(first['completed']);self.assertEqual(api.call_count,1)

            env.close();env=m['environment'].Environment(ep['environment'],out/'views')
            with patch.object(mod,'call_model',side_effect=response) as api:
                model=mod.Model(out/'requests','qwen3.5-flash','test')
                second=m['engine'].Agent(ep,env,model,m['tools'].SAM3(),out).run()
                self.assertEqual(second['termination'],'give_up')
                self.assertEqual(api.call_count,1);self.assertEqual(second['model_requests'],2)

                np.testing.assert_allclose(second['actual_stop_xy_m'],[220,200])
            env.close()
    def test_evaluation_counts_only_observed_views_and_missing_prediction(self):
        for d in ('AVDN','CityNav'):
            result,trace,model,m,ep,out=self.execute(d,[
                dict(action='explore',mode='offset',turn_deg=90,step_scale=1.,reason='east region')],rounds=1)
            # New position is at the goal, but no image was observed there before the limit.
            gt=dict(episode_id=ep['episode_id'],dataset=d,goal_xy_m=[300,200],
                    goal_polygon_xy_m=[[295,195],[305,195],[305,205],[295,205]],
                    evaluation_polygon_xy_m=[[290,190],[310,190],[310,210],[290,210]])
            folder=self.root/(d+'-eval');folder.mkdir()
            m['io'].write_jsonl(folder/'episodes.jsonl',[ep])
            m['io'].write_jsonl(folder/'evaluator_only.jsonl',[gt])
            import shutil
            shutil.copytree(out,folder/'episodes'/ep['episode_id'])
            summary=m['evaluate'].evaluate(folder)
            row=m['io'].load_json(folder/'evaluation.json')['rows'][0]
            self.assertFalse(row['center_covered']);self.assertFalse(row['success'])
            self.assertEqual(row['actual_stop_distance_m'],0.)
            self.assertEqual(summary['valid_predictions'],0)
            self.assertEqual(row['failure_category'],'5_limit_or_interface')
            self.assertEqual(row['geometric_failure_category'],'1_not_covered')


    def test_public_navigation_bearings_and_heading_up_views(self):
        from CityNav.tools.context import navigation_context, heading_of_view
        m=self.modules('CityNav');ep,env,out=self.fixture('CityNav',m)
        ep['agent_input']['named_landmarks']=[dict(id='lm-fiction',name='library',type='Building',
            xy_m=[140.,280.],bbox_xy_m=[130,270,150,290])]
        v=env.main([200,200],90,'east-view')
        n=navigation_context(ep['agent_input']['named_landmarks'],[200,200],90,v,[],env.spec,[])
        ref=n['landmarks'][0]
        self.assertAlmostEqual(ref['distance_to_center_m'],100)
        self.assertAlmostEqual(ref['bearing_deg'],323.13,places=2)
        self.assertAlmostEqual(ref['turn_from_current_heading_deg'],-126.87,places=2)
        self.assertFalse(ref['center_in_current_view'])
        self.assertFalse(ref['center_previously_observed'])
        self.assertAlmostEqual(heading_of_view(v),90)
        v2=env.main([140,280],270,'arrived')
        n2=navigation_context(ep['agent_input']['named_landmarks'],[140,280],270,v2,[v2],env.spec,[])
        ref=n2['landmarks'][0]
        self.assertEqual(ref['distance_to_center_m'],0)
        self.assertIsNone(ref['bearing_deg']);self.assertIsNone(ref['turn_from_current_heading_deg'])
        self.assertTrue(ref['center_in_current_view']);self.assertTrue(ref['center_previously_observed'])
        np.testing.assert_allclose(ref['center_image_point'],[500,500])
        self.assertEqual(ep['agent_input']['named_landmarks'][0]['xy_m'],[140.,280.])
        env.close()

    def test_public_extent_can_overlap_without_center_and_without_target(self):
        from CityNav.tools.context import navigation_context
        m=self.modules('CityNav');ep,env,out=self.fixture('CityNav',m)
        v=env.main([200,200],45,'main')
        landmarks=[dict(id='road',name='fictional road',type='TrafficRoad',
                        xy_m=[80.,200.],bbox_xy_m=[70,195,180,205])]
        n=navigation_context(landmarks,[200,200],45,v,[],env.spec,[])
        self.assertTrue(n['landmarks'][0]['extent_overlaps_current_view'])
        self.assertFalse(n['landmarks'][0]['center_in_current_view'])
        self.assertAlmostEqual(n['landmarks'][0]['distance_to_extent_m'],20)
        self.assertNotIn('goal_xy_m',json.dumps(n))
        env.close()

    def test_city_landmark_context_updates_after_real_move(self):
        result,trace,model,m,ep,out=self.execute('CityNav',[
            dict(action='explore',mode='landmark',landmark_id='lm-1',reason='inspect reference'),
            dict(action='give_up',reason='no visual match')])
        before,after=model.contexts
        self.assertEqual(before['allowed_actions'],['explore','locate','tool','give_up'])
        self.assertEqual(before['navigation']['landmarks'][0]['distance_to_center_m'],60)
        self.assertEqual(after['navigation']['landmarks'][0]['distance_to_center_m'],0)
        self.assertEqual(after['navigation']['last_executed_moves'][-1]['source'],'explore_landmark')
        self.assertIsNone(result['final'])  # Visiting public coordinates cannot confirm a target.


    def test_city_redundant_landmark_view_label_does_not_block_movement(self):
        from CityNav.loop import validate_action
        raw=dict(action='explore',mode='landmark',landmark_id='lm-1',view_id='r00-main',reason='known reference')
        canonical=validate_action(raw)
        self.assertNotIn('view_id',canonical)
        self.assertIn('view_id',raw)  # Preserve raw evidence.
        with self.assertRaises(ValueError):
            validate_action(dict(raw,point=[1,2]))  # Conflicting coordinates are not silently ignored.
        result,trace,*_=self.execute('CityNav',[raw,dict(action='give_up',reason='no evidence')])
        np.testing.assert_allclose(result['actual_stop_xy_m'],[260,200])
        self.assertTrue(any(e['kind']=='format_normalization' for e in trace['events']))
        self.assertIsNone(result['final'])


    def test_reference_extent_probes_are_public_unobserved_positions(self):
        from CityNav.tools.context import navigation_context
        from shapely.geometry import Polygon, Point
        m=self.modules('CityNav');ep,env,out=self.fixture('CityNav',m)
        view=env.main([200,200],0,'main')
        landmarks=[dict(id='wide',name='fictional campus',type='Building',xy_m=[200,200],bbox_xy_m=[80,160,320,240])]
        before=set(env.views)
        n=navigation_context(landmarks,[200,200],0,view,[],env.spec,[])
        self.assertEqual(set(env.views),before)  # No remote RGB preview is generated.
        ref=n['landmarks'][0]
        self.assertTrue(ref['center_in_current_view'])
        self.assertLess(ref['extent_observed_fraction'],.5)
        self.assertEqual(len(ref['unobserved_extent_probes']),2)
        seen=Polygon(view['world_corners_xy_m'])
        for suggestion in ref['unobserved_extent_probes']:
            p=suggestion['xy_m']
            self.assertTrue(80<p[0]<320 and 160<p[1]<240)
            self.assertFalse(seen.covers(Point(p)))
            computed,_=m['geometry'].offset([200,200],0,suggestion['turn_deg'],suggestion['step_scale'],100)
            np.testing.assert_allclose(computed,p,atol=.001)
            env.check_position(computed)
        covered=env.render(m['geometry'].box_points([80,240,320,160]),'whole','test')
        n2=navigation_context(landmarks,[200,200],0,view,[covered],env.spec,[])
        self.assertEqual(n2['landmarks'][0]['extent_observed_fraction'],1.)
        self.assertEqual(n2['landmarks'][0]['unobserved_extent_probes'],[])
        env.close()


    def test_grounding_wire_keeps_all_proposals_without_transport_duplicates(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds):
                from importlib import import_module
                compact=import_module(ds+'.tools.model').model_context
                proposals=[dict(candidate_id=str(i),view_id='v',bbox=[i,1,i+1,5],
                    score=.5,mask_path='local-mask.png',public_geometry={'bearing_deg':i})
                    for i in range(24)]
                original=dict(backend='sam3',status='ok',candidates=proposals,
                    raw={'candidates':[{'bbox_2d':[1,2,3,4]}]},
                    input={'views':[{'image_path':'local.png'}]},signature='transport')
                result=compact({'tool_result':original})['tool_result']
                self.assertEqual(len(result['candidates']),24)
                self.assertEqual(result['candidates'][-1]['bbox'],[23,1,24,5])
                self.assertEqual(result['candidates'][-1]['public_geometry'],{'bearing_deg':23})
                self.assertNotIn('mask_path',result['candidates'][0])
                self.assertNotIn('raw',result);self.assertNotIn('input',result)
                self.assertIn('mask_path',original['candidates'][0])
                self.assertIn('raw',original)  # Full evidence remains unchanged in journals.


    def test_explicit_tool_alias_executes_same_zoom_and_preserves_raw_action(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds):
                m=self.modules(ds)
                raw=dict(action='zoom',view_id='r00-main',bbox=[300,300,600,600],reason='inspect')
                canonical=m['contracts'].validate_action(raw)
                self.assertEqual(canonical,dict(raw,action='tool',tool='zoom'))
                self.assertEqual(raw['action'],'zoom')
                for bad in [dict(raw,tool='dino'),dict(raw,point=[1,2])]:
                    with self.assertRaises(ValueError):m['contracts'].validate_action(bad)
                with self.assertRaises(ValueError):m['contracts'].validate_action(raw,review=True)
                result,trace,*_=self.execute(ds,[raw,dict(action='give_up',reason='ambiguous')])
                self.assertEqual(len(trace['trajectory']),1)
                self.assertIsNone(result['final'])
                self.assertTrue(any(e['kind']=='format_normalization' for e in trace['events']))
                self.assertTrue(any(e['kind']=='tool' and e['input']['tool']=='zoom' for e in trace['events']))


    def test_thinking_tokens_are_sent_and_part_of_replay_identity(self):
        for ds in ('AVDN','CityNav'):
            m=self.modules(ds);mod=m['model'];root=self.root/ds;root.mkdir()

            response=dict(status=200,text='{"action":"give_up","reason":"ambiguous"}',usage=dict(input_tokens=100,output_tokens=1100))
            with patch.object(mod,'call_model',return_value=response) as api:
                runner=mod.Model(root/'requests','qwen3.8-flash','test',enable_thinking=True,thinking_budget=1024)
                record=runner.request({'mode':'decision'},[])

                self.assertTrue(api.call_args.kwargs['enable_thinking'])
                self.assertEqual(api.call_args.kwargs['thinking_budget'],1024)
                self.assertEqual(api.call_args.kwargs['response_format'],{'type':'json_object'})
                self.assertEqual(record['thinking_budget'],1024)
                with self.assertRaisesRegex(mod.Halt,'resume_input_mismatch'):
                    mod.Model(root/'requests','qwen3.8-flash','test',enable_thinking=False).request({'mode':'decision'},[])
                self.assertEqual(api.call_count,1)



    def test_uncertain_review_box_cannot_become_fallback_localization(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds):
                actions=[dict(action='locate',target='final',view_id='r00-main',bbox=[400,400,600,600],reason='candidate'),
                         dict(action='review',verdict='uncertain',view_id='r00-review',bbox=[100,100,800,800],reason='alternatives'),
                         dict(action='give_up',reason='insufficient evidence')]
                result,trace,*_=self.execute(ds,actions)
                self.assertIsNone(result['final'])
                self.assertEqual(len(trace['trajectory']),1)
                self.assertTrue(any(e['kind']=='format_normalization' and e['raw_action']['verdict']=='uncertain' for e in trace['events']))
                self.assertFalse(any(e['kind']=='final_localization' for e in trace['events']))


    def test_candidate_original_heading_frame_is_independent_of_prior_and_current_pose(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds):
                from importlib import import_module
                frame=import_module(ds+'.tools.context').candidate_context
                ep={'agent_input':{'initial_pose':{'xy_m':[200,200],'heading_deg':90},'named_landmarks':[]},
                    'initialization':{'direction':345}}
                a=frame([230,240],ep,[400,500],270)
                self.assertAlmostEqual(a['initial_forward_m'],30)
                self.assertAlmostEqual(a['initial_right_m'],-40)
                ep['initialization']['direction']=10
                b=frame([230,240],ep,[100,200],0)
                self.assertEqual((a['initial_forward_m'],a['initial_right_m']),
                                 (b['initial_forward_m'],b['initial_right_m']))
                ep['agent_input']['initial_pose']['heading_deg']=None
                c=frame([230,240],ep,[200,200],180)
                self.assertIsNone(c['initial_forward_m']);self.assertIsNone(c['initial_right_m'])


    def test_stationary_review_reuses_view_and_successful_proposals(self):
        for ds in ('AVDN','CityNav'):
            for first_error in (False,True):
                with self.subTest(dataset=ds,first_error=first_error):
                    m=self.modules(ds);ep,env,out=self.fixture(ds,m)
                    out=out/('error' if first_error else 'ok');out.mkdir()
                    calls=[]
                    def detect(payload):
                        calls.append(payload)
                        if first_error and len(calls)==1:return {'error':'transient'}
                        return {'candidates':[]}
                    detector=importlib.import_module(ds+'.tools.grounding').GroundingDINO(transport=detect)
                    script=[dict(action='locate',target='final',view_id='r00-main',bbox=[400,400,600,600],reason='check'),
                            dict(action='review',verdict='uncertain',reason='two alternatives'),
                            dict(action='give_up',reason='no discriminating evidence')]
                    model=ScriptModel(script,m)
                    result=m['engine'].Agent(ep,env,model,m['tools'].SAM3(),out,grounders={'dino':detector},observation_grounder='dino').run()
                    self.assertIsNone(result['final']);self.assertEqual(len(calls),2 if first_error else 1)
                    self.assertEqual(len([v for v in env.views.values() if v['source']=='main']),1)
                    trace=m['io'].load_json(out/'trace.json')
                    self.assertEqual(len(trace['memory']['visits']),1)
                    self.assertEqual(len(result['tool_calls']),2 if first_error else 1)
                    self.assertTrue(any('observation' in f for f in model.contexts[-1]['feedback']))
                    self.assertEqual(model.contexts[-1]['optional_tools_remaining'],1 if first_error else 2)
                    moved=env.main([210,200],0,'moved-main')
                    self.assertEqual(moved['view_id'],'moved-main')
                    rotated=env.main([210,200],90,'rotated-main')
                    self.assertEqual(rotated['view_id'],'rotated-main')
                    env.close()


    def test_stationary_cached_observation_resumes_without_repeating_detection_or_request(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds):
                m=self.modules(ds);ep,env,out=self.fixture(ds,m);mod=m['model']

                detections=[]
                def detect(payload):detections.append(payload);return {'candidates':[]}
                detector=importlib.import_module(ds+'.tools.grounding').GroundingDINO(transport=detect)
                actions=[dict(action='locate',target='final',view_id='r00-main',bbox=[400,400,600,600],reason='candidate'),
                         dict(action='review',verdict='uncertain',reason='ambiguous'),
                         dict(action='give_up',reason='no new evidence')]
                def respond(*args,**kwargs):return dict(status=200,text=json.dumps(actions.pop(0)),usage=dict(input_tokens=100,output_tokens=20))
                original=mod.Model.request;count=[0]
                def stop_third(*args,**kwargs):
                    count[0]+=1
                    if count[0]>2:raise mod.Halt('synthetic_interrupt')
                    return original(*args,**kwargs)
                with patch.object(mod,'call_model',side_effect=respond) as api,patch.object(mod.Model,'request',new=stop_third):
                    result=m['engine'].Agent(ep,env,mod.Model(out/'requests','qwen3.8-flash','test'),m['tools'].SAM3(),out,grounders={'dino':detector},observation_grounder='dino').run()
                    self.assertEqual(result['termination'],'synthetic_interrupt');self.assertEqual(api.call_count,2)
                env.close();env=m['environment'].Environment(ep['environment'],out/'views')
                with patch.object(mod,'call_model',side_effect=respond) as api:
                    result=m['engine'].Agent(ep,env,mod.Model(out/'requests','qwen3.8-flash','test'),m['tools'].SAM3(),out,grounders={'dino':detector},observation_grounder='dino').run()
                    self.assertEqual(result['termination'],'give_up');self.assertEqual(api.call_count,1)
                self.assertEqual(len(detections),1)

                self.assertIsNone(result['final']);env.close()

if __name__=='__main__':unittest.main()

