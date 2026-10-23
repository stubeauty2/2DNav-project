"""Native-protocol, abstention, replay and full review integration without inference."""
import base64
import copy
import importlib
import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL = 'OneJev1.6-0.8B'


class Response:
    def __init__(self, data, status=200):
        self.data, self.status_code = data, status
        self.text = json.dumps(data)

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError('test error')


def answer(p=.8, choice='confirm'):
    options = {'confirm': (1-p)/2, 'reject': (1-p)/2, 'uncertain': (1-p)/2}
    options[choice] = p
    return dict(model=MODEL, answers={'review': dict(type='choice', choice=choice,
        probabilities=options, confidence=(p-1/3)/(2/3))}, usage={'input_tokens': 10, 'output_tokens': 1})


def health(model=MODEL, prompt='v1', manifest=None):
    data=dict(status='ok', model=model, fork_mode='block', prompt_version=prompt)
    if manifest is not None:
        from AVDN.tools.runtime import digest
        data.update(service_manifest_sha256=digest(manifest),checkpoint_revision=manifest.get('checkpoint_revision'))
    return Response(data)


class JevProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.modules = [importlib.import_module(d+'.tools.jev') for d in ('AVDN', 'CityNav')]
        self.views = []
        for i in range(2):
            path = self.root/f'image{i}.png'
            Image.new('RGB', (30, 30), ('grey' if i == 0 else 'yellow')).save(path)
            self.views.append(dict(view_id=f'v{i}', image_path=str(path), source='main' if i == 0 else 'mandatory_review',
                                   world_corners_xy_m=[[0,30],[30,30],[30,0],[0,0]]))
        self.context = dict(mode='review', instruction_raw='the pair of white buildings',
            candidate=dict(view_id='v0', bbox=[100, 100, 300, 300], target='final'),
            memory=dict(recent_decisions=['untrusted earlier guess'], confirmed_auxiliary=[]))

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_images_public_state_and_no_base64_journal(self):
        for module in self.modules:
            wire, logged, assets = module.build_review_payload(self.context, self.views, MODEL)
            self.assertEqual(set(wire), {'model', 'state', 'questions', 'media'})
            self.assertEqual(set(wire['questions']['review']['criteria']), {'confirm','reject','uncertain'})
            self.assertEqual(wire['state']['context']['memory'], {'confirmed_auxiliary':[]})
            for i, media in enumerate(wire['media']):
                self.assertEqual(base64.b64decode(media['data'].split(',', 1)[1]), Path(self.views[i]['image_path']).read_bytes())
                self.assertEqual(assets[i]['view_id'], self.views[i]['view_id'])
            self.assertNotIn('base64', json.dumps(logged))
            bad = copy.deepcopy(self.context);bad['nested']={'goal_xy_m':[1,2]}
            with self.assertRaises(ValueError):
                module.build_review_payload(bad, self.views, MODEL)

    def test_threshold_is_max_probability_not_native_confidence(self):
        for module in self.modules:
            self.assertEqual(module.validate_answer(answer(.7),MODEL)['verdict'],'confirm')
            self.assertLess(module.validate_answer(answer(.7),MODEL)['confidence'],.7)
            self.assertEqual(module.validate_answer(answer(.699),MODEL)['verdict'],'uncertain')
            self.assertEqual(module.validate_answer(answer(.8,'reject'),MODEL)['verdict'],'reject')
            self.assertEqual(module.validate_answer(answer(.8,'uncertain'),MODEL)['verdict'],'uncertain')

    def test_invalid_probability_protocol_is_rejected(self):
        malformed=[]
        for value in (float('nan'), float('inf'), -.1, 1.1, True):
            raw=answer();raw['answers']['review']['probabilities']['confirm']=value;malformed.append(raw)
        raw=answer();raw['answers']['review']['probabilities']['unexpected']=.0;malformed.append(raw)
        raw=answer();raw['answers']['review']['probabilities']['reject']=.4;malformed.append(raw)
        raw=answer();raw['answers']['review']['choice']='reject';malformed.append(raw)
        raw=answer();raw['answers']['review']['confidence']=.9;malformed.append(raw)
        raw=answer();raw['model']='different';malformed.append(raw)
        for module in self.modules:
            for raw in malformed:
                with self.subTest(module=module.__name__,raw=str(raw)):
                    with self.assertRaises(ValueError):module.validate_answer(raw,MODEL)

    def test_native_http_journal_and_offline_replay(self):
        for module in self.modules:
            output=self.root/module.__name__/'review.json'
            manifest={'served_model':MODEL,'checkpoint_revision':'test','checkpoint_sha256':'test'}
            client=module.Jev('http://127.0.0.1:8770',MODEL,service_manifest=manifest)
            with patch.object(module.requests,'get',return_value=health(manifest=manifest)) as get, patch.object(module.requests,'post',return_value=Response(answer(.7))) as post:
                record=client.verify(self.context,self.views,output)
                self.assertEqual(record['verdict'],'confirm')
                self.assertEqual(get.call_args.args[0],'http://127.0.0.1:8770/health')
                self.assertEqual(post.call_args.args[0],'http://127.0.0.1:8770/v1/systemone')
                self.assertEqual(post.call_args.kwargs['json']['model'],MODEL)
                self.assertEqual(record['raw'],answer(.7))
                self.assertIsNotNone(record['latency_ms'])
                self.assertNotIn('base64',output.read_text())
                get.reset_mock();post.reset_mock()
                self.assertEqual(client.verify(self.context,self.views,output),record)
                get.assert_not_called();post.assert_not_called()

    def test_pending_and_changed_input_do_not_reissue(self):
        module=self.modules[0];output=self.root/'pending.json';client=module.Jev('http://local',MODEL)
        with patch.object(module.requests,'get',return_value=health()), patch.object(module.requests,'post',side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):client.verify(self.context,self.views,output)
        with patch.object(module.requests,'post') as post:
            with self.assertRaisesRegex(module.Halt,'jev_uncertain_request_pending'):client.verify(self.context,self.views,output)
            changed=dict(self.context,instruction_raw='changed')
            with self.assertRaisesRegex(module.Halt,'jev_resume_input_mismatch'):client.verify(changed,self.views,output)
            post.assert_not_called()

    def test_service_and_malformed_response_are_uncertain_without_fallback(self):
        module=self.modules[0]
        for i, effect in enumerate([requests.Timeout('timeout'), Response({'bad':'answer'}), Response(answer(),503)]):
            with patch.object(module.requests,'get',return_value=health()), patch.object(module.requests,'post',side_effect=effect if isinstance(effect,Exception) else None,return_value=effect):
                result=module.Jev('http://local',MODEL).verify(self.context,self.views,self.root/f'error{i}.json')
                self.assertEqual(result['status'],'error');self.assertEqual(result['verdict'],'uncertain')
                self.assertNotIn('bbox',module.review_action(result,self.context['candidate']))

    def test_identity_frozen_and_wrong_manifest_rejected(self):
        module=self.modules[0]
        with self.assertRaises(ValueError):module.Jev('http://local',MODEL,service_manifest={'served_model':'other'})
        client=module.Jev('http://local',MODEL)
        with patch.object(module.requests,'get',return_value=health('wrong')), patch.object(module.requests,'post') as post:
            with self.assertRaisesRegex(module.Halt,'jev_model_identity_mismatch'):client.verify(self.context,self.views,self.root/'wrong.json')
            post.assert_not_called()
        with patch.object(module.requests,'get',return_value=health()), patch.object(module.requests,'post',return_value=Response(answer())):
            client.verify(self.context,self.views,self.root/'good.json')
        with patch.object(module.requests,'get',return_value=health(prompt='changed')), patch.object(module.requests,'post') as post:
            with self.assertRaisesRegex(module.Halt,'jev_service_identity_changed'):client.verify(self.context,self.views,self.root/'changed.json')
            post.assert_not_called()

    def test_frozen_provenance_checked_against_actual_service(self):
        module=self.modules[0]
        manifest={'served_model':MODEL,'checkpoint_revision':'commit-a'}
        client=module.Jev('http://local',MODEL,service_manifest=manifest)
        with patch.object(module.requests,'get',return_value=health()),patch.object(module.requests,'post') as post:
            with self.assertRaisesRegex(module.Halt,'jev_service_manifest_mismatch'):
                client.verify(self.context,self.views,self.root/'provenance.json')
            post.assert_not_called()

    def test_manifest_file_binds_canonical_content_not_file_format(self):
        module=self.modules[0]
        manifest={'served_model':MODEL,'checkpoint_revision':'commit-a'}
        manifest_path=self.root/'manifest.json'
        manifest_path.write_text(json.dumps(manifest,indent=7),encoding='utf-8')
        client=module.Jev('http://local',MODEL,service_manifest=manifest_path)
        with patch.object(module.requests,'get',return_value=health(manifest=manifest)),patch.object(module.requests,'post',return_value=Response(answer())):
            record=client.verify(self.context,self.views,self.root/'path_manifest'/'review.json')
        self.assertEqual(record['status'],'ok')


class NoGrounder:
    def available(self):return False


class DecisionModel:
    def __init__(self, give_up_after=False):
        self.cursor=0;self.successful_views=set();self.give_up_after=give_up_after

    def request(self, context, views):
        if context['mode']=='review':raise AssertionError('Jev must never call MLLM review')
        self.cursor+=1;self.successful_views.update(v['view_id'] for v in views)
        action=dict(action='locate',target='final',view_id=views[0]['view_id'],bbox=[300,300,500,500],reason='proposal')
        if self.cursor>1 and self.give_up_after:action=dict(action='give_up',reason='still uncertain')
        return dict(status='ok',response=dict(text=json.dumps(action)))


class AgentIntegrationTests(unittest.TestCase):
    def setup_agent(self, ds, root, backend='jev'):
        loop=importlib.import_module(ds+'.loop');views=importlib.import_module(ds+'.tools.views')
        jev=importlib.import_module(ds+'.tools.jev')
        raster=root/'map.png';Image.new('RGB',(512,512),'grey').save(raster)
        spec=dict(image_path=str(raster),image_size=[512,512],world_to_raster=[[1,0,0],[0,1,0],[0,0,1]],main_width_m=100.,main_height_m=100.)
        agent_input=dict(dataset=ds,setting='visual',instruction_raw='the white building next to the road',turns=[],
            initial_pose=dict(xy_m=[256.,256.],heading_deg=0.),coordinate_frame='east north',named_landmarks=[],views=[])
        episode=dict(episode_id='test',agent_input=agent_input,environment=spec,
                     initialization=dict(direction=0.,final_target='white building next to road',auxiliary=[]))
        env=views.Environment(spec,root/'views');model=DecisionModel(give_up_after=True)
        client=jev.Jev('http://local',MODEL)
        try:
            agent=loop.Agent(episode,env,model,NoGrounder(),root,max_rounds=2,review_backend=backend,jev=client if backend=='jev' else None)
        except BaseException:
            env.close();raise
        return agent,env,jev

    def test_full_agent_jev_confirmation_preserves_box_and_does_not_move(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as td:
                agent,env,module=self.setup_agent(ds,Path(td))
                try:
                    with patch.object(module.requests,'get',return_value=health()),patch.object(module.requests,'post',return_value=Response(answer(.9))) as post:
                        result=agent.run()
                    self.assertEqual(result['termination'],'confirmed_final')
                    self.assertEqual(result['final']['bbox'],[300,300,500,500])
                    self.assertEqual(result['final']['view_id'],'r00-main')
                    self.assertEqual(result['actual_stop_xy_m'],[256.,256.])
                    self.assertEqual(result['model_requests'],1);self.assertEqual(result['jev_requests'],1)
                    self.assertEqual(result['jev_error_count'],0)
                    payload=post.call_args.kwargs['json']
                    self.assertEqual(len(payload['media']),2)
                    self.assertIn('yellow',payload['state']['context']['views'][1]['annotation'])
                finally:env.close()

    def test_failed_jev_review_preserves_uncertainty_and_continues_planning(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as td:
                agent,env,module=self.setup_agent(ds,Path(td))
                try:
                    with patch.object(module.requests,'get',side_effect=requests.ConnectionError('offline')):
                        result=agent.run()
                    self.assertEqual(result['termination'],'give_up');self.assertIsNone(result['final'])
                    self.assertEqual(result['model_requests'],2)
                    self.assertEqual(result['jev_error_count'],1)
                    self.assertEqual(result['jev_calls'][0]['status'],'error')
                    self.assertEqual(agent.memory['candidates'][0]['status'],'uncertain')
                finally:env.close()

    def test_fixed_box_controls_accept_original_zoom_coordinates(self):
        for ds in ('AVDN','CityNav'):
            for backend in ('mllm_fixed','jev'):
                with self.subTest(dataset=ds,backend=backend),tempfile.TemporaryDirectory() as td:
                    agent,env,module=self.setup_agent(ds,Path(td),backend)
                    try:
                        main=env.main([256.,256.],0.,'main');agent.current_main=main
                        zoom=env.zoom(main,[100,100,700,700],'zoom');agent.allowed.update(('main','zoom'))
                        candidate=env.locate(zoom,[200,200,400,400])
                        with patch.object(module.requests,'get',return_value=health()),patch.object(module.requests,'post',return_value=Response(answer(.9))),patch.object(agent,'ask',return_value=dict(action='review',verdict='confirm',view_id='main',bbox=[1,2,3,4],reason='would refine')) as ask:
                            agent.review(dict(target='final'),main,candidate)
                        self.assertEqual(agent.final['view_id'],'zoom');self.assertEqual(agent.final['bbox'],candidate['bbox'])
                        self.assertEqual(ask.call_count,1 if backend=='mllm_fixed' else 0)
                    finally:env.close()

    def test_default_mllm_review_still_refines(self):
        for ds in ('AVDN','CityNav'):
            with self.subTest(dataset=ds),tempfile.TemporaryDirectory() as td:
                agent,env,module=self.setup_agent(ds,Path(td),'mllm')
                try:
                    main=env.main([256.,256.],0.,'main');agent.current_main=main
                    candidate=env.locate(main,[200,200,400,400])
                    with patch.object(agent,'ask',return_value=dict(action='review',verdict='confirm',view_id='main',bbox=[300,300,550,550],reason='refined')):
                        agent.review(dict(target='final'),main,candidate)
                    self.assertEqual(agent.final['bbox'],[300,300,550,550])
                    self.assertIsNone(agent.jev)
                finally:env.close()

    def test_fixed_prompt_is_selected_and_original_prompt_unchanged(self):
        for ds in ('AVDN','CityNav'):
            model_module=importlib.import_module(ds+'.tools.model')
            prompts=importlib.import_module(ds+'.prompts')
            verifier=importlib.import_module(ds+'.tools.jev')
            self.assertTrue(prompts.REVIEW_FIXED_SYSTEM.startswith(verifier.REVIEW_QUESTION))
            self.assertIn('MUST apply to the original candidate',prompts.REVIEW_FIXED_SYSTEM)
            self.assertIn('must be refined',prompts.REVIEW_SYSTEM)
            for backend in ('mllm','mllm_fixed'):
                with self.subTest(dataset=ds,backend=backend),tempfile.TemporaryDirectory() as td:
                    root=Path(td);agent,env,_=self.setup_agent(ds,root,backend)
                    try:
                        main=env.main([256.,256.],0.,'main');agent.current_main=main
                        candidate=env.locate(main,[200,200,400,400])
                        detail=env.mark_review(env.review(candidate,main,'detail'),candidate)
                        agent.model=model_module.Model(root/'requests','test','test',enable_thinking=False)
                        response=dict(status=200,text=json.dumps(dict(action='review',verdict='uncertain',reason='insufficient')))
                        with patch.object(model_module,'call_model',return_value=response) as transport:
                            agent.ask('review',[main,detail],0,candidate=candidate)
                        messages=transport.call_args.args[3]
                        expected=prompts.REVIEW_FIXED_SYSTEM if backend=='mllm_fixed' else prompts.REVIEW_SYSTEM
                        self.assertEqual(messages[0]['content'],expected)
                        context=json.loads(messages[1]['content'][0]['text'])
                        self.assertEqual(context.get('review_policy'),'fixed_proposal' if backend=='mllm_fixed' else None)
                    finally:env.close()


if __name__=='__main__':unittest.main()
