"""Offline transport, request and response contracts for the only current modes."""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from fixtures import NavigationFixture
from andh_core.tools.model import ModelClient, ModelCallError, infer, messages, rgb_view, validate
from andh_core.tools.views import acquire
from andh_core.tools.geometry import polygon_iou, relative_pose
from andh_core.memory import TrajectoryMemory
from test_candidate_navigation import proposal


class ModelTests(NavigationFixture, unittest.TestCase):
    def test_transport_settings_and_response_content(self):
        post = Mock(return_value=SimpleNamespace(
            headers={'Content-Type': 'application/json'},
            json=lambda: {'choices': [{'message': {'content': '  recorded  '}}]}))
        client = ModelClient(api_key='offline-key', url='offline://model', model='recorded',
                             transport=SimpleNamespace(post=post))
        self.assertEqual(client.complete([], timeout=600), 'recorded')
        kwargs = post.call_args.kwargs
        self.assertEqual(kwargs['timeout'], 600)
        self.assertEqual(kwargs['json'], {'model':'recorded','temperature':0,'stream':False,'messages':[]})
        self.assertEqual(kwargs['headers']['Authorization'], 'Bearer offline-key')

    def test_transport_failures_are_explicit(self):
        responses = [SimpleNamespace(headers={'Content-Type':'text/html'}, text='unavailable'),
                     SimpleNamespace(headers={'Content-Type':'application/json'}, json=lambda: {'error':'offline'})]
        for response in responses:
            client = ModelClient(api_key='',url='',model='',transport=SimpleNamespace(post=lambda *a,**k:response))
            with self.assertRaises(ModelCallError): client.complete([])
        client = ModelClient(api_key='',url='',model='',transport=SimpleNamespace(post=Mock(side_effect=TimeoutError('offline'))))
        with self.assertRaises(ModelCallError): client.complete([])

    def test_invalid_retry_archives_every_raw_attempt(self):
        obs=acquire(self.ob,[.05,.05],0,5,'s1',1,'current',str(self.out))
        client=SimpleNamespace(complete=Mock(side_effect=['invalid JSON',proposal(obs.observation_id+':rgb')]))
        audit={}
        result=infer(client,'candidate_proposal','roof',[rgb_view(obs)],extra={'candidate_count':3},retries=1,audit=audit)
        self.assertEqual(result['decision'],'proposed')
        self.assertEqual([a['status'] for a in audit['attempts']],['parse_error','returned'])
        self.assertEqual(audit['attempts'][0]['response_text'],'invalid JSON')

    def test_source_aliases_and_auxiliary_evidence(self):
        obs=acquire(self.ob,[.05,.05],0,5,'s1',1,'current',str(self.out))
        views=[rgb_view(obs),dict(rgb_view(obs),view_id='context',selectable=False)]
        raw=json.loads(proposal(obs.observation_id))
        raw['candidates'][0]['evidence_view_ids']=[obs.observation_id,'context']
        result=validate(raw,'candidate_proposal',views,{'candidate_count':3})
        self.assertEqual(result['candidates'][0]['source_view_id'],obs.observation_id+':rgb')
        self.assertEqual(result['candidates'][0]['evidence_view_ids'],[obs.observation_id+':rgb','context'])

    def test_variable_views_have_attached_captions_without_fixed_image_numbers(self):
        obs=acquire(self.ob,[.05,.05],0,5,'s1',1,'current',str(self.out))
        for mode in ('candidate_proposal','candidate_confirmation'):
            for n in (1,2,5):
                views=[dict(rgb_view(obs),view_id=f'v{i}',selectable=i%2==0) for i in range(n)]
                request=messages(mode,'roof',views)
                self.assertNotRegex(request[0]['content'],r'(?:Image|IMAGE) [1-5]')
                parts=request[1]['content']
                self.assertEqual(sum(p['type']=='image_url' for p in parts),n)
                for index,part in enumerate(parts):
                    if part['type']=='image_url':
                        self.assertEqual(parts[index-1]['type'],'text')
                        self.assertIn('image_size: 768x768',parts[index-1]['text'])

    def test_relative_coordinates_and_heading_wrap(self):
        def pose(lat=0,lon=0,heading=0):
            return SimpleNamespace(position=[lat,lon],heading=heading,observation_id='fixture')
        self.assertAlmostEqual(relative_pose(pose(),pose(lat=.001))['forward_m'],111.19,places=2)
        self.assertAlmostEqual(relative_pose(pose(),pose(lon=.001))['right_m'],111.19,places=2)
        self.assertEqual(relative_pose(pose(heading=350),pose(heading=5))['clockwise_heading_delta_deg'],15)
        facing_east=relative_pose(pose(heading=90),pose(lon=.001,heading=90))
        self.assertAlmostEqual(facing_east['forward_m'],111.19,places=2)
        self.assertAlmostEqual(facing_east['right_m'],0,places=2)

    def test_polygon_iou_and_stage_isolation(self):
        a=[[30,120],[30,120.001],[30.001,120.001],[30.001,120]]
        b=[[30,120.0005],[30,120.0015],[30.001,120.0015],[30.001,120.0005]]
        self.assertAlmostEqual(polygon_iou(a,a),1,places=5)
        self.assertAlmostEqual(polygon_iou(a,b),1/3,places=5)
        memory=TrajectoryMemory()
        memory.event('one','uncertain',round=1)
        memory.stage('two','roof')
        self.assertEqual(memory.summary('two')['recent_search_and_confirmation'],[])

    def test_preview_outside_map_rejected_but_actual_crop_clips(self):
        with self.assertRaises(ValueError):
            acquire(self.ob,[0,0],0,5,'s1',1,'left',str(self.out),acquisition='map_preview')
        self.assertIsNotNone(acquire(self.ob,[0,0],0,5,'s1',1,'current',str(self.out)))


if __name__=='__main__':
    unittest.main()
