"""Offline candidate search contracts, geometry, history and movement budgets."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np
from fixtures import NavigationFixture
from andh_core import navigation as nav
from andh_core.config import NavigationOptions
from andh_core.tools.views import acquire, search_preview_poses
from andh_core.memory import TrajectoryMemory
from andh_core.tools.mosaic import build_mosaic
from andh_core.tools.model import validate, messages, rgb_view
from andh_core.tools.geometry import project_bbox, local_xy


BOX = [354,354,414,414]


def proposal(source=None, bbox=BOX):
    candidate = None if source is None else dict(source_view_id=source, bbox_2d=bbox,
                                                support=['roof and nearby road'], conflicts=[], evidence_view_ids=[source])
    return json.dumps({'candidates':[candidate,None,None]})


def confirmation(decision='confirmed', round_n=1, cid='s1:c1', ids=None, bbox=BOX):
    ids = ids or [cid]
    return json.dumps({'decision': decision,
        'candidate_id': cid if decision == 'confirmed' else None,
        'target_view_id': f's1:r{round_n}:confirm_{cid.rsplit(":",1)[-1]}:rgb' if decision == 'confirmed' else None,
        'bbox_2d': bbox if decision == 'confirmed' else None,
        'support': ['matches full description'], 'conflicts': [], 'evidence_view_ids': [],
        'candidate_assessments': [{'candidate_id': c,
            'status': 'confirmed' if c == cid and decision == 'confirmed' else 'uncertain',
            'reason': 'visual appearance checked', 'unresolved': []} for c in ids]})


class CandidateTests(NavigationFixture, unittest.TestCase):

    def run_nav(self, replies, **settings):
        config = dict(strict_confirmation=True, enable_trace=True,
                      history_size=1, model_retries=0)
        config.update(settings)
        with patch.object(nav.ModelClient,'complete',side_effect=replies) as model:
            result = nav.search_and_reach_destination(instr_id='s1',ob=self.ob,dest_desc='roof',via_desc='',
                start_pos_latlng=[.05,.05],heading_deg=0,scale_factor=5,out_dir=str(self.out),
                step_meters=120,max_steps=3,api_key='',url='',model='',options=NavigationOptions(**config))
        record = json.loads((self.out/'s1_search.json').read_text(encoding='utf-8'))
        return result,record,model

    def test_early_selection_has_no_view_selection_and_audits_inputs(self):
        result,record,model = self.run_nav([proposal('s1:r1:current:rgb'),confirmation()])
        self.assertTrue(result[0]); self.assertEqual(model.call_count,2)
        self.assertEqual([c['mode'] for c in record['model_calls']],['candidate_proposal','candidate_confirmation'])
        self.assertNotIn('spatial_map',record)
        views = record['model_calls'][0]['views']
        self.assertEqual(len(views),5)
        self.assertEqual(sum(v['selectable'] for v in views),4)
        for call in record['model_calls']:
            self.assertIn('memory',json.loads(call['text_input'][0]))
            for view in call['views']:
                self.assertTrue(Path(view['image_path']).is_file())
        actions = [e['action'] for e in record['memory']['stages']['s1']['events'] if e['kind']=='actual_move']
        self.assertNotIn('forward',actions)

    def test_empty_proposals_stop_after_three_rounds_and_two_moves(self):
        result,record,model = self.run_nav([proposal()]*3)
        self.assertFalse(result[0]); self.assertEqual(model.call_count,3)
        self.assertEqual(len(result[2]),3)
        self.assertEqual(record['query_counts']['preview_acquired'],9)
        actions = [e['action'] for e in record['memory']['stages']['s1']['events'] if e['kind']=='actual_move']
        self.assertEqual(actions,['forward','forward'])

    def test_old_candidate_still_selectable_without_new_detection(self):
        result,record,model = self.run_nav([proposal('s1:r1:current:rgb'),confirmation('continue_search'),
                                           proposal(),confirmation(round_n=2)])
        self.assertTrue(result[0]); self.assertEqual(model.call_count,4)
        candidate = record['memory']['stages']['s1']['candidates']['s1:c1']
        self.assertEqual(candidate['source_observation_id'],'s1:r1:current')
        self.assertEqual(len(record['memory']['stages']['s1']['candidates']),1)

    def test_history_original_can_supply_bbox_and_merges_repeat(self):
        result,record,_ = self.run_nav([proposal('s1:r1:current:rgb'),confirmation('continue_search'),
                                       proposal('s1:r1:current:rgb'),confirmation(round_n=2)])
        self.assertTrue(result[0])
        self.assertEqual(len(record['memory']['stages']['s1']['candidates']),1)
        second = record['model_calls'][2]
        self.assertTrue(next(v for v in second['views'] if v['view_id']=='s1:r1:current:rgb')['selectable'])
        self.assertIn('candidate_merged',[e['kind'] for e in record['memory']['stages']['s1']['events']])

    def test_final_round_cannot_continue_or_invent_candidate(self):
        context={'candidate_views':{'s1:c1':'s1:r1:confirm_c1:rgb'},'can_continue_search':False}
        for raw in (confirmation('continue_search'),confirmation(cid='missing')):
            with self.assertRaises(ValueError):
                validate(raw,'candidate_confirmation',[],context)
        self.assertEqual(validate(confirmation('no_match'),'candidate_confirmation',[],context)['decision'],'no_match')

    def test_auxiliary_cannot_supply_bbox_and_slot_count_is_enforced(self):
        obs=acquire(self.ob,[.05,.05],0,5,'s1',1,'current',str(self.out))
        views=[rgb_view(obs),dict(rgb_view(obs),view_id='sam3',selectable=False)]
        with self.assertRaises(ValueError):
            validate(proposal('sam3'),'candidate_proposal',views,{'candidate_count':3})
        with self.assertRaises(ValueError):
            validate(proposal(obs.observation_id+':rgb'),'candidate_proposal',views,{'candidate_count':2})
        for count in (1,2):
            req=messages('candidate_proposal','roof',views[:count],extra={'candidate_count':3})
            self.assertEqual(sum(c['type']=='image_url' for c in req[1]['content']),count)

    def test_preview_layout_keeps_heading_and_half_footprint_offsets(self):
        obs=acquire(self.ob,[.05,.05],32,5,'s1',1,'current',str(self.out))
        poses=list(search_preview_poses(obs))
        centers=local_xy([p[1] for p in poses],obs.position,obs.heading)
        self.assertEqual([p[2] for p in poses],[32]*3)
        self.assertAlmostEqual(centers[0,0],-centers[0,1],delta=.1)
        self.assertAlmostEqual(centers[2,0],centers[2,1],delta=.1)
        self.assertAlmostEqual(centers[1,1],centers[0,1],delta=.1)
        self.assertAlmostEqual(centers[1,0],0,delta=.1)

    def test_mosaic_keeps_unobserved_gap_blank(self):
        base=np.zeros((32,32,3),np.uint8); base[:]=(10,70,130)
        obs=[]
        for i,x in enumerate((0,.004)):
            corners=[[.001,x],[.001,x+.001],[0,x+.001],[0,x]]
            obs.append(SimpleNamespace(view_corners=corners,position=[.0005,x+.0005],heading=0,
                                       image=base,observation_id=f's:r1:{i}'))
        view=build_mosaic(obs,obs[0],self.out/'gap.png',long_side=512)
        h,w=view['image'].shape[:2]
        center=view['image'][h//2,w//2]
        self.assertTrue(np.all(center==225) or np.all(center==240))
        self.assertFalse(view['selectable'])

    def test_merge_uses_map_geometry_and_preserves_rejected_status(self):
        first=acquire(self.ob,[.05,.05],0,5,'s1',1,'current',str(self.out))
        second=acquire(self.ob,[.05,.05],0,5,'s1',2,'current',str(self.out))
        polygon,center=project_bbox(BOX,first,self.ob)
        memory=TrajectoryMemory()
        c,created=memory.merge_candidate('s1',first,BOX,polygon,center,{'support':['one']})
        c['status']='rejected'
        merged,created=memory.merge_candidate('s1',second,BOX,polygon,center,{'support':['two']})
        self.assertFalse(created); self.assertEqual(merged['candidate_id'],c['candidate_id'])
        self.assertEqual(merged['status'],'rejected'); self.assertEqual(len(merged['observations']),2)
        for _ in range(3):
            memory.candidate('s1',first,BOX,polygon,center,{})
        self.assertEqual(len(memory.summary('s1')['candidates']),4)
        self.assertNotIn('map_center',json.dumps(memory.summary('s1')))

    def test_bad_candidate_source_exhausts_round_budget(self):
        result,record,model=self.run_nav([proposal('invented')]*6,model_retries=1)
        self.assertFalse(result[0]); self.assertEqual(model.call_count,6)
        self.assertEqual(record['status'],'parse_error')
        actions=[e['action'] for e in record['memory']['stages']['s1']['events'] if e['kind']=='actual_move']
        self.assertEqual(actions,['forward','forward'])

    def test_rejected_candidate_is_not_offered_again(self):
        rejected=json.loads(confirmation('continue_search'))
        rejected['candidate_assessments'][0]['status']='rejected'
        result,record,model=self.run_nav([proposal('s1:r1:current:rgb'),json.dumps(rejected),
                                         proposal('s1:r1:current:rgb'),proposal()])
        self.assertFalse(result[0]); self.assertEqual(model.call_count,4)
        stage=record['memory']['stages']['s1']
        self.assertEqual(stage['candidates']['s1:c1']['status'],'rejected')
        self.assertEqual(sum(c['mode']=='candidate_confirmation' for c in record['model_calls']),1)
        actions=[e['action'] for e in stage['events'] if e['kind']=='actual_move']
        self.assertEqual(actions,['forward','forward'])

    def test_boundary_preview_reduces_offset_without_moving(self):
        original=nav.acquire
        def bounded(*args,**kwargs):
            if kwargs.get('acquisition')=='map_preview' and nav._haversine_m(.05,.05,*args[1])>1500:
                raise ValueError('simulated boundary')
            return original(*args,**kwargs)
        with patch.object(nav,'acquire',side_effect=bounded):
            result,record,_=self.run_nav([proposal('s1:r1:current:rgb'),confirmation()])
        self.assertTrue(result[0])
        self.assertGreater(record['query_counts']['preview_queries'],3)
        self.assertEqual(record['query_counts']['preview_acquired'],3)
        observations=record['memory']['observations']
        self.assertEqual(observations['s1:r1:current']['position'],[.05,.05])
        for name in ('left','forward','right'):
            self.assertLess(nav._haversine_m(.05,.05,*observations[f's1:r1:{name}']['position']),1500)

    def test_multiple_previews_then_one_move_and_no_old_boxes_in_memory(self):
        proposed=json.loads(proposal('s1:r1:current:rgb'))
        second=dict(proposed['candidates'][0], source_view_id='s1:r1:left:rgb',
                    evidence_view_ids=['s1:r1:left:rgb'])
        proposed['candidates'][1]=second
        result,record,model=self.run_nav([json.dumps(proposed),
            confirmation(cid='s1:c2',ids=['s1:c1','s1:c2'])],strict_confirmation=False)
        self.assertTrue(result[0]); self.assertEqual(model.call_count,2)
        call=record['model_calls'][1]
        self.assertEqual(sum(v['selectable'] for v in call['views']),2)
        self.assertEqual(len([v for v in call['views'] if ':detail' in v['view_id']]),2)
        summary=json.dumps(json.loads(call['text_input'][0])['memory'])
        for forbidden in ('bbox', 'map_polygon', 'source_view_id', 'evidence_view_ids'):
            self.assertNotIn(forbidden,summary)
        stage=record['memory']['stages']['s1']
        self.assertEqual(stage['candidates']['s1:c1']['status'],'uncertain')
        actions=[e['action'] for e in stage['events'] if e['kind']=='actual_move']
        self.assertEqual(actions,['approach_confirmed_target'])
        self.assertEqual(len(result[2]),2)  # actual starting view plus final target box
        np.testing.assert_allclose(result[1],np.mean(result[2][-1],axis=0),atol=1e-7)
        self.assertIn('source_bbox_2d', stage['candidates']['s1:c2'])
        self.assertIn('confirmation_result', stage['candidates']['s1:c2'])

    def test_confirmation_cannot_mix_candidate_and_view_or_use_auxiliary(self):
        obs=acquire(self.ob,[.05,.05],0,5,'s1',1,'confirm_c1',str(self.out))
        views=[rgb_view(obs),dict(rgb_view(obs),view_id='detail',selectable=False)]
        extra={'candidate_views':{'s1:c1':obs.observation_id+':rgb'},'can_continue_search':True}
        for target in ['detail','invented']:
            raw=json.loads(confirmation());raw['target_view_id']=target
            with self.assertRaises(ValueError):validate(raw,'candidate_confirmation',views,extra)
        extra['candidate_views']['s1:c2']='other:rgb'
        raw=json.loads(confirmation(ids=['s1:c1','s1:c2']));raw['candidate_id']='s1:c2'
        with self.assertRaises(ValueError):validate(raw,'candidate_confirmation',views,extra)

    def test_invalid_slot_does_not_discard_valid_candidate(self):
        obs=acquire(self.ob,[.05,.05],0,5,'s1',1,'current',str(self.out))
        raw=json.loads(proposal(obs.observation_id+':rgb'))
        raw['candidates'][1]=dict(raw['candidates'][0],bbox_2d=[10,10,768,768])
        result=validate(raw,'candidate_proposal',[rgb_view(obs)],{'candidate_count':3})
        self.assertIsNotNone(result['candidates'][0]);self.assertIsNone(result['candidates'][1])
        self.assertEqual(result['invalid_candidates'][0]['slot'],1)

    def test_third_round_confirmation_keeps_search_budget(self):
        result,record,model=self.run_nav([
            proposal('s1:r1:current:rgb'),confirmation('continue_search'),
            proposal(),confirmation('continue_search',round_n=2),
            proposal(),confirmation(round_n=3)])
        self.assertTrue(result[0]); self.assertEqual(model.call_count,6)
        self.assertEqual(record['status'],'confirmed')

    def test_confirmation_resolves_conflicts_without_leaking_geometry(self):
        obs=acquire(self.ob,[.05,.05],0,5,'s1',1,'current',str(self.out))
        polygon,center=project_bbox(BOX,obs,self.ob)
        memory=TrajectoryMemory()
        candidate=memory.candidate('s1',obs,BOX,polygon,center,{'conflicts':['truncated roof']})
        memory.record_confirmation('s1',json.loads(confirmation('continue_search')),1)
        self.assertEqual(candidate['conflicts'],[])
        self.assertEqual(candidate['status'],'uncertain')
        text=json.dumps(memory.summary('s1'))
        self.assertNotIn('bbox',text); self.assertNotIn('source_view_id',text)
        self.assertEqual(candidate['observations'][0]['conflicts'],['truncated roof'])

    def test_cli_exposes_candidate_controls(self):
        from andh_core import runner
        with patch.object(runner,'run_turn_and_crop_full_trajectory') as run:
            runner.main(['--out-dir',str(self.out),
                         '--candidate-count','2','--preview-offset-ratio','0.4'])
        options=run.call_args.kwargs['navigation_options']
        self.assertEqual(options.candidate_count,2)
        self.assertEqual(options.preview_offset_ratio,.4)


if __name__=='__main__':
    unittest.main()
