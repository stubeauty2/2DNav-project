"""G0--G4 geometry, observed-only coverage, evidence and protocol contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PSC-AVDN/script'),str(ROOT/'src')]
import cv2
import numpy as np
from andn_core.tools.spatial_map import SpatialMap, transform, build_spatial_request
from andn_core.tools.candidates import normalize_reply
from andn_core.tools import geometry, views


class SpatialGeometry(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'out');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);(self.root/'train_images').mkdir()
        cv2.imwrite(str(self.root/'train_images/test.tif'),np.zeros((1000,1200,3),np.uint8))
        self.ob=dict(dataset_dir=str(self.root),map_name='test',map_size=[1000,1200],
                     gps_botm_left=[44.99,9.99],gps_top_right=[45.01,10.01])
        self.state=SpatialMap([45,10],self.ob['gps_botm_left'],self.ob['gps_top_right'],
                              self.ob['map_size'],self.root/'spatial',mode='g4')
        self.state.begin_stage('s1','blue building',[45,10],45)

    def add(self,name,xy=(0,0),round_n=1,role='main',side=150,**kwargs):
        xy=np.asarray(xy)
        quad=xy+np.array([[-side,side],[side,side],[side,-side],[-side,-side]])
        c=self.state.to_latlng(quad)
        image=np.full((768,768,3),120,np.uint8)
        return self.state.observe(name,image,c,self.state.to_latlng(xy),45,round_n,role=role,**kwargs)

    def test_pixel_map_roundtrip_and_legacy_projection_agree_at_edges(self):
        for heading in [0,45,90,180,270]:
            for center in [[45,10],[45.009,10.009]]:
                corners=views.generate_view_corners_with_scale(center,self.ob,scale_factor=3,angle_deg=heading)
                record=self.state.observe('v%d_%s'%(heading,center),np.zeros((768,768,3),np.uint8),corners,center,heading,1)
                points=np.array([[0,0],[767,767],[400,280],[100,700]],float)
                mapped=transform(points,record['pixel_to_map'])
                np.testing.assert_allclose(transform(mapped,np.linalg.inv(record['pixel_to_map'])),points,atol=1e-5)
                expected=[geometry._patch_xy_to_latlng(x,y,corners,self.ob) for x,y in points]
                np.testing.assert_allclose(self.state.to_latlng(mapped),expected,rtol=0,atol=1e-8)

    def test_mask_preserves_gap_between_disjoint_observations(self):
        self.add('a',(-350,0),side=80);self.add('b',(350,0),side=80)
        image,mask=self.state.render(['a','b'],self.root/'overview.png')
        self.assertEqual(mask[mask.shape[0]//2,mask.shape[1]//2],0)
        self.assertAlmostEqual(self.state.coverage().area,2*160**2,delta=5)

    def test_preview_is_not_actual_movement(self):
        old=list(self.state.path);self.add('preview',(150,150),acquisition='preview')
        self.assertEqual(old,self.state.path)
        np.testing.assert_array_equal(self.state.agent['position_xy'],[0,0])

    def test_direction_uses_stage_and_agent_geometry(self):
        self.state.begin_stage('s1','x',[45,10],90)
        rel=self.state.relative([100,-50])
        self.assertEqual(rel['stage_forward_m'],100)
        self.assertEqual(rel['stage_right_m'],50)
        self.state.move(self.state.to_latlng([80,0]),90,'move')
        self.assertEqual(self.state.relative([100,-50])['agent_forward_m'],20)

    def test_retrieval_excludes_other_stage_current_round_and_auxiliary(self):
        self.add('old');self.add('detail',role='detail');self.add('current',round_n=2)
        self.state.record_search(['old'],dict(candidates=[]),1)
        self.assertEqual(self.state.retrieve(['current'],2),['old'])
        self.state.begin_stage('s2','another target',[45,10],0)
        self.add('new',round_n=1)
        self.assertEqual(self.state.retrieve(['new'],2),['new'])
        self.assertEqual(self.state.summary(['new'],['old'])['retrieved_search_records'],[])

    def test_history_summary_contains_no_old_boxes_or_generated_analysis(self):
        self.add('old');self.add('current',round_n=2)
        self.state.record_search(['old'],dict(candidates=[dict(source_view_id='old',bbox_2d=[10,20,30,40],
            confidence=.99,supporting_evidence='unverified model conclusion')]),1)
        s=json.dumps(self.state.summary(['current'],['old']))
        self.assertNotIn('bbox_2d',s);self.assertNotIn('unverified model conclusion',s)
        self.assertIn('proposals',s)

    def test_operational_failures_are_not_retrieved_as_negative_evidence(self):
        self.add('old');self.add('current',round_n=2)
        for reply in [dict(reason='fallback: request/json error'),
                      dict(candidates=[],validation_errors=['invalid_source_view'])]:
            self.state.record_search(['old'],reply,1)
        self.assertTrue(all(e['result']=='operational_error' for e in self.state.events))
        self.assertEqual(self.state.summary(['current'],['old'])['retrieved_search_records'],[])
        # Images remain valid observations, while a genuine empty model reply is retained.
        self.assertEqual(self.state.retrieve(['current'],2),['old'])
        self.state.record_search(['old'],dict(candidates=[],validation_errors=[]),1)
        records=self.state.summary(['current'],['old'])['retrieved_search_records']
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]['result'],'no_supported_candidate')

    def test_map_reuses_observation_pixels_without_reading_full_map(self):
        self.add('a')
        with patch('cv2.imread',side_effect=AssertionError('Map imagery read forbidden')):
            self.state.render(['a'],self.root/'map.png')

    def test_preview_pose_scales_with_footprint_and_has_fixed_crop_heading(self):
        self.add('main')
        poses=list(self.state.preview_poses('main'))
        self.assertEqual([p[0] for p in poses],['left','forward','right'])
        self.assertTrue(all(p[2]==45 for p in poses))
        self.assertLess(poses[0][3],0);self.assertGreater(poses[2][3],0)
        self.assertAlmostEqual(poses[0][4],poses[2][4],places=6)

    def test_snapshot_serializable_and_keeps_cross_stage_geometry(self):
        self.add('old');self.state.begin_stage('s2','new',[45,10],90);self.add('new')
        saved=json.loads(Path(self.state.snapshot('state')).read_text(encoding='utf8'))
        self.assertEqual(set(saved['views']),{'old','new'})
        self.assertNotIn('gt_path_corners',json.dumps(saved))

    def test_g4_real_navigation_binds_preview_coordinates_and_independent_confirmation(self):
        from andn_core import navigation
        from andn_core.tools.model import qwen_locate_bbox_in_view
        self.ob['gt_path_corners']=[views.generate_view_corners_with_scale([45,10],self.ob,1,0).tolist()]
        payloads=[]
        class Reply:
            headers={'Content-Type':'application/json'}
            def __init__(self,obj):self.obj=obj
            def json(self):return {'choices':[{'message':{'content':json.dumps(self.obj)}}]}
        def post(url,**kwargs):
            payloads.append(kwargs['json'])
            n=len(payloads)
            if n==1:return Reply(dict(candidates=[],grid_5x5=[]))
            if n==2:return Reply(dict(candidates=[dict(source_view_id='s1:r2:left',bbox_2d=[300,180,400,280],
                confidence=.9,supporting_evidence='test evidence')],grid_5x5=[]))
            return Reply(dict(dest_present=True,selected_candidate_id='r2_c1',view_id='r2_c1_confirm_main',
                bbox_2d=[300,300,400,400],confidence=.9,reason='independent visual match',grid_5x5=[]))
        with patch('requests.post',post),patch.object(navigation,'_sam3_image5',return_value=(None,None)):
            result=navigation.search_and_reach_destination('s1',self.ob,'blue building',None,[45,10],0,3,
                str(self.root/'nav'),candidate_count=2,spatial_state=self.state)
        self.assertTrue(result[0]);self.assertEqual(len(payloads),3)
        expected=self.state.locate('s1:r2:left',[300,180,400,280])['map_center']
        np.testing.assert_allclose(result[1],expected,rtol=0,atol=1e-8)
        search_text=' '.join(b.get('text','') for b in payloads[1]['messages'][1]['content'])
        self.assertIn('retrieved_search_records',search_text)
        self.assertIn('s1:r1:main',search_text)
        confirmation=json.dumps(payloads[2],ensure_ascii=False)
        self.assertNotIn('Spatial evidence',confirmation)
        self.assertNotIn('test evidence',confirmation)
        self.assertNotIn('retrieved_search_records',confirmation)


class SpatialProtocol(unittest.TestCase):
    def test_auxiliary_and_unknown_sources_rejected(self):
        candidate=lambda i:dict(source_view_id=i,bbox_2d=[5,5,20,20])
        for bad in ['history','overview','sam3','unknown']:
            r=normalize_reply(dict(candidates=[candidate(bad)]),'',allowed_source_ids=['left','main'])
            self.assertEqual(r['candidates'],[])
        r=normalize_reply(dict(candidates=[candidate('left'),candidate('main')]),'',allowed_source_ids=['left','main'])
        self.assertEqual([p['source_view_id'] for p in r['candidates']],['left','main'])

    def test_context_does_not_require_fixed_image_count(self):
        context=dict(mode='g4',main_ids=['v:main','v:left'],spatial_summary={'stage_direction_deg':0},views=[
            dict(view_id='v:main',role='main',image=np.zeros((3,3,3),np.uint8),localization_allowed=True,description='当前位置获取的视图。'),
            dict(view_id='v:left',role='main',image=np.zeros((3,3,3),np.uint8),localization_allowed=True,description='左前方裁剪。')])
        system,content=build_spatial_request('Original analysis requirements','target',context)
        self.assertIn('Original analysis requirements',system)
        self.assertIn('earlier model assessments',system)
        self.assertEqual(sum(x['type']=='image_url' for x in content),2)
        self.assertIn('v:left',content[0]['text'])


if __name__=='__main__':unittest.main()
