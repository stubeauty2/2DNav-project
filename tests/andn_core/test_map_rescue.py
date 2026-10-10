"""G5 preserves accepted SAM3 stages and uses one bounded spatial rescue."""
import csv,json
from copy import deepcopy
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PSC-AVDN/script'),str(ROOT/'src')]
from andn_core import runner,navigation
import test_spatial_map as spatial_tests


class RescueGeometry(unittest.TestCase):
    setUp=spatial_tests.SpatialGeometry.setUp
    add=spatial_tests.SpatialGeometry.add

    def prepare(self):
        self.state.mode='g0'
        for n in (1,2,3):
            vid='s1:r%d:main'%n;self.add(vid,round_n=n)
            self.state.record_search([vid],dict(candidates=[]),n)

    def run_rescue(self,match):
        self.prepare();payloads=[]
        class Reply:
            headers={'Content-Type':'application/json'}
            def __init__(self,obj):self.obj=obj
            def json(self):return {'choices':[{'message':{'content':json.dumps(self.obj)}}]}
        def post(url,**kwargs):
            payloads.append(kwargs['json'])
            if len(payloads)==1:
                return Reply(dict(candidates=[dict(source_view_id='s1:r4:left',bbox_2d=[300,180,400,280],confidence=.9)] if match else [],grid_5x5=[]))
            return Reply(dict(dest_present=True,selected_candidate_id='r4_c1',view_id='r4_c1_confirm_main',
                              bbox_2d=[300,300,400,400],confidence=.9,reason='independent match',grid_5x5=[]))
        before=deepcopy((self.state.agent,self.state.path,self.state.stage))
        with patch('requests.post',post),patch.object(navigation,'_sam3_image5',return_value=(None,None)):
            result=navigation.rescue_failed_stage(instr_id='s1',ob=self.ob,dest_desc='blue building',via_desc=None,
                heading_deg=45,scale_factor=3,out_dir=str(self.root/'rescue'),spatial_state=self.state,
                api_key='',url='offline://test',model='test',vision_tool_url='offline://test')
        return result,payloads,before

    def test_rescue_reuses_history_and_binds_new_round_view(self):
        result,requests,before=self.run_rescue(True)
        self.assertTrue(result[0]);self.assertEqual(len(requests),2)
        self.assertEqual(result[-1]['round_offset'],3)
        self.assertEqual(self.state.mode,'g0');self.assertEqual(self.state.stage,before[2])
        text=' '.join(c.get('text','') for c in requests[0]['messages'][1]['content'])
        self.assertIn('s1:r4:left',text);self.assertIn('s1:r3:main',text)
        self.assertIn('retrieved_search_records',text)
        self.assertNotIn('retrieved_search_records',json.dumps(requests[1]))

    def test_unconfirmed_rescue_does_not_move_and_stops_after_one_search(self):
        result,requests,before=self.run_rescue(False)
        self.assertEqual(result[:4],(False,None,[],None));self.assertEqual(len(requests),1)
        self.assertEqual((self.state.agent,self.state.path,self.state.stage),before)
        self.assertEqual(self.state.mode,'g0')

    def test_tool_exception_preserves_primary_state(self):
        self.prepare();before=deepcopy((self.state.agent,self.state.path))
        with patch.object(navigation,'_search_candidates_and_reach_destination',side_effect=RuntimeError('probe')):
            result=navigation.rescue_failed_stage(instr_id='s1',ob=self.ob,dest_desc='blue building',via_desc=None,
                heading_deg=45,scale_factor=3,out_dir=str(self.root),spatial_state=self.state,
                api_key='',url='offline://test',model='test',vision_tool_url='offline://test')
        self.assertEqual(result[:4],(False,None,[],None))
        self.assertEqual(result[-1]['status'],'operational_error_keep_baseline')
        self.assertEqual((self.state.agent,self.state.path),before);self.assertEqual(self.state.mode,'g0')


class RescueRunner(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'out');self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.start=np.array([[45.001,9.999],[45.001,10.001],[44.999,10.001],[44.999,9.999]])
        self.goal=self.start+[.006,0]
        self.ob=dict(map_name='test',route_index=0,starting_angle=0,gps_botm_left=[44.99,9.99],
            gps_top_right=[45.02,10.02],map_size=[1000,1000],gt_path_corners=[self.start.tolist(),self.goal.tolist()])
        self.csv=self.root/'parse.csv'
        with self.csv.open('w',newline='',encoding='utf8') as f:
            w=csv.DictWriter(f,fieldnames=['instr_id','step_idx','dest','move_dir','instruction']);w.writeheader()
            w.writerow(dict(instr_id='test__0',step_idx=1,dest='building',move_dir='12:00',instruction='go to building'))

    def run_case(self,name,primary,aux,mode='g5'):
        ob=deepcopy(self.ob)
        class Env:
            def __init__(self,**kwargs):pass
            def _get_obs(self,t):return [ob]
            def eval_metrics(self,*a,**k):return {},{}
        with patch.object(runner,'ANDHNavBatch',Env),patch.object(runner,'DataLoader',return_value=[None]),\
             patch.object(runner,'create_view_image',return_value=np.zeros((8,8,3),np.uint8)),\
             patch.object(runner,'draw_pos_on_patch'),patch.object(runner,'save_traj_boxes_debug_image'),\
             patch.object(runner,'search_and_reach_destination',return_value=primary) as baseline,\
             patch.object(navigation,'rescue_failed_stage',return_value=aux) as rescue:
            predictions,_=runner.run_turn_and_crop_full_trajectory(anno_dir=str(self.root),dataset_dir=str(self.root),
                results_csv=str(self.csv),out_dir=str(self.root/name),spatial_mode=mode,trace=False)
        return predictions['test__0'],baseline,rescue

    def test_confirmed_primary_is_never_rewritten_even_if_evaluation_fails(self):
        for i,box in enumerate([self.start,self.goal]):
            p,baseline,rescue=self.run_case('accepted'+str(i),(True,[45,10],[box],0),None)
            rescue.assert_not_called();self.assertEqual(baseline.call_args.kwargs['candidate_count'],0)
            self.assertEqual(p['final_iou'],p['g5']['baseline_final_iou'])
            self.assertEqual(p['g5']['rescue_attempts'],[])

    def test_successful_rescue_records_paired_primary_failure(self):
        aux=(True,[45.006,10],[self.goal],0,dict(committed=True,status='confirmed'))
        p,_,rescue=self.run_case('rescued',(False,None,[self.start],None),aux)
        rescue.assert_called_once();self.assertGreater(p['final_iou'],.3)
        self.assertLessEqual(p['g5']['baseline_final_iou'],.3)

    def test_failed_rescue_leaves_complete_baseline_result_intact(self):
        primary=(False,None,[self.start],None)
        plain,_,_=self.run_case('plain',primary,None,mode='off')
        p,_,rescue=self.run_case('failed',primary,(False,None,[],None,dict(committed=False)))
        rescue.assert_called_once();meta=p.pop('g5')
        self.assertEqual(p,plain);self.assertEqual(meta['baseline_final_iou'],plain['final_iou'])

    def test_legacy_found_without_position_is_not_reinterpreted(self):
        p,_,rescue=self.run_case('legacy',(True,None,[self.start],None),None)
        rescue.assert_not_called();self.assertEqual(p['final_iou'],p['g5']['baseline_final_iou'])

    def test_confirmed_rescue_can_reduce_iou_of_a_successful_fallback(self):
        """The failure gate is not a guarantee of monotonic evaluation success."""
        primary=(False,None,[self.start],None)
        plain,_,_=self.run_case('fallback_probe',primary,None,mode='off')
        self.ob['gt_path_corners'][-1]=plain['path_corners'][-1][0]
        wrong_box=self.start+[.015,0]
        aux=(True,[45.015,10],[wrong_box],0,dict(committed=True,status='confirmed'))
        p,_,rescue=self.run_case('wrong_rescue',primary,aux)
        rescue.assert_called_once()
        self.assertGreater(p['g5']['baseline_final_iou'],.3)
        self.assertLessEqual(p['final_iou'],.3)


if __name__=='__main__':unittest.main()
