"""STEP 3: geometry, protocol, evidence failure and review request contracts."""
from pathlib import Path
import json,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PSC-AVDN/script'),str(ROOT/'src')]
from andn_core.tools.geometry import confirmation_signals
from andn_core.tools.candidates import normalize_review,verify_confirmation
from andn_core.tools import telemetry as audit
from andn_core.tools.model import review_candidate_with_evidence

def quad(n):return [[0,0],[0,n],[n,n],[n,0]]
def reply(action='keep',box=None,**kw):
    return dict(action=action,bbox_2d=box,selected_candidate_id='r1_c1',view_id='r1_c1_confirm_main',
                confidence=.8,reason='Visible boundary evidence',**kw)

class Verification(unittest.TestCase):
    def test_expansion_signal_but_correct_shrink_is_not_automatically_rejected(self):
        expanded=confirmation_signals(quad(.001),quad(.003),[100,100,200,200],[100,100,400,400])
        self.assertAlmostEqual(expanded['area_ratio'],9)
        self.assertIn('expanded_while_covering_search',expanded['trigger_reasons'])
        shrink=confirmation_signals(quad(.003),quad(.001),[100,100,400,400],[100,100,200,200])
        self.assertLess(shrink['map_union_iou'],.2)
        self.assertEqual(shrink['trigger_reasons'],[])

    def test_review_schema_preserves_identity_and_distinguishes_rejection(self):
        for action in ['keep','reject']:
            self.assertEqual(normalize_review(reply(action),'r1_c1','r1_c1_confirm_main')['action'],action)
        self.assertEqual(normalize_review(reply('revise',[10,20,40,60]),'r1_c1','r1_c1_confirm_main')['bbox_2d'],[10,20,40,60])
        for obj in [reply('keep',[1,2,3,4]),reply('revise',[0,0,900,800]),reply('revise',[1,2,float('nan'),4]),
                    dict(reply(),selected_candidate_id='r1_c2'),dict(reply(),view_id='sam3'),dict(reply(),confidence=True)]:
            with self.assertRaises(ValueError):normalize_review(obj,'r1_c1','r1_c1_confirm_main')

    def test_review_request_metadata_and_no_gt(self):
        image=np.zeros((768,768,3),np.uint8)
        vv=[dict(role=role,candidate_id='r1_c1',view_id='r1_c1_confirm_'+role,
            image=image,corners=quad(.001),description=role+' view') for role in ['main','detail','context','comparison','sam3']]
        class Response:
            status_code=200;headers={'Content-Type':'application/json'}
            def json(self):return {'choices':[{'message':{'content':json.dumps(reply())}}]}
        with tempfile.TemporaryDirectory() as folder:
            recorder=audit.Recorder(folder,{});token=audit._run.set(recorder)
            try:
                initial=dict(selected_candidate_id='r1_c1',view_id='r1_c1_confirm_main',bbox_2d=[1,2,3,4])
                with patch('requests.post',return_value=Response()) as post:
                    result=review_candidate_with_evidence('roof',initial,{'area_ratio':3},vv,api_key='test-only')
                self.assertEqual(result['action'],'keep')
                self.assertEqual(post.call_args.kwargs['json']['temperature'],0)
                self.assertEqual(recorder.counts['review'],1);self.assertEqual(recorder.errors,0)
                call=json.loads((recorder.root/'calls/call00001.json').read_text(encoding='utf8'))
                self.assertEqual(len(call['images']),5)
                self.assertEqual([i['role'] for i in call['images'] if i['localization_allowed']],['main'])
                self.assertNotIn('gt_path',json.dumps(call['request']))
            finally:audit._run.reset(token)

    def test_observe_and_exhausted_budget_do_not_call_tools_and_errors_keep_initial(self):
        image=np.zeros((768,768,3),np.uint8)
        candidate=dict(candidate_id='r1_c1',bbox_2d=[10,10,20,20],source_corners=quad(.001))
        initial=dict(dest_present=True,selected_candidate_id='r1_c1',view_id='r1_c1_confirm_main',bbox_2d=[10,10,40,40],confidence=.9)
        view=dict(candidate_id='r1_c1',view_id='r1_c1_confirm_main',role='main',image=image,corners=quad(.001),path='unused',description='main')
        with tempfile.TemporaryDirectory() as folder, patch('andn_core.tools.geometry._patch_xy_to_latlng',side_effect=lambda x,y,*a:[y/10000,x/10000]), patch('andn_core.tools.geometry._latlng_to_patch_xy',return_value=[[10,10],[20,10],[20,20],[10,20]]), patch('andn_core.tools.sam3._sam3_image5',side_effect=RuntimeError('tool offline')) as tool:
            args=('roof',candidate,initial,[view],{},folder,'stage','test','url','model','tool')
            for mode,budget in [('observe',True),('once',False)]:
                result,record=verify_confirmation(*args,mode,budget)
                self.assertEqual(result,initial);self.assertFalse(record['attempted'])
            tool.assert_not_called()
            result,record=verify_confirmation(*args,'once',True)
            self.assertEqual(result,initial);self.assertTrue(record['attempted'])
            self.assertEqual(record['status'],'operational_error_keep_initial')
            tool.assert_called_once()

if __name__=='__main__':unittest.main()
