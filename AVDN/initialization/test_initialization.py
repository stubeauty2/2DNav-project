import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from .contracts import SCHEMA, validate_result, parse_model_output, resolve_direction, exploration_seed, grounding_query
from .geometry import angular_error
from .views import homography, heading_corners, crop
from .cli import run, evaluate
from .client import consume_native, native_messages

DATASET = 'AVDN'

def agent():
    return {'dataset':DATASET, 'setting':'visual', 'instruction_raw':'white building beside a road',
            'turns':[{'turn_id':'t0','role':'INS','text':'white building beside a road'}],
            'initial_pose':{'xy_m':[0,0],'heading_deg':10}, 'coordinate_frame':'east/north',
            'named_landmarks':[], 'views':[]}

def answer():
    return {'direction':45, 'final_target':'white building beside a road','auxiliary':['road is a reference']}

class InitializationTests(unittest.TestCase):
    def test_complete_json_survives_missing_answer_wrapper_end(self):
        result,_=parse_model_output('<answer>'+json.dumps(answer()),agent())
        self.assertEqual(result,answer())
        with self.assertRaises(ValueError):
            parse_model_output('<answer>{"direction":',agent())

    def test_request_budget_reserves_before_sending_and_survives_reopen(self):
        from .budget import RequestBudget, BudgetExceeded
        from .cli import save_json
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'budget.json';b=RequestBudget(path,0.005,save_json)
            messages=[{'role':'user','content':'small test'}]
            token=b.reserve('qwen3.5-flash',messages,2048)
            with self.assertRaises(BudgetExceeded):b.reserve('qwen3.5-flash',messages,2048)
            reopened=RequestBudget(path,0.005,save_json)
            with self.assertRaises(BudgetExceeded):reopened.reserve('qwen3.5-flash',messages,2048)
            reopened.settle(token,{'input_tokens':100,'output_tokens':10})
            self.assertAlmostEqual(reopened.read()['charged_cny'],0.00004)
            token=reopened.reserve('qwen3.5-flash',messages,2048)
            reopened.settle(token,None)
            self.assertLessEqual(reopened.read()['charged_cny'],0.005)

    def test_plan_followed_by_plain_json_answer(self):
        raw='<plan>{"stages":[],"assumptions":[]}</plan>'+json.dumps(answer())
        result,_=parse_model_output(raw,agent(),require_plan=True)
        self.assertEqual(result,answer())

    def test_exactly_three_fields(self):
        a=answer();validate_result(a,agent())
        a['extra']='unwanted'
        with self.assertRaises(Exception):validate_result(a,agent())
        self.assertEqual(set(SCHEMA['properties']),{'direction','final_target','auxiliary'})

    def test_null_does_not_turn_into_predicted_heading(self):
        a=answer();a['direction']=None
        self.assertIsNone(resolve_direction(a,agent())['heading_deg'])
        seed=exploration_seed(a,agent())
        self.assertEqual(seed['heading_deg'],10)
        self.assertEqual(seed['role'],'exploration_only')

    def test_nonfinite_angle_rejected(self):
        for direction in [float('nan'),float('inf'),-1,360]:
            a=answer();a['direction']=direction
            with self.assertRaises(Exception):validate_result(a,agent())

    def test_answer_and_plan_are_separate(self):
        raw='<answer>'+json.dumps(answer())+'</answer>'
        a,diagnostics=parse_model_output(raw,agent())
        self.assertEqual(set(a),{'direction','final_target','auxiliary'})
        self.assertIn('plan',diagnostics)
        self.assertIn('beside a road',grounding_query(a))
        with self.assertRaises(ValueError):parse_model_output(raw+raw,agent())

    def test_native_sse_discards_private_reasoning(self):
        frames=[json.dumps({'output':{'choices':[{'message':{'content':[part]}}]}})
                for part in [{'reasoning_content':'private'}, {'text':'{"ok":'}, {'text':'true}'}]]
        text,_=consume_native(['data: '+x for x in frames])
        self.assertEqual(text,'{"ok":true}')

    def test_native_sse_heartbeat_cannot_extend_deadline(self):
        import requests
        with patch('AVDN.initialization.client.time.monotonic', return_value=101):
            with self.assertRaises(requests.Timeout):
                consume_native([': heartbeat', ''], deadline=100)

    def test_transform_roundtrip(self):
        corners=heading_corners([100,200],63,100)
        m=homography([(0,0),(511,0),(511,511),(0,511)],corners)
        p=m@np.array([255.5,255.5,1])
        np.testing.assert_allclose(p[:2]/p[2],[100,200],atol=1e-8)
        self.assertEqual(angular_error(355,5),10)

    def test_outside_pixels_invalid(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);src=folder/'source.png';Image.new('RGB',(20,20),'white').save(src)
            corners=[[-10,-10],[10,-10],[10,10],[-10,10]]
            view=crop(src,corners,corners,folder/'view.png',size=20)
            self.assertLess(view['valid_fraction'],0.4)

    def test_missing_prediction_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            truths=[{'episode_id':str(i),'dataset':DATASET,'start_xy_m':[0,0],
                     'goal_xy_m':[0,10],'initial_heading_deg':90} for i in range(2)]
            pred=[{'episode_id':'0','status':'ok','setting':'visual','resolved_direction':{'heading_deg':0,'source':'model'}}]
            (p/'truth').write_text('\n'.join(json.dumps(x) for x in truths),encoding='utf-8')
            (p/'pred').write_text('\n'.join(json.dumps(x) for x in pred),encoding='utf-8')
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate(SimpleNamespace(truth=p/'truth',predictions=p/'pred',out=p/'report'))
            report=json.loads((p/'report').read_text(encoding='utf-8'))['summary'][DATASET+'/visual']['model']
            self.assertEqual(report['within_deg_all_eligible']['45'],0.5)

    def test_run_exports_only_three_label_fields(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            (p/'input').write_text(json.dumps({'episode_id':'test','agent_input':agent()}),encoding='utf-8')
            args=SimpleNamespace(inputs=p/'input',out=p/'run',model=None,url=None,protocol='native',
                                 timeout=1,workers=1,direction_mode='direct')
            response={'status':200,'text':'<answer>'+json.dumps(answer())+'</answer>','usage':{}}
            with patch.dict('os.environ',{'API_KEY':'unit-test-placeholder'}), patch(DATASET+'.initialization.client.call_model',return_value=response), contextlib.redirect_stdout(io.StringIO()):
                run(args)
            self.assertEqual(json.loads((p/'run/labels/test.json').read_text(encoding='utf-8')),answer())


    def test_vector_composition_uses_lengths(self):
        from .contracts import compose_stages
        p={'stages':[{'heading_deg':90,'weight':5,'target':'road'},
                     {'heading_deg':0,'weight':1,'target':'goal'}],'assumptions':['relative length estimate']}
        d=compose_stages(p)
        self.assertAlmostEqual(d['weighted_heading_deg'],78.6900675,places=6)
        self.assertAlmostEqual(d['equal_weight_heading_deg'],45)

    def test_cancellation_unknown_and_absolute_headings(self):
        from .contracts import compose_stages
        p={'stages':[{'heading_deg':350,'weight':1,'target':'road'},
                     {'heading_deg':170,'weight':1,'target':'goal'}],'assumptions':[]}
        self.assertIsNone(compose_stages(p)['weighted_heading_deg'])
        p['stages'][0]['heading_deg']=None
        self.assertIsNone(compose_stages(p)['weighted_heading_deg'])
        p['stages']=[{'heading_deg':162,'weight':1,'target':'goal'}]
        self.assertAlmostEqual(compose_stages(p)['weighted_heading_deg'],162)

    def test_staged_mode_requires_plan(self):
        with self.assertRaises(ValueError):
            parse_model_output(json.dumps(answer()),agent(),require_plan=True)

    def test_original_parser_clock_and_absolute_conversion(self):
        from .contracts import stage_heading
        self.assertEqual(stage_heading('3:00',158),248)
        self.assertEqual(stage_heading('5:00',155),305)
        self.assertEqual(stage_heading('315deg',214),315)
        self.assertEqual(stage_heading('11:00',97),67)
        self.assertIsNone(stage_heading('3:00',None))
        with self.assertRaises(ValueError):stage_heading('13:00',0)

    def test_staged_label_uses_computed_vector(self):
        a=agent();a['initial_pose']['heading_deg']=303
        p={'stages':[{'move_dir':'6:00','weight':1,'target':'approach'},
                     {'move_dir':'3:00','weight':1.5,'target':'blue roofs'}], 'assumptions':[]}
        raw='<plan>'+json.dumps(p)+'</plan><answer>'+json.dumps(answer())+'</answer>'
        result,d=parse_model_output(raw,a,True)
        self.assertAlmostEqual(result['direction'],179.309932474,places=6)
        self.assertEqual(d['direction_source'],'stage_vector')

    def test_symmetric_north_composition_stays_in_angle_domain(self):
        p={'stages':[{'move_dir':'NORTHEAST','weight':1,'target':'first'},
                     {'move_dir':'9:00','weight':1,'target':'second'}], 'assumptions':[]}
        raw='<plan>'+json.dumps(p)+'</plan><answer>'+json.dumps(answer())+'</answer>'
        result,_=parse_model_output(raw,agent(),True)
        self.assertEqual(result['direction'],0.0)
        validate_result(result,agent())

    def test_completed_run_resumes_without_reissuing_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder); inp=p/'inputs.jsonl'
            inp.write_text(json.dumps({'episode_id':'test','agent_input':agent()}),encoding='utf-8')
            args=SimpleNamespace(inputs=inp,out=p/'run',model=None,url=None,protocol='native',
                                 timeout=1,workers=1,direction_mode='direct')
            response={'status':200,'text':json.dumps(answer()),'usage':{}}
            with patch.dict('os.environ',{'API_KEY':'unit-test-placeholder'}), patch(
                DATASET+'.initialization.client.call_model',return_value=response) as call, contextlib.redirect_stdout(io.StringIO()):
                run(args)
                (p/'run/predictions.jsonl').unlink()
                run(args)
                self.assertEqual(call.call_count,1)
                inp.write_text(json.dumps({'episode_id':'different','agent_input':agent()}),encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'different prompt/input/runtime'):run(args)

    def test_atomic_write_retries_transient_windows_reader_lock(self):
        import os
        from .cli import save_json
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'progress.json';replace=os.replace;attempts=[]
            def transient(src,dst):
                attempts.append(1)
                if len(attempts)<3:raise PermissionError('temporary reader lock')
                replace(src,dst)
            with patch('AVDN.initialization.cli.os.replace',side_effect=transient), patch('AVDN.initialization.cli.time.sleep'):
                save_json(output,{'completed':7})
            self.assertEqual(len(attempts),3)
            self.assertEqual(json.loads(output.read_text()),{'completed':7})

if __name__=='__main__':
    unittest.main()
