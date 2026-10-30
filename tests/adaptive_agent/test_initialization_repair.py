import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from fixtures import episode
from AVDN import analysis
from AVDN.tools import model


VALID = '<plan>{"stages":[{"move_dir":"5:00","weight":1,"target":"grey field"}],"assumptions":[]}</plan><answer>{"direction":null,"final_target":"grey field near blue building","auxiliary":["reference"]}</answer>'
BROKEN = VALID.replace('["reference"]', '["reference"')


class InitializationRepairTests(unittest.TestCase):
    def test_repair_continues_and_completed_requests_replay(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);agent=episode(root,'AVDN')['agent_input'];calls=[]
            def transport(*args, **kwargs):
                calls.append(kwargs)
                return dict(status=200,text=BROKEN if len(calls)<3 else VALID,usage={})
            with patch.object(model,'call_model',side_effect=transport):
                result=analysis.analyze(agent,root/'analysis','test',model='qwen3.5-flash',url='test')
            self.assertEqual(result['final_target'],'grey field near blue building')
            self.assertEqual(len(calls),3)
            self.assertEqual(calls[2]['temperature'],0)
            self.assertFalse(calls[2]['enable_thinking'])
            with patch.object(model,'call_model',side_effect=AssertionError('Must replay')):
                self.assertEqual(analysis.analyze(agent,root/'analysis','test',model='qwen3.5-flash',url='test'),result)
            self.assertEqual(json.loads((root/'analysis/result.json').read_text())['request_count'],3)

    def test_exhaustion_and_pending_never_fabricate_initialization(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);agent=episode(root,'AVDN')['agent_input']
            with patch.object(model,'call_model',return_value=dict(status=200,text=BROKEN,usage={})) as call:
                with self.assertRaisesRegex(model.Halt,'analysis_repair_exhausted'):
                    analysis.analyze(agent,root/'analysis','test',model='qwen3.5-flash',url='test')
            self.assertEqual(call.call_count,8)
            self.assertFalse((root/'analysis/result.json').exists())
            p=root/'analysis/000.json';record=json.loads(p.read_text(encoding='utf-8'));record['status']='pending';p.write_text(json.dumps(record),encoding='utf-8')
            with patch.object(model,'call_model',side_effect=AssertionError('Uncertain request must not repeat')):
                with self.assertRaisesRegex(model.Halt,'analysis_uncertain_request_pending'):
                    analysis.analyze(agent,root/'analysis','test',model='qwen3.5-flash',url='test')
