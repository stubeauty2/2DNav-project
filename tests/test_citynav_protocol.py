import json
from pathlib import Path
import sys
import unittest
import tempfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from CityNav.agent.contracts import validate_action
from CityNav.agent.prompts import SYSTEM, REVIEW_SYSTEM
from CityNav.tools.llm import parse_action, Model, AgentBudget


class ProtocolTests(unittest.TestCase):
    def test_review_thinking_disabled_decision_retains_budget(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            budget=AgentBudget(root/'budget.json',None,lambda p,v:Path(p).write_text(json.dumps(v)))
            model=Model(root/'requests',budget,'qwen3.8-flash','test',enable_thinking=True,thinking_budget=2048)
            calls=[]
            def fake(url,name,key,messages,**options):
                calls.append((name,options['enable_thinking'],options['thinking_budget']))
                return dict(status=200,text='{}',usage=dict(input_tokens=1,output_tokens=1))
            with patch('CityNav.tools.llm.call_model',side_effect=fake):
                decision=model.request(dict(mode='decision'),[])
                review=model.request(dict(mode='review'),[])
            self.assertEqual(calls,[('qwen3.5-flash',True,2048),('qwen3.8-flash',False,0)])
            self.assertEqual(decision['reserved_output_tokens'],4096)
            self.assertEqual(review['reserved_output_tokens'],2048)
            self.assertFalse(review['enable_thinking'])

    def test_decision_prompt_separates_review(self):
        self.assertNotIn('"action":"review"',SYSTEM)
        self.assertIn('"action":"review"',REVIEW_SYSTEM)

    def test_review_in_decision_requires_explicit_repair(self):
        with self.assertRaisesRegex(ValueError,'DECISION stage'):
            validate_action(dict(action='review',verdict='confirm',view_id='v',bbox=[0,0,10,10],reason='visible'))

    def test_missing_view_never_inferred(self):
        with self.assertRaisesRegex(ValueError,'Missing view_id'):
            validate_action(dict(action='tool',tool='sam3',query='building',reason='find'))

    def test_long_reason_does_not_change_valid_action(self):
        action=dict(action='locate',target='final',view_id='v',bbox=[1,2,3,4],reason='x'*2100)
        self.assertEqual(validate_action(parse_action(json.dumps(action))),action)

    def test_offset_metadata_preserves_movement(self):
        action=dict(action='explore',mode='offset',view_id='v',point=None,turn_deg=10,step_scale=.5,reason='move')
        result=validate_action(action)
        self.assertEqual(result,{k:v for k,v in action.items() if k not in ['view_id','point']})
        action['point']=[5,6]
        with self.assertRaises(ValueError):validate_action(action)

    def test_review_remains_required_and_strict(self):
        action=dict(action='review',verdict='confirm',view_id='v',bbox=[1,2,3,4],reason='visible')
        self.assertEqual(validate_action(action,review=True),action)
        with self.assertRaises(ValueError):validate_action(dict(action,extra=True),review=True)


if __name__=='__main__':unittest.main()
