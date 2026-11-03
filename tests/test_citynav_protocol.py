import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from CityNav.agent.contracts import validate_action
from CityNav.agent.prompts import SYSTEM, REVIEW_SYSTEM
from CityNav.tools.llm import parse_action


class ProtocolTests(unittest.TestCase):
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
