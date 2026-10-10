"""Stop-only instructions retain a prediction without fabricating a target."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'PSC-AVDN/script'),str(ROOT/'src')]
from andn_core.tools.termination import stop_only_instruction
from andn_core.tools.parsing import request_dialogue_sequence,parse_dialogue_sequence
import test_map_rescue as fixtures

class StopSemantics(unittest.TestCase):
    def test_only_complete_terminal_commands_match(self):
        for text in ('[INS] You are already at the destination.\n','Stop here.','you have already reached your destination'):
            self.assertTrue(stop_only_instruction(text))
        for text in ('', 'go to the destination', 'Stop here then go north',
                     '[INS] go north [INS] You are already at the destination.',
                     'you are not at the destination', '1912__7'):
            self.assertFalse(stop_only_instruction(text))

    def test_terminal_parse_needs_no_model_and_other_empty_results_still_fail(self):
        with patch('andn_core.tools.parsing.analyze_dialogue_with_prompt') as call:
            steps,_=request_dialogue_sequence('Stop here.',0)
        call.assert_not_called();self.assertEqual(steps[0]['move_dir'],'Stop')
        with self.assertRaises(ValueError):parse_dialogue_sequence('{"steps":[]}')

class StopRunner(unittest.TestCase):
    setUp=fixtures.RescueRunner.setUp
    run_case=fixtures.RescueRunner.run_case
    def test_stop_uses_initial_observed_footprint_without_search_or_rescue(self):
        self.ob['instructions']='[INS] You are already at the destination.'
        for mode in ('off','g5'):
            p,baseline,rescue=self.run_case('terminal_'+mode,None,None,mode=mode)
            baseline.assert_not_called();rescue.assert_not_called()
            self.assertEqual(len(p['path_corners']),1)
            self.assertEqual(p['path_corners'][0][0],self.start.tolist())

if __name__=='__main__':unittest.main()
