import json
import unittest
from SatNav.initialization.parser import parse


def response(turn):
    return json.dumps(dict(stages=[dict(turn_before_deg=0,distance_m=100),dict(turn_before_deg=turn,distance_m=160)],
        estimated_direction=None,final_target='white dock',auxiliary=[],assumptions=[],stop_condition='dock visible'))


class TurnConsistencyTests(unittest.TestCase):
    def test_explicit_right_turn_repairs_sign_and_recomposes_heading(self):
        _,d=parse(response(-120),90.,'Move 100m; a 120-degree right turn follows. Proceed 160m.')
        self.assertEqual(d['stages'][1]['heading_deg'],210.)
        self.assertEqual(d['language_consistency_corrections'][0]['model_value'],-120)

    def test_reverse_word_order_left_turn_is_negative(self):
        _,d=parse(response(75),90.,'Move 100m then turn left by 75 degrees. Proceed 160m.')
        self.assertEqual(d['stages'][1]['heading_deg'],15.)
        _,d=parse(response(-120),90.,'Turn right 120°. Proceed 160m.')
        self.assertEqual(d['stages'][1]['heading_deg'],210.)

    def test_unknown_visual_turn_and_magnitude_conflicts_are_not_filled(self):
        _,d=parse(response(None),90.,'Turn right until the roof is upper left.')
        self.assertIsNone(d['stages'][1]['heading_deg'])
        _,d=parse(response(-90),90.,'Turn right 120 degrees.')
        self.assertEqual(d['stages'][1]['heading_deg'],0.)
        self.assertNotIn('language_consistency_corrections',d)

    def test_ambiguous_multiple_turns_and_missing_instruction_keep_original_parse(self):
        for instruction in [None,'Turn right 120 degrees, then turn left 60 degrees.']:
            _,d=parse(response(-120),90.,instruction)
            self.assertEqual(d['stages'][1]['heading_deg'],330.)
            self.assertNotIn('language_consistency_corrections',d)


if __name__=='__main__':unittest.main()
