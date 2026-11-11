import copy
import unittest
from SatNav.semantic_plan import validate_plan, aligned_heading, region_bearing


class SemanticPlanTests(unittest.TestCase):
    def plan(self):
        return dict(unsupported_reason=None,legs=[
            dict(landmark='red roof',distance_m=None,arrival_region='lower_right',turn_direction='right',turn_deg=None,align_region='upper_center'),
            dict(landmark='blue pool',distance_m=90,arrival_region='lower_left',turn_direction=None,turn_deg=None,align_region=None)])

    def test_unknown_distances_preserved_and_private_fields_rejected(self):
        value=validate_plan(self.plan())
        self.assertIsNone(value['legs'][0]['distance_m'])
        contaminated=copy.deepcopy(value);contaminated['legs'][0]['goal_xy_m']=[0,1]
        with self.assertRaises(ValueError):validate_plan(contaminated)

    def test_ambiguous_turn_and_invalid_number_rejected(self):
        for field,value in [('align_region',None),('distance_m',float('nan'))]:
            plan=self.plan();plan['legs'][0][field]=value
            with self.assertRaises(ValueError):validate_plan(plan)
        plan=self.plan();plan['legs'][-1]['turn_direction']='right'
        with self.assertRaises(ValueError):validate_plan(plan)

    def test_region_alignment_uses_observed_landmark_only(self):
        self.assertEqual(region_bearing('middle_right'),90.)
        self.assertAlmostEqual(aligned_heading([0,0],[10,0],'upper_center'),90.)
        self.assertAlmostEqual(aligned_heading([0,0],[10,0],'lower_center'),270.)
        with self.assertRaises(ValueError):aligned_heading([0,0],[0,0],'upper_left')

    def test_unsupported_plan_is_explicit(self):
        self.assertEqual(validate_plan(dict(legs=[],unsupported_reason='Ambiguous route'))['legs'],[])
        with self.assertRaises(ValueError):validate_plan(dict(legs=[],unsupported_reason=None))
