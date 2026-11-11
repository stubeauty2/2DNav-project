import json
import unittest
from .semantic_plan import parse_plan, validate_plan
from .search_prior import search_region


class SemanticPlanSyntaxTests(unittest.TestCase):
    def legs(self):
        return [dict(landmark='roof',distance_m=None,arrival_region='lower_middle',
                     turn_direction='left',turn_deg=None,align_region=None),
                dict(landmark='pool',distance_m=110,arrival_region=None,
                     turn_direction=None,turn_deg=None,align_region=None)]

    def test_uncertain_geometry_is_not_promoted_to_exact_geometry(self):
        plan=parse_plan(json.dumps(self.legs()),allow_qualitative=True)
        self.assertEqual(plan['legs'][0]['arrival_region'],'lower_center')
        self.assertIsNone(plan['legs'][0]['turn_deg'])
        with self.assertRaises(ValueError):validate_plan(plan)
        prior=search_region(0,plan,dict(distance_median_m=150,angle_median_deg=105))
        self.assertTrue(prior['uncertain'])
        self.assertEqual(len(prior['imputed']),2)
        self.assertNotIn('endpoint_xy_m',prior)

    def test_unknown_region_and_final_turn_still_rejected(self):
        legs=self.legs();legs[0]['arrival_region']='somewhere'
        with self.assertRaises(ValueError):parse_plan(json.dumps(legs),allow_qualitative=True)
        legs=self.legs();legs[-1]['turn_direction']='right'
        with self.assertRaises(ValueError):parse_plan(json.dumps(legs),allow_qualitative=True)

    def test_exact_explicit_values_preserved(self):
        legs=self.legs();legs[0].update(turn_deg=75,align_region='upper_left')
        plan=parse_plan(json.dumps(dict(legs=legs,unsupported_reason=None)))
        self.assertEqual(plan['legs'][0]['turn_deg'],75)
        self.assertEqual(plan['legs'][1]['distance_m'],110)


if __name__=='__main__':unittest.main()
