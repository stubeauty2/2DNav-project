import copy
import unittest
from SatNav.initialization.prior import estimate_bearing
from SatNav.route_geometry import numeric_route


class PriorTests(unittest.TestCase):
    def test_qualitative_left_turn_is_only_an_uncertain_bearing(self):
        legs=[dict(distance_m=None,turn_direction='left',turn_deg=None),
              dict(distance_m=None,turn_direction=None,turn_deg=None)]
        original=copy.deepcopy(legs)
        result=estimate_bearing(0.,legs,150.,105.)
        self.assertAlmostEqual(result['direction'],307.5)
        self.assertEqual(legs,original)
        self.assertTrue(result['uncertain'])
        self.assertNotIn('endpoint_xy_m',result)
        self.assertEqual(len(result['imputed']),3)
        self.assertIsNone(numeric_route({'stages':[dict(heading_deg=0.,distance_m=None),dict(heading_deg=None,distance_m=None)]}))

    def test_explicit_values_override_priors(self):
        legs=[dict(distance_m=100.,turn_direction='right',turn_deg=90.),
              dict(distance_m=100.,turn_direction=None,turn_deg=None)]
        result=estimate_bearing(30.,legs,150.,105.)
        self.assertAlmostEqual(result['direction'],75.)
        self.assertEqual(result['imputed'],[])

    def test_positional_priors_preserve_explicit_leg_lengths(self):
        legs=[dict(distance_m=None,turn_direction='left',turn_deg=None),
              dict(distance_m=90.,turn_direction=None,turn_deg=None)]
        result=estimate_bearing(0.,legs,[160.,150.],[105.])
        self.assertEqual(result['imputed'],[dict(leg=0,field='distance_m',value=160.),dict(leg=0,field='turn_deg',value=105.)])
        self.assertGreater(result['direction'],300.)
        self.assertLess(result['direction'],340.)
        with self.assertRaises(ValueError):estimate_bearing(0.,legs,[150.],[105.])
