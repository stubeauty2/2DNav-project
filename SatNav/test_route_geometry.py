import unittest
from .route_geometry import numeric_route, guidance, final_ray, project_ray


class RouteTests(unittest.TestCase):
    def test_partial_ray_preserves_unknown_length_and_clamps_backward_projection(self):
        stages=[{'heading_deg':0.,'distance_m':100.},{'heading_deg':90.,'distance_m':None}]
        ray=final_ray({'stages':stages})
        self.assertEqual(ray['anchor_xy_m'],[0.,100.])
        point,distance=project_ray(ray,[50.,120.])
        self.assertAlmostEqual(point[0],50.)
        self.assertAlmostEqual(point[1],100.)
        self.assertAlmostEqual(distance,20.)
        self.assertEqual(project_ray(ray,[-50.,100.])[0],[0.,100.])
        self.assertIsNone(stages[-1]['distance_m'])
        stages[0]['distance_m']=None
        self.assertIsNone(final_ray({'stages':stages}))
        stages[0]['distance_m']=100.;stages[-1]['heading_deg']=None
        self.assertIsNone(final_ray({'stages':stages}))

    def test_relative_route_endpoint_and_turn(self):
        route=numeric_route({'stages':[{'heading_deg':0,'distance_m':100},
                                       {'heading_deg':90,'distance_m':100}]})
        self.assertAlmostEqual(route['endpoint_xy_m'][0],100)
        self.assertAlmostEqual(route['endpoint_xy_m'][1],100)
        g=guidance(route,[100,0],90)
        self.assertAlmostEqual(g['relative_turn_deg'],-90)
        self.assertAlmostEqual(g['remaining_distance_m'],100)

    def test_unknown_geometry_never_filled(self):
        self.assertIsNone(numeric_route({'stages':[{'heading_deg':0,'distance_m':None}]}))
        self.assertIsNone(numeric_route({'stages':[{'heading_deg':None,'distance_m':100}]}))
        self.assertIsNone(numeric_route({}))


if __name__=='__main__':unittest.main()
