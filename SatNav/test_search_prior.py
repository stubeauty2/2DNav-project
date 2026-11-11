import copy
import math
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .search_prior import search_region,spiral_offsets
from .prior_navigation import PriorSearchAgent


class SearchPriorTests(unittest.TestCase):
    def test_real_loop_reobserves_after_approach_then_stops_at_observed_target(self):
        from AVDN.agent.io import read_jsonl
        from AVDN.tools.sam3 import SAM3
        from .environment import Environment
        source=Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():self.skipTest('Prepared raster fixture unavailable')
        entry=dict(read_jsonl(source)[0],initialization=dict(direction=None,final_target='test',auxiliary=[]))
        class Model:
            cursor=0
            def __init__(self):self.successful_views=set()
            def request(self,context,views):
                self.cursor+=1;self.successful_views.update(v['view_id'] for v in views)
                action=dict(action='review',verdict='confirm',view_id=views[0]['view_id'],bbox=[600,400,800,600],reason='visual match') if context['mode']=='review' else dict(action='locate',target='final',view_id=views[0]['view_id'],bbox=[600,400,800,600],reason='candidate')
                return dict(status='ok',response={'text':json.dumps(action)})
        with tempfile.TemporaryDirectory() as directory:
            env=Environment(entry['environment'],Path(directory)/'views')
            try:
                agent=PriorSearchAgent(entry,env,Model(),SAM3(),directory,
                    search_prior=dict(search_center_xy_m=[0.,100.],bearing_deg=0.,estimated_final_heading_deg=0.,uncertain=True),
                    route_diagnostics={'stages':[{'heading_deg':None,'distance_m':None}]})
                result=agent.run()
                self.assertEqual(result['termination'],'confirmed_final')
                self.assertEqual([x['xy_m'][1] for x in agent.trajectory[:3]],[0.,70.,100.])
                self.assertAlmostEqual(result['actual_stop_xy_m'][0],20.)
                self.assertAlmostEqual(result['actual_stop_xy_m'][1],100.)
                self.assertEqual(sum(e['kind']=='review_result' for e in agent.events),1)
            finally:env.close()

    def test_region_is_uncertain_and_preserves_explicit_distances(self):
        plan={'legs':[dict(landmark='roof',distance_m=None,arrival_region='lower_right',turn_direction='left',turn_deg=None,align_region='upper_left'),
                      dict(landmark='pool',distance_m=90.,arrival_region=None,turn_direction=None,turn_deg=None,align_region=None)],'unsupported_reason':None}
        original=copy.deepcopy(plan)
        prior=search_region(0.,plan,dict(distance_median_m=150.,angle_median_deg=105.))
        self.assertAlmostEqual(prior['search_center_xy_m'][0],-90*math.sin(math.radians(105)))
        self.assertAlmostEqual(prior['search_center_xy_m'][1],150+90*math.cos(math.radians(105)))
        self.assertTrue(prior['uncertain']);self.assertEqual(plan,original)
        self.assertNotIn('endpoint_xy_m',prior)

    def test_spiral_observations_overlap_and_do_not_repeat(self):
        offsets=spiral_offsets(50.)
        self.assertEqual(len(offsets),len({tuple(x) for x in offsets}))
        for a,b in zip([[0.,0.]]+offsets,offsets):self.assertAlmostEqual(math.dist(a,b),50.)

    def test_stop_uses_observed_center_never_search_prior(self):
        agent=PriorSearchAgent.__new__(PriorSearchAgent)
        agent.search_prior={'search_center_xy_m':[999.,999.]};agent.heading=0.;agent.position=[1.,2.];agent.round=0
        agent.env=SimpleNamespace(main=lambda *args:{'view_id':'stop'})
        agent.event=lambda *args,**kwargs:None
        with patch('AVDN.navigation.Agent.final_confirmed') as confirm,patch('AVDN.navigation.Agent.move') as move:
            agent.final_confirmed({'center_xy_m':[10.,20.]},'visual evidence',{})
        confirm.assert_called_once()
        self.assertEqual(move.call_args.args[1],[10.,20.])
        self.assertEqual(move.call_args.args[3],'observed_target_stop')


if __name__=='__main__':unittest.main()
