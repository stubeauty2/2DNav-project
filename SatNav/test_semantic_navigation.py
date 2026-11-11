import json
import tempfile
import unittest
import math
from types import SimpleNamespace
from pathlib import Path
from AVDN.agent.io import read_jsonl
from AVDN.tools.sam3 import SAM3
from SatNav.environment import Environment
from SatNav.semantic_navigation import SemanticAgent
from SatNav.test_runtime import ScriptedModel


class PhaseTests(unittest.TestCase):
    def geometry_agent(self, region='lower_right', heading=0.):
        agent=SemanticAgent.__new__(SemanticAgent)
        agent.legs=[dict(arrival_region=region,distance_m=None,align_region=None)]
        agent.phase=0;agent.phase_origin=[0.,0.];agent.position=[0.,0.]
        agent.phase_heading=heading;agent.local_search_steps=0
        agent.env=SimpleNamespace(spec=dict(main_width_m=100.,main_height_m=100.))
        return agent

    def test_wrong_side_cannot_be_made_valid_by_lateral_alignment(self):
        agent=self.geometry_agent()
        self.assertIsNotNone(agent.candidate_conflict({'center_xy_m':[-10.,40.]}))
        self.assertIsNone(agent.candidate_conflict({'center_xy_m':[20.,40.]}))
        self.assertEqual(agent.desired_camera({'center_xy_m':[20.,40.]}),[0.,70.])
        # The same rule must rotate with the phase, rather than use global east.
        agent=self.geometry_agent(heading=90.)
        self.assertIsNotNone(agent.candidate_conflict({'center_xy_m':[40.,10.]}))
        target=agent.desired_camera({'center_xy_m':[40.,-20.]})
        self.assertAlmostEqual(target[0],70.)
        self.assertAlmostEqual(target[1],0.)

    def test_visible_stop_does_not_invent_object_center_arrival(self):
        agent=self.geometry_agent(region=None)
        self.assertEqual(agent.desired_camera({'center_xy_m':[25.,40.]},final=True),[0.,0.])

    def test_local_search_stays_bounded_and_escapes_repeated_boundary(self):
        agent=self.geometry_agent()
        endpoint=[0.,100.];agent.position=endpoint[:]
        for _ in range(8):
            target=agent.bounded_local_search([0.,1000.],endpoint)
            self.assertLessEqual(math.dist(endpoint,target),50.+1e-6)
            self.assertLessEqual(math.dist(agent.position,target),50.+1e-6)
            self.assertNotEqual(target,agent.position)
            agent.position=target

    def test_semantic_prompts_do_not_inherit_opposing_route_policy(self):
        from SatNav.semantic_prompts import SEMANTIC_SYSTEM, SEMANTIC_REVIEW_SYSTEM
        from AVDN.agent.prompts import SYSTEM
        self.assertNotIn('No fixed stage order is required',SEMANTIC_SYSTEM)
        self.assertNotIn('not a compulsory sequence',SEMANTIC_SYSTEM)
        self.assertNotIn('auxiliary center',SEMANTIC_REVIEW_SYSTEM)
        self.assertIn('No fixed stage order is required',SYSTEM)
    def test_intermediate_region_then_turn_and_final_reobservation(self):
        source=Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():self.skipTest('Prepare pilot first')
        entry=read_jsonl(source)[0]
        entry=dict(entry,agent_input=dict(entry['agent_input'],initial_pose=dict(xy_m=[0.,0.],heading_deg=0.)),
                   initialization=dict(direction=200.,final_target='pool',auxiliary=['roof']))
        plan=dict(unsupported_reason=None,legs=[
            dict(landmark='roof',distance_m=None,arrival_region='middle_right',turn_direction='right',turn_deg=None,align_region='upper_center'),
            dict(landmark='pool',distance_m=20.,arrival_region=None,turn_direction=None,turn_deg=None,align_region=None)])
        class PhaseModel(ScriptedModel):
            def request(self,context,views):
                self.successful_views.update(v['view_id'] for v in views);self.cursor+=1
                phase=context['semantic_route']['phase_index']
                bbox=[750,450,850,550] if phase==0 else [400,400,600,600]
                if context['mode']=='review':action=dict(action='review',verdict='confirm',view_id=views[0]['view_id'],bbox=bbox,reason='matching current landmark')
                else:action=dict(action='locate',target='final',view_id=views[0]['view_id'],bbox=bbox,reason='candidate')
                return dict(status='ok',response=dict(text=json.dumps(action)))
        with tempfile.TemporaryDirectory() as folder:
            env=Environment(entry['environment'],Path(folder)/'views')
            try:
                agent=SemanticAgent(entry,env,PhaseModel(),SAM3(),folder,semantic_plan=plan)
                self.assertEqual(agent.heading,0.)
                self.assertEqual(agent.observation_target(),'roof')
                self.assertNotIn('Use the final-direction prior',agent.context('decision',[],0)['navigation_contract']['exploration'])
                old_key=agent.observation_cache_key({'view_id':'same'})
                result=agent.run()
                self.assertEqual(result['termination'],'confirmed_final')
                self.assertEqual(agent.phase,1)
                self.assertEqual(agent.observation_target(),'pool')
                self.assertNotEqual(old_key,agent.observation_cache_key({'view_id':'same'}))
                turn=next(p for p in agent.trajectory if p['source']=='semantic_turn')
                self.assertAlmostEqual(turn['xy_m'][0],0.,places=5)
                self.assertAlmostEqual(turn['heading_deg'],90.,places=5)
                self.assertAlmostEqual(result['actual_stop_xy_m'][0],20.,places=5)
                self.assertAlmostEqual(result['actual_stop_xy_m'][1],0.,places=5)
                self.assertEqual(sum(e['kind']=='review_result' for e in agent.events),3)
                self.assertTrue(any(e['kind']=='semantic_stop_deferred' for e in agent.events))
                self.assertNotIn('goal_xy_m',json.dumps(agent.context('decision',[],0)))
            finally:env.close()
