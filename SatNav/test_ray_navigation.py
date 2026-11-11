import unittest
from unittest.mock import patch
from types import SimpleNamespace
from .ray_navigation import RayAgent, stopping_side, stopping_vertical
from .route_geometry import final_ray


class RayTests(unittest.TestCase):
    def test_region_stop_uses_observation_thirds_without_filling_distance(self):
        agent=RayAgent.__new__(RayAgent)
        agent.ray=final_ray({'stages':[{'heading_deg':0.,'distance_m':100.},{'heading_deg':0.,'distance_m':None}]})
        agent.env=SimpleNamespace(spec={'main_height_m':90.})
        agent.region_stop=True;agent.ray_vertical=stopping_vertical('lower right')
        self.assertEqual(agent.stopping_position([10.,150.]),[0.,180.])
        agent.ray_vertical=stopping_vertical('upper middle')
        self.assertEqual(agent.stopping_position([10.,150.]),[0.,120.])
        self.assertEqual(agent.stopping_position([10.,110.]),[0.,100.])
        agent.region_stop=False
        self.assertEqual(agent.stopping_position([10.,150.]),[0.,150.])

    def test_grounding_proposals_are_filtered_by_public_geometry_not_identity(self):
        agent=RayAgent.__new__(RayAgent)
        agent.ray=final_ray({'stages':[{'heading_deg':0.,'distance_m':100.},{'heading_deg':0.,'distance_m':None}]})
        agent.ray_ready=True;agent.ray_side='left';agent.round=0
        agent.env=SimpleNamespace(locate=lambda view,box:{'center_xy_m':box[:2]},
                                  tool_overlay=lambda view,items,name:{'candidates':items})
        candidates=[{'candidate_id':str(i),'bbox':[x,150.,x+1,151.]} for i,x in enumerate([20.,-10.,-40.])]
        result={'status':'ok','candidates':candidates,'rejected_candidates':[]}
        with patch('SatNav.navigation.Agent.ground',return_value=(result,None)):
            filtered,overlay=agent.ground({},'sam3','roof',1)
        self.assertEqual([x['candidate_id'] for x in filtered['candidates']],['1'])
        self.assertEqual(len(filtered['rejected_candidates']),2)
        self.assertEqual(overlay['candidates'],filtered['candidates'])
        self.assertIn('not identity',filtered['note'])

    def test_visual_rejections_cannot_stall_at_one_pose_indefinitely(self):
        agent=RayAgent.__new__(RayAgent)
        agent.ray={'source':'test'};agent.ray_ready=True;agent.review_stalls=0
        agent.position=[0.,0.];agent.termination=None;agent.event=lambda *args,**kwargs:None
        with patch('AVDN.navigation.Agent.review'),patch.object(agent,'advance_ray') as advance:
            agent.review({}, {}, {})
            advance.assert_not_called()
            agent.review({}, {}, {})
            advance.assert_called_once_with('partial_review_recovery')

    def test_side_uses_stopping_phrase_not_general_left_turn(self):
        self.assertEqual(stopping_side('pool in the lower-left'),'left')
        self.assertEqual(stopping_side('appears middle right'),'right')
        self.assertIsNone(stopping_side('turn left then stop at the roof'))

    def test_conflicting_candidate_retries_current_view_before_moving(self):
        agent=RayAgent.__new__(RayAgent)
        agent.ray=final_ray({'stages':[{'heading_deg':0.,'distance_m':100.}, {'heading_deg':0.,'distance_m':None}]})
        agent.ray_ready=True;agent.ray_side='left';agent.ray_rejections=0
        agent.ray_main={};agent.memory={'candidates':[]};agent.region_stop=False
        agent.event=lambda *args,**kwargs:None
        refined={'center_xy_m':[20.,150.]}
        with patch('SatNav.ray_navigation.normalized',return_value=[[500.,500.]]),patch.object(agent,'advance_ray') as advance:
            agent.final_confirmed(refined,'lookalike',{})
            agent.final_confirmed(refined,'lookalike',{})
            advance.assert_not_called()
            self.assertTrue(agent.memory['candidates'][-1]['geometry']['side_conflict'])
            agent.final_confirmed(refined,'lookalike',{})
            advance.assert_called_once()


if __name__=='__main__':unittest.main()
