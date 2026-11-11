"""Offline control-flow checks; these are not online model success evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from AVDN.agent.io import read_jsonl
from AVDN.tools.sam3 import SAM3
from .environment import Environment
from .navigation import Agent


class ScriptedModel:
    def __init__(self):
        self.cursor = 0
        self.successful_views = set()

    def request(self, context, views):
        self.successful_views.update(v['view_id'] for v in views)
        view = views[0]['view_id']
        if self.cursor == 0:
            action = dict(action='explore', mode='offset', turn_deg=0, step_scale=.1, reason='test movement')
        elif self.cursor == 1:
            action = dict(action='locate', target='final', view_id=view, bbox=[400,400,600,600], reason='test proposal')
        elif self.cursor == 2:
            action = dict(action='review', verdict='confirm', view_id=view, bbox=[400,400,600,600], reason='test confirmation')
        else:
            action = dict(action='explore', mode='point', view_id=view, point=[600,500], reason='test stopping pose')
        self.cursor += 1
        return dict(status='ok', response=dict(text=json.dumps(action)))


class RuntimeTests(unittest.TestCase):
    def test_partial_route_requires_reobservation_after_known_prefix(self):
        from .ray_navigation import RayAgent
        source=Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():self.skipTest('Prepare pilot first')
        entry=dict(read_jsonl(source)[0],initialization=dict(direction=45.,final_target='test',auxiliary=[]))
        class ConfirmModel(ScriptedModel):
            def request(self,context,views):
                self.successful_views.update(v['view_id'] for v in views)
                self.cursor+=1
                if context['mode']=='review':
                    action=dict(action='review',verdict='confirm',view_id=views[0]['view_id'],bbox=[400,400,600,600],reason='visual confirmation')
                else:
                    action=dict(action='locate',target='final',view_id=views[0]['view_id'],bbox=[400,400,600,600],reason='visible candidate')
                return dict(status='ok',response=dict(text=json.dumps(action)))
        with tempfile.TemporaryDirectory() as directory:
            env=Environment(entry['environment'],Path(directory)/'views')
            try:
                agent=RayAgent(entry,env,ConfirmModel(),SAM3(),directory,
                    route_diagnostics={'stages':[{'heading_deg':0.,'distance_m':100.},{'heading_deg':90.,'distance_m':None}]})
                result=agent.run()
                self.assertEqual(result['termination'],'confirmed_final')
                self.assertEqual([p['xy_m'][1] for p in agent.trajectory[:3]],[0.,50.,100.])
                self.assertEqual(sum(e['kind']=='partial_prefix_candidate_deferred' for e in agent.events),2)
                self.assertEqual(sum(e['kind']=='review_result' for e in agent.events),1)
                self.assertEqual(agent.trajectory[-1]['source'],'partial_route_stop')
                self.assertNotIn('goal_xy_m',json.dumps(agent.context('decision',[],0)))
            finally:env.close()

    def test_uncertain_route_exploration_cannot_skip_unobserved_footprints(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        agent=Agent.__new__(Agent)
        agent.bounded_exploration=True;agent.language_route=None;agent.route_guard=False
        agent.position=[0.,0.];agent.env=SimpleNamespace(spec={'main_height_m':100.})
        events=[];agent.event=lambda kind,**details:events.append((kind,details))
        with patch('AVDN.navigation.Agent.move') as move:
            agent.move([0.,250.],0.,'explore_offset')
            self.assertEqual(move.call_args.args[0],[0.,50.])
            self.assertEqual(events[0][0],'exploration_stride_clipped')
            agent.language_route={'endpoint_xy_m':[0.,250.]}
            agent.move([0.,250.],0.,'explore_offset')
            self.assertEqual(move.call_args.args[0],[0.,250.])
            agent.language_route=None;agent.bounded_exploration=False
            agent.move([0.,250.],0.,'explore_offset')
            self.assertEqual(move.call_args.args[0],[0.,250.])

    def test_stop_gate_requires_new_confirmation_after_bounded_moves(self):
        source=Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():self.skipTest('Prepare pilot first')
        entry=dict(read_jsonl(source)[0],initialization=dict(direction=0.,final_target='test',auxiliary=[]))
        class ConfirmModel(ScriptedModel):
            def request(self, context, views):
                self.successful_views.update(v['view_id'] for v in views)
                self.cursor+=1
                if context['mode']=='review':
                    action=dict(action='review',verdict='confirm',view_id=views[0]['view_id'],bbox=[400,400,600,600],reason='visual confirmation')
                else:
                    action=dict(action='locate',target='final',view_id=views[0]['view_id'],bbox=[400,400,600,600],reason='visible candidate')
                return dict(status='ok',response=dict(text=json.dumps(action)))
        with tempfile.TemporaryDirectory() as directory:
            env=Environment(entry['environment'],Path(directory)/'views')
            try:
                agent=Agent(entry,env,ConfirmModel(),SAM3(),directory,
                            route_diagnostics={'stages':[{'heading_deg':0.,'distance_m':150.}]},route_stop_gate=True)
                result=agent.run()
                self.assertEqual(result['termination'],'confirmed_final')
                self.assertEqual([p['xy_m'][1] for p in agent.trajectory[:3]],[0.,70.,140.])
                self.assertAlmostEqual(result['actual_stop_xy_m'][1],150.,places=5)
                self.assertEqual(sum(e['kind']=='stop_deferred_for_route' for e in agent.events),2)
                self.assertEqual(sum(e['kind']=='review_result' for e in agent.events),3)
                self.assertEqual(sum(e['kind']=='stop' for e in agent.events),1)
                for e in agent.events:
                    if e['kind']=='stop_deferred_for_route':self.assertNotIn('goal_xy_m',json.dumps(e))
            finally:env.close()

    def test_stop_gate_rejection_does_not_reuse_old_confirmation(self):
        source=Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():self.skipTest('Prepare pilot first')
        entry=dict(read_jsonl(source)[0],initialization=dict(direction=0.,final_target='test',auxiliary=[]))
        class RejectModel(ScriptedModel):
            def request(self,context,views):
                self.successful_views.update(v['view_id'] for v in views)
                i=self.cursor;self.cursor+=1
                if i==4:action=dict(action='give_up',reason='No matching object after reobservation')
                elif context['mode']=='review':action=dict(action='review',verdict='confirm' if i==1 else 'reject',view_id=views[0]['view_id'],bbox=[400,400,600,600],reason='scripted visual judgment')
                else:action=dict(action='locate',target='final',view_id=views[0]['view_id'],bbox=[400,400,600,600],reason='candidate')
                return dict(status='ok',response=dict(text=json.dumps(action)))
        with tempfile.TemporaryDirectory() as directory:
            env=Environment(entry['environment'],Path(directory)/'views')
            try:
                agent=Agent(entry,env,RejectModel(),SAM3(),directory,
                            route_diagnostics={'stages':[{'heading_deg':0.,'distance_m':100.}]},route_stop_gate=True)
                result=agent.run()
                self.assertEqual(result['termination'],'give_up')
                self.assertIsNone(agent.final)
                self.assertFalse(any(e['kind']=='stop' for e in agent.events))
            finally:env.close()

    def test_motion_guard_prevents_overshoot_then_releases(self):
        source=Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():self.skipTest('Prepare pilot first')
        entry=dict(read_jsonl(source)[0],initialization=dict(direction=0.,final_target='test',auxiliary=[]))
        with tempfile.TemporaryDirectory() as directory:
            env=Environment(entry['environment'],Path(directory)/'views')
            try:
                agent=Agent(entry,env,ScriptedModel(),SAM3(),directory,
                            route_diagnostics={'stages':[{'heading_deg':0.,'distance_m':100.}]},route_guard=True)
                agent.move([0.,200.],0.,'explore_offset')
                self.assertEqual(agent.position,[0.,70.])
                agent.move([0.,270.],0.,'explore_offset')
                self.assertEqual(agent.position,[0.,100.])
                self.assertTrue(agent.route_area_visited)
                agent.move([10.,100.],90.,'explore_offset')
                self.assertEqual(agent.position,[10.,100.])
                self.assertIsNone(agent.termination)
            finally:env.close()

    def test_numeric_stop_follows_visual_review(self):
        source=Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():
            self.skipTest('Prepare pilot first')
        entry=dict(read_jsonl(source)[0],initialization=dict(direction=0.,final_target='test object',auxiliary=[]))
        route=dict(stages=[dict(heading_deg=0.,distance_m=10.),dict(heading_deg=90.,distance_m=10.)])
        with tempfile.TemporaryDirectory() as directory:
            env=Environment(entry['environment'],Path(directory)/'views')
            try:
                agent=Agent(entry,env,ScriptedModel(),SAM3(),directory,route_diagnostics=route)
                result=agent.run()
                self.assertEqual(result['termination'],'confirmed_final')
                self.assertEqual(agent.trajectory[-1]['source'],'numeric_language_stop')
                self.assertAlmostEqual(result['actual_stop_xy_m'][0],10.,places=5)
                self.assertAlmostEqual(result['actual_stop_xy_m'][1],10.,places=5)
                kinds=[e['kind'] for e in agent.events]
                self.assertLess(kinds.index('review_result'),kinds.index('stop_pose_from_language'))
                self.assertEqual(agent.model.cursor,3)
            finally:env.close()

    def test_prompt_selection_does_not_change_base_defaults(self):
        from AVDN.tools.llm import Model
        from AVDN.agent.prompts import SYSTEM, REVIEW_SYSTEM
        from .prompts import SYSTEM as SATNAV_SYSTEM, EXPLORATION_SYSTEM
        with tempfile.TemporaryDirectory() as directory:
            original = Model(directory, None, 'test', 'unused')
            adapted = Model(directory, None, 'test', 'unused', decision_prompt=SATNAV_SYSTEM)
            exploration = Model(directory, None, 'test', 'unused', decision_prompt=EXPLORATION_SYSTEM)
            self.assertEqual(original.decision_prompt, SYSTEM)
            self.assertEqual(original.review_prompt, REVIEW_SYSTEM)
            self.assertEqual(adapted.decision_prompt, SATNAV_SYSTEM)
            self.assertEqual(exploration.review_prompt, REVIEW_SYSTEM)
            self.assertNotIn('STOPPING-POSE REQUEST', exploration.decision_prompt)

    def test_loop_moves_then_localizes_then_physically_stops(self):
        source = Path('out/satnav/pilot_20261109/inputs.jsonl')
        if not source.exists():
            self.skipTest('Prepare the frozen pilot first')
        entry = read_jsonl(source)[0]
        entry = dict(entry, initialization=dict(direction=0., final_target='test object', auxiliary=[]))
        with tempfile.TemporaryDirectory() as directory:
            env = Environment(entry['environment'], Path(directory)/'views')
            try:
                agent = Agent(entry, env, ScriptedModel(), SAM3(), directory)
                result = agent.run()
                self.assertEqual(result['termination'], 'confirmed_final')
                self.assertEqual(len(agent.trajectory), 3)
                self.assertEqual(agent.trajectory[-1]['source'], 'language_conditioned_stop')
                self.assertAlmostEqual(result['actual_stop_xy_m'][0], 10., places=5)
                self.assertAlmostEqual(result['actual_stop_xy_m'][1], 10., places=5)
                self.assertTrue(any(e['kind']=='stop' for e in agent.events))
                self.assertNotIn('goal_xy_m', json.dumps(agent.context('decision', [], 0)))
            finally:
                env.close()


if __name__ == '__main__':
    unittest.main()
