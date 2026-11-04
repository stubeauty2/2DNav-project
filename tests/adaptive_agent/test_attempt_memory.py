"""Regression for historical tool attempts lost by adaptive presentation."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from AVDN.tools.memory import attempt_index, presentation_adaptive


class AttemptMemoryTests(unittest.TestCase):
    def test_current_pose_and_compass_are_separate_from_previous_view(self):
        shown=presentation_adaptive(dict(mode='decision',heading_deg=260,position_xy_m=[4,5],
            views=[dict(view_id='old',source='main_previous',image_up_heading_deg=0),
                   dict(view_id='now',source='main',image_up_heading_deg=90)]))
        current=shown['current_request']
        self.assertEqual(current['view_id'],'now')
        self.assertEqual(current['heading_deg'],260)
        self.assertEqual(current['compass_image_vectors']['north'],[-1.,0.])
        self.assertEqual(current['compass_image_vectors']['east'],[0.,-1.])

    def test_duplicate_attempts_retain_parameters_status_and_count(self):
        action = dict(action='tool', tool='visual_question', view_id='r00-main',
                      question='Compare the roof openings', regions=[], reason='inspect')
        events = [dict(kind='tool', round=i, input=action,
                       output=dict(status='ok', cache_hit=i>0, evidence_id='same')) for i in range(3)]
        context = dict(mode='decision', optional_tools_remaining=4,
                       capabilities=dict(zoom=True), tool_attempts=attempt_index(events))
        shown = presentation_adaptive(context)
        self.assertEqual(len(shown['tool_attempts']), 1)
        attempt = shown['tool_attempts'][0]
        self.assertEqual(attempt['attempts'], 3)
        self.assertEqual(attempt['question'], action['question'])
        self.assertTrue(attempt['cache_hit'])
        self.assertEqual(attempt['evidence_id'], 'same')
        self.assertEqual(len(context['tool_attempts']), 3)

    def test_error_and_success_are_distinct_and_old_views_remain(self):
        events = [dict(kind='tool', round=i,
                       input=dict(tool='dino', view_id=view, query='building'),
                       output=dict(status=status))
                  for i,(view,status) in enumerate([('r00-main','error'),('r00-main','ok'),('r01-main','ok')])]
        shown = presentation_adaptive(dict(mode='decision', tool_attempts=attempt_index(events)))
        self.assertEqual(len(shown['tool_attempts']), 3)
        self.assertEqual(shown['tool_attempts'][0]['status'], 'error')


if __name__ == '__main__':
    unittest.main()
