"""Synthetic read-only journal analysis; no inference, network, or evaluator data."""
import copy
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('review_usage', Path(__file__).with_name('review_usage_report.py'))
usage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(usage)


class ReviewUsageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root/'episodes/AVDN/test-model/jev/episode'
        (self.folder/'tools').mkdir(parents=True)
        self.record = dict(status='ok', signature='one', verdict='uncertain', choice='confirm',
            max_probability=.6, abstained=True, assets=[{'sha256': 'image1'}, {'sha256': 'image2'}],
            input={'state': {'context': {'round': 1, 'instruction_raw': 'Find the pair.', 'turns': [],
                'initialization': {'final_target': 'pair', 'auxiliary': ['bridge']},
                'candidate': {'target': 'final', 'view_id': 'main', 'bbox': [1, 2, 3, 4],
                              'projection_in_review': {'view_id': 'r01-review'}}}}},
            raw={'answers': {'review': {'choice': 'confirm'}}, 'usage': {'input_tokens': 100}})

    def write(self, name, value):
        (self.folder/'tools'/name).write_text(json.dumps(value), encoding='utf-8')

    def test_same_evidence_different_round_is_not_same_full_request(self):
        self.write('r01-jev-review.json', self.record)
        second = copy.deepcopy(self.record)
        second['signature'] = 'two'
        second['input']['state']['context']['round'] = 2
        second['input']['state']['context']['candidate']['projection_in_review']['view_id'] = 'r02-review'
        self.write('r02-jev-review.json', second)
        report = usage.summarize(self.root)
        group = report['episodes'][0]['groups'][0]
        self.assertEqual((group['count'], group['repeated_calls']), (2, 1))
        self.assertEqual((group['distinct_full_journal_inputs'], group['distinct_request_signatures']), (2, 2))
        self.assertEqual(report['summary']['raw_to_verdict'], {'confirm->uncertain': 2})

    def test_candidate_description_image_order_or_role_changes_split(self):
        self.write('r00-jev-review.json', self.record)
        for index in range(4):
            altered = copy.deepcopy(self.record)
            context = altered['input']['state']['context']
            if index == 0: context['candidate']['bbox'][0] += 1
            if index == 1: context['initialization']['final_target'] = 'another pair'
            if index == 2: altered['assets'].reverse()
            if index == 3: context['candidate']['target'] = 'auxiliary'
            self.write(f'r{index+1:02d}-jev-review.json', altered)
        self.assertEqual(usage.summarize(self.root)['summary']['evidence_groups'], 5)

    def test_pending_and_broken_logs_not_counted_and_no_evaluator_reads(self):
        self.write('r01-jev-review.json', self.record)
        pending = dict(self.record, status='pending')
        self.write('r02-jev-review.json', pending)
        (self.folder/'tools/r03-jev-review.json').write_text('{', encoding='utf-8')
        (self.root/'evaluator_only.json').write_text('DO NOT READ', encoding='utf-8')
        read = usage.load
        with patch.object(usage, 'load', side_effect=lambda path: read(path) if path.name != 'evaluator_only.json' else self.fail('Read GT')):
            report = usage.summarize(self.root)
        self.assertEqual(report['summary']['terminal_reviews'], 1)
        self.assertEqual(report['skipped_pending'], 1)
        self.assertEqual(len(report['read_errors']), 1)

    def test_move_count_and_zero_review_episode(self):
        (self.folder/'trace.json').write_text(json.dumps({'termination': 'step_limit', 'events': [
            {'kind': 'move'}, {'kind': 'decision'}, {'kind': 'move'}]}), encoding='utf-8')
        report = usage.summarize(self.root)
        self.assertEqual(report['episodes'][0]['move_count'], 2)
        self.assertEqual(report['episodes'][0]['terminal_review_count'], 0)

    def test_error_separate_from_semantic_uncertainty_and_quantiles(self):
        record = dict(self.record, status='error', raw=None, choice=None)
        self.write('r01-jev-review.json', record)
        report = usage.summarize(self.root)
        self.assertEqual(report['summary']['statuses'], {'error': 1})
        self.assertEqual(report['summary']['raw_to_verdict'], {'unavailable->uncertain': 1})
        self.assertEqual(report['summary']['input_tokens'], {'n': 0})
        self.assertEqual(usage.percentiles([100, 200])['p50'], 150)

    def test_plan_excludes_old_directories_and_binds_exact_file_bytes(self):
        self.write('r01-jev-review.json', self.record)
        old = self.root/'episodes/AVDN/test-model/jev/old-episode/tools'
        old.mkdir(parents=True)
        (old/'r01-jev-review.json').write_text('{', encoding='utf-8')
        jobs = [dict(dataset='AVDN', model='test-model', arm='jev', episode_id='episode'),
                dict(dataset='AVDN', model='test-model', arm='jev', episode_id='not-started'),
                dict(dataset='AVDN', model='test-model', arm='mllm', episode_id='old-episode')]
        plan = self.root/'plan.json'
        plan.write_text(json.dumps({'jobs': jobs}, indent=2), encoding='utf-8')
        report = usage.summarize(self.root)
        self.assertEqual(report['binding'], 'plan')
        self.assertEqual(report['plan_sha256'], hashlib.sha256(plan.read_bytes()).hexdigest())
        self.assertEqual(report['planned_jev_jobs'], 2)
        self.assertEqual(report['summary']['episodes'], 1)
        self.assertEqual(report['summary']['terminal_reviews'], 1)
        self.assertEqual(report['read_errors'], [])

    def test_no_plan_explicitly_unbound_with_scan_end_time(self):
        report = usage.summarize(self.root)
        self.assertEqual(report['binding'], 'unbound')
        self.assertIsNone(report['plan_sha256'])
        self.assertGreaterEqual(datetime.fromisoformat(report['snapshot_finished_utc']),
                                datetime.fromisoformat(report['snapshot_utc']))
        self.assertIn('Non-transactional', report['snapshot_consistency'])

    def test_existing_invalid_plan_cannot_silently_fall_back_to_scan(self):
        (self.root/'plan.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'jobs list'):
            usage.summarize(self.root)

    def test_plan_job_cannot_read_outside_episode_directory(self):
        plan = {'jobs': [dict(dataset='AVDN', model='test-model', arm='jev', episode_id='../elsewhere')]}
        (self.root/'plan.json').write_text(json.dumps(plan), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'single path components'):
            usage.summarize(self.root)


if __name__ == '__main__':
    unittest.main()
