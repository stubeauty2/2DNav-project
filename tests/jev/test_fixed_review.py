"""Meaningful leakage/identity and abstention accounting checks."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import requests

spec = importlib.util.spec_from_file_location('fixed_review', Path(__file__).with_name('fixed_review.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class FixedReviewTests(unittest.TestCase):
    def test_nested_truth_rejected(self):
        with self.assertRaises(ValueError):
            module.no_truth({'feedback': [{'tool': {'goal_xy_m': [0, 0]}}]})
        module.no_truth({'candidate': {'center_xy_m': [0, 0], 'bbox': [0, 0, 1, 1]}})

    def test_uncertain_and_invalid_are_not_true_negatives(self):
        rows = [(True, 'confirm'), (True, 'reject'), (True, 'uncertain'),
                (True, 'invalid'), (False, 'reject'), (False, 'confirm'),
                (False, 'uncertain'), (False, 'invalid')]
        result = module.classify_counts(rows)
        self.assertEqual(result['TP'], 1)
        self.assertEqual(result['TN'], 1)
        self.assertEqual(result['FN'], 1)
        self.assertEqual(result['FP'], 1)
        self.assertEqual(result['rejection_of_qualifying_rate'], .25)
        self.assertEqual(result['nonconfirmation_of_qualifying_rate'], .75)
        self.assertEqual(result['decisive_count'], 4)

    def test_historical_verdict_strict(self):
        self.assertEqual(module.parse_verdict('{"action":"review","verdict":"confirm"}'), 'confirm')
        self.assertEqual(module.parse_verdict('{"verdict":"confirm"}'), 'invalid')
        self.assertEqual(module.parse_verdict('not JSON'), 'invalid')

    def test_public_image_identity_and_no_labels_needed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.png'
            image.write_bytes(b'original bytes')
            case = {'case_id': 'opaque', 'context': {'candidate': {'bbox': [1, 1, 2, 2]}},
                    'assets': [{'image_path': str(image), 'sha256': module.sha(image), 'view_id': 'v'}],
                    'original_system': 'review'}
            case['input_sha256'] = module.digest({k: case[k] for k in ('context', 'assets', 'original_system')})
            (root / 'public_requests.jsonl').write_text(module.json.dumps(case) + '\n', encoding='utf-8')
            module.save(root / 'design.json', {'threshold': .7, 'public_sha256': module.digest([case])})
            module.save(root / 'decision_policy.json', module.DECISION_POLICY)
            module.validate_public(root)
            self.assertFalse((root / 'evaluator_only.json').exists())
            image.write_bytes(b'modified bytes')
            with self.assertRaises(ValueError):
                module.validate_public(root)

    def test_run_never_needs_labels_and_does_not_resend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.png'
            image.write_bytes(b'fixed image bytes')
            context = {'mode': 'review', 'candidate': {'target': 'final', 'bbox': [1, 1, 2, 2]},
                       'views': [{'view_id': 'main'}, {'view_id': 'review'}]}
            assets = [{'view_id': name, 'image_path': str(image), 'sha256': module.sha(image)}
                      for name in ('main', 'review')]
            case = dict(case_id='fixed', context=context, assets=assets, original_system='original')
            case['input_sha256'] = module.digest({k: case[k] for k in ('context', 'assets', 'original_system')})
            (root / 'public_requests.jsonl').write_text(module.json.dumps(case) + '\n', encoding='utf-8')
            module.save(root / 'design.json', {'threshold': .7, 'public_sha256': module.digest([case])})
            module.save(root / 'decision_policy.json', module.DECISION_POLICY)
            calls = []
            class FakeJev:
                def __init__(self, **kwargs):
                    self.manifest = None
                def verify(self, ctx, views, output):
                    calls.append(ctx)
                    return {'status': 'ok', 'verdict': 'confirm'}
            fake = SimpleNamespace(Jev=FakeJev, __file__=module.__file__, Halt=RuntimeError,
                                   build_review_payload=lambda ctx, views, model: ({}, {'state': {'context': ctx}}, assets))
            health = SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'status': 'ok', 'model': 'fixed-model'})
            with patch.object(module.importlib, 'import_module', return_value=fake), patch.object(requests, 'get', return_value=health):
                module.run(root, model='fixed-model')
                module.run(root, model='fixed-model')
            self.assertEqual(len(calls), 1)
            self.assertFalse((root / 'evaluator_only.json').exists())
            record = module.read(root / 'responses/fixed.json')
            record['status'] = 'pending'
            module.save(root / 'responses/fixed.json', record)
            with patch.object(module.importlib, 'import_module', return_value=fake), patch.object(requests, 'get', return_value=health):
                with self.assertRaises(RuntimeError):
                    module.run(root, model='fixed-model')
            self.assertEqual(len(calls), 1)


if __name__ == '__main__':
    unittest.main()
