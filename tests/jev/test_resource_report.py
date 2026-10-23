import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('resource_report', Path(__file__).with_name('resource_report.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ResourceReportTest(unittest.TestCase):
    def test_image_tokens_are_a_subset_and_missing_usage_is_explicit(self):
        usage = module.token_usage({'response': {'usage': {'input_tokens': 10, 'output_tokens': 2,
            'total_tokens': 12, 'image_tokens': 3, 'input_tokens_details': {'image_tokens': 3}}}})
        self.assertEqual(usage['total_tokens'], 12)
        self.assertEqual(usage['image_tokens'], 3)
        self.assertIsNone(usage['cached_tokens'])
        self.assertIsNone(module.token_usage({'status': 'pending'}))
        self.assertIsNone(module.token_usage({'response': {'usage': {'total_tokens': float('nan')}}})['total_tokens'])

    def test_pending_usage_only_in_observed_totals_and_paired_changes_use_terminal(self):
        online = module.online
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ids = {dataset: [dataset+'-one'] for dataset in online.POOLS}
            jobs = online.make_jobs(ids)
            plan = dict(jobs=jobs, selected=[dict(dataset=d, episode_id=i[0]) for d,i in ids.items()])
            online.save(root / 'plan.json', plan)
            targeted = [next(j for j in jobs if j['dataset']=='AVDN' and j['model']==online.MODELS[0] and j['arm']==a)
                        for a in ('mllm', 'jev', 'mllm_fixed')]
            for job, n, state in zip(targeted, (3, 2, 1), ('complete', 'complete', 'pending')):
                folder = online.job_folder(root, job)
                online.save(folder/'attempt.json', dict(status=state, started_unix=10, finished_unix=20))
                for i in range(n):
                    online.save(folder/'requests'/f'{i}.json', dict(status='ok', assets=['image'], context={'mode':'navigate'},
                        response={'usage':dict(input_tokens=10, output_tokens=2, total_tokens=12, image_tokens=3)}))
            with patch.object(online, 'validate_frozen', return_value=plan):
                report = module.summarize(root)
            self.assertEqual(report['observed_mllm_requests'], 6)
            self.assertEqual(report['observed_reported_tokens']['total_tokens'], 72)
            pending_group = next(g for g in report['groups'] if (g['dataset'],g['model'],g['arm'])==
                                 ('AVDN',online.MODELS[0],'mllm_fixed'))
            self.assertEqual(pending_group['terminal'], 0)
            self.assertIsNone(pending_group['mean_mllm_requests'])
            pair = next(p for p in report['paired_request_changes'] if (p['dataset'],p['model'],p['baseline'])==
                        ('AVDN',online.MODELS[0],'mllm'))
            self.assertEqual(pair['n'], 1)
            self.assertEqual(pair['mean_mllm_request_change_jev_minus_baseline'], -1)
            self.assertFalse(report['all_terminal'])
            self.assertFalse((root/'evaluator_only.json').exists())


if __name__ == '__main__':
    unittest.main()
