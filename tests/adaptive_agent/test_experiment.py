"""Check the paired-report scorer against independently audited real trajectories."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import experiment


class ReportTests(unittest.TestCase):
    def test_scoring_matches_archived_audit_without_model_or_detector_calls(self):
        source=experiment.ROOT/'visible_02'
        if not (source/'plan.json').exists():
            self.skipTest('Requires the local frozen visible_02 audit inputs')
        plan=experiment.read(source/'plan.json')
        audit=experiment.read(experiment.AUDIT)
        expected={r['episode_id']:r for r in audit['rows'] if r['group']=='historical'}
        with tempfile.TemporaryDirectory(prefix='2dnav-evaluation-') as td:
            root=Path(td);batch=root/'fixture'
            experiment.save(batch/'plan.json',plan)
            experiment.save(batch/'evaluator_only.json',experiment.read(source/'evaluator_only.json'))
            for job in plan['jobs']:
                old=Path(job['historical_folder'])
                for arm in ('before','after'):
                    dest=batch/'runs'/arm/job['episode_id']
                    (dest/'requests').mkdir(parents=True)
                    for name in ('trace.json','result.json'):
                        shutil.copyfile(old/name,dest/name)
                    for request in (old/'requests').glob('*.json'):
                        shutil.copyfile(request,dest/'requests'/request.name)
            with patch.object(experiment,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):
                experiment.report(SimpleNamespace(batch='fixture'))
            result=experiment.read(batch/'evaluation.json')
            self.assertEqual(len(result['rows']),2*len(plan['jobs']))
            for row in result['rows']:
                reference=expected[row['episode_id']]
                self.assertEqual(row['success'],reference['success'])
                if reference['final_metric'] is None:self.assertIsNone(row['metric'])
                else:self.assertAlmostEqual(row['metric'],reference['final_metric'],places=8)
                self.assertEqual(row['correct_proposal'],reference['correct_proposal'])
                self.assertEqual(row['target_center_in_main'],reference['main_center_covered'])
                rejected=sum(p['target']=='final' and p['success'] and
                    p.get('review',{}).get('verdict') in ('reject','uncertain') for p in reference['proposals'])
                self.assertEqual(row['correct_proposal_rejected'],rejected)


if __name__=='__main__':unittest.main()
