"""Paired gap diagnostics: sign reversals, joint declines, and incomplete cells."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location('gap_report', Path(__file__).with_name('gap_report.py'))
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture(values):
    """Input columns: baseline flash/max, Jev flash/max; both baselines equal."""
    selected = [dict(dataset='AVDN', episode_id=str(i)) for i in range(len(values))]
    jobs, rows = [], []
    for i, (bf, bm, jf, jm) in enumerate(values):
        for model, base, jev in zip(report.MODELS, (bf, bm), (jf, jm)):
            for arm in report.ARMS:
                job = dict(dataset='AVDN', episode_id=str(i), model=model, arm=arm)
                jobs.append(job)
                rows.append(dict(job, status='complete', success=bool(jev if arm == 'jev' else base)))
    plan = dict(seed=20261022, selected=selected, jobs=jobs,
                settings=dict(models=list(report.MODELS), arms=list(report.ARMS)))
    return plan, dict(rows=rows)


class GapReportTests(unittest.TestCase):
    def analyze(self, values):
        plan, analysis = fixture(values)
        return report.diagnostics(plan, analysis, repeats=200)['rows'][0]

    def test_signed_gap_falls_but_absolute_gap_grows(self):
        row = self.analyze([(0, 1, 1, 0), (1, 1, 1, 0), (1, 1, 1, 1), (0, 0, 1, 1)])
        self.assertEqual([row[k] for k in report.RATE_KEYS], [.5, .75, 1., .5])
        self.assertEqual(row['signed_gap_change'], -.75)
        self.assertEqual(row['absolute_gap_change'], .25)
        self.assertTrue(row['interpretation_flags']['signed_gap_fell_while_absolute_gap_rose'])

    def test_both_models_decline_while_gap_narrows(self):
        row = self.analyze([(1, 1, 1, 1), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0)])
        self.assertEqual(row['flash_change'], -.25)
        self.assertEqual(row['max_change'], -.75)
        self.assertEqual(row['absolute_gap_change'], -.5)
        self.assertTrue(row['interpretation_flags']['gap_narrowed_with_both_models_declining'])

    def test_pending_in_other_baseline_excludes_episode_from_both(self):
        plan, analysis = fixture([(1, 1, 1, 1), (0, 0, 0, 0)])
        next(r for r in analysis['rows'] if r['episode_id'] == '1' and
             r['model'] == report.MODELS[0] and r['arm'] == 'mllm_fixed')['status'] = 'pending'
        result = report.diagnostics(plan, analysis, repeats=20)
        self.assertFalse(result['all_terminal'])
        for row in result['rows']:
            self.assertEqual((row['n'], row['planned_pairs'], row['missing_pairs']), (1, 2, 1))
            self.assertEqual(row['excluded_episode_ids'], ['1'])
            self.assertEqual([row[k] for k in report.RATE_KEYS], [1.]*4)

    def test_empty_intersection_produces_null_rates_and_intervals(self):
        plan, analysis = fixture([(1, 1, 1, 1)])
        analysis['rows'][0].update(status='unstarted', success=False)
        row = report.diagnostics(plan, analysis, repeats=20)['rows'][0]
        self.assertEqual(row['n'], 0)
        self.assertTrue(all(row[k] is None and row['ci95'][k] is None for k in report.METRIC_KEYS))

    def test_failed_attempt_remains_in_denominator(self):
        plan, analysis = fixture([(1, 1, 1, 1), (0, 0, 0, 0)])
        analysis['rows'][-1]['status'] = 'failed'
        row = report.diagnostics(plan, analysis, repeats=20)['rows'][0]
        self.assertEqual(row['n'], 2)
        self.assertEqual(row['jev_max_sr'], .5)
        analysis['rows'][-1]['success'] = True
        with self.assertRaisesRegex(ValueError, 'Failed attempt'):
            report.diagnostics(plan, analysis, repeats=20)

    def test_absolute_gap_is_abs_mean_gap_not_mean_abs_per_episode(self):
        row = self.analyze([(0, 0, 0, 1), (0, 0, 1, 0)])
        self.assertEqual(row['jev_absolute_gap'], 0.)
        self.assertEqual(row['absolute_gap_change'], 0.)
        self.assertEqual(row['ci95']['absolute_gap_change'], [0., 1.])
        self.assertEqual(row['ci95']['signed_gap_change'], [-1., 1.])

    def test_bootstrap_is_deterministic_and_preserves_paired_cells(self):
        values = [(0, 1, 0, 1), (1, 0, 1, 0), (1, 1, 1, 1)]
        row = self.analyze(values)
        self.assertEqual(row['ci95'], self.analyze(values)['ci95'])
        for key in ('flash_change', 'max_change', 'absolute_gap_change', 'signed_gap_change'):
            self.assertEqual(row['ci95'][key], [0., 0.])

    def test_source_gap_consistency_checked(self):
        plan, analysis = fixture([(0, 1, 1, 0)])
        analysis['gap_changes'] = [dict(dataset='AVDN', baseline='mllm', n=1,
            signed_gap_change_jev_minus_baseline=-2.)]
        self.assertTrue(report.diagnostics(plan, analysis, repeats=20)['rows'][0]['source_gap_change_verified'])
        analysis['gap_changes'][0]['n'] = 0
        with self.assertRaisesRegex(ValueError, 'another intersection'):
            report.diagnostics(plan, analysis, repeats=20)

    def test_duplicate_missing_and_nonboolean_outcomes_rejected(self):
        plan, analysis = fixture([(0, 1, 1, 0)])
        bad = copy.deepcopy(analysis)
        bad['rows'].append(bad['rows'][0])
        with self.assertRaisesRegex(ValueError, 'planned six-cell'):
            report.diagnostics(plan, bad, repeats=20)
        bad = copy.deepcopy(analysis)
        bad['rows'].pop()
        with self.assertRaisesRegex(ValueError, 'planned six-cell'):
            report.diagnostics(plan, bad, repeats=20)
        analysis['rows'][0]['success'] = 0
        with self.assertRaisesRegex(ValueError, 'scored boolean'):
            report.diagnostics(plan, analysis, repeats=20)

    def test_build_reads_only_two_sources_and_rejects_wrong_plan_hash(self):
        plan, analysis = fixture([(0, 1, 1, 0)])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan_bytes = json.dumps(plan).encode()
            (root/'plan.json').write_bytes(plan_bytes)
            analysis['plan_sha256'] = hashlib.sha256(plan_bytes).hexdigest()
            (root/'analysis.json').write_text(json.dumps(analysis), encoding='utf-8')
            result = report.build(root, repeats=20)
            self.assertTrue((root/'gap_diagnostics.json').is_file())
            self.assertEqual(result['plan_sha256'], analysis['plan_sha256'])
            self.assertEqual(result['analysis_sha256'], hashlib.sha256((root/'analysis.json').read_bytes()).hexdigest())
            self.assertTrue(result['posthoc_descriptive'])
            self.assertEqual(set(p.name for p in root.iterdir()),
                             {'plan.json', 'analysis.json', 'gap_diagnostics.json'})
            analysis['plan_sha256'] = 'wrong'
            (root/'analysis.json').write_text(json.dumps(analysis), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                report.build(root, repeats=20)


if __name__ == '__main__':
    unittest.main()
