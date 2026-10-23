"""Protocol integrity tests with synthetic public pools; no service or model calls."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('jev_online', Path(__file__).with_name('online_experiment.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OnlineExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.work = self.base / 'workspace'
        self.work.mkdir()
        self.root = self.base / 'experiment'
        self.image = self.work / 'map.bin'
        self.image.write_bytes(b'fixed public pixels')
        self.pools = {}
        self.fixed = self.work / 'fixed.jsonl'
        self.prior = self.work / 'prior.json'
        fixed, prior = [], []
        runtime = '''import json
from pathlib import Path
def run_episode(entry, directory, **kwargs):
    assert kwargs['thinking_budget'] == 0
    assert kwargs['max_requests'] == 48
    assert kwargs['max_rounds'] == 12 and kwargs['max_tools'] == 2
    result = dict(completed=True, termination='step_limit', final=None, model_requests=0,
                  actual_stop_xy_m=[0, 0], jev_calls=[], jev_requests=0, jev_error_count=0,
                  tool_capabilities={'sam3':'available', 'dino':'available'})
    Path(directory, 'result.json').write_text(json.dumps(result), encoding='utf-8')
    return result
'''
        for ds, count in [('AVDN', 432), ('CityNav', 50)]:
            source = self.work / ds
            source.mkdir()
            (source / '__init__.py').write_text('', encoding='utf-8')
            (source / 'run.py').write_text(runtime, encoding='utf-8')
            folder = self.work / (ds + '_pool')
            folder.mkdir()
            self.pools[ds] = folder
            public, truth, raw = [], [], []
            for i in range(count):
                ident = f'{ds.lower()}-{i:03d}'
                instruction = f'public instruction {i}'
                public.append(dict(episode_id=ident,
                    agent_input=dict(instruction_raw=instruction, views=[]), initialization=dict(direction=0),
                    environment=dict(image_path=str(self.image))))
                truth.append(dict(episode_id=ident, source_row=i, goal_xy_m=[0, 0],
                                  evaluation_polygon_xy_m=[[0, 0], [1, 0], [1, 1], [0, 1]]))
                raw.append(dict(instructions=instruction, descriptions=[instruction]))
            for name, rows in [('episodes.jsonl', public), ('evaluator_only.jsonl', truth)]:
                (folder / name).write_text('\n'.join(json.dumps(r) for r in rows)+'\n', encoding='utf-8')
            module.save(folder / 'annotation.json', raw)
            module.save(folder / 'sample_manifest.json', dict(annotation=str(folder / 'annotation.json'),
                annotation_sha256=module.sha(folder / 'annotation.json'),
                split='test_unseen_full' if ds == 'AVDN' else 'test_unseen'))
            excluded_count = 20 if ds == 'AVDN' else 6
            fixed.extend(dict(dataset=ds, episode_id=f'{ds.lower()}-{i:03d}') for i in range(excluded_count))
            prior.extend(dict(dataset=ds, episode_id=f'{ds.lower()}-{i:03d}') for i in range(20, 30))
        self.fixed.write_text('\n'.join(json.dumps(r) for r in fixed), encoding='utf-8')
        module.save(self.prior, dict(selected=prior))
        self.service = self.work / 'service.json'
        module.save(self.service, dict(served_model=module.JEV_MODEL,
            repository='OmniJev/OneJev1.6-4B', checkpoint_revision=module.JEV_REVISION,
            settings=dict(fork_mode='sequential', debias=1), weight_files={'weights': {'sha256': 'a'}},
            source_sha256={'engine.py': 'b'}))
        self.patchers = [patch.object(module, name, value) for name, value in [
            ('WORK', self.work), ('POOLS', self.pools), ('FIXED', self.fixed), ('REFACTOR', self.prior)]]
        for item in self.patchers:
            item.start()
            self.addCleanup(item.stop)
        self.addCleanup(self.clear_runtime)

    @staticmethod
    def clear_runtime():
        for name in list(module.sys.modules):
            if name.split('.')[0] in ('AVDN', 'CityNav'):
                del module.sys.modules[name]

    def prepare(self):
        return module.prepare(self.root, self.service)

    def test_random_selection_exclusions_and_budget(self):
        plan = self.prepare()
        self.assertEqual(len(plan['jobs']), 240)
        self.assertEqual(plan['budget']['max_paid_mllm_requests'], 11520)
        self.assertEqual(plan['budget']['max_generated_tokens'], 23592960)
        self.assertEqual(plan['pools']['AVDN']['eligible_size'], 402)
        self.assertEqual(plan['pools']['CityNav']['eligible_size'], 34)
        for ds in self.pools:
            selected = {r['episode_id'] for r in plan['selected'] if r['dataset'] == ds}
            self.assertEqual(len(selected), 20)
            self.assertFalse(selected & set(plan['pools'][ds]['excluded_ids']))
        self.assertEqual(plan, self.prepare())
        for row in plan['selected']:
            six = [j for j in plan['jobs'] if j['dataset'] == row['dataset'] and j['episode_id'] == row['episode_id']]
            self.assertEqual(len(six), 6)
            self.assertEqual({j['public_input'] for j in six}, {row['public_input']})

    def test_leakage_is_rejected_recursively(self):
        with self.assertRaisesRegex(ValueError, 'Evaluator'):
            module.public_only({'memory': [{'goal_xy_m': [1, 2]}]})
        module.public_only({'candidate': {'center_xy_m': [1, 2]}})

    def test_source_and_plan_mutation_refuse_resume(self):
        self.prepare()
        source = self.work / 'AVDN/run.py'
        original = source.read_bytes()
        source.write_bytes(original + b'\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'Source code changed'):
            module.validate_frozen(self.root)
        source.write_bytes(original)
        plan = module.read(self.root / 'plan.json')
        plan['settings']['endpoint'] = 'https://different.example'
        module.save(self.root / 'plan.json', plan)
        with self.assertRaisesRegex(ValueError, 'plan/config changed'):
            module.validate_frozen(self.root)

    def test_image_and_service_mutation_refuse(self):
        self.prepare()
        self.image.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Map image changed'):
            module.validate_frozen(self.root, check_images=True)
        self.image.write_bytes(b'fixed public pixels')
        service = module.read(self.service)
        service['settings']['fork_mode'] = 'batched'
        module.save(self.service, service)
        with self.assertRaisesRegex(ValueError, 'Service manifest source changed'):
            module.validate_frozen(self.root)

    def test_worker_runs_without_gt_and_never_retries(self):
        plan = self.prepare()
        module.save(self.root / 'tool_health.json', {})
        (self.root / 'evaluator_only.json').unlink()
        job = next(j for j in plan['jobs'] if j['arm'] == 'mllm')
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'secret-never-written'}), patch.object(module, 'healthcheck', return_value={}) as health:
            first = module.worker(self.root, job['job_id'])
            second = module.worker(self.root, job['job_id'])
        self.assertEqual(first['status'], 'complete')
        self.assertTrue(second['skipped'])
        journal = (module.job_folder(self.root, job) / 'attempt.json').read_text()
        self.assertNotIn('secret-never-written', journal)
        health.assert_called_once_with(plan, include_jev=False)

    def test_worker_refuses_grounder_identity_change(self):
        plan = self.prepare()
        module.save(self.root / 'tool_health.json', {'sam3': {'checkpoint': 'frozen'}})
        job = next(j for j in plan['jobs'] if j['arm'] == 'mllm')
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'test-secret'}), patch.object(
                module, 'healthcheck', return_value={'sam3': {'checkpoint': 'changed'}}):
            result = module.worker(self.root, job['job_id'])
        self.assertEqual(result['status'], 'failed')
        attempt = module.read(module.job_folder(self.root, job) / 'attempt.json')
        self.assertIn('service identity/configuration changed', attempt['detail'])
        self.assertFalse((module.job_folder(self.root, job) / 'result.json').exists())

    def test_pending_attempt_not_reissued(self):
        plan = self.prepare()
        job = plan['jobs'][0]
        folder = module.job_folder(self.root, job)
        self.assertTrue(module.claim(folder, job, module.sha(self.root / 'plan.json')))
        self.assertFalse(module.claim(folder, job, module.sha(self.root / 'plan.json')))
        with patch.dict(os.environ, {'DASHSCOPE_API_KEY': 'secret'}), patch.object(module, 'healthcheck') as health:
            result = module.worker(self.root, job['job_id'])
        self.assertEqual(result['status'], 'pending')
        self.assertTrue(result['skipped'])
        health.assert_not_called()

    def test_pending_analysis_is_not_false_sr(self):
        self.prepare()
        report = module.analyze(self.root)
        self.assertFalse(report['all_terminal'])
        self.assertEqual(report['statuses'], {'unstarted': 240})
        self.assertTrue(all(r['sr'] is None and r['observed_sr_bounds'] == [0, 1] for r in report['summaries']))

    def test_failed_pair_retained_and_runtime_jev_error_count_used(self):
        plan = self.prepare()
        ident = next(r['episode_id'] for r in plan['selected'] if r['dataset'] == 'CityNav')
        for arm, status in [('mllm_fixed', 'complete'), ('jev', 'failed')]:
            job = next(j for j in plan['jobs'] if j['dataset'] == 'CityNav' and
                       j['episode_id'] == ident and j['model'] == module.MODELS[0] and j['arm'] == arm)
            folder = module.job_folder(self.root, job)
            result = dict(final={'center_xy_m': [0, 0]}, actual_stop_xy_m=[30, 0],
                          jev_calls=[dict(status='error')]*2 if arm == 'jev' else [],
                          jev_requests=2 if arm == 'jev' else 0, jev_error_count=2 if arm == 'jev' else 0,
                          tool_capabilities={'sam3':'unavailable', 'dino':'available'})
            module.save(folder / 'result.json', result)
            module.save(folder / 'attempt.json', dict(status=status, job_id=job['job_id'],
                plan_sha256=module.sha(self.root / 'plan.json'), result_sha256=module.sha(folder / 'result.json')))
        report = module.analyze(self.root)
        effect = next(r for r in report['within_model_effects'] if r['dataset'] == 'CityNav' and
                      r['model'] == module.MODELS[0] and r['left'] == 'mllm_fixed' and r['right'] == 'jev')
        self.assertEqual(effect['primary']['n'], 1)
        self.assertEqual(effect['primary']['lost'], 1)
        failed = next(r for r in report['rows'] if r['episode_id'] == ident and
                      r['model'] == module.MODELS[0] and r['arm'] == 'jev')
        self.assertEqual(failed['jev_errors'], 2)
        self.assertEqual(failed['jev_requests'], 2)
        self.assertEqual(failed['unavailable_planned_grounders'], ['sam3'])
        self.assertTrue(failed['observation_grounder_unavailable'])
        self.assertEqual(failed['tool_errors'], 0)
        self.assertFalse(failed['success'])
        self.assertEqual(failed['actual_stop_distance_m'], 30)

    def test_paired_direction_discordance_and_zero_variation(self):
        stats = module.paired_stats([True, False, False, True], [False, True, True, True], repeats=1000)
        self.assertEqual((stats['gained'], stats['lost'], stats['discordant']), (2, 1, 3))
        self.assertEqual(stats['difference_right_minus_left'], .25)
        constant = module.paired_stats([False]*20, [True]*20, repeats=100)
        self.assertEqual(constant['difference_ci95'], [1, 1])
        self.assertLess(constant['exact_mcnemar_two_sided'], .001)

    def test_metrics_do_not_substitute_actual_stop_for_prediction(self):
        truth = dict(goal_xy_m=[0, 0])
        scored = module.evaluate_result('CityNav', dict(final={'center_xy_m': [21, 0]}, actual_stop_xy_m=[0, 0]), truth)
        self.assertFalse(scored['success'])
        self.assertTrue(scored['actual_stop_within20m'])
        edge = module.evaluate_result('CityNav', dict(final={'center_xy_m': [20, 0]}), truth)
        self.assertTrue(edge['success'])
        avdn_truth = dict(goal_xy_m=[0, 0], evaluation_polygon_xy_m=[[0, 0], [1, 0], [1, 1], [0, 1]])
        scored = module.evaluate_result('AVDN', dict(final={'evaluation_polygon_xy_m': [[0, 0], [.4, 0], [.4, 1], [0, 1]]}), avdn_truth)
        self.assertFalse(scored['success'])
        self.assertTrue(scored['secondary_success'])

    def test_no_more_than_two_workers_and_key_redaction(self):
        with self.assertRaisesRegex(ValueError, 'max_workers'):
            module.run(self.root, max_workers=3)
        self.assertEqual(module.sanitized('Bearer abc-secret', 'abc-secret'), 'Bearer [REDACTED]')

    def test_infrastructure_stop_keeps_queue_unstarted(self):
        import threading
        started, gate = [], threading.Barrier(2)
        def launch(job):
            started.append(job)
            gate.wait(timeout=3)
            return {'job': job}
        completed, stopped = module.dispatch_jobs(list(range(10)), launch,
            lambda value, rows: None, lambda job, value: True, 2)
        self.assertTrue(stopped)
        self.assertEqual(set(started), {0, 1})
        self.assertEqual(len(completed), 2)

    def test_optional_disabled_tool_does_not_stop_dispatch(self):
        plan = self.prepare()
        job = plan['jobs'][0]
        folder = module.job_folder(self.root, job)
        module.save(folder / 'attempt.json', {'status': 'complete'})
        module.save(folder / 'result.json', dict(tool_capabilities={
            'sam3': 'available', 'dino': 'available', 'locate_anything': 'unavailable'}))
        self.assertFalse(module.infrastructure_failure(self.root, job))
        module.save(folder / 'result.json', dict(jev_error_count=1))
        self.assertTrue(module.infrastructure_failure(self.root, job))

    def test_child_exits_when_parent_is_terminated(self):
        import subprocess
        import sys
        import time
        parent = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        ready = self.base / 'parent_watch_ready'
        code = ("import importlib.util,time,pathlib; "
                f"s=importlib.util.spec_from_file_location('h',{str(Path(module.__file__).resolve())!r}); "
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
                f"m.guard_parent({parent.pid});pathlib.Path({str(ready)!r}).write_text('ready');time.sleep(60)")
        child = subprocess.Popen([sys.executable, '-c', code])
        try:
            deadline = time.monotonic()+5
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.02)
            self.assertTrue(ready.exists())
            self.assertTrue(module.process_alive(child.pid))
            parent.terminate(); parent.wait(timeout=3)
            self.assertEqual(child.wait(timeout=3), 70)
            self.assertFalse(module.process_alive(child.pid))
        finally:
            for process in (parent, child):
                if process.poll() is None:
                    process.kill(); process.wait()


if __name__ == '__main__':
    unittest.main()
