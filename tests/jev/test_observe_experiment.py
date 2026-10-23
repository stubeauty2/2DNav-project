"""Synthetic observer boundaries; never start navigation, services, or live observation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('observe_experiment', Path(__file__).with_name('observe_experiment.py'))
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


class ObserverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.jobs = [dict(job_id='j1', dataset='AVDN', model='model', arm='jev', episode_id='e1')]
        (self.root/'plan.json').write_text(json.dumps({'jobs': self.jobs}), encoding='utf-8')
        self.now = 0
        self.states = []
        self.sleeps = []
        self.status = 'pending'
        self.online = SimpleNamespace(job_folder=lambda root, job: root/'episodes'/job['episode_id'],
                                      job_state=lambda folder: self.status)
        self.refresh = Mock(return_value={'status': 'ok', 'steps': []})

    def emit(self, root, state):
        self.states.append(copy.deepcopy(state))

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def run_observer(self, alive, sleep=None):
        return observer.observe(123, self.root, self.online, alive, refresh=self.refresh,
            sleep=sleep or self.sleep, monotonic=lambda: self.now, emit=self.emit)

    def expand_jobs(self, count):
        self.jobs = [dict(self.jobs[0], job_id=f'j{i+1}', episode_id=f'e{i+1}') for i in range(count)]
        (self.root/'plan.json').write_text(json.dumps({'jobs': self.jobs}), encoding='utf-8')

    def test_already_finished_refreshes_once_even_when_launcher_dead(self):
        self.status = 'complete'
        self.assertEqual(self.run_observer(lambda: False), 0)
        self.refresh.assert_called_once()
        self.assertEqual(self.states[-1]['status'], 'finished')
        self.assertEqual(self.sleeps, [])

    def test_launcher_stopped_pending_forces_refresh_and_error_exit(self):
        self.assertEqual(self.run_observer(lambda: False), 3)
        self.refresh.assert_called_once()
        self.assertEqual(self.states[-1]['status'], 'launcher_stopped')
        self.assertEqual(self.states[-1]['statuses'], {'pending': 1})
        self.assertIn('will not restart', self.states[-1]['error'])

    def test_short_sleeps_and_no_unchanged_periodic_refresh(self):
        self.expand_jobs(3)
        code = self.run_observer(lambda: self.now < 122)
        self.assertEqual(code, 3)
        self.assertEqual(self.refresh.call_count, 2)  # initial, stopped; no unchanged refresh
        self.assertTrue(all(s <= 2 for s in self.sleeps))

    def test_terminal_change_waits_interval_and_finish_forces_final_refresh(self):
        self.expand_jobs(4)
        self.online.job_state = lambda folder: ('complete' if self.now >= (2 if folder.name == 'e1' else 62) else 'pending')
        refresh_times = []
        self.refresh.side_effect = lambda: refresh_times.append(self.now) or {'status': 'ok'}
        self.assertEqual(self.run_observer(lambda: self.now < 64), 0)
        self.assertEqual(refresh_times, [0, 60, 64])
        self.assertEqual(self.states[-1]['terminal'], 4)
        self.assertIn('waiting_finalizer', [s['status'] for s in self.states])

    def test_refresh_failure_is_explicit_and_stops(self):
        self.expand_jobs(3)
        self.refresh.return_value = {'status': 'error', 'steps': []}
        self.assertEqual(self.run_observer(lambda: True), 4)
        self.assertEqual(self.states[-1]['status'], 'refresh_failed')
        self.assertEqual(self.sleeps, [])

    def test_plan_mutation_stops_without_running_more_reports(self):
        self.expand_jobs(3)
        def mutate(seconds):
            self.sleep(seconds)
            (self.root/'plan.json').write_text(json.dumps({'jobs': self.jobs, 'changed': True}), encoding='utf-8')
        self.assertEqual(self.run_observer(lambda: True, mutate), 5)
        self.assertEqual(self.refresh.call_count, 1)
        self.assertEqual(self.states[-1]['status'], 'observer_error')

    def test_lock_rejects_duplicate_and_releases_for_next_observer(self):
        with observer.observer_lock(self.root):
            with self.assertRaisesRegex(RuntimeError, 'another_observer'):
                with observer.observer_lock(self.root):
                    self.fail('Duplicate observer acquired lock')
        with observer.observer_lock(self.root):
            pass

    def test_only_fixed_offline_commands_and_fixed_cwd(self):
        names = [Path(cmd[2]).name for cmd in observer.commands()]
        self.assertEqual(names, ['online_experiment.py', 'audit_metrics.py', 'stage_report.py',
            'resource_report.py', 'review_usage_report.py', 'gap_report.py', 'build_report.py'])
        self.assertEqual(observer.commands()[0][3:], ('analyze',))
        for cmd in observer.commands():
            self.assertEqual(cmd[:2], (observer.sys.executable, '-B'))
            self.assertFalse({'prepare', 'run', 'worker', '--ask-key'}.intersection(cmd))
        with patch.object(observer.subprocess, 'run', return_value=SimpleNamespace(
                returncode=0, stdout=b'{}', stderr=b'')) as run, \
             patch.object(observer.Path, 'is_file', return_value=True), \
             patch.object(observer, 'sha', return_value='hash'):
            result = observer.refresh_reports()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(run.call_count, 7)
        for call in run.call_args_list:
            self.assertIn(call.args[0], observer.commands())
            self.assertEqual(call.kwargs['cwd'], observer.WORK)
            self.assertIs(call.kwargs['shell'], False)

    def test_failed_refresh_does_not_dispatch_dependent_reports(self):
        with patch.object(observer.subprocess, 'run', return_value=SimpleNamespace(
                returncode=1, stdout=b'', stderr=b'failure')) as run, \
             patch.object(observer.time, 'sleep') as sleep:
            result = observer.refresh_reports()
        self.assertEqual(run.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(all(call.args[0] == observer.commands()[0] for call in run.call_args_list))
        self.assertEqual(result['status'], 'error')
        self.assertNotIn('failure', json.dumps(result))  # raw process output is not logged

    def test_near_end_waits_for_owner_finalizer_before_only_final_refresh(self):
        def finish(seconds):
            self.sleep(seconds)
            self.status = 'complete'
        refresh_times = []
        self.refresh.side_effect = lambda: refresh_times.append(self.now) or {'status': 'ok'}
        self.assertEqual(self.run_observer(lambda: self.now < 4, finish), 0)
        self.assertEqual(refresh_times, [4])
        self.assertEqual(self.states[0]['refresh_deferred'], 'final_jobs_running')
        self.assertIn('waiting_finalizer', [s['status'] for s in self.states])

    def test_windows_probe_uses_online_helper_and_original_process_handle(self):
        kernel = Mock()
        kernel.WaitForSingleObject.return_value = 258
        online = SimpleNamespace(process_alive=Mock(return_value=True),
                                 process_handle=Mock(return_value=(kernel, 'held-handle')))
        with patch.object(observer.os, 'name', 'nt'):
            with observer.launcher_probe(online, 123) as alive:
                self.assertTrue(alive())
                kernel.WaitForSingleObject.return_value = 0
                self.assertFalse(alive())  # Even if the numeric PID is reused.
        online.process_alive.assert_called_with(123)
        kernel.CloseHandle.assert_called_once_with('held-handle')

    def test_transient_report_failure_retries_only_offline_step_then_continues(self):
        failed = SimpleNamespace(returncode=1, stdout=b'', stderr=b'')
        passed = SimpleNamespace(returncode=0, stdout=b'{}', stderr=b'')
        with patch.object(observer.subprocess, 'run', side_effect=[failed]+[passed]*7) as run, \
             patch.object(observer.Path, 'is_file', return_value=True), \
             patch.object(observer, 'sha', return_value='hash'), \
             patch.object(observer.time, 'sleep'):
            result = observer.refresh_reports()
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(run.call_count, 8)
        self.assertEqual(run.call_args_list[0].args[0], run.call_args_list[1].args[0])
        self.assertEqual(len(result['steps'][0]['attempts']), 2)


if __name__ == '__main__':
    unittest.main()
