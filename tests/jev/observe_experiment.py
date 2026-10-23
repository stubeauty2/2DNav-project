"""Observe one existing launcher and refresh offline reports; never launch navigation."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


WORK = Path(__file__).resolve().parents[2]
ROOT = WORK/'out/jev_study_20261022/online'
REFRESH_SECONDS = 60
POLL_SECONDS = 2
MAX_REFRESH_ATTEMPTS = 3
TERMINAL = {'complete', 'failed'}
# No arbitrary command/root argument: audit_metrics and stage_report use this study's fixed root.
REFRESH_STEPS = (
    ('online_experiment.py', ('analyze',), 'analysis.json'),
    ('audit_metrics.py', (), 'metric_audit.json'),
    ('stage_report.py', (), 'stage_analysis.json'),
    ('resource_report.py', (), 'workload.json'),
    ('review_usage_report.py', (), 'review_usage.json'),
    ('gap_report.py', (), 'gap_diagnostics.json'),
    ('build_report.py', (), '../index.html'),
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_online():
    spec = importlib.util.spec_from_file_location('observer_online', WORK/'tests/jev/online_experiment.py')
    online = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(online)
    return online


def commands():
    return tuple((sys.executable, '-B', str(WORK/'tests/jev'/name), *arguments)
                 for name, arguments, _ in REFRESH_STEPS)


def source_hashes():
    paths = [Path(__file__), *(WORK/'tests/jev'/step[0] for step in REFRESH_STEPS)]
    return {str(path.relative_to(WORK)): sha(path) for path in paths}


def refresh_reports():
    """Execute only the literal offline allowlist; do not read or log credentials."""
    started = utc_now()
    results = []
    for command, (name, _, artifact) in zip(commands(), REFRESH_STEPS):
        source = WORK/'tests/jev'/name
        item = dict(script=name, command=list(command), source_sha256=sha(source), started_utc=utc_now(), attempts=[])
        for attempt_number in range(1, MAX_REFRESH_ATTEMPTS+1):
            attempt = dict(number=attempt_number, started_utc=utc_now())
            try:
                # Inherit the environment without inspecting it. These fixed entrypoints
                # read saved files; no headers, environment values, or raw output are logged.
                result = subprocess.run(command, cwd=WORK, shell=False, capture_output=True, timeout=300)
                attempt.update(returncode=result.returncode,
                    stdout_sha256=hashlib.sha256(result.stdout).hexdigest(), stdout_bytes=len(result.stdout),
                    stderr_sha256=hashlib.sha256(result.stderr).hexdigest(), stderr_bytes=len(result.stderr))
                attempt['status'] = 'ok' if result.returncode == 0 else 'error'
            except (OSError, subprocess.TimeoutExpired) as exc:
                attempt.update(status='error', error=type(exc).__name__)
            attempt['finished_utc'] = utc_now()
            item['source_unchanged'] = sha(source) == item['source_sha256']
            if not item['source_unchanged']:
                attempt.update(status='error', error='report_source_changed_during_execution')
            output = (ROOT/artifact).resolve()
            item.update(artifact=str(output), artifact_sha256=sha(output) if output.is_file() else None)
            if attempt['status'] == 'ok' and not output.is_file():
                attempt.update(status='error', error='expected_report_missing')
            item['attempts'].append(attempt)
            item['status'] = attempt['status']
            if item['status'] == 'ok' or not item['source_unchanged']:
                break
            if attempt_number < MAX_REFRESH_ATTEMPTS:
                time.sleep(POLL_SECONDS)  # retry only this offline allowlisted report
        item['finished_utc'] = utc_now()
        results.append(item)
        if item['status'] != 'ok':
            break
    return dict(status='ok' if len(results) == len(REFRESH_STEPS) and all(r['status'] == 'ok' for r in results) else 'error',
                started_utc=started, finished_utc=utc_now(), steps=results)


@contextmanager
def observer_lock(root):
    """Separate from the launcher's run.lock; released by the OS after a crash."""
    path = Path(root)/'observer.lock'
    stream = path.open('a+b')
    locked = False
    try:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise RuntimeError('another_observer_holds_lock') from exc
        yield
    finally:
        if locked:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


@contextmanager
def launcher_probe(online, pid):
    """Use the online helper, plus a held Windows handle to avoid PID reuse."""
    initial_alive = online.process_alive(pid)
    kernel = handle = None
    if os.name == 'nt' and initial_alive:
        kernel, handle = online.process_handle(pid)
    def alive():
        if not initial_alive:
            return False
        if os.name == 'nt' and (handle is None or kernel.WaitForSingleObject(handle, 0) != 258):
            return False
        return online.process_alive(pid)
    try:
        yield alive
    finally:
        if handle is not None:
            kernel.CloseHandle(handle)


def read_plan(root):
    payload = (Path(root)/'plan.json').read_bytes()
    plan = json.loads(payload)
    jobs = plan['jobs']
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Expected a nonempty frozen job list')
    ids, paths = set(), set()
    for job in jobs:
        if not isinstance(job, dict):
            raise ValueError('Plan job must be an object')
        if not isinstance(job.get('job_id'), str) or job['job_id'] in ids:
            raise ValueError('Invalid or duplicate job_id')
        ids.add(job['job_id'])
        parts = tuple(job.get(k) for k in ('dataset', 'model', 'arm', 'episode_id'))
        if any(not isinstance(p, str) or not p or p in ('.', '..')
               or any(c in p for c in ('/', '\\', ':')) for p in parts):
            raise ValueError('Unsafe job path')
        if parts in paths:
            raise ValueError('Duplicate job path')
        paths.add(parts)
    return jobs, hashlib.sha256(payload).hexdigest()


def snapshot(root, jobs, plan_hash, online):
    if sha(Path(root)/'plan.json') != plan_hash:
        raise ValueError('plan_changed_during_observation')
    counts, terminal = Counter(), []
    for job in jobs:
        folder = online.job_folder(root, job)
        if not folder.resolve().is_relative_to((Path(root)/'episodes').resolve()):
            raise ValueError('Job directory leaves episode root')
        status = online.job_state(folder)
        if status not in TERMINAL | {'pending', 'unstarted', 'untracked'}:
            raise ValueError('unknown_attempt_status')
        counts[status] += 1
        if status in TERMINAL:
            terminal.append((job['job_id'], status))
    return dict(planned=len(jobs), terminal=len(terminal), statuses=dict(counts),
                all_terminal=len(terminal) == len(jobs),
                terminal_signature=hashlib.sha256(json.dumps(sorted(terminal)).encode()).hexdigest())


def persist(root, state):
    """Observer files are the only direct writes, apart from offline report CLIs."""
    root = Path(root)
    state['updated_utc'] = utc_now()
    output = root/'observer_state.json'
    temporary = output.with_name(output.name+f'.{os.getpid()}.writing')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(output)
    compact = {k: state.get(k) for k in ('updated_utc', 'status', 'launcher_pid', 'launcher_alive',
               'planned', 'terminal', 'statuses', 'refresh_count', 'plan_sha256', 'error', 'exit_code')}
    line = json.dumps(compact, ensure_ascii=False, allow_nan=False)
    with (root/'observer.log.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(line+'\n')
    print(line, flush=True)


def observe(launcher_pid, root, online, alive, *, refresh=refresh_reports, sleep=time.sleep,
            monotonic=time.monotonic, emit=persist):
    """Dependency injection supports offline boundary tests, never custom CLI commands."""
    state = dict(schema='jev-observer-v1', observer_pid=os.getpid(), launcher_pid=launcher_pid,
                 started_utc=utc_now(), refresh_interval_seconds=REFRESH_SECONDS,
                 poll_seconds=POLL_SECONDS, refresh_count=0, source_sha256=source_hashes(),
                 status='starting', non_transactional=True)
    last_refresh = last_emit = float('-inf')
    last_terminal = last_statuses = None
    try:
        jobs, plan_hash = read_plan(root)
        state['plan_sha256'] = plan_hash
        while True:
            current = snapshot(root, jobs, plan_hash, online)
            state.update(current, launcher_alive=alive())
            now = monotonic()
            final_status = (('finished' if current['all_terminal'] else 'launcher_stopped')
                            if not state['launcher_alive'] else None)
            # The credential owner performs its own final analyze after its child
            # launcher exits. Wait for that owner before our final report chain.
            waiting_finalizer = current['all_terminal'] and state['launcher_alive']
            final_jobs_running = current['planned']-current['terminal'] <= 2
            state['refresh_deferred'] = ('owner_finalizer' if waiting_finalizer else
                                         'final_jobs_running' if final_jobs_running and not final_status else None)
            changed = current['terminal_signature'] != last_terminal
            if final_status or (not final_jobs_running and changed and now-last_refresh >= REFRESH_SECONDS):
                state['status'] = 'refreshing'
                emit(root, state)
                state['last_refresh'] = refresh()
                state['refresh_count'] += 1
                last_refresh = monotonic()
                last_terminal = current['terminal_signature']
                if state['last_refresh']['status'] != 'ok':
                    state.update(status='refresh_failed', exit_code=4)
                    emit(root, state)
                    return 4
                if final_status:
                    code = 0 if final_status == 'finished' else 3
                    state.update(status=final_status, exit_code=code)
                    if final_status == 'launcher_stopped':
                        state['error'] = 'launcher_exited_before_all_planned_jobs_terminal; observer will not restart or resume it'
                    emit(root, state)
                    return code
            state['status'] = 'waiting_finalizer' if waiting_finalizer else 'observing'
            now = monotonic()
            if current['statuses'] != last_statuses or now-last_emit >= REFRESH_SECONDS:
                emit(root, state)
                last_emit = now
                last_statuses = dict(current['statuses'])
            sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        state.update(status='observer_interrupted', exit_code=130)
        emit(root, state)
        return 130
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        state.update(status='observer_error', error=type(exc).__name__, exit_code=5)
        emit(root, state)
        return 5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher-pid', type=int, required=True)
    args = parser.parse_args()
    if args.launcher_pid <= 0:
        parser.error('--launcher-pid must be positive')
    online = load_online()
    try:
        with observer_lock(ROOT), launcher_probe(online, args.launcher_pid) as alive:
            return observe(args.launcher_pid, ROOT, online, alive)
    except (OSError, RuntimeError) as exc:
        # A second observer must not overwrite the active observer's state/log.
        print(json.dumps(dict(status='observer_start_failed', error=type(exc).__name__,
                              reason='lock or launcher lifetime could not be acquired', exit_code=2)), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
