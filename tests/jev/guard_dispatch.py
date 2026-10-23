"""Operational guard: suspend an existing launcher on a new terminal API failure.

Does not send requests, read credentials, change runtime code, or retry episodes.
Existing failures are the startup baseline; subsequent API-exhausted episodes
trigger one reversible process suspension. There is deliberately no auto-resume.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import time
from ctypes import wintypes

WORK = Path(__file__).resolve().parents[2]
ROOT = WORK/'out/jev_study_20261022/online'


def api_exhausted(folder):
    path = folder/'result.json'
    if not path.exists():
        return False
    result = json.loads(path.read_bytes())
    if result.get('termination') != 'interface_error':
        return False
    return any(json.loads(p.read_bytes()).get('status') == 'api_error'
               for p in (folder/'requests').glob('*.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher-pid', type=int, required=True)
    args = parser.parse_args()
    if os.name != 'nt' or args.launcher_pid <= 0:
        raise ValueError('This operational guard requires a positive Windows launcher PID')
    spec = importlib.util.spec_from_file_location('guard_online', WORK/'tests/jev/online_experiment.py')
    online = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(online)
    plan = online.validate_frozen(ROOT)
    plan_hash = online.sha(ROOT/'plan.json')
    folders = {j['job_id']:online.job_folder(ROOT,j) for j in plan['jobs']}
    known = {ident for ident,folder in folders.items() if api_exhausted(folder)}
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.WaitForSingleObject.argtypes = [wintypes.HANDLE,wintypes.DWORD]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    n = ctypes.WinDLL('ntdll')
    n.NtSuspendProcess.argtypes = [wintypes.HANDLE]
    n.NtSuspendProcess.restype = ctypes.c_long
    handle = k.OpenProcess(0x00100000 | 0x0800, False, args.launcher_pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Cannot hold launcher process handle')
    state = dict(schema='jev-dispatch-guard-v1', pid=os.getpid(), launcher_pid=args.launcher_pid,
                 plan_sha256=plan_hash, baseline_api_exhausted=sorted(known),
                 status='watching', started_utc=datetime.now(timezone.utc).isoformat())
    def emit():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        online.save(ROOT/'dispatch_guard.json', state)
        print(json.dumps(state), flush=True)
    try:
        emit()
        while k.WaitForSingleObject(handle,0) == 258:
            if online.sha(ROOT/'plan.json') != plan_hash:
                raise RuntimeError('Plan changed')
            new = [ident for ident,folder in folders.items() if ident not in known and api_exhausted(folder)]
            if new:
                result = n.NtSuspendProcess(handle)
                if result != 0:
                    raise OSError(result, 'Could not suspend launcher')
                state.update(status='launcher_suspended', triggering_jobs=new)
                online.save(ROOT/'operational_hold.json', dict(state,
                    reason='New terminal interface_error with recorded API failure; no automatic resume/retry'))
                emit()
                return 3
            time.sleep(1)
        state['status'] = 'launcher_exited'
        emit()
        return 0
    finally:
        k.CloseHandle(handle)


if __name__ == '__main__':
    raise SystemExit(main())
