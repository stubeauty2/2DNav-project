"""Process-only API credential session for explicitly queued frozen experiments.

Queue files contain only a safe batch name. No credentials are read from or written
to files. The input console closes after starting an independent hidden broker.
Create stop.json to stop it, or use --exit-after-batch to exit after a successful
batch. Failed runs retain the session for an explicitly queued follow-up.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'out/16_adaptive_agent/20261027'
SESSION=ROOT/'api_session_03'


def save(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(temp,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',default='api_session_03')
    parser.add_argument('--max-cases',type=int,default=6)
    parser.add_argument('--exit-after-batch',action='store_true')
    parser.add_argument('--serve',action='store_true')
    args=parser.parse_args()
    if not re.fullmatch('[a-z0-9_-]+',args.session):parser.error('Safe session name required')
    if not 1<=args.max_cases<=100:parser.error('max-cases must be 1..100')
    SESSION=ROOT/args.session
    SESSION.mkdir(parents=True,exist_ok=True)
    ready=SESSION/'ready.json'
    if ready.exists():raise RuntimeError('Session already recorded; inspect its process rather than starting another')
    serving=args.serve
    if not serving:
        print('2DNav: enter the key once; a hidden worker will retain it in memory for queued tests.',flush=True)
    key=os.getenv('DASHSCOPE_API_KEY') or (None if serving else getpass.getpass('API key (hidden): '))
    if not key:raise RuntimeError('No credential supplied')
    env=dict(os.environ,DASHSCOPE_API_KEY=key,PYTHONDONTWRITEBYTECODE='1')
    if not serving:
        # The input window can close without killing the credential worker.
        with (SESSION/'broker.stdout.log').open('ab') as out,(SESSION/'broker.stderr.log').open('ab') as err:
            proc=subprocess.Popen([sys.executable,'-u','-B',str(Path(__file__).resolve()),*sys.argv[1:],'--serve'],
                cwd=WORK,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
        save(SESSION/'launcher.json',dict(broker_pid=proc.pid,started=time.time()))
        print('Hidden worker started. This input window will close automatically.',flush=True)
        return
    save(ready,dict(pid=os.getpid(),ready_at=time.time(),status='ready',max_cases=args.max_cases))
    print('READY. Waiting for explicitly queued frozen batches.',flush=True)
    try:
        while not (SESSION/'stop.json').exists():
            for command in sorted(SESSION.glob('queue-*.json')):
                result=command.with_name(command.stem+'-result.json')
                if command.stem.endswith('-result') or result.exists():continue
                batch=json.loads(command.read_text(encoding='utf-8')).get('batch')
                if not isinstance(batch,str) or not re.fullmatch('[a-z0-9_-]+',batch):raise ValueError('Unsafe batch name')
                folder=ROOT/batch
                plan=json.loads((folder/'plan.json').read_text(encoding='utf-8'))
                if len(plan['jobs'])>args.max_cases or plan['decision_model']!='qwen3.5-flash':
                    raise ValueError('This session permits only bounded batches with the 3.5 decision model')
                print('RUN',batch,flush=True)
                save(result,dict(batch=batch,status='running',started=time.time()))
                with (SESSION/(command.stem+'.stdout.log')).open('wb') as out,(SESSION/(command.stem+'.stderr.log')).open('wb') as err:
                    proc=subprocess.Popen([sys.executable,'-u','-B',str(folder/'harness.py'),'run','--batch',batch],
                        cwd=WORK,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
                    save(result,dict(batch=batch,status='running',pid=proc.pid,started=time.time()))
                    code=proc.wait()
                save(result,dict(batch=batch,status='finished',exit_code=code,finished=time.time()))
                print('FINISHED',batch,'exit',code,flush=True)
                # A preflight failure must not discard the credential session.
                # Preserve the attempt and wait for an explicitly queued retry.
                if args.exit_after_batch and code==0:return
            time.sleep(1)
    finally:
        save(SESSION/'closed.json',dict(pid=os.getpid(),closed_at=time.time()))


if __name__=='__main__':main()
