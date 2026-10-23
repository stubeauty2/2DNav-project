"""Run an explicit andn_core configuration and monitor results against SAM3."""
import argparse
import csv
import getpass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'out/02_sam3/search/preds_sam3_test_unseen_full/turn_and_crop_eval_results_full_traj.json'
SAMPLE = ROOT / 'datasets/sample2'
PARSE = ROOT / 'out/01_parse_only/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def compare(predictions, baseline, total):
    rows = []
    for iid, pred in predictions.items():
        previous = float(baseline[iid]['final_iou'])
        current = float(pred['final_iou'])
        delta = current - previous
        transition = ('success' if previous > .3 else 'failure') + '->' + ('success' if current > .3 else 'failure')
        rows.append(dict(instr_id=iid, baseline_final_iou=previous, final_iou=current,
                         delta=delta, change='improved' if delta > 1e-9 else 'regressed' if delta < -1e-9 else 'unchanged',
                         transition=transition))
    n = len(rows)
    summary = dict(completed=n, total=total, success_threshold='final_iou > 0.3',
                   mean_final_iou=sum(r['final_iou'] for r in rows)/n if n else None,
                   matched_baseline_mean_final_iou=sum(r['baseline_final_iou'] for r in rows)/n if n else None,
                   success=sum(r['final_iou'] > .3 for r in rows),
                   matched_baseline_success=sum(r['baseline_final_iou'] > .3 for r in rows),
                   improved=sum(r['change']=='improved' for r in rows),
                   regressed=sum(r['change']=='regressed' for r in rows),
                   unchanged=sum(r['change']=='unchanged' for r in rows),
                   success_to_failure=[r['instr_id'] for r in rows if r['transition']=='success->failure'],
                   failure_to_success=[r['instr_id'] for r in rows if r['transition']=='failure->success'])
    return dict(summary=summary, trajectories=rows)


def main():
    import requests
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--candidate-count', type=int, choices=(0, 2), default=0)
    args = parser.parse_args()
    key = os.getenv('API_KEY') or getpass.getpass('API key (hidden, memory only): ')
    if not key:
        raise SystemExit('API key missing')
    service = 'http://127.0.0.1:8765'
    health = requests.get(service+'/health', timeout=10).json()
    if not health.get('prewarmed'):
        raise SystemExit('SAM3 not prewarmed; experiment not started')
    url = os.getenv('QWEN_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
    model = os.getenv('QWEN_MODEL', 'qwen-vl-max')
    response = requests.post(url, headers={'Authorization':'Bearer '+key}, json={
        'model':model, 'messages':[{'role':'user','content':'Reply OK.'}], 'max_tokens':4}, timeout=90)
    if response.status_code != 200 or 'error' in response.json():
        raise SystemExit('Model preflight failed: HTTP '+str(response.status_code)+' '+str(response.json()).replace(key, '[REDACTED]'))
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
    out = args.out_dir.resolve() if args.out_dir else ROOT / ('out/andn_core/sample2_sam3_'+stamp)
    out.mkdir(parents=True)
    baseline = read(BASE)['predictions']
    ids = {str(row['map_name'])+'__'+str(row['route_index']) for row in read(SAMPLE/'test_unseen_full_data.json')}
    assert len(ids)==20 and ids <= baseline.keys()
    package = ROOT/'PSC-AVDN/script/andn_core'
    write(out/'run_manifest.json', dict(started_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        baseline=str(BASE), annotations=str(SAMPLE), parse_csv=str(PARSE), candidate_count=args.candidate_count,
        parse_sha256=hashlib.sha256(PARSE.read_bytes()).hexdigest(), model=model, url=url, sam3=health,
        source_hashes={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest() for p in package.rglob('*.py')},
        baseline_sample2_mean=sum(baseline[i]['final_iou'] for i in ids)/len(ids),
        baseline_sample2_success=sum(baseline[i]['final_iou']>.3 for i in ids),
        land='Unmodified util.py defaults, including its original endpoint and credential lookup.'))
    env = dict(os.environ, API_KEY=key, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    command = [sys.executable, '-B', '-u', str(package/'runner.py'), '--anno-dir', str(SAMPLE),
               '--dataset-dir', str(SAMPLE), '--results-csv', str(PARSE), '--out-dir', str(out),
               '--vision-tool-url', service, '--qwen-url', url, '--qwen-model', model,
               '--candidate-count', str(args.candidate_count)]
    seen = set()
    print('RUN_DIR', out, flush=True)
    write(out/'live_vs_sam3.json', compare({}, baseline, len(ids)))
    with (out/'run.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(out/'process.json', dict(controller_pid=os.getpid(), runner_pid=child.pid))
        while True:
            done = child.poll() is not None
            source = out/('turn_and_crop_eval_results_full_traj.json' if done and (out/'turn_and_crop_eval_results_full_traj.json').exists() else 'turn_and_crop_eval_results_full_traj.in_progress.json')
            try:
                predictions = read(source)['predictions']
            except (FileNotFoundError, json.JSONDecodeError):
                predictions = None
            if predictions is not None:
                result = compare(predictions, baseline, len(ids))
                write(out/'live_vs_sam3.json', result)
                with (out/'live_vs_sam3.csv').open('w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=['instr_id','baseline_final_iou','final_iou','delta','change','transition'])
                    writer.writeheader(); writer.writerows(result['trajectories'])
                for row in result['trajectories']:
                    if row['instr_id'] not in seen:
                        print('TRAJECTORY', json.dumps(row, ensure_ascii=False), flush=True)
                        seen.add(row['instr_id'])
                        print('LIVE', json.dumps(result['summary'], ensure_ascii=False), flush=True)
                if done:
                    result['exit_code'] = child.returncode
                    result['complete'] = len(predictions)==len(ids) and child.returncode==0
                    write(out/'final_vs_sam3.json', result)
                    print('FINAL', json.dumps(result['summary']), 'exit_code', child.returncode, flush=True)
            if done:
                return child.returncode
            time.sleep(2)


if __name__ == '__main__':
    sys.exit(main())
