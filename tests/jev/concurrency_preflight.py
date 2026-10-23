"""Replay two frozen public requests concurrently and measure health responsiveness."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import time
import requests

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
from AVDN.tools.jev import Jev


def main():
    root = WORK / 'out/jev_study_20261022'
    source = root / 'fixed_review_4b'
    records = [json.loads(p.read_text()) for p in (source / 'responses').glob('*.json')]
    records.sort(key=lambda r: r['response']['raw']['qev']['prefix_tokens'])
    selected = [records[-1], records[0]]
    public = {r['case_id']: r for r in map(json.loads, (source / 'public_requests.jsonl').read_text().splitlines())}
    folder = root / 'concurrency_preflight_v2'
    folder.mkdir(exist_ok=True)
    def invoke(record):
        client = Jev('http://127.0.0.1:8770', 'OneJev1.6-4B',
            service_manifest=root / 'services/OneJev1.6-4B-worker.manifest.json')
        item = public[record['case_id']]
        views = [dict(next(v for v in item['context']['views'] if v['view_id'] == a['view_id']),
                      image_path=a['image_path']) for a in item['assets']]
        response = client.verify(item['context'], views, folder / item['case_id'] / 'request.json')
        old = record['response']
        return dict(case_id=item['case_id'], status=response['status'],
            same_probabilities=old['probabilities'] == response['probabilities'],
            same_verdict=old['verdict'] == response['verdict'], latency_ms=response['latency_ms'])
    checks = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(invoke, r) for r in selected]
        while not all(f.done() for f in futures):
            start = time.perf_counter()
            try:
                response = requests.get('http://127.0.0.1:8770/health', timeout=5)
                response.raise_for_status()
                checks.append(dict(ok=response.json()['status']=='ok', seconds=time.perf_counter()-start))
            except requests.RequestException as exc:
                checks.append(dict(ok=False, error=type(exc).__name__, seconds=time.perf_counter()-start))
            time.sleep(.5)
        results = [f.result() for f in futures]
    report = dict(cases=results, health_checks=len(checks), health_failures=sum(not c['ok'] for c in checks),
                  max_health_seconds=max(c['seconds'] for c in checks), health_details=checks)
    (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'health_details'}))
    if report['health_failures'] or any(r['status'] != 'ok' or not r['same_probabilities'] for r in results):
        raise RuntimeError('Concurrency preflight failed')


if __name__ == '__main__':
    main()
