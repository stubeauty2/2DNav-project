"""Read-only workload accounting from request journals; never reads evaluation labels."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, median

SPEC = importlib.util.spec_from_file_location('jev_online', Path(__file__).with_name('online_experiment.py'))
online = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(online)


def token_usage(request):
    usage = request.get('response', {}).get('usage')
    if not isinstance(usage, dict):
        return None
    result = {k: usage.get(k) for k in ('input_tokens', 'output_tokens', 'total_tokens')}
    result['image_tokens'] = usage.get('image_tokens', usage.get('input_tokens_details', {}).get('image_tokens'))
    result['cached_tokens'] = usage.get('prompt_tokens_details', {}).get('cached_tokens')
    return {k: v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 else None
            for k, v in result.items()}


def summarize(root):
    root = Path(root).resolve()
    plan = online.validate_frozen(root)
    rows = []
    token_keys = ('input_tokens', 'output_tokens', 'total_tokens', 'image_tokens', 'cached_tokens')
    for job in plan['jobs']:
        folder = online.job_folder(root, job)
        state = online.job_state(folder)
        attempt = online.read(folder / 'attempt.json') if (folder / 'attempt.json').exists() else {}
        requests = [online.read(p) for p in (folder / 'requests').glob('*.json')]
        jev = [online.read(p) for p in (folder / 'tools').glob('*-jev-review.json')]
        usages = [token_usage(r) for r in requests]
        usage = {key: sum(u[key] for u in usages if u is not None and u[key] is not None) for key in token_keys}
        missing = {key: sum(u is None or u[key] is None for u in usages) for key in token_keys}
        latencies = [r['latency_ms']/1000 for r in jev if isinstance(r.get('latency_ms'), (int, float))]
        seconds = attempt.get('finished_unix', 0)-attempt.get('started_unix', 0) if state in online.TERMINAL else None
        rows.append(dict(job, status=state, mllm_requests=len(requests),
            mllm_review_requests=sum(r.get('context', {}).get('mode') == 'review' for r in requests),
            mllm_image_requests=sum(bool(r.get('assets')) for r in requests),
            request_statuses=dict(Counter(r.get('status', 'missing') for r in requests)),
            reported_tokens=usage, token_fields_missing=missing,
            jev_requests=len(jev), jev_statuses=dict(Counter(r.get('status') for r in jev)),
            jev_verdicts=dict(Counter(r.get('verdict') for r in jev if r.get('status') == 'ok')),
            jev_call_seconds=sum(latencies), jev_latency_samples=len(latencies), episode_seconds=seconds))
    groups = []
    for dataset in online.POOLS:
        for model in online.MODELS:
            for arm in online.ARMS:
                planned = [r for r in rows if (r['dataset'], r['model'], r['arm']) == (dataset, model, arm)]
                terminal = [r for r in planned if r['status'] in online.TERMINAL]
                seconds = [r['episode_seconds'] for r in terminal if r['episode_seconds'] is not None]
                groups.append(dict(dataset=dataset, model=model, arm=arm, planned=len(planned),
                    terminal=len(terminal), complete=len(terminal)==len(planned),
                    mean_mllm_requests=mean(r['mllm_requests'] for r in terminal) if terminal else None,
                    mean_review_requests=mean(r['mllm_review_requests'] for r in terminal) if terminal else None,
                    mean_jev_requests=mean(r['jev_requests'] for r in terminal) if terminal else None,
                    mean_episode_seconds=mean(seconds) if seconds else None,
                    median_episode_seconds=median(seconds) if seconds else None,
                    reported_tokens={k:sum(r['reported_tokens'][k] for r in terminal) for k in token_keys},
                    token_fields_missing={k:sum(r['token_fields_missing'][k] for r in terminal) for k in token_keys}))
    pairs = []
    index = {(r['dataset'], r['episode_id'], r['model'], r['arm']):r for r in rows}
    for dataset in online.POOLS:
        ids = [r['episode_id'] for r in plan['selected'] if r['dataset'] == dataset]
        for model in online.MODELS:
            for baseline in ('mllm', 'mllm_fixed'):
                joined = [(index[(dataset, ident, model, baseline)], index[(dataset, ident, model, 'jev')]) for ident in ids]
                joined = [(a,b) for a,b in joined if a['status'] in online.TERMINAL and b['status'] in online.TERMINAL]
                differences = [b['mllm_requests']-a['mllm_requests'] for a,b in joined]
                pairs.append(dict(dataset=dataset, model=model, baseline=baseline, n=len(joined), planned=len(ids),
                    mean_mllm_request_change_jev_minus_baseline=mean(differences) if differences else None,
                    descriptive_ci95=online.bootstrap_mean(differences)))
    return dict(schema='jev-workload-v1', plan_sha256=online.sha(root / 'plan.json'),
        all_terminal=all(r['status'] in online.TERMINAL for r in rows),
        statuses=dict(Counter(r['status'] for r in rows)),
        observed_mllm_requests=sum(r['mllm_requests'] for r in rows),
        observed_reported_tokens={k:sum(r['reported_tokens'][k] for r in rows) for k in token_keys},
        observed_missing_token_fields={k:sum(r['token_fields_missing'][k] for r in rows) for k in token_keys},
        groups=groups, paired_request_changes=pairs, rows=rows,
        limitations=['Token values are provider-reported telemetry, not independently measured cost.',
            'Request counts are persisted request attempts, not a billing statement; pending journals may lack responses.',
            'Image and cached tokens are subsets of input tokens; never add them to total tokens.',
            'Observed totals include in-progress work; group comparisons include terminal attempts only.',
            'Jev call time includes local HTTP and queue waiting. Episode time also includes CPU grounders and remote APIs.',
            'Different episode lengths mean removing review calls need not reduce total MLLM calls.',
            'No success-rate labels are read by this workload report.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=online.ROOT)
    args = parser.parse_args()
    result = summarize(args.root)
    online.save(args.root / 'workload.json', result)
    print(json.dumps({k:result[k] for k in ('all_terminal', 'statuses', 'observed_mllm_requests', 'observed_reported_tokens')}))


if __name__ == '__main__':
    main()
