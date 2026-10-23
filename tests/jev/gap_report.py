"""Post-hoc descriptive gap diagnostics from saved analysis, with no GT or API access.

The same terminal six-cell episode intersection as online_experiment.gap_changes
is used for both baselines. Absolute gaps are absolute differences between sample
success rates, not the mean absolute difference between individual outcomes.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


ROOT = Path(__file__).resolve().parents[2] / 'out' / 'jev_study_20261022' / 'online'
MODELS = ('qwen3.8-flash', 'qwen3.8-max')
ARMS = ('mllm', 'mllm_fixed', 'jev')
TERMINAL = {'complete', 'failed'}
STATUSES = TERMINAL | {'pending', 'unstarted'}
RATE_KEYS = ('baseline_flash_sr', 'baseline_max_sr', 'jev_flash_sr', 'jev_max_sr')
METRIC_KEYS = RATE_KEYS + (
    'baseline_signed_gap', 'baseline_absolute_gap', 'jev_signed_gap',
    'jev_absolute_gap', 'signed_gap_change', 'absolute_gap_change',
    'flash_change', 'max_change')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(row):
    return tuple(row[k] for k in ('dataset', 'episode_id', 'model', 'arm'))


def statistics(values):
    """Columns are baseline flash/max, Jev flash/max on paired episodes."""
    if not values:
        return dict.fromkeys(METRIC_KEYS)
    rates = [sum(row[col] for row in values) / len(values) for col in range(4)]
    bf, bm, jf, jm = rates
    base_gap, jev_gap = bm - bf, jm - jf
    return dict(zip(RATE_KEYS, rates), baseline_signed_gap=base_gap,
        baseline_absolute_gap=abs(base_gap), jev_signed_gap=jev_gap,
        jev_absolute_gap=abs(jev_gap), signed_gap_change=jev_gap-base_gap,
        absolute_gap_change=abs(jev_gap)-abs(base_gap),
        flash_change=jf-bf, max_change=jm-bm)


def bootstrap(values, seed, repeats):
    if not values:
        return dict.fromkeys(METRIC_KEYS)
    rng, n = random.Random(seed), len(values)
    samples = {key: [] for key in METRIC_KEYS}
    for _ in range(repeats):
        # One shared episode draw retains all four cells and their dependence.
        result = statistics([values[rng.randrange(n)] for _ in range(n)])
        for key in METRIC_KEYS:
            samples[key].append(result[key])
    lo, hi = int(.025 * (repeats-1)), int(.975 * (repeats-1))
    return {key: [ordered[lo], ordered[hi]]
            for key, values in samples.items() if (ordered := sorted(values))}


def diagnostics(plan, analysis, *, repeats=10000, seed=None):
    """Pure analysis of already scored booleans; never opens result/GT paths."""
    require(isinstance(repeats, int) and not isinstance(repeats, bool) and repeats > 0,
            'Bootstrap repeats must be a positive integer')
    seed = plan['seed'] if seed is None else seed
    require(isinstance(seed, int) and not isinstance(seed, bool), 'Invalid bootstrap seed')
    require(tuple(plan['settings']['models']) == MODELS, 'Unexpected model identities/order')
    require(set(plan['settings']['arms']) == set(ARMS), 'Unexpected experimental arms')
    selected = [(r['dataset'], r['episode_id']) for r in plan['selected']]
    require(len(selected) == len(set(selected)), 'Duplicate selected episode')
    expected = {(ds, episode, model, arm) for ds, episode in selected
                for model in MODELS for arm in ARMS}
    planned = [identity(row) for row in plan['jobs']]
    require(len(planned) == len(set(planned)) and set(planned) == expected,
            'Plan is not the selected complete six-cell design')
    rows = analysis['rows']
    keys = [identity(row) for row in rows]
    require(len(keys) == len(set(keys)) and set(keys) == expected,
            'Analysis rows differ from the planned six-cell design')
    for row in rows:
        require(row['status'] in STATUSES, 'Unknown analysis status')
        if row['status'] in TERMINAL:
            require(type(row.get('success')) is bool, 'Terminal outcome is not a scored boolean')
            require(row['status'] != 'failed' or row['success'] is False,
                    'Failed attempt must remain a failure')
    index = {identity(row): row for row in rows}
    source_gaps = {(row['dataset'], row['baseline']): row for row in analysis.get('gap_changes', [])}
    require(len(source_gaps) == len(analysis.get('gap_changes', [])), 'Duplicate source gap comparison')
    output = []
    for dataset in dict.fromkeys(ds for ds, _ in selected):
        ids = [episode for ds, episode in selected if ds == dataset]
        included = [episode for episode in ids if all(
            index[(dataset, episode, model, arm)]['status'] in TERMINAL
            for model in MODELS for arm in ARMS)]
        excluded = [episode for episode in ids if episode not in included]
        counts = [dict(model=model, arm=arm, statuses=dict(Counter(
            index[(dataset, episode, model, arm)]['status'] for episode in ids)))
            for model in MODELS for arm in ARMS]
        for baseline in ('mllm', 'mllm_fixed'):
            cells = ((MODELS[0], baseline), (MODELS[1], baseline),
                     (MODELS[0], 'jev'), (MODELS[1], 'jev'))
            values = [[int(index[(dataset, episode, model, arm)]['success'])
                       for model, arm in cells] for episode in included]
            metrics = statistics(values)
            source = source_gaps.get((dataset, baseline))
            if source is not None:
                require(source['n'] == len(included), 'Source gap_changes uses another intersection')
                old, new = source['signed_gap_change_jev_minus_baseline'], metrics['signed_gap_change']
                require(old is None and new is None or old is not None and new is not None
                        and math.isclose(old, new, abs_tol=1e-12),
                        'Source signed gap does not match the saved outcomes')
            flags = dict(signed_gap_fell_while_absolute_gap_rose=None,
                         gap_narrowed_with_both_models_declining=None)
            if values:
                flags.update(signed_gap_fell_while_absolute_gap_rose=(
                    metrics['signed_gap_change'] < -1e-12 and metrics['absolute_gap_change'] > 1e-12),
                    gap_narrowed_with_both_models_declining=(metrics['absolute_gap_change'] < -1e-12
                        and metrics['flash_change'] < -1e-12 and metrics['max_change'] < -1e-12))
            output.append(dict(dataset=dataset, baseline=baseline, n=len(included),
                planned_pairs=len(ids), missing_pairs=len(excluded),
                all_pairs_terminal=not excluded, included_episode_ids=included,
                excluded_episode_ids=excluded, cell_status_counts=counts,
                successes=dict(zip(RATE_KEYS, [sum(v[c] for v in values) for c in range(4)])),
                **metrics, ci95=bootstrap(values, seed, repeats), interpretation_flags=flags,
                source_gap_change_verified=source is not None))
    return dict(schema='jev-online-gap-diagnostics-v1', posthoc_descriptive=True,
        all_terminal=all(row['status'] in TERMINAL for row in rows),
        statuses=dict(Counter(row['status'] for row in rows)),
        outcome='Saved primary success only: AVDN square IoU > 0.4; CityNav predicted center <= 20 m',
        units='Rates and differences are proportions; multiply by 100 for percent/percentage points',
        pairing='Same terminal complete/failed intersection across all six cells; both baselines share it',
        bootstrap=dict(seed=seed, repeats=repeats, method='Paired episode percentile bootstrap',
            confidence_level=.95, absolute_gap='Recompute abs(max SR - flash SR) within every draw'),
        rows=output, limitations=[
            'Post-hoc descriptive supplement; not a prespecified equivalence or noninferiority criterion.',
            'Pending/unstarted episodes are excluded, never imputed as failures; interim intersections may be selective.',
            'Failed attempts remain failures in all terminal denominators.',
            'Gap narrowing can accompany declines in both models; inspect all four rates and both model effects.',
            'Absolute-gap bootstrap intervals are exploratory and may behave poorly near zero with small samples.',
            'Small intersections, all-zero outcomes, or no discordance can yield empirical [0, 0] intervals; this does not establish zero population uncertainty or equivalence.',
            'The episode bootstrap does not capture dependence between episodes on the same map.',
            'This snapshot does not read live attempts, ground truth, images, or raw model results.'])


def build(root=ROOT, *, repeats=10000, seed=None):
    root = Path(root).resolve()
    plan_path, analysis_path = root/'plan.json', root/'analysis.json'
    plan_bytes, analysis_bytes = plan_path.read_bytes(), analysis_path.read_bytes()
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    analysis = json.loads(analysis_bytes)
    require(analysis['plan_sha256'] == plan_hash, 'Analysis plan hash mismatch')
    result = diagnostics(json.loads(plan_bytes), analysis, repeats=repeats, seed=seed)
    result.update(plan_sha256=plan_hash, analysis_sha256=hashlib.sha256(analysis_bytes).hexdigest(),
        source_paths=dict(plan=str(plan_path), analysis=str(analysis_path)),
        generated_utc=datetime.now(timezone.utc).isoformat())
    # The source analysis may refresh during this run; hashes identify the exact
    # snapshot consumed. A renderer can refuse stale diagnostics by checking them.
    output = root/'gap_diagnostics.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--repeats', type=int, default=10000)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    result = build(args.root, repeats=args.repeats, seed=args.seed)
    print(json.dumps(dict(output=str(args.root.resolve()/'gap_diagnostics.json'),
        all_terminal=result['all_terminal'], intersections=[dict(dataset=r['dataset'],
            baseline=r['baseline'], n=r['n'], planned=r['planned_pairs']) for r in result['rows']]),
        ensure_ascii=False))
