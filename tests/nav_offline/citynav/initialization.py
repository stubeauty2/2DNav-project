"""Offline initial input preparation and direction evaluation."""
import hashlib
import json
import random
import statistics
from pathlib import Path
from collections import defaultdict
from CityNav.tools.runtime import read_jsonl, write_jsonl, save_json
from CityNav.analysis import SCHEMA
from CityNav.prompts import initialization_prompt
from .geometry import bearing, angular_error
DATASET = 'CityNav'

def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from .adapter import load_episodes, render_initial
    entries = list(load_episodes(args.annotation, args.objects, args.landmarks))
    ids = sorted({x[2]['source_row'] for x in entries})
    if args.limit:
        ids = random.Random(args.seed).sample(ids, min(args.limit, len(ids)))
    selected = set(ids)
    public, private = [], []
    view_cache = {}
    for ident, agent, gt, context in entries:
        row_id = gt['source_row']
        if row_id not in selected:
            continue
        if args.image_dir:
            if row_id not in view_cache:
                view_cache[row_id] = render_initial(context, args.image_dir, out/'images'/f'{row_id}.png')
            agent['views'] = [view_cache[row_id]]
        public.append({'episode_id': ident, 'agent_input': agent})
        private.append(gt)
    write_jsonl(out/'agent_inputs.jsonl', public)
    write_jsonl(out/'evaluator_only.jsonl', private)
    save_json(out/'schema.json', SCHEMA)
    (out/'system_prompt.txt').write_text(initialization_prompt(), encoding='utf-8')
    save_json(out/'manifest.json', {'dataset': DATASET, 'annotation': str(Path(args.annotation).resolve()),
              'annotation_sha256': hashlib.sha256(Path(args.annotation).read_bytes()).hexdigest(),
              'seed': args.seed, 'sampled_source_rows': sorted(selected), 'episodes': len(public),
              'input_regime': 'initial RGB + text' if args.image_dir else 'text/pose only, no RGB',
              'evaluation_isolation': 'run reads agent_inputs.jsonl only; evaluate reads evaluator_only.jsonl',
              'initialization_only': True})
    print(json.dumps({'episodes': len(public), 'out': str(out.resolve())}))

def evaluate(args):
    truths = read_jsonl(args.truth)
    if any(row['dataset'] != DATASET for row in truths):
        raise ValueError(f'This evaluator accepts {DATASET} ground truth only')
    preds = read_jsonl(args.predictions)
    if len({r['episode_id'] for r in preds}) != len(preds):
        raise ValueError('Duplicate prediction ids')
    indexed = {r['episode_id']: r for r in preds}
    unknown_ids = set(indexed) - {r['episode_id'] for r in truths}
    if unknown_ids:
        raise ValueError('Predictions have ids outside evaluation manifest')
    groups = defaultdict(list)
    details = []
    for gt in truths:
        pred = indexed.get(gt['episode_id'], {})
        goal_bearing = bearing(gt['start_xy_m'], gt['goal_xy_m'])
        resolved = pred.get('resolved_direction', {})
        heading = resolved.get('heading_deg') if pred.get('status') == 'ok' else None
        baseline = gt.get('initial_heading_deg')
        e = angular_error(heading, goal_bearing) if heading is not None and goal_bearing is not None else None
        b = angular_error(baseline, goal_bearing) if baseline is not None and goal_bearing is not None else None
        seed = pred.get('exploration_seed', {}).get('heading_deg')
        seed_error = angular_error(seed, goal_bearing) if seed is not None and goal_bearing is not None else None
        setting = pred.get('setting') or ('landmark_assisted' if gt['episode_id'].endswith('landmark_assisted') else 'visual')
        rec = {'episode_id': gt['episode_id'], 'goal_bearing_deg': goal_bearing,
               'predicted_heading_deg': heading, 'error_deg': e, 'initial_heading_error_deg': b,
               'exploration_seed_error_deg': seed_error,
               'status': pred.get('status', 'missing'), 'source': resolved.get('source'),
               'sector_contains_goal': e <= resolved['half_width_deg'] if e is not None and resolved.get('half_width_deg') is not None else None,
               'half_width_deg': resolved.get('half_width_deg')}
        groups[gt['dataset']+'/'+setting].append(rec)
        details.append(rec)

    def stats(records, field):
        eligible = [r for r in records if r['goal_bearing_deg'] is not None]
        values = [r[field] for r in eligible if r[field] is not None]
        n = len(eligible)
        return {'eligible_count': n, 'undefined_bearing_excluded': len(records)-n,
                'resolved_count': len(values), 'unresolved_or_failed': n-len(values),
                'resolution_rate': len(values)/n if n else None,
                'median_error_on_resolved_deg': statistics.median(values) if values else None,
                'within_deg_all_eligible': {str(t): sum(v <= t for v in values)/n if n else None for t in (30,45,60,90)},
                'within_deg_resolved_only': {str(t): sum(v <= t for v in values)/len(values) if values else None for t in (30,45,60,90)}}
    summary = {}
    for name, records in groups.items():
        paired = [r for r in records if r['error_deg'] is not None and r['initial_heading_error_deg'] is not None]
        widths = [r['half_width_deg'] for r in records if r['error_deg'] is not None and r['half_width_deg'] is not None]
        summary[name] = {'total': len(records), 'model': stats(records, 'error_deg'),
                         'exploration_policy_seed': stats(records, 'exploration_seed_error_deg'),
                         'initial_heading_baseline': stats(records, 'initial_heading_error_deg'),
                         'paired_resolved_count': len(paired),
                         'paired_mean_improvement_deg': statistics.mean(r['initial_heading_error_deg']-r['error_deg'] for r in paired) if paired else None,
                         'mean_sector_half_width_deg': statistics.mean(widths) if widths else None}
    save_json(args.out, {'summary': summary, 'records': details,
                        'limitations': ['Initialization bearing only, not navigation SR.',
                                        'Unknown, failed and missing predictions stay in eligible denominator.',
                                        'Zero start-goal displacement has no bearing and is excluded explicitly.',
                                        'Landmark centers are coarse geographic priors; target offsets remain unknown.']})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
