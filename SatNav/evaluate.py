"""Offline-only scoring. No evaluator data is imported by the policy."""
import math
import statistics
from pathlib import Path
from AVDN.agent.io import read_jsonl, load_json, save_json


def evaluate(root):
    root = Path(root)
    truth = read_jsonl(root / 'evaluator_only.jsonl')
    init_path = root / 'initialization_results.jsonl'
    predictions = {x['episode_id']: x for x in read_jsonl(init_path)} if init_path.exists() else {}
    rows = []
    for gt in truth:
        ident = gt['episode_id']
        prediction = predictions.get(ident, {})
        heading = prediction.get('initialization', {}).get('direction')
        gx, gy = gt['goal_xy_m']
        goal_bearing = math.degrees(math.atan2(gx, gy)) % 360
        error = abs((heading-goal_bearing+180) % 360-180) if heading is not None else None
        row = dict(episode_id=ident, split=gt['split'], initialization_status=prediction.get('status', 'not_run'),
                   direction_error_deg=error, direction_within_90=error < 90 if error is not None else None,
                   navigation_status='not_run', success=None,
                   target_visibility='requires visual audit; goal-point containment is not target visibility')
        path = root / 'episodes' / ident / 'result.json'
        if path.exists():
            result = load_json(path)
            x, y = result['actual_stop_xy_m']
            distance = math.hypot(x-gx, y-gy)
            stopped = result['termination'] == 'confirmed_final'
            row.update(navigation_status=result['termination'], stop_distance_m=distance,
                       success=stopped and distance <= gt['success_radius_m'],
                       target_confirmed=result['final'] is not None,
                       rounds=result['rounds'], tool_calls=result['tool_calls'])
            trace = load_json(path.with_name('trace.json'))
            distances = [math.hypot(p['xy_m'][0]-gx, p['xy_m'][1]-gy) for p in trace['trajectory']]
            row.update(movement_count=max(0,len(trace['trajectory'])-1),
                       closest_visited_distance_m=min(distances),
                       ever_within_success_radius=min(distances)<=gt['success_radius_m'])
        rows.append(row)
    summary = {}
    for split in ('val_seen', 'val_unseen'):
        subset = [r for r in rows if r['split'] == split]
        resolved = [r for r in subset if r['direction_error_deg'] is not None]
        ran = [r for r in subset if r['navigation_status'] != 'not_run']
        qualified = [r for r in ran if r['direction_within_90'] is True]
        summary[split] = dict(sample_count=len(subset), resolved_directions=len(resolved),
            within_90_count=sum(r['direction_within_90'] for r in resolved),
            median_resolved_direction_error_deg=statistics.median(r['direction_error_deg'] for r in resolved) if resolved else None,
            navigation_runs=len(ran), successes=sum(bool(r['success']) for r in ran),
            direction_qualified_navigation_runs=len(qualified),
            direction_qualified_successes=sum(bool(r['success']) for r in qualified),
            success_rate_completed=sum(bool(r['success']) for r in ran)/len(ran) if ran else None)
    report = dict(summary=summary, episodes=rows,
                  protocol='project continuous-action adapter; not native SatSim benchmark parity')
    save_json(root / 'evaluation.json', report)
    return summary
