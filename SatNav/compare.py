"""Paired prompt comparison with pending episodes explicitly retained."""
from pathlib import Path
from .evaluate import evaluate
from AVDN.agent.io import read_jsonl, load_json, save_json, digest


def compare(baseline, adapted, output):
    baseline, adapted = Path(baseline), Path(adapted)
    if digest(read_jsonl(baseline/'episodes.jsonl')) != digest(read_jsonl(adapted/'episodes.jsonl')):
        raise ValueError('Paired prompts require identical episode inputs and initialization')
    reports = []
    for root in (baseline, adapted):
        evaluate(root)
        reports.append(load_json(root/'evaluation.json'))
    a, b = ({r['episode_id']: r for r in report['episodes']} for report in reports)
    if set(a) != set(b):
        raise ValueError('Evaluation episode sets differ')
    rows = []
    for ident in a:
        first, second = a[ident], b[ident]
        complete = all(r['navigation_status']!='not_run' for r in (first,second))
        rows.append(dict(episode_id=ident, split=first['split'], paired_complete=complete,
                         baseline_success=first['success'], adapted_success=second['success'],
                         baseline_stop_distance_m=first.get('stop_distance_m'),
                         adapted_stop_distance_m=second.get('stop_distance_m')))
    done = [r for r in rows if r['paired_complete']]
    report = dict(total=len(rows), paired_complete=len(done),
                  gained=sum(r['adapted_success'] and not r['baseline_success'] for r in done),
                  lost=sum(r['baseline_success'] and not r['adapted_success'] for r in done),
                  both_success=sum(r['baseline_success'] and r['adapted_success'] for r in done),
                  both_failure=sum(not r['baseline_success'] and not r['adapted_success'] for r in done),
                  baseline_summary=reports[0]['summary'], adapted_summary=reports[1]['summary'],
                  note='Pending episodes are excluded from paired counts, never treated as failures.', rows=rows)
    save_json(output,report)
    return {k:v for k,v in report.items() if k not in ('rows','baseline_summary','adapted_summary')}


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True)
    p.add_argument('--adapted',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(compare(a.baseline,a.adapted,a.output))
