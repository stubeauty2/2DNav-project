"""Offline paired audit stratified by availability of complete language geometry."""
from pathlib import Path
from .compare import compare
from .route_geometry import numeric_route
from AVDN.agent.io import read_jsonl, load_json, save_json


def audit(baseline, adapted, output):
    baseline,adapted=Path(baseline),Path(adapted)
    compare(baseline,adapted,output)
    report=load_json(output)
    predictions={p['episode_id']:p for p in read_jsonl(adapted/'initialization_results.jsonl')}
    strata={name:dict(total=0,completed=0,baseline_successes=0,adapted_successes=0,
                     numeric_stops=0,motion_corrections=0) for name in ('complete_numeric','other')}
    for row in report['rows']:
        route=numeric_route(predictions[row['episode_id']].get('diagnostics'))
        name='complete_numeric' if route else 'other'
        row['language_geometry_group']=name
        group=strata[name];group['total']+=1
        if not row['paired_complete']:continue
        group['completed']+=1
        group['baseline_successes']+=bool(row['baseline_success'])
        group['adapted_successes']+=bool(row['adapted_success'])
        trace=load_json(adapted/'episodes'/row['episode_id']/'trace.json')
        kinds=[e['kind'] for e in trace['events']]
        group['numeric_stops']+=kinds.count('stop_pose_from_language')
        group['motion_corrections']+=kinds.count('language_motion_correction')
        if 'stop_pose_from_language' in kinds:
            prior=trace['events'][:kinds.index('stop_pose_from_language')]
            assert any(e['kind']=='review_result' and e['answer']['verdict']=='confirm' for e in prior)
    report['language_geometry_strata']=strata
    save_json(output,report)
    return strata


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True);p.add_argument('--adapted',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(audit(a.baseline,a.adapted,a.output))
