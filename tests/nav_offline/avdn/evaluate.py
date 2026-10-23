"""Offline evaluator only: navigation runtime never imports this module."""
from pathlib import Path
from collections import Counter
import numpy as np
from shapely.geometry import Polygon, Point
from shapely import make_valid
from AVDN.tools.runtime import read_jsonl, load_json, save_json
from AVDN.tools.geometry import project, square_bbox, world, box_points

def polygon(points):
    return make_valid(Polygon(points)) if len(points)>=3 else Polygon()

def iou(a,b):
    a,b=polygon(a),polygon(b)
    union=a.union(b).area
    return float(a.intersection(b).area/union) if union else 0.

def candidate_correct(candidate, dataset, gt, views):
    if dataset=='CityNav':
        return np.linalg.norm(np.asarray(candidate['center_xy_m'])-gt['goal_xy_m'])<=20
    v=views[candidate['view_id']]
    square=project(v['pixel_to_world'],square_bbox(candidate['bbox_pixels']))
    return iou(square,gt['evaluation_polygon_xy_m'])>0.3

def evaluate(folder):
    folder=Path(folder)
    public={x['episode_id']:x for x in read_jsonl(folder/'episodes.jsonl')}
    private=read_jsonl(folder/'evaluator_only.jsonl')
    rows=[]
    for gt in private:
        ident=gt['episode_id'];path=folder/'episodes'/ident
        if not (path/'result.json').exists():
            rows.append(dict(episode_id=ident,success=False,termination='not_started',
                             failure_category='5_limit_or_interface',prediction_valid=False))
            continue
        result=load_json(path/'result.json');trace=load_json(path/'trace.json')
        dataset=gt['dataset'];final=result['final'];env=public[ident]['environment']
        w,h=env['image_size']
        map_polygon=polygon(project(np.linalg.inv(env['world_to_raster']),[[0,0],[w,0],[w,h],[0,h]]))
        views={v['view_id']:v for v in trace['views']}
        target=polygon(gt.get('goal_polygon_xy_m',[]));center=Point(gt['goal_xy_m'])
        coverage=[]
        for ident_view in trace['delivered_view_ids']:
            valid=polygon(views[ident_view]['world_corners_xy_m']).intersection(map_polygon)
            coverage.append(dict(view_id=ident_view,center_covered=bool(valid.covers(center)),
                target_area_coverage=float(valid.intersection(target).area/target.area) if target.area else None))
        covered=any(v['center_covered'] for v in coverage)
        good_proposals=[e for e in trace['events'] if e['kind']=='candidate' and e['target']=='final'
                        and candidate_correct(e['candidate'],dataset,gt,views)]
        rejected=[e for e in trace['events'] if e['kind']=='review_result' and
                  e['answer']['verdict']!='confirm' and
                  any(e['candidate']==p['candidate'] for p in good_proposals)]
        tool_proposals=[]
        for event in trace['events']:
            if event['kind']!='tool' or event['output'].get('status')!='ok':continue
            for c in event['output'].get('candidates',[]):
                if c['view_id'] not in trace['delivered_view_ids']:continue
                view=views[c['view_id']];b=c['bbox']
                candidate=dict(view_id=c['view_id'],bbox_pixels=[v*767/1000 for v in b],
                    center_xy_m=world(view,[[(b[0]+b[2])/2,(b[1]+b[3])/2]])[0])
                tool_proposals.append(dict(backend=event['input']['tool'],view_id=c['view_id'],
                    candidate_id=c['candidate_id'],correct=bool(candidate_correct(candidate,dataset,gt,views))))
        row=dict(episode_id=ident,dataset=dataset,termination=result['termination'],
                 prediction_valid=final is not None,model_requests=result['model_requests'],
                 center_covered=covered,max_target_area_coverage=max((v['target_area_coverage'] or 0 for v in coverage),default=0),
                 coverage=coverage,correct_final_candidate_proposed=bool(good_proposals),
                 correct_tool_proposal=any(p['correct'] for p in tool_proposals),tool_proposals=tool_proposals,
                 correct_candidate_review_rejected=bool(rejected),
                 actual_stop_distance_m=float(np.linalg.norm(np.asarray(result['actual_stop_xy_m'])-gt['goal_xy_m'])))
        if dataset=='AVDN':
            value=iou(final['evaluation_polygon_xy_m'],gt['evaluation_polygon_xy_m']) if final else None
            row.update(iou=value,success=value is not None and value>0.3)
        else:
            value=float(np.linalg.norm(np.asarray(final['center_xy_m'])-gt['goal_xy_m'])) if final else None
            row.update(prediction_distance_m=value,success=value is not None and value<=20)
        geometric=('1_not_covered' if not covered else '3_candidate_review_failed' if rejected
                   else '2_covered_no_correct_candidate' if not good_proposals else '4_localization_or_stop_error')
        resource=result['termination'] not in ('confirmed_final','give_up')
        row['failure_category']=None if row['success'] else ('5_limit_or_interface' if resource else
                                '4_localization_or_stop_error' if final else geometric)
        row['geometric_failure_category']=None if row['success'] else geometric
        row['resource_limit']=result['termination'] if resource else None
        rows.append(row)
    distances=[r['prediction_distance_m'] for r in rows if r.get('prediction_distance_m') is not None]
    stopped=[r['actual_stop_distance_m'] for r in rows if 'actual_stop_distance_m' in r]
    report=dict(episodes=len(rows),run_episodes=sum(r['termination']!='not_started' for r in rows),
        successes=sum(r['success'] for r in rows),success_rate=sum(r['success'] for r in rows)/len(rows) if rows else 0,
        valid_predictions=sum(r['prediction_valid'] for r in rows),
        prediction_distance_mean_m=float(np.mean(distances)) if distances else None,
        prediction_distance_valid_count=len(distances),
        actual_stop_distance_mean_m=float(np.mean(stopped)) if stopped else None,
        actual_stop_distance_count=len(stopped),
        failure_categories=dict(Counter(r['failure_category'] for r in rows if not r['success'])),
        note='Coverage is geometric observation opportunity, not proof of visible facade attributes or a grounding fault.',
        tool_evidence_note='Tool availability and calls are recorded per episode; availability alone is not acceptance.',
        successes_iou_gt04=sum(r.get('iou') is not None and r['iou']>0.4 for r in rows) if private and private[0]['dataset']=='AVDN' else None,
        rows=rows)
    save_json(folder/'evaluation.json',report)
    return {k:v for k,v in report.items() if k!='rows'}

