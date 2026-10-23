"""Offline geometry-based stage diagnostics; never called by navigation workers."""
from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import numpy as np
from shapely.geometry import Point, Polygon


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


online = module_at('stage_online', Path(__file__).with_name('online_experiment.py'))


def invalid_action_reasons(events):
    """Categorize recorded validation messages; never infer a visual failure cause."""
    counts = Counter()
    for event in events:
        if event.get('kind') != 'invalid_action':
            continue
        error = str(event.get('error',''))
        if error.startswith('Requested position ') and ' is outside map.' in error:
            category = 'outside_map'
        elif error == 'Image point is in the invalid raster area':
            category = 'invalid_raster'
        elif error == 'Optional tool limit reached; explore, locate or give_up':
            category = 'tool_budget_exhausted'
        else:
            category = 'other'
        counts[category] += 1
    return dict(counts)


def jev_raw_review(review, responses, folder):
    """Read only the matching in-job journal; missing evidence never becomes a choice."""
    details = dict(status='missing', choice=None, probabilities=None, abstained=None,
                   journal_path=None, journal_sha256=None)
    if not responses:
        return dict(details, reason='missing_jev_response')
    if len(responses) != 1:
        return dict(details, status='invalid', reason='ambiguous_jev_response')
    response = responses[0]
    if folder is None or not response.get('journal'):
        return dict(details, reason='missing_job_folder_or_journal_path')
    folder = Path(folder).resolve()
    path = Path(response['journal'])
    path = (path if path.is_absolute() else folder/path).resolve()
    if not path.is_relative_to(folder):
        return dict(details, status='invalid', reason='journal_outside_job')
    details['journal_path'] = str(path)
    if not path.is_file():
        return dict(details, reason='missing_journal_file')
    try:
        data = path.read_bytes()
        details['journal_sha256'] = hashlib.sha256(data).hexdigest()
        journal = json.loads(data)
        candidate = review['candidate']
        context = journal['input']['state']['context']
        saved_candidate = context['candidate']
        if (response['round'] != review['round'] or context.get('round') != review['round']+1 or
                context.get('mode') != 'review' or saved_candidate.get('target') != 'final'):
            raise ValueError('journal_round_or_target_mismatch')
        for field in ('view_id', 'bbox', 'center_xy_m', 'polygon_xy_m'):
            if field in candidate and saved_candidate.get(field) != candidate[field]:
                raise ValueError('journal_candidate_mismatch')
        if not all(field in candidate and field in saved_candidate for field in ('view_id', 'bbox', 'center_xy_m')):
            raise ValueError('journal_candidate_identity_missing')
        if (journal.get('status') != response.get('status') or
                journal.get('verdict') != response.get('verdict') or
                response.get('verdict') != review['answer']['verdict']):
            raise ValueError('journal_status_or_verdict_mismatch')
        if journal['status'] != 'ok':
            return dict(details, status='service_error', reason='jev_response_not_ok')
        raw = journal.get('raw', {}).get('answers', {}).get('review', {})
        required = ('choice', 'probabilities', 'abstained')
        if any(key not in journal for key in required) or not all(key in raw for key in ('choice', 'probabilities')):
            return dict(details, reason='missing_raw_choice_fields')
        choice, probabilities, abstained = journal['choice'], journal['probabilities'], journal['abstained']
        choices = {'confirm', 'reject', 'uncertain'}
        if (choice not in choices or not isinstance(probabilities, dict) or set(probabilities) != choices or
                not isinstance(abstained, bool) or raw['choice'] != choice or raw['probabilities'] != probabilities):
            raise ValueError('invalid_or_inconsistent_raw_choice')
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or
               not 0 <= v <= 1 for v in probabilities.values()) or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-4):
            raise ValueError('invalid_raw_probabilities')
        if journal.get('min_probability') != .7 or abstained != (max(probabilities.values()) < .7):
            raise ValueError('journal_threshold_policy_mismatch')
        if journal['verdict'] != ('uncertain' if abstained else choice):
            raise ValueError('journal_choice_verdict_mismatch')
        return dict(details, status='ok', choice=choice, probabilities=probabilities, abstained=abstained)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return dict(details, status='invalid', reason=str(exc))


def qualified_jev_choices(reviews):
    selected = [r['jev_raw'] for r in reviews if r['proposal_qualified'] and 'jev_raw' in r]
    choices = Counter(r['choice'] for r in selected if r['status'] == 'ok')
    return dict(unit='review_events_not_independent_episodes', total=len(selected),
                choices={key: choices[key] for key in ('confirm', 'reject', 'uncertain')},
                journal_statuses=dict(Counter(r['status'] for r in selected)),
                threshold_abstentions=sum(r['status']=='ok' and r['abstained'] for r in selected))


def interrupted_reviews(events, arm, final_keys, termination):
    """Identify failed review attempts without treating planning errors as reviews."""
    interrupted = []
    for index, event in enumerate(events):
        if event['kind'] != 'review_created':
            continue
        key = (event['round'], online.digest(event['candidate']))
        view_id = event.get('view', {}).get('view_id')
        if key not in final_keys or view_id is None:
            continue
        following = []
        for later in events[index+1:]:
            if later['round'] != event['round'] or later['kind'] in ('candidate', 'review_created'):
                break
            following.append(later)
            if later['kind'] == 'review_result':
                break
        # Only a trailing, unresolved review can have terminated the trajectory.
        if any(e['kind'] == 'review_result' for e in following) or index+1+len(following) != len(events):
            continue
        if arm == 'jev':
            halted = next((e for e in following if e['kind'] == 'jev_halt' and
                           view_id in e.get('view_ids', [])), None)
            if halted is not None:
                interrupted.append((event, 'halted', halted.get('reason', 'jev_halt')))
        elif termination == 'interface_error':
            responses = [e for e in following if e['kind'] == 'model_response' and
                         view_id in e.get('view_ids', [])]
            # ask() emits review-created -> matching model responses (including
            # retries/invalid formats) -> Halt, with no review_result on exhaustion.
            # A request-budget halt or errors on main/zoom planning views do not qualify.
            if responses:
                interrupted.append((event, 'interface_error', termination))
    return interrupted


def diagnose(dataset, public, truth, trace, summary, geometry, folder=None):
    views = {v['view_id']:v for v in trace.get('views', [])}
    delivered = set(trace.get('delivered_view_ids', []))
    env = public['environment']
    w, h = env['image_size']
    map_polygon = Polygon(geometry.project(np.linalg.inv(env['world_to_raster']), [[0,0],[w,0],[w,h],[0,h]]))
    goal = Point(truth['goal_xy_m'])
    covered = any(Polygon(views[i]['world_corners_xy_m']).intersection(map_polygon).covers(goal)
                  for i in delivered if i in views)
    def score(candidate):
        if dataset == 'CityNav':
            value = math.dist(candidate['center_xy_m'], truth['goal_xy_m'])
            return value, value <= 20
        view = views[candidate['view_id']]
        predicted = Polygon(geometry.project(view['pixel_to_world'], geometry.square_bbox(candidate['bbox_pixels'])))
        target = Polygon(truth['evaluation_polygon_xy_m'])
        union = predicted.union(target).area
        value = predicted.intersection(target).area / union if union else 0
        return value, value > .4
    events = trace.get('events', [])
    finals = {(e['round'], online.digest(e['candidate'])):e for e in events
              if e['kind']=='candidate' and e['target']=='final'}
    proposals = [dict(round=e['round'], metric=score(e['candidate'])[0], qualified=score(e['candidate'])[1])
                 for e in finals.values()]
    jev_status = {e['round']:e.get('status', 'missing') for e in events if e['kind']=='jev_response'}
    reviews = []
    for event in events:
        if event['kind'] != 'review_result' or (event['round'], online.digest(event['candidate'])) not in finals:
            continue
        value, qualified = score(event['candidate'])
        refined = score(event['refined']) if event.get('refined') else (None, None)
        status = jev_status.get(event['round'], 'missing') if summary['arm']=='jev' else 'ok'
        reviews.append(dict(round=event['round'], proposal_metric=value, proposal_qualified=qualified,
            verdict=event['answer']['verdict'], reviewer_status=status,
            refined_metric=refined[0], refined_qualified=refined[1], result_recorded=True))
        if summary['arm'] == 'jev':
            responses = [e for e in events if e['kind']=='jev_response' and e['round']==event['round']]
            reviews[-1]['jev_raw'] = jev_raw_review(event, responses, folder)
    for event, status, error in interrupted_reviews(events, summary['arm'], finals, trace.get('termination')):
        value, qualified = score(event['candidate'])
        reviews.append(dict(round=event['round'], proposal_metric=value, proposal_qualified=qualified,
            verdict=None, reviewer_status=status, reviewer_error=error,
            refined_metric=None, refined_qualified=None, result_recorded=False))
        if summary['arm'] == 'jev':
            reviews[-1]['jev_raw'] = dict(status='service_error', reason='jev_halt_without_response',
                choice=None, probabilities=None, abstained=None, journal_path=None, journal_sha256=None)
    tool_qualified = 0
    for event in events:
        if event['kind'] != 'tool' or event.get('output', {}).get('status') != 'ok':
            continue
        for candidate in event['output'].get('candidates', []):
            ident = candidate['view_id']
            if ident not in delivered or ident not in views:
                continue
            view, box = views[ident], candidate['bbox']
            corners = geometry.pixels(view, [[box[0],box[1]],[box[2],box[3]]])
            pixels = [*corners[0], *corners[1]]
            center = geometry.world(view, [[(box[0]+box[2])/2,(box[1]+box[3])/2]])[0]
            tool_qualified += score(dict(view_id=ident, bbox_pixels=pixels, center_xy_m=center))[1]
    qualified_reviews = [r for r in reviews if r['proposal_qualified']]
    missed = [r for r in qualified_reviews if r['reviewer_status']=='ok' and r['verdict']!='confirm']
    return dict(geometric_center_covered=bool(covered), final_proposals=len(proposals),
        qualified_final_proposals=sum(p['qualified'] for p in proposals),
        qualified_tool_proposals=int(tool_qualified), final_reviews=len(reviews),
        final_review_results=sum(r['result_recorded'] for r in reviews),
        interrupted_final_reviews=sum(not r['result_recorded'] for r in reviews),
        qualified_proposal_reviews=len(qualified_reviews),
        qualified_reviews_not_confirmed=len(missed),
        qualified_reviews_service_error=sum(r['reviewer_status']!='ok' for r in qualified_reviews),
        qualified_review_verdicts=dict(Counter(r['verdict'] for r in qualified_reviews if r['reviewer_status']=='ok')),
        qualified_jev_raw_choices=qualified_jev_choices(reviews),
        refinement_damaged=sum(r['proposal_qualified'] and r['refined_qualified'] is False for r in reviews),
        refinement_repaired=sum(not r['proposal_qualified'] and r['refined_qualified'] is True for r in reviews),
        invalid_actions=sum(e['kind']=='invalid_action' for e in events),
        invalid_action_reasons=invalid_action_reasons(events),
        invalid_formats=sum(e['kind']=='invalid_format' for e in events),
        proposals=proposals, reviews=reviews)


def main():
    root = online.ROOT
    plan = online.validate_frozen(root)
    if online.sha(root/'evaluator_only.json') != plan['evaluator_sha256']:
        raise ValueError('Evaluation labels changed')
    labels = online.read(root/'evaluator_only.json')
    analysis = online.read(root/'analysis.json')
    if analysis['plan_sha256'] != online.sha(root/'plan.json'):
        raise ValueError('Different analysis plan')
    geometry_path = root/'runtime/AVDN/tools/geometry.py'
    geometry = module_at('frozen_geometry', geometry_path)
    rows = []
    for item in analysis['rows']:
        if item['status'] not in online.TERMINAL:
            continue
        folder = online.job_folder(root, item)
        trace_path = folder/'trace.json'
        if not trace_path.exists():
            rows.append(dict(job_id=item['job_id'], dataset=item['dataset'], model=item['model'], arm=item['arm'],
                             missing_trace=True, status=item['status']))
            continue
        detail = diagnose(item['dataset'], online.read(root/item['public_input']),
            labels[item['dataset']][item['episode_id']], online.read(trace_path), item, geometry, folder)
        rows.append(dict(job_id=item['job_id'], dataset=item['dataset'], model=item['model'], arm=item['arm'],
            status=item['status'], success=item['success'], trace_sha256=online.sha(trace_path), **detail))
    groups = []
    fields = ('geometric_center_covered','qualified_tool_proposals','qualified_final_proposals',
              'qualified_reviews_not_confirmed','qualified_reviews_service_error','refinement_damaged','refinement_repaired')
    for dataset in online.POOLS:
        for model in online.MODELS:
            for arm in online.ARMS:
                selected = [r for r in rows if (r['dataset'],r['model'],r['arm'])==(dataset,model,arm)]
                groups.append(dict(dataset=dataset, model=model, arm=arm, terminal=len(selected),
                    missing_trace=sum(r.get('missing_trace',False) for r in selected),
                    invalid_action_events=sum(r.get('invalid_actions',0) for r in selected),
                    invalid_format_events=sum(r.get('invalid_formats',0) for r in selected),
                    invalid_action_reasons=dict(sum((Counter(r.get('invalid_action_reasons',{})) for r in selected),Counter())),
                    qualified_jev_raw_choices=qualified_jev_choices([v for r in selected for v in r.get('reviews', [])]),
                    **{field:sum(bool(r.get(field)) for r in selected) for field in fields}))
    report = dict(schema='jev-offline-stage-v1', plan_sha256=online.sha(root/'plan.json'),
        source_analysis_sha256=online.sha(root/'analysis.json'), geometry_sha256=online.sha(geometry_path),
        terminal=len(rows), planned=len(plan['jobs']), all_terminal=analysis['all_terminal'], rows=rows, groups=groups,
        qualified_jev_raw_choices=qualified_jev_choices([v for r in rows for v in r.get('reviews', [])]),
        limitations=['Coverage is geometric inclusion in delivered valid map footprint, not proof of human-visible identity.',
            'Qualification is the frozen geometric success proxy; it is not a semantic label.',
            'Only final-target proposals are evaluated against final-target ground truth; auxiliary targets are excluded.',
            'Review counts include evidenced attempts interrupted by Jev halt or MLLM review interface exhaustion; these have no verdict and are not abstentions.',
            'Stage columns count episodes and overlap; they are not a mutually exclusive causal failure classification.',
            'qualified_jev_raw_choices counts qualified final-review events, including repeated reviews within an episode; missing/invalid journals and service errors are separate from choices.',
            'A qualified candidate not confirmed is evidence of an observed bottleneck, not a counterfactual success guarantee.',
            'No online inputs, thresholds, models, or sampling decisions are changed by this offline report.'])
    online.save(root/'stage_analysis.json', report)
    print(json.dumps(dict(terminal=len(rows), planned=len(plan['jobs']),
                          missing_traces=sum(r.get('missing_trace',False) for r in rows))))


if __name__ == '__main__':
    main()
