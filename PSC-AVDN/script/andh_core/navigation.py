"""Multi-candidate Search and Confirmation; previews never move the agent."""
import json
from copy import deepcopy
from pathlib import Path
import cv2
import numpy as np
from util import _move_forward_geo
from .config import FIXED_CROP_SIDE, NavigationOptions
from .tools.views import acquire, search_preview_poses
from .memory import TrajectoryMemory
from .tools.mosaic import build_mosaic
from .tools.model import ModelClient, infer, rgb_view
from .tools.geometry import project_bbox, polygon_iou, relative_pose
from .tools.views import render_context_views, create_view_image
from .tools.geometry import _minimal_square_uv_from_bbox, _patch_xy_to_latlng, _haversine_m
from .tools.telemetry import append_event
from util import generate_view_corners_with_scale


def search_and_reach_destination(*, instr_id, ob, dest_desc, via_desc, start_pos_latlng,
                     heading_deg, scale_factor, out_dir, step_meters, max_steps,
                     api_key, url, model, options=None,
                     trajectory_context=None):
    options = options if options is not None else NavigationOptions.from_env()
    root = Path(out_dir); root.mkdir(parents=True, exist_ok=True)
    trace = str(root / f'{instr_id}_observations.jsonl') if options.enable_trace else None
    pos = list(start_pos_latlng); heading = float(heading_deg or 0)
    context = trajectory_context if trajectory_context is not None else {}
    memory = context.setdefault('memory', TrajectoryMemory())
    observation_cache = {}
    model_calls = []
    stage = memory.stage(instr_id, dest_desc)
    client = ModelClient(api_key=api_key, url=url, model=model)
    history = []; boxes = []
    found = False; destination = None; zoom_idx = None
    final_status = 'budget_exhausted'
    counts = {"preview_queries": 0, "preview_acquired": 0}

    def event(kind, **data):
        record = memory.event(instr_id, kind, **data)
        append_event(trace, {"event": kind, "stage_id": instr_id, **record})

    def move(after, action):
        nonlocal pos
        before = list(pos); pos = list(after)
        event('actual_move', from_position=before, to_position=list(pos), action=action,
              distance_m=_haversine_m(*before, *pos))

    def capture(position, crop_heading, round_n, name, **kw):
        try:
            obs = acquire(ob, position, crop_heading, scale_factor, instr_id,
                          round_n, name, out_dir, **kw)
        except (ValueError, OSError) as exc:
            event('acquisition_error', name=name, round=round_n, reason=str(exc))
            return None
        memory.observe(obs)
        observation_cache[obs.observation_id] = obs
        event('observation_acquired', observation_id=obs.observation_id,
              acquisition=obs.acquisition)
        return obs

    def call(mode, views, observations, extra=None):
        details = dict(extra or {})
        if options.use_relative_pose:
            from types import SimpleNamespace
            anchor = SimpleNamespace(position=pos, heading=heading, observation_id='agent_now')
            details['views_relative_to_agent'] = [relative_pose(anchor, o) for o in observations]
        # Archive every supplied image, including generated detail/context views.
        for index, view in enumerate(views):
            if not view.get('image_path'):
                path = root / 'model_inputs' / f'{instr_id}_call{len(model_calls)+1:03d}_{index}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), view['image']):
                    raise OSError('Cannot archive model input')
                view['image_path'] = str(path.resolve())
        audit = {}
        result = infer(client, mode, dest_desc, views, via=via_desc,
                       memory=memory.summary(instr_id, details.get('candidate_views')),
                       extra=details, retries=options.model_retries, trace_path=trace, audit=audit)
        model_calls.append(audit)
        event('model_result', mode=mode, result=deepcopy(result),
              observation_ids=[o.observation_id for o in observations])
        return result

    def propose_and_confirm(views, observations, round_n):
        from types import SimpleNamespace
        anchor = SimpleNamespace(position=pos, heading=heading, observation_id='agent_now')
        proposal = call('candidate_proposal', views, observations,
                        extra={'candidate_count': options.candidate_count, 'round': round_n,
                               'views_relative_to_agent': [relative_pose(anchor,o) for o in observations],
                               'view_labels': next((v['source_views'] for v in views if 'source_views' in v), {})})
        if proposal['decision'] in ('tool_error','parse_error'):
            return proposal, None
        for item in proposal['candidates']:
            if item is None:
                continue
            source = next(o for o in observations if o.observation_id+':rgb' == item['source_view_id'])
            polygon, center = project_bbox(item['bbox_2d'], source, ob)
            candidate, created = memory.merge_candidate(instr_id, source, item['bbox_2d'], polygon, center, item)
            event('candidate_proposed' if created else 'candidate_merged',
                  candidate_id=candidate['candidate_id'], source_view_id=item['source_view_id'],
                  round=round_n, support=item['support'], conflicts=item['conflicts'])
        eligible = [c for c in stage['candidates'].values() if c['status'] in ('pending','proposed','uncertain')]
        if not eligible:
            return {'decision':'no_match'}, None
        # Crop every eligible candidate without changing the agent's actual position.
        confirm_views = []
        confirm_observations = []
        candidate_views = {}
        for candidate in eligible:
            cid = candidate['candidate_id']
            source = observation_cache[candidate['source_observation_id']]
            obs = capture(candidate['map_center'], source.heading, round_n,
                          'confirm_' + cid.rsplit(':', 1)[-1],
                          acquisition='candidate_confirmation_preview',
                          anchor_id=source.observation_id,
                          description=f'候选 {cid}：以Search目标框中心裁剪的确认视图。')
            if obs is None:
                continue
            main = rgb_view(obs)
            candidate_views[cid] = main['view_id']
            confirm_views.append(main)
            confirm_observations.append(obs)
            entry = {'round': round_n, 'main_view_id': main['view_id'],
                     'observation_id': obs.observation_id}
            detail, surrounding = render_context_views(obs.position, ob, obs.heading,
                render=create_view_image, generate_corners=generate_view_corners_with_scale)
            for name, image, label in (('detail', detail, '同位置的放大视图。'),
                                       ('context', surrounding, '同位置的扩大范围视图。')):
                if image is not None:
                    view_id = obs.observation_id + ':' + name
                    confirm_views.append({'view_id': view_id, 'image': image,
                        'selectable': False, 'description': f'候选 {cid}：{label}'})
                    entry[name + '_view_id'] = view_id
            candidate.setdefault('confirmation_observations', []).append(entry)
            candidate['attempt_count'] += 1
        if not candidate_views:
            return {'decision': 'uncertain'}, None
        # Original search crops and mosaic retain the wider relational context,
        # but only newly centred confirmation main views may supply a final bbox.
        confirm_views.extend(dict(v, selectable=False) for v in views)
        relative = []
        for candidate in eligible:
            if candidate['candidate_id'] in candidate_views:
                target = SimpleNamespace(position=candidate['map_center'], heading=heading,
                                         observation_id=candidate['candidate_id'])
                relative.append(relative_pose(anchor, target))
        result = call('candidate_confirmation', confirm_views, confirm_observations, extra={
            'candidate_views': candidate_views, 'round': round_n,
            'can_continue_search': round_n < max_steps,
            'candidate_positions_relative_to_agent': relative})
        event('candidate_confirmation', round=round_n, result=result)
        if result['decision'] in ('parse_error', 'tool_error'):
            return result, None
        if result['decision'] == 'confirmed' and options.strict_confirmation:
            candidate = stage['candidates'][result['candidate_id']]
            oid = result['target_view_id'][:-4]
            polygon, _ = project_bbox(result['bbox_2d'], observation_cache[oid], ob)
            agreement = polygon_iou(polygon, candidate['map_polygon'])
            candidate['confirmation_iou'] = agreement
            if agreement < .1:
                event('confirmation_uncertain', candidate_id=candidate['candidate_id'],
                      round=round_n, agreement=agreement, reason='geometry_mismatch')
                result = dict(result, decision='continue_search' if round_n < max_steps else 'no_match',
                              candidate_id=None, target_view_id=None, bbox_2d=None)
                result['candidate_assessments'] = [dict(a, status='uncertain',
                    reason='Confirmation geometry does not support the proposed region.',
                    unresolved=['Candidate identity remains uncertain.'])
                    if a['candidate_id'] == candidate['candidate_id'] else a
                    for a in result['candidate_assessments']]
        memory.record_confirmation(instr_id, result, round_n)
        if result['decision'] != 'confirmed':
            return {'decision': 'uncertain' if result['decision'] == 'continue_search' else 'no_match'}, None
        candidate = stage['candidates'][result['candidate_id']]
        return {**result, 'decision': 'found'}, candidate

    event('stage_start', options=vars(options))
    try:
        for round_n in range(1, max_steps + 1):
            current = capture(pos, heading, round_n, 'current')
            if current is None:
                final_status = 'acquisition_error'
                break
            # Only actual agent observations enter the runner's trajectory boxes.
            boxes.append(np.asarray(current.view_corners))
            observations = [current]
            # Retry smaller displacement near map boundaries without moving the agent.
            for name in ('left','forward','right'):
                for shrink in (1., .5, .25, .125):
                    pose = next(p for p in search_preview_poses(current, options.preview_offset_ratio, shrink) if p[0]==name)
                    _, center, crop_heading, offset, distance = pose
                    counts['preview_queries'] += 1
                    direction = '正前方' if offset == 0 else ('左偏45度' if offset < 0 else '右偏45度')
                    preview = capture(center, crop_heading, round_n, name,
                        acquisition='map_preview', anchor_id=current.observation_id,
                        description=f'这是从当前位置{direction}、前进{distance:.0f}m处获取的视图裁剪。')
                    if preview is not None:
                        observations.append(preview)
                        counts['preview_acquired'] += 1
                        break
            # Prefer the latest actual main and a prior proposal's source; then other previews.
            prioritized = [o for o in reversed(history) if o.acquisition == 'actual_pose_crop'][:1]
            source_ids = {c['source_observation_id'] for c in stage['candidates'].values()}
            prioritized += [o for o in reversed(history) if o.observation_id in source_ids and o not in prioritized]
            prioritized += [o for o in reversed(history) if o not in prioritized]
            recent = prioritized[:options.history_size]
            # A historical preview must not be described as the agent's old location.
            views = [rgb_view(o) for o in observations]
            mosaic = build_mosaic(observations, current, root / f'{instr_id}_r{round_n:02d}_mosaic.png')
            views.insert(0, mosaic)
            for old in recent:
                label = (f'第{old.search_step}轮获取的历史视图。' if old.acquisition != 'map_preview'
                         else f'第{old.search_step}轮获取的历史预览视图。')
                observations.append(old); views.append(rgb_view(old, label))
            history.extend(o for o in observations if o.search_step == round_n)
            result, selected_candidate = propose_and_confirm(views, observations, round_n)
            if result['decision'] in ('tool_error', 'parse_error'):
                final_status = result['decision']
                event(final_status, round=round_n)
                if round_n < max_steps:
                    move(_move_forward_geo(pos, heading, meters=step_meters), 'forward')
                    continue
                break
            if result['decision'] != 'found':
                final_status = 'budget_exhausted'
                event('searched_no_match' if result['decision'] == 'no_match' else 'uncertain',
                      round=round_n, observation_ids=[o.observation_id for o in observations])
                if round_n < max_steps:
                    move(_move_forward_geo(pos, heading, meters=step_meters), 'forward')
                continue
            source = observation_cache[result['target_view_id'][:-4]]
            # The comparison already returned a new bbox in a confirmation crop.
            # There is no preselection call or second single-candidate confirmation.
            polygon, center = project_bbox(result['bbox_2d'], source, ob)
            selected_candidate['confirmed_map_polygon'] = polygon
            selected_candidate['confirmed_map_center'] = center
            move(center, 'approach_confirmed_target')
            destination = center
            uv = _minimal_square_uv_from_bbox(result['bbox_2d'], side=FIXED_CROP_SIDE)
            zoom_corners = [_patch_xy_to_latlng(x, y, source.view_corners, ob) for x, y in uv]
            create_view_image(zoom_corners, ob, save_path=str(root / f'{instr_id}_dest_zoom.jpg'), out_px=FIXED_CROP_SIDE)
            boxes.append(np.asarray(zoom_corners)); zoom_idx = len(boxes)-1
            found = True; final_status = 'confirmed'
            event('confirmation_passed', candidate_id=selected_candidate['candidate_id'])
            break
    except Exception as exc:
        final_status = 'internal_error'
        event('internal_error', error_type=type(exc).__name__)
        raise
    finally:
        event('stage_end', found=found, status=final_status, final_position=pos,
              returned_destination=destination, **counts)
        record = {"instr_id": instr_id, "found": found, "dest_latlng": destination,
                  "final_pos": pos, "status": final_status, "options": vars(options),
                  "query_counts": counts, "memory": memory.serialize(), "model_calls": model_calls}
        (root / f'{instr_id}_search.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    return found, destination, boxes, zoom_idx
