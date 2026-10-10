"""Original SAM3 Search/Confirmation policy with passive STEP 0 trace hooks."""
import os
import json
import math
import cv2
import numpy as np
from .tools import telemetry as _audit
from .config import RESULTS_CSV, ANNO_DIR, DATASET_DIR, SPLIT, OUT_DIR, SCALE_FACTOR, FIXED_CROP_SIDE, DEFAULT_API_KEY, QWEN_URL, QWEN_MODEL, VISION_TOOL_URL, SEARCH_QWEN_TIMEOUT
from .tools.category_terms import extract_category_terms
from .tools.model import qwen_locate_bbox_in_view
from .tools.sam3 import _sam3_image5
from .tools.geometry import _haversine_m, _patch_xy_to_latlng
from .tools.views import create_view_image, _draw_bbox_on_patch, generate_view_corners_with_scale, draw_pos_on_patch, make_minimap_crop_basemap
from util import _corners_center, _move_forward_geo

def search_and_reach_destination(
    instr_id, ob, dest_desc, via_desc,
    start_pos_latlng, heading_deg, scale_factor,
    out_dir,
    step_meters=120.0,
    max_steps=3,
    api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL,
    target_terms=None, vision_tool_url=VISION_TOOL_URL,
    candidate_count=0,
    confirmation_review='off',
    spatial_state=None,
):
    if spatial_state is not None:
        spatial_state.begin_stage(instr_id, dest_desc, start_pos_latlng, heading_deg or 0.0)
        if spatial_state.mode != 'g0' and candidate_count != 2:
            raise ValueError('G1--G4 require candidate_count=2; G0 also supports the frozen baseline')
    if confirmation_review != 'off':
        if confirmation_review not in ('observe','once') or candidate_count != 2:
            raise ValueError('Review requires candidate_count=2 and mode observe or once')
    if candidate_count:
        return _search_candidates_and_reach_destination(**locals())

    def _haversine_m(lat1, lng1, lat2, lng2):
        R = 6371000.0
        phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlmb = math.radians(lng2 - lng1)
        a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2.0)**2
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
        return R * c

    start_pos = [float(start_pos_latlng[0]), float(start_pos_latlng[1])]
    pos = start_pos[:]
    heading = float(heading_deg or 0.0)

    steps_log = []
    found = False
    dest_latlng = None

    predicted_corners_seq = []
    zoom_idx = None
    patch_seq = []

    first_step_pos = None
    fallback_scale = 2.0

    last_grid_5x5 = None

    for k in range(max_steps):
        step_n = k + 1
        _audit.context(trajectory_id=instr_id.rsplit('_step', 1)[0], stage_id=instr_id, round=step_n, position=pos, heading=heading, role='search')
        _audit.event('search_round_start', destination=dest_desc, via=via_desc, scale=scale_factor, step_meters=step_meters, max_steps=max_steps)

        corners_k = generate_view_corners_with_scale(
            pos, ob, scale_factor=scale_factor, angle_deg=heading
        )
        predicted_corners_seq.append(np.array(corners_k, dtype=float))

        step_path = os.path.join(
            out_dir,
            f"{instr_id}_step{step_n:02d}_search_view_h{heading:.1f}.jpg"
        )
        patch_k = create_view_image(
            corners_k, ob,
            save_path=step_path,
            out_px=FIXED_CROP_SIDE
        )

        corners_k_s4 = generate_view_corners_with_scale(
            pos, ob, scale_factor=3, angle_deg=heading
        )
        patch_k_s4 = create_view_image(
            corners_k_s4, ob,
            save_path=None,
            out_px=FIXED_CROP_SIDE
        )

        corners_k_s8 = generate_view_corners_with_scale(
            pos, ob, scale_factor=7, angle_deg=heading
        )
        patch_k_s8 = create_view_image(
            corners_k_s8, ob,
            save_path=None,
            out_px=FIXED_CROP_SIDE
        )

        if patch_k is None:
            _audit.event('render_failure', path=step_path)
            steps_log.append({"k": k, "action": "render_fail", "pos": pos})
            pos = _move_forward_geo(pos, heading, meters=step_meters)
            _audit.move(pos, 'render_failure_forward')
            if spatial_state is not None:
                spatial_state.move(pos, heading, "navigation_move")
            if first_step_pos is None:
                first_step_pos = pos[:]
            continue

        draw_pos_on_patch(patch_k, corners_k, pos)
        cv2.imwrite(step_path, patch_k, [cv2.IMWRITE_JPEG_QUALITY, 95])

        patch_seq.append(patch_k)
        mini_path = os.path.join(
            out_dir,
            f"{instr_id}_step{step_n:02d}_minimap.jpg"
        )
        minimap_bgr = make_minimap_crop_basemap(
            ob,
            corners_seq=predicted_corners_seq,
            img_seq=patch_seq,
            out_path=mini_path,
            pad_m=40.0
        )

        segmentation_image = None
        sam3_meta = None
        try:
            terms = list(target_terms or extract_category_terms(dest_desc))
            sam3_dir = os.path.join(out_dir, "sam3_image5", f"{instr_id}_step{step_n:02d}")
            segmentation_image, sam3_meta = _sam3_image5(
                patch_k, terms, image_path=step_path,
                artifact_dir=sam3_dir, vision_tool_url=vision_tool_url,
            )
        except Exception as exc:
            _audit.event('sam3_unavailable', error=str(exc), baseline_action='continue_without_image5')
            print(f"[WARN][SAM3] Image 5 unavailable for {instr_id} step{step_n:02d}; using four-image Search: {exc}")

        _audit.context(trajectory_id=instr_id.rsplit('_step', 1)[0], stage_id=instr_id, round=step_n, position=pos, heading=heading, role='search', views={'main': corners_k, 'detail': corners_k_s4, 'context': corners_k_s8}, sam3=sam3_meta)
        if spatial_state is not None:
            spatial_state.prepare_search([('main',patch_k,corners_k),('detail',patch_k_s4,corners_k_s4),
                ('context',patch_k_s8,corners_k_s8)],pos,heading,step_n,scale_factor,ob)
        q = qwen_locate_bbox_in_view(
            dest_desc,
            patch_main_bgr=patch_k,
            patch_s4_bgr=patch_k_s4,
            patch_s8_bgr=patch_k_s8,
            minimap_bgr=minimap_bgr,
            api_key=api_key, url=url, model=model,
            timeout=SEARCH_QWEN_TIMEOUT,
            restrict_top_half=True,
            prev_grid_5x5=last_grid_5x5,
            segmentation_image=segmentation_image,
        )
        if sam3_meta is not None:
            q["sam3_image5"] = sam3_meta

        last_grid_5x5 = q.get("grid_5x5") or last_grid_5x5

        if spatial_state is not None:
            source_id=instr_id+':r%d:main'%step_n
            proposals=[dict(source_view_id=source_id,**{k:q.get(k) for k in ('bbox_2d','confidence')})] if q.get('dest_present') else []
            assessment=dict(candidates=proposals)
            if str(q.get('reason','')).startswith('fallback:'):
                assessment=dict(candidates=[],validation_errors=['baseline_operational_error'])
            spatial_state.record_search([source_id],assessment,step_n)
            spatial_state.snapshot('round_%s_%d'%(instr_id,step_n))

        h, w = patch_k.shape[:2]
        steps_log.append({"k": k, "qwen": q, "pos": pos, "w": w, "h": h})
        _audit.bbox('search_bbox', q.get('bbox_2d'), corners_k, ob, dest_present=q.get('dest_present'), view_path=step_path)

        if q.get("dest_present"):
            bbox = [float(v) for v in (q.get("bbox_2d") or [])][:4]
            if len(bbox) != 4 or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                _audit.event('invalid_search_bbox', bbox=bbox)
                pos = _move_forward_geo(pos, heading, meters=step_meters)
                _audit.move(pos, 'invalid_search_bbox_forward')
                if spatial_state is not None:
                    spatial_state.move(pos, heading, "navigation_move")
                if first_step_pos is None:
                    first_step_pos = pos[:]
                if step_n >= max_steps and not found:
                    fallback_pos = first_step_pos or _move_forward_geo(start_pos, heading, meters=step_meters)
                    pos = fallback_pos[:]
                    _audit.move(pos, 'return_to_first_step_after_invalid_bbox')
                    if spatial_state is not None:
                        spatial_state.move(pos, heading, "navigation_move")
                    corners_back = generate_view_corners_with_scale(
                        pos, ob,
                        scale_factor=fallback_scale,
                        angle_deg=heading
                    )
                    back_path = os.path.join(
                        out_dir,
                        f"{instr_id}_step{step_n:02d}_return_to_first_step_hr.jpg"
                    )
                    patch_back = create_view_image(
                        corners_back, ob,
                        save_path=back_path,
                        out_px=FIXED_CROP_SIDE
                    )
                    if patch_back is not None:
                        draw_pos_on_patch(patch_back, corners_back, pos)
                        cv2.imwrite(back_path, patch_back, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    predicted_corners_seq.append(np.array(corners_back, dtype=float))
                    steps_log.append({"k": k, "action": "return_to_first_step", "pos": pos})
                    break
                continue

            try:
                side = float(FIXED_CROP_SIDE)
                x1f, y1f, x2f, y2f = bbox
                cx_px = 0.5 * (x1f + x2f)
                cy_px = 0.5 * (y1f + y2f)
                in_lower_half = (cy_px >= side * 0.5)

                if in_lower_half:
                    img_center_ll = _patch_xy_to_latlng(
                        side * 0.5, side * 0.5,
                        corners_k, ob
                    )
                    box_center_ll = _patch_xy_to_latlng(
                        cx_px, cy_px,
                        corners_k, ob
                    )
                    dist_m = _haversine_m(
                        img_center_ll[0], img_center_ll[1],
                        box_center_ll[0], box_center_ll[1]
                    )

                    if dist_m > 120.0:
                        start_box_ll = np.array(ob['gt_path_corners'][0], dtype=float)
                        predicted_corners_seq.append(start_box_ll)
                        steps_log.append({
                            "k": k,
                            "action": "override_to_start_box_on_first_detect_lower_half_far",
                            "dist_m": float(dist_m),
                            "bbox": [x1f, y1f, x2f, y2f]
                        })
                        pos = _corners_center(start_box_ll)
                        _audit.move(pos, 'lower_half_far_override_local_position')
                        if spatial_state is not None:
                            spatial_state.move(pos, heading, "navigation_move")
                        _audit.event('evaluation_region', source='start_box_override', map_polygon_latlng=start_box_ll, bbox_2d=None, actual_move=False)
                        found = True
                        zoom_idx = None
                        break
            except Exception as _e:
                _audit.event('first_detect_rule_error', error=str(_e))
                steps_log.append({"k": k, "action": "first_detect_rule_error", "error": str(_e)})

            det_vis_path = os.path.join(
                out_dir,
                f"{instr_id}_step{step_n:02d}_det.jpg"
            )
            patch_det = patch_k.copy()
            _draw_bbox_on_patch(
                patch_det, bbox,
                color=(0,0,255), thickness=3,
                label=f"{float(q.get('confidence',0.0)):.2f}"
            )
            cv2.imwrite(det_vis_path, patch_det, [cv2.IMWRITE_JPEG_QUALITY, 95])

            x1f, y1f, x2f, y2f = bbox
            cx_px = 0.5 * (x1f + x2f); cy_px = 0.5 * (y1f + y2f)
            dest_latlng = _patch_xy_to_latlng(cx_px, cy_px, corners_k, ob)
            pos = dest_latlng[:]
            _audit.move(pos, 'approach_search_center')
            if spatial_state is not None:
                spatial_state.move(pos, heading, "navigation_move")

            corners_confirm = generate_view_corners_with_scale(
                pos, ob,
                scale_factor=scale_factor,
                angle_deg=heading
            )
            predicted_corners_seq.append(np.array(corners_confirm, dtype=float))

            confirm_path = os.path.join(
                out_dir,
                f"{instr_id}_step{step_n:02d}_dest_confirm_hr.jpg"
            )
            patch_confirm = create_view_image(
                corners_confirm, ob,
                save_path=confirm_path,
                out_px=FIXED_CROP_SIDE
            )
            if patch_confirm is not None:
                draw_pos_on_patch(patch_confirm, corners_confirm, pos)
                cv2.imwrite(confirm_path, patch_confirm, [cv2.IMWRITE_JPEG_QUALITY, 95])

            corners_confirm_s4 = generate_view_corners_with_scale(
                pos, ob, scale_factor=3, angle_deg=heading
            )
            patch_confirm_s4 = create_view_image(
                corners_confirm_s4, ob,
                save_path=None,
                out_px=FIXED_CROP_SIDE
            )

            corners_confirm_s8 = generate_view_corners_with_scale(
                pos, ob, scale_factor=7, angle_deg=heading
            )
            patch_confirm_s8 = create_view_image(
                corners_confirm_s8, ob,
                save_path=None,
                out_px=FIXED_CROP_SIDE
            )

            _audit.context(trajectory_id=instr_id.rsplit('_step', 1)[0], stage_id=instr_id, round=step_n, position=pos, heading=heading, role='confirmation', views={'main': corners_confirm, 'detail': corners_confirm_s4, 'context': corners_confirm_s8})
            _audit.event('confirmation_view', view_path=confirm_path, view_corners_latlng=corners_confirm, actual_move=False)
            q_confirm = qwen_locate_bbox_in_view(
                dest_desc,
                patch_main_bgr=patch_confirm,
                patch_s4_bgr=patch_confirm_s4,
                patch_s8_bgr=patch_confirm_s8,
                minimap_bgr=None,
            api_key=api_key, url=url, model=model,
            restrict_top_half=False,
            prev_grid_5x5=last_grid_5x5,
            )
            steps_log.append({"k": k, "confirm": q_confirm, "pos": pos})

            last_grid_5x5 = q_confirm.get("grid_5x5") or last_grid_5x5

            confirm_box  = None
            conf_confirm = 0.0

            if q_confirm.get("dest_present"):
                cb = [float(v) for v in q_confirm.get("bbox_2d", [])][:4]
                if len(cb) == 4 and cb[2] > cb[0] and cb[3] > cb[1]:
                    side2 = float(FIXED_CROP_SIDE); eps = 1.0
                    bx1, by1, bx2, by2 = cb
                    bx1 = max(eps, min(bx1, side2-1.0-eps))
                    by1 = max(eps, min(by1, side2-1.0-eps))
                    bx2 = max(eps, min(bx2, side2-1.0-eps))
                    by2 = max(eps, min(by2, side2-1.0-eps))
                    confirm_box  = [bx1, by1, bx2, by2]
                    conf_confirm = float(q_confirm.get("confidence", 0.0))

                    confirm_vis_path = os.path.join(
                        out_dir,
                        f"{instr_id}_step{step_n:02d}_dest_confirm_det.jpg"
                    )
                    patch_confirm_det = patch_confirm.copy()
                    _draw_bbox_on_patch(
                        patch_confirm_det,
                        confirm_box,
                        color=(0,255,0),
                        thickness=3,
                        label=f"C:{conf_confirm:.2f}"
                    )
                    cv2.imwrite(
                        confirm_vis_path,
                        patch_confirm_det,
                        [cv2.IMWRITE_JPEG_QUALITY, 95]
                    )

            _audit.bbox('confirmation_bbox', q_confirm.get('bbox_2d'), corners_confirm, ob, applied_bbox=confirm_box, dest_present=q_confirm.get('dest_present'), view_path=confirm_path)
            _audit.event('confirmation_resolution', source='confirmation' if confirm_box is not None else 'search_fallback', reason=q_confirm.get('reason'))
            x1f, y1f, x2f, y2f = [float(v) for v in bbox]
            if confirm_box is not None:
                bx1, by1, bx2, by2 = confirm_box
                use_box = [bx1, by1, bx2, by2]
                cx_use = 0.5 * (bx1 + bx2)
                cy_use = 0.5 * (by1 + by2)
                center_latlng = _patch_xy_to_latlng(
                    cx_use, cy_use,
                    corners_confirm, ob
                )
            else:
                use_box = [x1f, y1f, y2f, y2f]
                use_box = [x1f, y1f, x2f, y2f]
                center_latlng = dest_latlng

            def _minimal_square_uv_from_bbox(bbox, side=FIXED_CROP_SIDE, pad_px=0):
                x1, y1, x2, y2 = [float(v) for v in bbox]
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                edge = max(w, h) * 0.5
                edge += float(pad_px)

                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)

                def _clamp_square(cx, cy, half, S):
                    left  = cx - half; right = cx + half
                    top   = cy - half; bot   = cy + half

                    if left < 0:      cx += -left
                    if right > S-1:   cx -= (right - (S-1))
                    if top  < 0:      cy += -top
                    if bot  > S-1:    cy -= (bot - (S-1))

                    half = min(half, cx, S-1-cx, cy, S-1-cy)

                    left  = cx - half; right = cx + half
                    top   = cy - half; bot   = cy + half
                    return cx, cy, half, (left, top, right, bot)

                cx, cy, edge, (l, t, r, b) = _clamp_square(
                    cx, cy, edge, float(side)
                )
                return [(l, t), (r, t), (r, b), (l, b)]

            corners_base = corners_confirm if (confirm_box is not None) else corners_k

            ZOOM_PAD_PX = 0
            sq_uv = _minimal_square_uv_from_bbox(
                use_box,
                side=FIXED_CROP_SIDE,
                pad_px=ZOOM_PAD_PX
            )

            corners_zoom = [
                _patch_xy_to_latlng(u, v, corners_base, ob)
                for (u, v) in sq_uv
            ]

            zoom_path = os.path.join(
                out_dir,
                f"{instr_id}_step{step_n:02d}_dest_zoom_hr.jpg"
            )
            patch_zoom = create_view_image(
                corners_zoom, ob,
                save_path=zoom_path,
                out_px=FIXED_CROP_SIDE
            )
            predicted_corners_seq.append(np.array(corners_zoom, dtype=float))
            _audit.event('evaluation_region', source='confirmation' if confirm_box is not None else 'search_fallback', source_bbox=use_box, square_uv=sq_uv, view_corners_latlng=corners_base, map_polygon_latlng=corners_zoom, actual_move=False)
            zoom_idx = len(predicted_corners_seq) - 1

            found = True
            break

        else:
            pos = _move_forward_geo(pos, heading, meters=step_meters)
            _audit.move(pos, 'search_no_match_forward')
            if spatial_state is not None:
                spatial_state.move(pos, heading, "navigation_move")
            if first_step_pos is None:
                first_step_pos = pos[:]

            if step_n >= max_steps and not found:
                fallback_pos = first_step_pos or _move_forward_geo(start_pos, heading, meters=step_meters)
                pos = fallback_pos[:]
                _audit.move(pos, 'return_to_first_step_no_match')
                if spatial_state is not None:
                    spatial_state.move(pos, heading, "navigation_move")
                corners_back = generate_view_corners_with_scale(
                    pos, ob,
                    scale_factor=fallback_scale,
                    angle_deg=heading
                )
                back_path = os.path.join(
                    out_dir,
                    f"{instr_id}_step{step_n:02d}_return_to_first_step_hr.jpg"
                )
                patch_back = create_view_image(
                    corners_back, ob,
                    save_path=back_path,
                    out_px=FIXED_CROP_SIDE
                )
                if patch_back is not None:
                    draw_pos_on_patch(patch_back, corners_back, pos)
                    cv2.imwrite(back_path, patch_back, [cv2.IMWRITE_JPEG_QUALITY, 95])
                predicted_corners_seq.append(np.array(corners_back, dtype=float))
                steps_log.append({"k": k, "action": "return_to_first_step", "pos": pos})
                break

    search_json_path = os.path.join(out_dir, f"{instr_id}_search.json")
    try:
        out = {
            "instr_id": instr_id,
            "found": bool(found),
            "dest_latlng": dest_latlng,
            "final_pos": pos,
            "heading_deg": heading,
            "step_meters": step_meters,
            "max_steps": max_steps,
            "steps": steps_log
        }
        with open(search_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write search JSON: {search_json_path} -> {e}")

    _audit.event('search_stage_end', found=found, returned_position=dest_latlng, local_position=pos, predicted_corners_seq=predicted_corners_seq, zoom_idx=zoom_idx)
    if spatial_state is not None:
        spatial_state.snapshot("stage_"+instr_id)
    return found, dest_latlng, predicted_corners_seq, zoom_idx


def rescue_failed_stage(*, instr_id, ob, dest_desc, via_desc, heading_deg, scale_factor,
                        out_dir, spatial_state, api_key, url, model, vision_tool_url):
    """G5: one observation-only G4 call after the original SAM3 stage failed.

    No evaluation labels or IoU enter the trigger. Reuse observed images; preserve
    the primary stage anchor. Only a confirmed candidate commits a new position.
    """
    from copy import deepcopy
    record = dict(stage=instr_id, policy='G5_baseline_first', search_budget=1,
                  candidate_budget=2, confirmation_review='off', committed=False)
    empty = (False, None, [], None)
    if spatial_state is None or not spatial_state.stage or spatial_state.stage['stage_id'] != instr_id:
        record['status'] = 'spatial_state_unavailable'
        return (*empty, record)
    old_mode = spatial_state.mode
    old_agent, old_path = deepcopy(spatial_state.agent), deepcopy(spatial_state.path)
    offset = max((v['round'] for v in spatial_state.views.values() if v['stage_id']==instr_id), default=0)
    start = spatial_state.to_latlng(old_agent['position_xy']).tolist()
    record.update(round_offset=offset, start_position=start)
    result = empty
    _audit.context(phase='g5_rescue')
    _audit.event('g5_rescue_started', **record)
    try:
        spatial_state.mode = 'g4'
        result = _search_candidates_and_reach_destination(
            instr_id, ob, dest_desc, via_desc, start, heading_deg, scale_factor,
            os.path.join(out_dir, 'g5_rescue'), step_meters=0, max_steps=1,
            api_key=api_key, url=url, model=model, vision_tool_url=vision_tool_url,
            candidate_count=2, confirmation_review='off', spatial_state=spatial_state,
            round_offset=offset)
        record['committed'] = bool(result[0] and result[1])
        record['status'] = 'confirmed' if record['committed'] else 'no_selection'
    except Exception as exc:
        record.update(status='operational_error_keep_baseline', error_type=type(exc).__name__)
    finally:
        spatial_state.mode = old_mode
        if not record['committed']:
            spatial_state.agent, spatial_state.path = old_agent, old_path
            result = empty
        try:
            spatial_state.snapshot('g5_'+instr_id)
        except Exception as exc:
            record['recording_error_type'] = type(exc).__name__
        _audit.event('g5_rescue_finished', **record)
        _audit.context(phase='sam3_primary')
    return (*result, record)


def _search_candidates_and_reach_destination(
    instr_id, ob, dest_desc, via_desc, start_pos_latlng, heading_deg, scale_factor,
    out_dir, step_meters=120.0, max_steps=3, api_key=DEFAULT_API_KEY,
    url=QWEN_URL, model=QWEN_MODEL, target_terms=None,
    vision_tool_url=VISION_TOOL_URL, candidate_count=2,
    confirmation_review='off',
    spatial_state=None,
    round_offset=0,
):
    """C1: propose on the current Search view, independently confirm previews.

    Only a valid selection commits a candidate center. Pending candidates are
    local to this round; archived evidence is never supplied as semantic memory.
    """
    from .tools.candidates import confirmation_crops, square_uv, verify_confirmation
    if candidate_count != 2:
        raise ValueError('candidate_count must be 0 (baseline) or 2 (C1)')
    os.makedirs(out_dir, exist_ok=True)
    pos = [float(v) for v in start_pos_latlng]
    heading = float(heading_deg or 0.0)
    predicted_corners_seq, patch_seq, steps_log = [], [], []
    last_grid_5x5 = None
    first_step_pos = None
    found, dest_latlng, zoom_idx = False, None, None
    review_used = False

    for k in range(max_steps):
        step_n = k + 1 + round_offset
        anchor = pos[:]
        prefix = '%s_step%02d' % (instr_id, step_n)
        _audit.context(trajectory_id=instr_id.rsplit('_step', 1)[0], stage_id=instr_id,
                       round=step_n, position=anchor, heading=heading, role='search', sam3=None)
        _audit.event('search_round_start', destination=dest_desc, via=via_desc,
                     scale=scale_factor, step_meters=step_meters, max_steps=max_steps, candidate_count=2)
        corners = generate_view_corners_with_scale(anchor, ob, scale_factor=scale_factor, angle_deg=heading)
        predicted_corners_seq.append(np.array(corners, dtype=float))
        search_path = os.path.join(out_dir, prefix + '_search_view_h%.1f.jpg' % heading)
        main = create_view_image(corners, ob, save_path=search_path, out_px=FIXED_CROP_SIDE)
        entry = dict(k=k, pos=anchor, candidates=[], previews=[])
        if round_offset:
            entry['round'] = step_n
        steps_log.append(entry)
        if main is not None:
            detail_corners = generate_view_corners_with_scale(anchor, ob, scale_factor=3, angle_deg=heading)
            context_corners = generate_view_corners_with_scale(anchor, ob, scale_factor=7, angle_deg=heading)
            detail = create_view_image(detail_corners, ob, save_path=None, out_px=FIXED_CROP_SIDE)
            context_image = create_view_image(context_corners, ob, save_path=None, out_px=FIXED_CROP_SIDE)
            draw_pos_on_patch(main, corners, anchor)
            cv2.imwrite(search_path, main, [cv2.IMWRITE_JPEG_QUALITY, 95])
            patch_seq.append(main)
            minimap = make_minimap_crop_basemap(ob, corners_seq=predicted_corners_seq,
                img_seq=patch_seq, out_path=os.path.join(out_dir, prefix+'_minimap.jpg'), pad_m=40.0)
            segmentation, sam3_meta = None, None
            try:
                segmentation, sam3_meta = _sam3_image5(main,
                    list(target_terms or extract_category_terms(dest_desc)), image_path=search_path,
                    artifact_dir=os.path.join(out_dir, 'sam3_image5', prefix), vision_tool_url=vision_tool_url)
            except Exception as exc:
                _audit.event('sam3_unavailable', error=str(exc), baseline_action='continue_without_image5')
                print('[WARN][SAM3] Image 5 unavailable for %s: %s' % (prefix, exc))
            _audit.context(trajectory_id=instr_id.rsplit('_step', 1)[0], stage_id=instr_id,
                round=step_n, position=anchor, heading=heading, role='search',
                views={'main': corners, 'detail': detail_corners, 'context': context_corners}, sam3=sam3_meta)
            search_context=None
            if spatial_state is not None:
                prepared=spatial_state.prepare_search([('main',main,corners),('detail',detail,detail_corners),
                    ('context',context_image,context_corners)],anchor,heading,step_n,scale_factor,ob,
                    sam3_image=segmentation,minimap=minimap)
                if spatial_state.mode!='g0':search_context=prepared
                _audit.event('spatial_search_input',mode=spatial_state.mode,
                    summary=prepared['spatial_summary'],main_ids=prepared['main_ids'],
                    historical_ids=prepared['historical_ids'],
                    snapshot=spatial_state.snapshot('input_%s_%d'%(instr_id,step_n)))
            q = qwen_locate_bbox_in_view(dest_desc, patch_main_bgr=main, patch_s4_bgr=detail,
                patch_s8_bgr=context_image, minimap_bgr=minimap, api_key=api_key, url=url, model=model,
                timeout=SEARCH_QWEN_TIMEOUT, restrict_top_half=True, prev_grid_5x5=last_grid_5x5,
                segmentation_image=segmentation, candidate_count=2,
                **({'search_context':search_context} if search_context is not None else {}))
            if sam3_meta is not None:
                q['sam3_image5'] = sam3_meta
            last_grid_5x5 = q.get('grid_5x5') or last_grid_5x5
            entry.update(qwen=q, w=main.shape[1], h=main.shape[0])
            if spatial_state is not None:
                recorded=dict(q)
                if spatial_state.mode=='g0':
                    recorded['candidates']=[dict(c,source_view_id=prepared['main_ids'][0]) for c in q.get('candidates',[])]
                spatial_state.record_search(prepared['main_ids'],recorded,step_n)
                entry['spatial_snapshot']=spatial_state.snapshot('search_%s_%d'%(instr_id,step_n))
            views = []
            for index, proposal in enumerate(q.get('candidates', [])[:2], 1):
                source_corners=corners
                source_path=search_path
                if search_context is not None:
                    source=next(v for v in search_context['views'] if v['view_id']==proposal['source_view_id'] and v['localization_allowed'])
                    source_corners=source['corners'];source_path=source['path']
                candidate = dict(proposal, candidate_id='r%d_c%d' % (step_n, index), status='pending',
                                 source_corners=np.asarray(source_corners).tolist())
                box = candidate['bbox_2d']
                candidate['map_center'] = _patch_xy_to_latlng((box[0]+box[2])/2, (box[1]+box[3])/2, source_corners, ob)
                entry['candidates'].append(candidate)
                _audit.bbox('search_bbox', box, source_corners, ob, candidate_id=candidate['candidate_id'],
                            source_view_id=proposal['source_view_id'], dest_present=True, view_path=source_path)
                try:
                    crops = confirmation_crops(candidate, ob, anchor, heading, scale_factor, out_dir, prefix)
                except Exception as exc:
                    crops = []
                    candidate['crop_error'] = str(exc)
                if not crops:
                    candidate['status'] = 'unavailable'
                views.extend(crops)
            for view in views:
                metadata = {key: value for key, value in view.items() if key != 'image'}
                metadata['corners'] = np.asarray(metadata['corners']).tolist()
                entry['previews'].append(metadata)
                _audit.event('candidate_preview', **metadata, actual_move=False)
            _audit.event('candidate_proposals', candidates=entry['candidates'])

            if views:
                _audit.context(trajectory_id=instr_id.rsplit('_step', 1)[0], stage_id=instr_id,
                    round=step_n, heading=heading, role='confirmation', position=anchor, views={}, sam3=None)
                # Pass only images/IDs/roles. No Search boxes, reasons, confidence or grid.
                confirmed = qwen_locate_bbox_in_view(dest_desc,
                    patch_main_bgr=views[0]['image'], api_key=api_key, url=url, model=model,
                    restrict_top_half=False, candidate_count=2, confirmation_views=views)
                entry['confirm'] = confirmed
                if spatial_state is not None:
                    spatial_state.events[-1]['confirmation']=dict(dest_present=confirmed.get('dest_present'),
                        selected_candidate_id=confirmed.get('selected_candidate_id'))
                if confirmed.get('dest_present') and confirmation_review != 'off':
                    selected = next(c for c in entry['candidates'] if c['candidate_id'] == confirmed['selected_candidate_id'])
                    selected_view = next(v for v in views if v['view_id'] == confirmed['view_id'])
                    _audit.bbox('initial_confirmation_bbox', confirmed['bbox_2d'], selected_view['corners'], ob,
                        candidate_id=selected['candidate_id'], view_id=selected_view['view_id'],
                        dest_present=True, view_path=selected_view['path'])
                    confirmed, verification = verify_confirmation(dest_desc, selected, confirmed, views, ob,
                        out_dir, prefix, api_key, url, model, vision_tool_url, confirmation_review, not review_used)
                    review_used = review_used or verification['attempted']
                    entry['verification'] = verification
                    entry['applied_confirmation'] = confirmed
                if confirmed.get('dest_present'):
                    selected = next(c for c in entry['candidates'] if c['candidate_id'] == confirmed['selected_candidate_id'])
                    selected_view = next(v for v in views if v['view_id'] == confirmed['view_id'])
                    selected['status'] = 'confirmed'
                    entry['selected_candidate_id'] = selected['candidate_id']
                    _audit.bbox('confirmation_bbox', confirmed['bbox_2d'], selected_view['corners'], ob,
                        candidate_id=selected['candidate_id'], view_id=selected_view['view_id'],
                        dest_present=True, applied_bbox=confirmed['bbox_2d'], view_path=selected_view['path'])
                    _audit.event('confirmation_resolution', source='confirmation', selected_candidate_id=selected['candidate_id'])

                    # Preserve the baseline lower-half/far override for the SELECTED proposal.
                    box = selected['bbox_2d']
                    behind=(box[1]+box[3])/2 >= FIXED_CROP_SIDE/2
                    if search_context is not None:
                        behind=spatial_state.relative(spatial_state.to_xy(selected['map_center']))['agent_forward_m']<=0
                    if behind:
                        image_center = _patch_xy_to_latlng(FIXED_CROP_SIDE/2, FIXED_CROP_SIDE/2, corners, ob)
                        distance = _haversine_m(*image_center, *selected['map_center'])
                        if distance > 120.0:
                            start_box = np.array(ob['gt_path_corners'][0], dtype=float)
                            predicted_corners_seq.append(start_box)
                            pos = _corners_center(start_box)
                            entry.update(action='override_to_start_box_on_first_detect_lower_half_far', dist_m=distance)
                            _audit.move(pos, 'lower_half_far_override_local_position')
                            if spatial_state is not None:
                                spatial_state.move(pos, heading, "navigation_move")
                            _audit.event('evaluation_region', source='start_box_override', map_polygon_latlng=start_box, actual_move=False)
                            found = True
                            break

                    dest_latlng = selected['map_center'][:]
                    pos = dest_latlng[:]
                    _audit.move(pos, 'approach_selected_search_center')
                    if spatial_state is not None:
                        spatial_state.move(pos, heading, "navigation_move")
                    predicted_corners_seq.append(np.array(selected_view['corners'], dtype=float))
                    square = square_uv(confirmed['bbox_2d'])
                    zoom_corners = [_patch_xy_to_latlng(u, v, selected_view['corners'], ob) for u, v in square]
                    det = selected_view['image'].copy()
                    _draw_bbox_on_patch(det, confirmed['bbox_2d'], color=(0,255,0), thickness=3,
                                       label='C:%.2f' % confirmed.get('confidence', 0.0))
                    cv2.imwrite(os.path.join(out_dir, prefix+'_dest_confirm_det.jpg'), det, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    create_view_image(zoom_corners, ob, save_path=os.path.join(out_dir, prefix+'_dest_zoom_hr.jpg'), out_px=FIXED_CROP_SIDE)
                    predicted_corners_seq.append(np.array(zoom_corners, dtype=float))
                    zoom_idx = len(predicted_corners_seq) - 1
                    _audit.event('evaluation_region', source='confirmation', source_bbox=confirmed['bbox_2d'],
                        square_uv=square, view_corners_latlng=selected_view['corners'], map_polygon_latlng=zoom_corners,
                        candidate_id=selected['candidate_id'], view_id=selected_view['view_id'], actual_move=False)
                    found = True
                    break
                for candidate in entry['candidates']:
                    if candidate['status'] == 'pending':
                        candidate['status'] = 'uncertain'
                _audit.event('confirmation_resolution', source='no_selection', reason=confirmed.get('reason'),
                             validation_errors=confirmed.get('validation_errors'))
        else:
            entry['action'] = 'render_fail'
            _audit.event('render_failure', path=search_path)

        # Failed previews/confirmation do not move the anchor to any candidate.
        pos = _move_forward_geo(anchor, heading, meters=step_meters)
        entry['next_position'] = pos[:]
        _audit.move(pos, 'candidate_round_unconfirmed_forward')
        if spatial_state is not None:
            spatial_state.move(pos, heading, "navigation_move")
        if first_step_pos is None:
            first_step_pos = pos[:]

    if not found and max_steps > 0:
        pos = first_step_pos[:]
        back = generate_view_corners_with_scale(pos, ob, scale_factor=2.0, angle_deg=heading)
        back_path = os.path.join(out_dir, prefix+'_return_to_first_step_hr.jpg')
        patch = create_view_image(back, ob, save_path=back_path, out_px=FIXED_CROP_SIDE)
        if patch is not None:
            draw_pos_on_patch(patch, back, pos)
            cv2.imwrite(back_path, patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        predicted_corners_seq.append(np.array(back, dtype=float))
        _audit.move(pos, 'return_to_first_step_no_match')
        if spatial_state is not None:
            spatial_state.move(pos, heading, "navigation_move")
        steps_log.append(dict(action='return_to_first_step', pos=pos))
    with open(os.path.join(out_dir, instr_id+'_search.json'), 'w', encoding='utf-8') as handle:
        json.dump(dict(instr_id=instr_id, policy='STEP2_C1', candidate_count=2, found=found,
            dest_latlng=dest_latlng, final_pos=pos, heading_deg=heading, step_meters=step_meters,
            max_steps=max_steps, steps=steps_log), handle, ensure_ascii=False, indent=2)
    _audit.event('search_stage_end', found=found, returned_position=dest_latlng, local_position=pos,
                 predicted_corners_seq=predicted_corners_seq, zoom_idx=zoom_idx)
    if spatial_state is not None:
        spatial_state.snapshot("stage_"+instr_id)
    return found, dest_latlng, predicted_corners_seq, zoom_idx

