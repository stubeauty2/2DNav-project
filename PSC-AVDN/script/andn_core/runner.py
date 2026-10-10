"""SAM3 baseline entry point; data paths may be selected without changing policy."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from andn_core.config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT / "src"))
import os
import argparse
import csv
import json
import cv2
import numpy as np
from collections import defaultdict
from env import ANDHNavBatch
from torch.utils.data import DataLoader
from andn_core.config import RESULTS_CSV, ANNO_DIR, DATASET_DIR, SPLIT, OUT_DIR, SCALE_FACTOR, FIXED_CROP_SIDE, DEFAULT_API_KEY, QWEN_URL, QWEN_MODEL, VISION_TOOL_URL, SEARCH_QWEN_TIMEOUT
from andn_core.navigation import search_and_reach_destination
from andn_core.tools.category_terms import extract_category_terms
from andn_core.tools.views import create_view_image
from util import (generate_view_corners_with_scale, decide_turn_from_land, draw_pos_on_patch, _corners_center, angle_to_clock, clock_to_angle_deg, save_traj_boxes_debug_image, compute_iou, _move_forward_geo)
from andn_core.tools import telemetry as _audit
from andn_core.tools.termination import stop_only_instruction, stop_step

decide_turn_from_land = _audit.external_model(decide_turn_from_land, 'land')

def hook_after_turn_and_crop(row, ob, instr_id, pos, new_heading, scale_factor, out_dir,
                             api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL,
                             vision_tool_url=VISION_TOOL_URL):
    dest_desc = (row.get("dest") or "").strip()
    via_desc  = (row.get("via") or "").strip()
    if via_desc.lower() in ("none","null","na","n/a",""):
        via_desc = None
    print(f"[DEST_DESC] {instr_id}: {dest_desc}")

    found, dest_latlng, predicted_corners_seq, zoom_idx = search_and_reach_destination(
        instr_id=instr_id,
        ob=ob,
        dest_desc=dest_desc,
        via_desc=via_desc,
        start_pos_latlng=pos,
        heading_deg=new_heading,
        scale_factor=scale_factor,
        out_dir=out_dir,
        step_meters=120,
        max_steps=3,
        api_key=api_key,
        url=url,
        model=model,
        vision_tool_url=vision_tool_url,
        target_terms=extract_category_terms(dest_desc),
    )

    if found:
        print(f"[OK][DEST] {instr_id}: destination found, center approx {dest_latlng}")
    else:
        print(f"[INFO][DEST] {instr_id}: destination not found within max steps; search log written.")

    return predicted_corners_seq, zoom_idx

def run_turn_and_crop(anno_dir=ANNO_DIR, dataset_dir=DATASET_DIR, split=SPLIT,
                      results_csv=RESULTS_CSV, out_dir=OUT_DIR,
                      scale_factor=SCALE_FACTOR,
                      vision_tool_url=VISION_TOOL_URL,
                      ):
    try:
        pass
    except Exception as e:
        raise RuntimeError("Cannot import DataLoader or ANDHNavBatch; check dependencies and PYTHONPATH.") from e

    os.makedirs(out_dir, exist_ok=True)

    tif_dataset_dir = os.path.join(dataset_dir, 'train_images')
    env = ANDHNavBatch(
        anno_dir=anno_dir,
        dataset_dir=tif_dataset_dir,
        splits=[split],
        tokenizer=None,
        max_instr_len=512,
        batch_size=1,
        seed=0,
        full_traj=False
    )
    loader = DataLoader(env, batch_size=1)
    preds = {}
    _metrics_acc = {"final_iou": [], "path_length": []}

    id2row = {}
    with open(results_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            instr_id = (r.get("instr_id") or "").strip()
            if instr_id:
                id2row[instr_id] = r

    processed = 0
    skipped = 0

    for _batch_idx, _ in enumerate(loader):
        obs_list = env._get_obs(t=0)
        if not obs_list:
            continue
        ob = obs_list[0]
        ob["dataset_dir"] = dataset_dir

        map_name  = ob.get("map_name", "")
        route_idx = ob.get("route_index", "")
        instr_id  = f"{map_name}__{route_idx}"

        if instr_id not in id2row:
            skipped += 1
            continue

        row = id2row[instr_id]
        instruction = row.get("instruction", "")
        move_dir = (row.get("move_dir") or "").strip()

        start_corners = np.array(ob['gt_path_corners'][0])
        pos     = np.mean(start_corners, axis=0)
        heading = float(ob.get("starting_angle", 0.0) or 0.0)

        corners0 = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=heading)
        step0_hr = os.path.join(out_dir, f"{instr_id}_h{heading:.1f}_step00_start_hr.jpg")
        patch0   = create_view_image(corners0, ob, save_path=step0_hr, out_px=FIXED_CROP_SIDE)
        if patch0 is None:
            print(f"[WARN] start patch failed, skipping: {instr_id}")
            skipped += 1
            continue
        draw_pos_on_patch(patch0, corners0, pos)
        cv2.imwrite(step0_hr, patch0, [cv2.IMWRITE_JPEG_QUALITY, 95])

        corners_abs  = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=None)
        step0_abs_hr = os.path.join(out_dir, f"{instr_id}_step00_abs_north_hr.jpg")
        patch_abs    = create_view_image(corners_abs, ob, save_path=step0_abs_hr, out_px=FIXED_CROP_SIDE)

        if move_dir.lower() == "land":
            use_img = patch_abs if patch_abs is not None else patch0
            clock_str, reason, meta = decide_turn_from_land(instruction, use_img, view="absolute", heading_deg=heading)
            if isinstance(meta, dict) and meta.get("rel_deg") is not None:
                clock_str = angle_to_clock(meta["rel_deg"])
        else:
            clock_str = move_dir
            reason = ""
            meta = {}

        delta_deg  = clock_to_angle_deg(clock_str)
        new_heading = (heading + delta_deg) % 360.0

        corners1 = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=new_heading)
        step1_hr = os.path.join(out_dir, f"{instr_id}_turn_{clock_str.replace(':','-')}_h{new_heading:.1f}_step01_after_turn_hr.jpg")
        patch1   = create_view_image(corners1, ob, save_path=step1_hr, out_px=FIXED_CROP_SIDE)
        if patch1 is not None:
            draw_pos_on_patch(patch1, corners1, pos)
            cv2.imwrite(step1_hr, patch1, [cv2.IMWRITE_JPEG_QUALITY, 95])

        dest_desc = (row.get("dest") or "").strip()
        print(f"[DEST_DESC] {instr_id}: {dest_desc}")
        forward_raw = str(row.get("forward", "")).strip().lower()
        forward_bool = forward_raw in ("true", "1", "yes", "y", "t")

        predicted_boxes = []
        if isinstance(corners0, (np.ndarray, list)):
            predicted_boxes.append(np.array(corners0, dtype=float))
        if isinstance(corners1, (np.ndarray, list)):
            predicted_boxes.append(np.array(corners1, dtype=float))

        zoom_highlight_idx = None

        if dest_desc.lower() == "destination":

            start_box = np.array(ob['gt_path_corners'][0], dtype=float)
            start_ctr = np.array(_corners_center(start_box), dtype=float)

            if forward_bool:
                end_ctr = np.array(_move_forward_geo(start_ctr.tolist(), new_heading, meters=60.0), dtype=float)
                delta = end_ctr - start_ctr
                final_box = (start_box + delta).astype(float)
            else:
                final_box = start_box.astype(float)

            predicted_boxes.append(final_box)

            out_traj_img = os.path.join(out_dir, f"{instr_id}_traj_boxes.jpg")
            save_traj_boxes_debug_image(
                tif_dataset_dir=tif_dataset_dir,
                map_name=map_name,
                ob=ob,
                predicted_boxes=predicted_boxes,
                out_path=out_traj_img,
                zoom_highlight_idx=None
            )

        else:
            try:
                predicted_seq_from_search, zoom_idx_local = hook_after_turn_and_crop(
                    row=row,
                    ob=ob,
                    instr_id=instr_id,
                    pos=pos,
                    new_heading=new_heading,
                    scale_factor=scale_factor,
                    out_dir=out_dir,
                    api_key=DEFAULT_API_KEY,
                    url=QWEN_URL,
                    model=QWEN_MODEL,
                    vision_tool_url=vision_tool_url,
                )
            except Exception as e:
                print(f"[WARN] search error: {instr_id} -> {e}")
                predicted_seq_from_search, zoom_idx_local = [], None

            for c in (predicted_seq_from_search or []):
                predicted_boxes.append(np.array(c, dtype=float))
            if zoom_idx_local is not None:
                zoom_highlight_idx = 2 + zoom_idx_local

            out_traj_img = os.path.join(out_dir, f"{instr_id}_traj_boxes.jpg")
            save_traj_boxes_debug_image(
                tif_dataset_dir=tif_dataset_dir,
                map_name=map_name,
                ob=ob,
                predicted_boxes=predicted_boxes,
                out_path=out_traj_img,
                zoom_highlight_idx=zoom_highlight_idx
            )

        goal_corners = np.array(ob['gt_path_corners'][-1], dtype=float)
        progress     = [float(compute_iou(pb, goal_corners)) for pb in predicted_boxes] if predicted_boxes else []
        trajectory   = [_corners_center(pb) for pb in predicted_boxes]
        path_corners = [(pb.tolist(), float(ob.get("starting_angle", 0.0) or 0.0)) for pb in predicted_boxes]

        final_iou   = float(progress[-1]) if progress else 0.0
        path_length = len(predicted_boxes)
        success     = final_iou > 0.3

        traj_entry = {
            "instr_id": instr_id,
            "trajectory": trajectory,
            "path_corners": path_corners,
            "progress": progress,
            "gt_progress": progress[:],
            "gt_path_corners": ob["gt_path_corners"],
            "reasoning": [],
            "final_iou": final_iou,
            "path_length": path_length,
            "success": success,
        }

        preds[instr_id] = traj_entry
        _metrics_acc["final_iou"].append(final_iou)
        _metrics_acc["path_length"].append(path_length)

        rel_info = ""
        abs_info = ""
        if move_dir.lower() == "land" and isinstance(meta, dict):
            if meta.get("rel_deg") is not None:
                rel_info = f" | rel≈{float(meta['rel_deg']):.1f}°"
                q_deg = clock_to_angle_deg(clock_str)
                diff = abs(((q_deg - float(meta['rel_deg']) + 540) % 360) - 180)
                if diff > 7.6:
                    print(f"[WARN] quantization mismatch: rel={float(meta['rel_deg']):.1f}°, clock={clock_str}({q_deg:.1f}°)")
            if meta.get("abs_deg") is not None:
                abs_info = f" | abs={float(meta['abs_deg']):.1f}°"

        if patch1 is None:
            print(f"[WARN] turn patch render failed: {instr_id} ({clock_str}) | reason: {reason[:120]}{rel_info}{abs_info}")
        else:
            print(f"[OK] {instr_id}: heading {heading:.1f} -> {new_heading:.1f} "
                  f"by {clock_str} ({delta_deg:.1f}°){rel_info}{abs_info} | reason: {reason[:120]}")

        analysis_path = os.path.join(out_dir, f"{instr_id}_analysis.json")
        try:
            out_json = {
                "instr_id": instr_id,
                "map_name": map_name,
                "route_idx": route_idx,
                "instruction": instruction,
                "heading_deg": heading,
                "clock": clock_str,
                "delta_deg": delta_deg,
                "new_heading_deg": new_heading,
                "used_view": "absolute" if move_dir.lower() == "land" else "heading",
                "reason": reason
            }
            if meta:
                out_json["meta"] = meta
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(out_json, f, ensure_ascii=False, indent=2)
        except Exception as _e:
            print(f"[WARN] failed to write analysis JSON: {analysis_path} -> {_e}")

        processed += 1

    print(f"\nDone: generated {processed} entries; skipped {skipped} entries. Output dir: {out_dir}")

    if _metrics_acc["final_iou"]:
        success_rate    = sum(1 for v in _metrics_acc["final_iou"] if v > 0.3) / len(_metrics_acc["final_iou"])
        avg_iou         = float(np.mean(_metrics_acc["final_iou"]))
        avg_path_length = float(np.mean(_metrics_acc["path_length"]))
    else:
        success_rate = 0.0
        avg_iou = 0.0
        avg_path_length = 0.0

    metrics = {
        "success_rate": success_rate,
        "avg_iou": avg_iou,
        "avg_path_length": avg_path_length,
    }

    result_file = os.path.join(out_dir, "turn_and_crop_eval_results.json")
    def _to_jsonable(x):
        if hasattr(x, "tolist"): return x.tolist()
        if isinstance(x, dict):  return {k: _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):  return [_to_jsonable(v) for v in x]
        return x

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable({"predictions": preds, "metrics": metrics}), f, ensure_ascii=False, indent=2)

    try:
        score_summary, detailed_metrics = env.eval_metrics(preds, human_att_eval=False)
        print("\n" + "="*50)
        print("Natural Path Eval summary:", score_summary)
        txt_file = os.path.join(out_dir, "turn_and_crop_score_summary.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            if isinstance(score_summary, dict):
                for k, v in score_summary.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(str(score_summary))
        print(f"Score summary saved to: {txt_file}")
    except Exception as e:
        detailed_metrics = {}
        print(f"[WARN] env.eval_metrics failed: {e}")

    print(f"Results saved to: {result_file}")

    return preds, detailed_metrics

def _to_bool_strict(x):
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    return s in ("true", "t", "1", "yes", "y")

@_audit.navigation_run
def run_turn_and_crop_full_trajectory(
    anno_dir=ANNO_DIR, dataset_dir=DATASET_DIR, split=SPLIT,
    results_csv=RESULTS_CSV, out_dir=OUT_DIR,
    scale_factor=SCALE_FACTOR,
    api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL,
    vision_tool_url=VISION_TOOL_URL,
    candidate_count=0,
    confirmation_review='off',
    spatial_mode='off',
):
    if spatial_mode != 'off':
        from .tools.spatial_map import SpatialMap, MODES
        if spatial_mode == 'g5':
            if candidate_count != 0 or confirmation_review != 'off':
                raise ValueError('G5 preserves the original SAM3 primary: candidate_count=0, review=off')
        elif spatial_mode not in MODES or (spatial_mode != 'g0' and candidate_count != 2):
            raise ValueError('Spatial mode must be G0--G4; G1--G4 require candidate_count=2')
    try:
        pass
    except Exception as e:
        raise RuntimeError("Cannot import DataLoader or ANDHNavBatch; check dependencies and PYTHONPATH.") from e

    os.makedirs(out_dir, exist_ok=True)

    groups = defaultdict(list)
    with open(results_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        has_step = ("step_idx" in reader.fieldnames)
        if not has_step:
            raise RuntimeError("CSV does not contain step_idx field; please confirm this is a full-trajectory CSV.")
        for r in reader:
            instr_id = (r.get("instr_id") or "").strip()
            if not instr_id:
                continue
            try:
                step_idx = int(float(r.get("step_idx", 0)))
            except Exception:
                step_idx = 0
            r["_step_idx_int"] = step_idx
            groups[instr_id].append(r)

    tif_dataset_dir = os.path.join(dataset_dir, 'train_images')
    env = ANDHNavBatch(
        anno_dir=anno_dir,
        dataset_dir=tif_dataset_dir,
        splits=[split],
        tokenizer=None,
        max_instr_len=512,
        batch_size=1,
        seed=0,
        full_traj=False
    )
    loader = DataLoader(env, batch_size=1)

    preds = {}
    _metrics_acc = {"final_iou": [], "path_length": []}

    processed, skipped = 0, 0

    for _batch_idx, _ in enumerate(loader):
        obs_list = env._get_obs(t=0)
        if not obs_list:
            continue
        ob = obs_list[0]
        ob["dataset_dir"] = dataset_dir

        map_name  = ob.get("map_name", "")
        route_idx = ob.get("route_index", "")
        instr_id  = f"{map_name}__{route_idx}"

        if stop_only_instruction(ob.get('instructions', '')):
            groups[instr_id] = [dict(stop_step(ob['instructions']), _step_idx_int=1)]
        if instr_id not in groups:
            skipped += 1
            continue

        start_corners = np.array(ob['gt_path_corners'][0], dtype=float)
        pos           = np.mean(start_corners, axis=0).tolist()
        heading       = float(ob.get("starting_angle", 0.0) or 0.0)

        predicted_boxes = []
        rows = sorted(groups[instr_id], key=lambda r: r["_step_idx_int"])
        _audit.context(trajectory_id=instr_id, position=pos, heading=heading)
        _audit.event("trajectory_start", map_name=map_name, route_index=route_idx, parsed_stages=rows)
        if spatial_mode != 'off':
            spatial_state=SpatialMap(pos,ob['gps_botm_left'],ob['gps_top_right'],ob['map_size'],
                os.path.join(out_dir,'spatial',instr_id),mode='g0' if spatial_mode=='g5' else spatial_mode)
        if spatial_mode == 'g5':
            g5_baseline_boxes, g5_attempts = None, []
            _audit.context(phase='sam3_primary')

        end_sample = False
        zoom_highlight_idx_global = None

        sample_fallback_corners_s2 = None
        sample_step1_first_pos = None
        sample_step1_heading = None

        for r in rows:
            if end_sample:
                break

            step_n = r["_step_idx_int"]
            instruction = r.get("instruction", "")
            move_dir    = (r.get("move_dir") or "").strip()
            dest_desc   = (r.get("dest") or "").strip()
            via_desc    = (r.get("via") or "").strip() or None

            fwd_raw = str(r.get("forward", "")).strip()
            if not fwd_raw:
                fwd_raw = str(r.get("forward_bool", "")).strip()
            forward_bool = _to_bool_strict(fwd_raw)
            _audit.context(trajectory_id=instr_id, stage_id=f"{instr_id}_step{step_n:02d}", position=pos, heading=heading)
            _audit.event("stage_start", parsed_instruction=r)

            if move_dir.lower() == 'stop' and stop_only_instruction(ob.get('instructions', '')):
                predicted_boxes.append(start_corners.copy())
                _audit.event('explicit_stop', position=pos, source='complete_stop_only_instruction',
                             map_polygon_latlng=start_corners)
                end_sample = True
                break

            corners0 = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=heading)
            step0_hr = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_start_h{heading:.1f}.jpg")
            patch0   = create_view_image(corners0, ob, save_path=step0_hr, out_px=FIXED_CROP_SIDE)
            if patch0 is not None:
                draw_pos_on_patch(patch0, corners0, pos)
                cv2.imwrite(step0_hr, patch0, [cv2.IMWRITE_JPEG_QUALITY, 95])

            corners_abs  = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=None)
            step0_abs_hr = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_abs_north.jpg")
            patch_abs    = create_view_image(corners_abs, ob, save_path=step0_abs_hr, out_px=FIXED_CROP_SIDE)

            if move_dir.lower() == "land":
                use_img = patch_abs if patch_abs is not None else patch0
                _audit.context(trajectory_id=instr_id, stage_id=f"{instr_id}_step{step_n:02d}", position=pos, heading=heading, role="land", views={"main": corners_abs if patch_abs is not None else corners0})
                clock_str, reason, meta = decide_turn_from_land(instruction, use_img, view="absolute", heading_deg=heading)
                _audit.event("land_resolution", clock=clock_str, reason=reason, metadata=meta)
                if isinstance(meta, dict) and meta.get("rel_deg") is not None:
                    clock_str = angle_to_clock(meta["rel_deg"])
            else:
                clock_str = move_dir
                reason, meta = "", {}

            delta_deg   = clock_to_angle_deg(clock_str)
            new_heading = (heading + delta_deg) % 360.0
            _audit.event("turn", from_heading=heading, to_heading=new_heading, clock=clock_str, delta_deg=delta_deg)

            if sample_step1_first_pos is None:
                sample_step1_heading = new_heading
                sample_step1_first_pos = _move_forward_geo(pos, sample_step1_heading, meters=120.0)
                sample_fallback_corners_s2 = generate_view_corners_with_scale(
                    sample_step1_first_pos, ob, scale_factor=2.0, angle_deg=sample_step1_heading
                )

            corners1 = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=new_heading)
            step1_hr = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_turn_{clock_str.replace(':','-')}_h{new_heading:.1f}.jpg")
            patch1   = create_view_image(corners1, ob, save_path=step1_hr, out_px=FIXED_CROP_SIDE)
            if patch1 is not None:
                draw_pos_on_patch(patch1, corners1, pos)
                cv2.imwrite(step1_hr, patch1, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if isinstance(corners0, (np.ndarray, list)):
                predicted_boxes.append(np.array(corners0, dtype=float))
            if isinstance(corners1, (np.ndarray, list)):
                predicted_boxes.append(np.array(corners1, dtype=float))

            if dest_desc.lower() == "destination":
                start_box = np.array(ob['gt_path_corners'][0], dtype=float)
                start_ctr = np.array(_corners_center(start_box), dtype=float)

                if forward_bool:
                    end_ctr = np.array(_move_forward_geo(start_ctr.tolist(), new_heading, meters=60.0), dtype=float)
                    delta   = end_ctr - start_ctr
                    final_box = (start_box + delta).astype(float)
                else:
                    final_box = start_box.astype(float)

                predicted_boxes.append(final_box)
                _audit.event("evaluation_region", source="destination_special_rule", forward=forward_bool, map_polygon_latlng=final_box, actual_move=False)

                analysis_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_analysis.json")
                try:
                    out_json = {
                        "instr_id": instr_id,
                        "step_idx": step_n,
                        "map_name": map_name,
                        "route_idx": route_idx,
                        "instruction": instruction,
                        "heading_deg": heading,
                        "clock": clock_str,
                        "delta_deg": delta_deg,
                        "new_heading_deg": new_heading,
                        "used_view": "absolute" if move_dir.lower() == "land" else "heading",
                        "reason": reason,
                        "meta": meta
                    }
                    with open(analysis_path, "w", encoding="utf-8") as f:
                        json.dump(out_json, f, ensure_ascii=False, indent=2)
                except Exception as _e:
                    print(f"[WARN] failed to write analysis JSON: {analysis_path} -> {_e}")

                end_sample = True
                heading = new_heading
                break

            try:
                found, dest_latlng, predicted_corners_seq, zoom_idx_local = search_and_reach_destination(
                    instr_id=f"{instr_id}_step{step_n:02d}",
                    ob=ob,
                    dest_desc=dest_desc,
                    via_desc=via_desc,
                    start_pos_latlng=pos,
                    heading_deg=new_heading,
                    scale_factor=scale_factor,
                    out_dir=out_dir,
                    step_meters=120,
                    max_steps=3,
                    api_key=api_key, url=url, model=model,
                    vision_tool_url=vision_tool_url,
                    target_terms=extract_category_terms(dest_desc),
                    candidate_count=candidate_count,
                    confirmation_review=confirmation_review,
                    spatial_state=spatial_state if spatial_mode != 'off' else None,
                )
            except Exception as e:
                _audit.event("search_exception", error_type=type(e).__name__, error=str(e), baseline_action="terminate_with_existing_fallback")
                print(f"[WARN] search error: {instr_id} step{step_n:02d} -> {e}")
                found, dest_latlng, predicted_corners_seq, zoom_idx_local = False, None, [], None

            for c in (predicted_corners_seq or []):
                predicted_boxes.append(np.array(c, dtype=float))
            if zoom_idx_local is not None:
                zoom_highlight_idx_global = len(predicted_boxes) - 1

            if spatial_mode == 'g5' and not found:
                # Preserve the result the primary SAM3 run would have returned.
                # IoU is evaluated only after navigation, never used by this gate.
                if g5_baseline_boxes is None:
                    g5_baseline_boxes = [np.array(c, dtype=float).copy() for c in predicted_boxes]
                    if sample_fallback_corners_s2 is not None:
                        g5_baseline_boxes.append(np.array(sample_fallback_corners_s2, dtype=float).copy())
                from .navigation import rescue_failed_stage
                rescue = rescue_failed_stage(instr_id=f'{instr_id}_step{step_n:02d}', ob=ob,
                    dest_desc=dest_desc, via_desc=via_desc, heading_deg=new_heading,
                    scale_factor=scale_factor, out_dir=out_dir, spatial_state=spatial_state,
                    api_key=api_key, url=url, model=model, vision_tool_url=vision_tool_url)
                rescued, rescue_position, rescue_boxes, rescue_zoom, rescue_record = rescue
                g5_attempts.append(rescue_record)
                if rescued and rescue_position:
                    found, dest_latlng = True, rescue_position
                    predicted_boxes.extend(np.array(c,dtype=float) for c in rescue_boxes)
                    if rescue_zoom is not None:
                        zoom_highlight_idx_global = len(predicted_boxes)-1

            analysis_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_analysis.json")
            try:
                out_json = {
                    "instr_id": instr_id,
                    "step_idx": step_n,
                    "map_name": map_name,
                    "route_idx": route_idx,
                    "instruction": instruction,
                    "heading_deg": heading,
                    "clock": clock_str,
                    "delta_deg": delta_deg,
                    "new_heading_deg": new_heading,
                    "used_view": "absolute" if move_dir.lower() == "land" else "heading",
                    "reason": reason
                }
                if meta:
                    out_json["meta"] = meta
                with open(analysis_path, "w", encoding="utf-8") as f:
                    json.dump(out_json, f, ensure_ascii=False, indent=2)
            except Exception as _e:
                print(f"[WARN] failed to write analysis JSON: {analysis_path} -> {_e}")

            if found and dest_latlng:
                pos = dest_latlng[:]
                heading = new_heading
                _audit.event("stage_position_commit", position=pos, heading=heading, source="returned_search_center")
                print(f"[OK][STEP {step_n:02d}] {instr_id}: destination found, center approx {dest_latlng}")
            else:
                if sample_fallback_corners_s2 is not None:
                    predicted_boxes.append(np.array(sample_fallback_corners_s2, dtype=float))
                    _audit.event("evaluation_region", source="runner_failure_fallback", map_polygon_latlng=sample_fallback_corners_s2, actual_move=False)

                print(f"[INFO][STEP {step_n:02d}] {instr_id}: destination not found; terminating sample.")
                _audit.event("stage_terminated", found=found, returned_position=dest_latlng)
                end_sample = True
                heading = new_heading
                break

        out_traj_img = os.path.join(out_dir, f"{instr_id}_traj_boxes_full.jpg")
        save_traj_boxes_debug_image(
            tif_dataset_dir=tif_dataset_dir,
            map_name=map_name,
            ob=ob,
            predicted_boxes=predicted_boxes,
            out_path=out_traj_img,
            zoom_highlight_idx=zoom_highlight_idx_global
        )

        goal_corners = np.array(ob['gt_path_corners'][-1], dtype=float)
        progress     = [float(compute_iou(pb, goal_corners)) for pb in predicted_boxes] if predicted_boxes else []
        trajectory   = [_corners_center(pb) for pb in predicted_boxes]
        path_corners = [(pb.tolist(), float(ob.get("starting_angle", 0.0) or 0.0)) for pb in predicted_boxes]

        final_iou   = float(progress[-1]) if progress else 0.0
        path_length = len(predicted_boxes)
        success     = final_iou > 0.3

        traj_entry = {
            "instr_id": instr_id,
            "trajectory": trajectory,
            "path_corners": path_corners,
            "progress": progress,
            "gt_progress": progress[:],
            "gt_path_corners": ob["gt_path_corners"],
            "reasoning": [],
            "final_iou": final_iou,
            "path_length": path_length,
            "success": success,
        }

        if spatial_mode == 'g5':
            primary_boxes = predicted_boxes if g5_baseline_boxes is None else g5_baseline_boxes
            primary_iou = float(compute_iou(primary_boxes[-1], goal_corners)) if primary_boxes else 0.0
            traj_entry['g5'] = dict(baseline_final_iou=primary_iou, baseline_success=primary_iou > 0.3,
                baseline_path_corners=[np.asarray(c).tolist() for c in primary_boxes],
                rescue_attempts=g5_attempts, gate='original SAM3 found=false; no GT or IoU in gate')
        preds[instr_id] = traj_entry
        _audit.context(trajectory_id=instr_id)
        _audit.event("trajectory_result", result=traj_entry)
        _metrics_acc["final_iou"].append(final_iou)
        _metrics_acc["path_length"].append(path_length)

        processed += 1

        # Keep a recoverable, incrementally updated snapshot for long full runs.
        progress_file = os.path.join(out_dir, "turn_and_crop_eval_results_full_traj.in_progress.json")
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump({"predictions": preds, "processed": processed, "skipped": skipped}, f,
                      ensure_ascii=False, indent=2,
                      default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))

    print(f"\nFull-trajectory done: generated {processed} entries; skipped {skipped} entries. Output dir: {out_dir}")

    if _metrics_acc["final_iou"]:
        success_rate    = sum(1 for v in _metrics_acc["final_iou"] if v > 0.3) / len(_metrics_acc["final_iou"])
        avg_iou         = float(np.mean(_metrics_acc["final_iou"]))
        avg_path_length = float(np.mean(_metrics_acc["path_length"]))
    else:
        success_rate = 0.0
        avg_iou = 0.0
        avg_path_length = 0.0

    metrics = {
        "success_rate": success_rate,
        "avg_iou": avg_iou,
        "avg_path_length": avg_path_length,
    }

    result_file = os.path.join(out_dir, "turn_and_crop_eval_results_full_traj.json")
    def _to_jsonable(x):
        if hasattr(x, "tolist"): return x.tolist()
        if isinstance(x, dict):  return {k: _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):  return [_to_jsonable(v) for v in x]
        return x

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(_to_jsonable({"predictions": preds, "metrics": metrics}), f, ensure_ascii=False, indent=2)

    try:
        score_summary, detailed_metrics = env.eval_metrics(preds, human_att_eval=False)
        print("\n" + "="*50)
        print("Natural Path Eval summary (full traj):", score_summary)
        txt_file = os.path.join(out_dir, "turn_and_crop_score_summary_full_traj.txt")
        with open(txt_file, "w", encoding="utf-8") as f:
            if isinstance(score_summary, dict):
                for k, v in score_summary.items():
                    f.write(f"{k}: {v}\n")
            else:
                f.write(str(score_summary))
        print(f"Score summary saved to: {txt_file}")
    except Exception as e:
        detailed_metrics = {}
        print(f"[WARN] env.eval_metrics failed: {e}")

    print(f"Results saved to: {result_file}")

    return preds, detailed_metrics

def main():
    parser = argparse.ArgumentParser(description="Original SAM3 baseline, modular layout")
    parser.add_argument("--anno-dir", default=ANNO_DIR)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--results-csv", default=RESULTS_CSV)
    parser.add_argument("--out-dir", default=OUT_DIR)
    parser.add_argument("--vision-tool-url", default=VISION_TOOL_URL)
    parser.add_argument("--qwen-url", default=QWEN_URL)
    parser.add_argument("--qwen-model", default=QWEN_MODEL)
    parser.add_argument("--candidate-count", type=int, choices=(0, 2), default=0,
                        help="0: original SAM3 baseline; 2: STEP 2 C1 current-round candidates")
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction, default=True,
                        help="STEP 0: archive requests, images and decisions without changing navigation")
    parser.add_argument('--confirmation-review', choices=('off','observe','once'), default='off',
                        help='STEP 3 C1: record geometry, or allow one evidence review per stage')
    parser.add_argument('--spatial-mode', choices=('off','g0','g1','g2','g3','g4','g5'), default='off',
                        help='G0 passive; G1 direction; G2 multiview; G3 map; G4 history; G5 SAM3 first, one map rescue after failure')
    args = parser.parse_args()
    run_turn_and_crop_full_trajectory(
        anno_dir=args.anno_dir, dataset_dir=args.dataset_dir, split=args.split,
        results_csv=args.results_csv, out_dir=args.out_dir, scale_factor=SCALE_FACTOR,
        api_key=DEFAULT_API_KEY, url=args.qwen_url, model=args.qwen_model,
        vision_tool_url=args.vision_tool_url,
        trace=args.trace,
        candidate_count=args.candidate_count,
        confirmation_review=args.confirmation_review,
        spatial_mode=args.spatial_mode,
    )

if __name__ == "__main__":
    main()
