"""Standalone Search/Confirmation runner; accepts existing Parse CSV outputs."""
import os
import argparse
import sys
import csv
import json
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import ANDHNavBatch
from torch.utils.data import DataLoader
from andh_core.config import (
    DEFAULT_API_KEY, QWEN_URL, QWEN_MODEL, FIXED_CROP_SIDE,
    NavigationOptions,
)
from andh_core.tools.views import create_view_image
from andh_core.navigation import search_and_reach_destination
from util import (
    generate_view_corners_with_scale,
    decide_turn_from_land,
    draw_pos_on_patch,
    _corners_center,
    angle_to_clock,
    clock_to_angle_deg,
    save_traj_boxes_debug_image,
    compute_iou,
    _move_forward_geo,
)

RESULTS_CSV = str(PROJECT_ROOT / "out" / "parse" / "preds_out_Parse_test_unseen_full" / "parsing_results_full.csv")

ANNO_DIR = str(PROJECT_ROOT / "datasets" / "FULL")
DATASET_DIR = str(PROJECT_ROOT / "datasets" / "AVDN")
SPLIT       = "test_unseen_full"
OUT_DIR     = str(PROJECT_ROOT / "out" / "andh_core" / "current")
SCALE_FACTOR       = 5


def _to_bool_strict(x):
    if isinstance(x, bool):
        return x
    s = str(x or "").strip().lower()
    return s in ("true", "t", "1", "yes", "y")

def run_turn_and_crop_full_trajectory(
    anno_dir=ANNO_DIR, dataset_dir=DATASET_DIR, split=SPLIT,
    results_csv=RESULTS_CSV, out_dir=OUT_DIR,
    scale_factor=SCALE_FACTOR,
    api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL,
    navigation_options=None,
):
    navigation_options = navigation_options or NavigationOptions.from_env()

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

        if instr_id not in groups:
            skipped += 1
            continue

        start_corners = np.array(ob['gt_path_corners'][0], dtype=float)
        pos           = np.mean(start_corners, axis=0).tolist()
        heading       = float(ob.get("starting_angle", 0.0) or 0.0)

        predicted_boxes = []
        rows = sorted(groups[instr_id], key=lambda r: r["_step_idx_int"])
        trajectory_context = {}

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
                clock_str, reason, meta = decide_turn_from_land(instruction, use_img, view="absolute", heading_deg=heading)
                if isinstance(meta, dict) and meta.get("rel_deg") is not None:
                    clock_str = angle_to_clock(meta["rel_deg"])
            else:
                clock_str = move_dir
                reason, meta = "", {}

            delta_deg   = clock_to_angle_deg(clock_str)
            new_heading = (heading + delta_deg) % 360.0

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
                    step_meters=navigation_options.step_meters,
                    max_steps=navigation_options.max_steps,
                    api_key=api_key, url=url, model=model,
                    options=navigation_options,
                    trajectory_context=trajectory_context,
                )
            except Exception as e:
                print(f"[WARN] search error: {instr_id} step{step_n:02d} -> {e}")
                found, dest_latlng, predicted_corners_seq, zoom_idx_local = False, None, [], None

            for c in (predicted_corners_seq or []):
                predicted_boxes.append(np.array(c, dtype=float))
            if zoom_idx_local is not None:
                zoom_highlight_idx_global = len(predicted_boxes) - 1

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
                print(f"[OK][STEP {step_n:02d}] {instr_id}: destination found, center approx {dest_latlng}")
            else:
                if sample_fallback_corners_s2 is not None:
                    predicted_boxes.append(np.array(sample_fallback_corners_s2, dtype=float))

                print(f"[INFO][STEP {step_n:02d}] {instr_id}: destination not found; terminating sample.")
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

        preds[instr_id] = traj_entry
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

def main(argv=None):
    parser = argparse.ArgumentParser(description="PSC-AVDN andh_core Search/Confirmation runner")
    parser.add_argument("--qwen-url", default=QWEN_URL)
    parser.add_argument("--qwen-model", default=QWEN_MODEL)
    parser.add_argument("--anno-dir", default=ANNO_DIR)
    parser.add_argument("--dataset-dir", default=DATASET_DIR)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--results-csv", default=RESULTS_CSV)
    parser.add_argument("--out-dir", default=OUT_DIR)
    defaults = NavigationOptions.from_env()
    parser.add_argument("--history-size", type=int, choices=(0, 1, 2), default=defaults.history_size)
    parser.add_argument("--relative-pose", action=argparse.BooleanOptionalAction,
                        default=defaults.use_relative_pose)
    parser.add_argument("--trace", action=argparse.BooleanOptionalAction,
                        default=defaults.enable_trace)
    parser.add_argument("--candidate-count", type=int, default=defaults.candidate_count)
    parser.add_argument("--preview-offset-ratio", type=float, default=defaults.preview_offset_ratio)
    parser.add_argument("--strict-confirmation", action=argparse.BooleanOptionalAction, default=defaults.strict_confirmation)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--step-meters", type=float, default=defaults.step_meters)
    parser.add_argument("--model-retries", type=int, default=defaults.model_retries)
    args = parser.parse_args(argv)
    options = NavigationOptions(history_size=args.history_size, use_relative_pose=args.relative_pose,
                                enable_trace=args.trace,
                                strict_confirmation=args.strict_confirmation,
                                max_steps=args.max_steps, step_meters=args.step_meters,
                                model_retries=args.model_retries,
                                candidate_count=args.candidate_count, preview_offset_ratio=args.preview_offset_ratio)
    os.makedirs(args.out_dir, exist_ok=True)
    run_turn_and_crop_full_trajectory(
        anno_dir=args.anno_dir,
        dataset_dir=args.dataset_dir,
        split=args.split,
        results_csv=args.results_csv,
        out_dir=args.out_dir,
        scale_factor=SCALE_FACTOR,
        api_key=DEFAULT_API_KEY, url=args.qwen_url, model=args.qwen_model,
        navigation_options=options,
    )


if __name__ == "__main__":
    main()
