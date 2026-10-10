"""Original dialogue Parse orchestration and CSV postprocessing."""
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from andn_core.config import PROJECT_ROOT
sys.path.append(str(PROJECT_ROOT / "src"))
import os

import sys

import re

import math

import csv

import json

import base64

from pathlib import Path

from typing import Any, Dict, Iterable, Optional, Tuple, List

import numpy as np

import cv2

import requests

from tqdm import tqdm

from torch.utils.data import DataLoader

import pandas as pd

from env import ANDHNavBatch
from andn_core.tools.parsing import (CFGPU_URL, CFGPU_MODEL, CFGPU_API_TOKEN, ANNO_DIR, DATASET_DIR, SPLIT, PRED_DIR, MAX_STEPS, SCALE_FACTOR, _DIALOGUE_SEQUENCE_SYSTEM_PROMPT, _CLOCK_OUTPUT_RE, _DEGREE_OUTPUT_RE, _BARE_NUMBER_OUTPUT_RE, analyze_dialogue_with_prompt, _extract_json_object, _normalize_move_dir, parse_dialogue_sequence, request_dialogue_sequence, append_jsonl_record, CLOCK_ALIAS, CLOCK_DEG_MAP, DEGREE_MARK, CLOCK_MARK, clock_to_angle_deg, quantize_deg_to_clock, generate_view_corners_with_scale, create_view_image, safe_read_csv, CLOCK_MAP, degree_to_clock, _pick_angle_baseline, convert_move_dir_row, INS_BLOCK, FORWARD_PATTERN, extract_ins, has_forward, CONTROL_CHARS, clean_text)

def run_multistep_min_rot(
    anno_dir: str,
    dataset_dir: str,
    split: str,
    pred_dir: str,
    max_steps: int = 8,
    scale_factor: float = 3.0,
    results_csv_name: str = "parsing_results_full_raw.csv",
    results_csv_name_post: str = "parsing_results_full_clock.csv",
):
    os.makedirs(pred_dir, exist_ok=True)
    step_dir = os.path.join(pred_dir, "stepwise_views")
    os.makedirs(step_dir, exist_ok=True)
    live_results_path = os.path.join(pred_dir, "parsing_results_live.jsonl")
    # The live file represents this run only. Each completed trajectory is
    # appended and fsynced below, so already parsed results survive interruption.
    with open(live_results_path, "w", encoding="utf-8"):
        pass
    print(f"Live JSONL initialized at: {live_results_path}")

    tif_dataset_dir = os.path.join(dataset_dir, 'train_images')
    env = ANDHNavBatch(
        anno_dir=anno_dir,
        dataset_dir=tif_dataset_dir,
        splits=[split],
        tokenizer=None,
        max_instr_len=512,
        batch_size=1,
        seed=0,
        full_traj=False,
    )
    loader = DataLoader(env, batch_size=1)

    try:
        total_samples = len(loader)
    except Exception:
        total_samples = "unknown"
    print(f"ANDHNavBatch loaded with {total_samples} instructions, using splits: {split}")

    rows = []
    parse_failures = []

    def _safe(s):
        return re.sub(r'[^\w\-.]+', '_', str(s))

    for batch_idx, _ in enumerate(tqdm(loader, desc="samples")):
        obs_list = env._get_obs(t=0)
        assert len(obs_list) == 1
        ob = obs_list[0]
        ob["dataset_dir"] = dataset_dir

        map_name = str(ob.get("map_name", ""))
        route_idx = ob.get("route_index", "")
        instr_id = f"{_safe(map_name)}__{_safe(route_idx)}"
        raw_dialog = str(ob.get("instructions", "") or "").strip()

        print(f"[SAMPLE {batch_idx+1}] map_name={map_name} route={route_idx}")

        start_corners = np.array(ob['gt_path_corners'][0])
        pos = np.mean(start_corners, axis=0)
        heading = float(ob.get("starting_angle", 0.0) or 0.0)
        starting_heading = heading

        corners0 = generate_view_corners_with_scale(pos, ob, scale_factor=scale_factor, angle_deg=heading)
        patch0 = create_view_image(corners0, ob)
        if patch0 is None:
            print("[WARN] start patch crop failed, skipping sample.")
            append_jsonl_record(live_results_path, {
                "sample_idx": batch_idx + 1,
                "instr_id": instr_id,
                "map_name": map_name,
                "route_idx": route_idx,
                "status": "skipped",
                "starting_heading": starting_heading,
                "dialogue": raw_dialog,
                "steps": [],
                "raw_output": "",
                "error": "start patch crop failed",
            })
            continue

        lat_min, lng_min = ob['gps_botm_left']
        lat_max, lng_max = ob['gps_top_right']
        h, w = ob['map_size'][:2]
        x_map = (pos[1] - lng_min) / (lng_max - lng_min) * w
        y_map = (lat_max - pos[0]) / (lat_max - lat_min) * h
        src_point = np.array([[x_map, y_map]], dtype=np.float32)

        src = []
        for lat, lng in corners0:
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            src.append([x, y])
        src = np.array(src, dtype=np.float32)
        dst = np.array([[0, 0], [223, 0], [223, 223], [0, 223]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        dst_point = cv2.perspectiveTransform(src_point[None, :, :], M)[0, 0]
        px, py = int(dst_point[0]), int(dst_point[1])
        if 0 <= px < 224 and 0 <= py < 224:
            cv2.circle(patch0, (px, py), radius=5, color=(0, 255, 0), thickness=-1)
        debug_name = f"{instr_id}_step00_start_with_point.jpg"
        cv2.imwrite(os.path.join(step_dir, debug_name), patch0)

        print("  [FULL DIALOGUE]\n", raw_dialog)
        try:
            planned_steps, output = request_dialogue_sequence(
                raw_dialog,
                starting_heading=heading,
                max_steps=max_steps,
                attempts=2,
            )
        except Exception as exc:
            message = str(exc)
            print(f"[ERROR] full-dialogue parse failed; skipping {instr_id}: {message}")
            parse_failures.append({
                "instr_id": instr_id,
                "map_name": map_name,
                "route_idx": route_idx,
                "error": message,
            })
            append_jsonl_record(live_results_path, {
                "sample_idx": batch_idx + 1,
                "instr_id": instr_id,
                "map_name": map_name,
                "route_idx": route_idx,
                "status": "parse_error",
                "starting_heading": starting_heading,
                "dialogue": raw_dialog,
                "steps": [],
                "raw_output": getattr(exc, "last_output", ""),
                "error": message,
            })
            continue

        print("  Model route sequence:\n", output)

        append_jsonl_record(live_results_path, {
            "sample_idx": batch_idx + 1,
            "instr_id": instr_id,
            "map_name": map_name,
            "route_idx": route_idx,
            "status": "ok",
            "starting_heading": starting_heading,
            "dialogue": raw_dialog,
            "steps": planned_steps,
            "raw_output": output,
            "error": "",
        })

        for step_idx, planned_step in enumerate(planned_steps, start=1):
            instruction = planned_step["instruction"]
            move_dir = planned_step["move_dir"]
            dest = planned_step["dest"]
            print(
                f"  [PLAN step {step_idx}] direction={move_dir} | "
                f"destination={dest}"
            )

            angle_before = float(heading)
            angle_after = float(heading)

            m_deg = DEGREE_MARK.match(move_dir)
            if m_deg:
                abs_angle = float(m_deg.group(1)) % 360
                angle_after = float(abs_angle)
                heading = angle_after
            else:
                m_clock = CLOCK_MARK.match(move_dir)
                if m_clock:
                    rel = clock_to_angle_deg(move_dir)
                    angle_after = float((heading + rel) % 360)
                    heading = angle_after
                else:
                    angle_after = float(heading)

            rows.append({
                "instr_id": instr_id,
                "map_name": map_name,
                "route_idx": route_idx,
                "step_idx": step_idx,
                "instruction": instruction,
                "move_dir": move_dir,
                "dest": dest,
                # Keep the legacy baseline CSV schema for the unchanged
                # Search_Confirmation reader; full-dialogue parsing no longer
                # asks the model to generate a separate waypoint/via field.
                "via": "",
                "angle_before": angle_before,
                "angle_after": angle_after,
                "raw_output": json.dumps(planned_step, ensure_ascii=False),
            })

    results_csv = os.path.join(pred_dir, results_csv_name)
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "instr_id","map_name","route_idx","step_idx","instruction",
                "move_dir","dest","via","angle_before","angle_after","raw_output"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    failures_path = os.path.join(pred_dir, "parsing_failures_full.jsonl")
    with open(failures_path, "w", encoding="utf-8") as f:
        for failure in parse_failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    print(f"\nCSV written to: {results_csv}")
    print(f"Live JSONL written to: {live_results_path}")
    print(f"Parse failures written to: {failures_path} ({len(parse_failures)} failures)")
    print(f"Screenshot dir: {os.path.join(pred_dir, 'stepwise_views')}")

    try:
        df = pd.read_csv(results_csv)
        def convert_row(row):
            s = str(row["move_dir"]).strip()
            m = DEGREE_MARK.match(s)
            if not m:
                return s
            abs_angle = float(m.group(1)) % 360
            angle_before = float(row["angle_before"])
            rel = (abs_angle - angle_before) % 360
            return quantize_deg_to_clock(rel)
        df["move_dir"] = df.apply(convert_row, axis=1)
        out_csv = os.path.join(pred_dir, results_csv_name_post)
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"Relative-clock CSV written to: {out_csv}")
    except Exception as e:
        print(f"[WARN] post-processing failed (skipping): {e}")

def postprocess_csv(pred_dir=PRED_DIR):
    input_csv = Path(pred_dir) / "parsing_results_full_raw.csv"
    intermediate_out = Path(pred_dir) / "parsing_results_full_clock.csv"
    final_out = Path(pred_dir) / "parsing_results_full.csv"

    df, used_enc = safe_read_csv(input_csv)
    if "move_dir" in df.columns:
        df["__angle_base__"] = [_pick_angle_baseline(row) for _, row in df.iterrows()]
        df["move_dir"] = [convert_move_dir_row(md, ang) for md, ang in zip(df["move_dir"], df["__angle_base__"])]
    else:
        print("Warning: missing column 'move_dir' - skip conversion.")
    df.to_csv(intermediate_out, index=False, encoding="utf-8-sig")
    print("Intermediate saved to:", intermediate_out)
    print("Read encoding (step1):", used_enc)

    df2, used_enc2 = safe_read_csv(intermediate_out)
    if "dest" in df2.columns:
        dest_series = df2["dest"].astype(str).str.strip().str.lower()
        is_dest = dest_series.eq("destination") | dest_series.eq("目的地")
    else:
        is_dest = pd.Series([False] * len(df2))
    ins_series = df2.get("instruction", pd.Series([""] * len(df2))).astype(str).map(extract_ins)
    forward_bool = is_dest & ins_series.map(has_forward)
    df2["forward_bool"] = forward_bool
    df2["forward"] = forward_bool.map(lambda v: "TRUE" if bool(v) else "FALSE")
    for col in df2.columns:
        if df2[col].dtype == object:
            df2[col] = df2[col].astype(str).map(clean_text)
    preferred_order = [
        "instr_id", "map_name", "route_idx", "step_idx", "instruction",
        "move_dir", "dest", "via", "forward", "forward_bool",
        "angle", "angle_before", "angle_after", "raw_output"
    ]
    cols = [c for c in preferred_order if c in df2.columns] + [c for c in df2.columns if c not in preferred_order]
    df2 = df2[cols]
    df2.to_csv(
        final_out,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\r\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    print("Read encoding (step2):", used_enc2)
    print("Final saved to:", final_out)

def main():
    os.makedirs(PRED_DIR, exist_ok=True)
    run_multistep_min_rot(
        anno_dir=ANNO_DIR,
        dataset_dir=DATASET_DIR,
        split=SPLIT,
        pred_dir=PRED_DIR,
        max_steps=MAX_STEPS,
        scale_factor=SCALE_FACTOR,
        results_csv_name="parsing_results_full_raw.csv",
        results_csv_name_post="parsing_results_full_clock.csv",
    )
    postprocess_csv(PRED_DIR)

if __name__ == "__main__":
    main()
