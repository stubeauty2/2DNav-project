"""Local rendering and Qwen + SAM3 search engine used by ANDH-Full."""

import os
import sys
import re
import csv
import json
import math
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import ANDHNavBatch
from torch.utils.data import DataLoader
from util import (
    generate_view_corners_with_scale,
    create_view_image,
    decide_turn_from_land,
    draw_pos_on_patch,
    _corners_center,
    angle_to_clock,
    clock_to_angle_deg,
    save_traj_boxes_debug_image,
    compute_iou,
    _move_forward_geo,
)
from visual_grounding.navigation_geometry import ViewGeoTransform
from reach_contracts import ReachMotionResult, TargetSelectionResult

RESULTS_CSV = str(PROJECT_ROOT / "preds_out" / "parsing_results.csv")

ANNO_DIR = str(PROJECT_ROOT / "datasets" / "AVDN" / "annotations")
DATASET_DIR = str(PROJECT_ROOT / "datasets" / "AVDN")
SPLIT       = "val_seen"
PRED_DIR = str(PROJECT_ROOT / "preds" / "andh")
OUT_DIR     = os.path.join(PRED_DIR, "search_output")
SCALE_FACTOR       = 5
FIXED_CROP_SIDE    = 768
SEARCH_VIEW_SCALES = {
    "main": 5.0,
}

DEFAULT_API_KEY = os.getenv("API_KEY", "")
QWEN_URL        = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
QWEN_MODEL      = os.getenv("QWEN_MODEL", "qwen3-vl-plus")
GROUNDING_BACKEND = os.getenv("GROUNDING_BACKEND", "sam3")
VISION_TOOL_URL = os.getenv("VISION_TOOL_URL", "http://127.0.0.1:8765")
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "4"))
QWEN_API_TIMEOUT = float(os.getenv("QWEN_API_TIMEOUT", "300"))
_GROUNDING_PREFLIGHT_CACHE = {}

def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c

def _patch_xy_to_latlng(u, v, view_corners, ob):
    del ob  # Kept in the signature for existing callers and tests.
    return ViewGeoTransform(view_corners, FIXED_CROP_SIDE, FIXED_CROP_SIDE).pixel_to_latlng([u, v])


def _coerce_latlng(value):
    if isinstance(value, str):
        parts = value.replace(",", " ").split()
        if len(parts) >= 2:
            return [float(parts[0]), float(parts[1])]
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return [float(value[0]), float(value[1])]
    raise ValueError(f"invalid latitude/longitude point: {value!r}")


def _agent_footprint_at_position(ob, position):
    raw_boxes = ob.get("gt_path_corners")
    if raw_boxes is None or len(raw_boxes) == 0:
        return []
    try:
        footprint = np.asarray([_coerce_latlng(item) for item in raw_boxes[0]], dtype=float)
    except (TypeError, ValueError, IndexError):
        return []
    if footprint.shape != (4, 2):
        return []
    translated = footprint + (np.asarray(position, dtype=float) - footprint.mean(axis=0))
    return translated.tolist()

def preflight_visual_grounding(
    vision_tool_url=VISION_TOOL_URL,
    grounding_backend=GROUNDING_BACKEND,
    qwen_api_timeout=QWEN_API_TIMEOUT,
):
    cache_key = (str(vision_tool_url).rstrip("/"), str(grounding_backend))
    if cache_key in _GROUNDING_PREFLIGHT_CACHE:
        return _GROUNDING_PREFLIGHT_CACHE[cache_key]
    from grounding_agent import GroundingAgent
    agent = GroundingAgent(
        api_key="",
        qwen_url=QWEN_URL,
        model=QWEN_MODEL,
        vision_tool_url=vision_tool_url,
        backend=grounding_backend,
        qwen_api_timeout=qwen_api_timeout,
        max_tool_calls=MAX_TOOL_CALLS,
    )
    health = agent.preflight()
    _GROUNDING_PREFLIGHT_CACHE[cache_key] = health
    return health


def qwen_locate_bbox_in_view(
    dest_desc,
    *,
    api_key=DEFAULT_API_KEY,
    url=QWEN_URL,
    model=QWEN_MODEL,
    grounding_backend=GROUNDING_BACKEND,
    vision_tool_url=VISION_TOOL_URL,
    max_tool_calls=MAX_TOOL_CALLS,
    qwen_api_timeout=QWEN_API_TIMEOUT,
    view_paths=None,
    view_scales=None,
    view_corners=None,
    view_sizes=None,
    current_position=None,
    heading_deg=0.0,
    agent_footprint=None,
    artifact_dir=None,
    entity_kind="LANDMARK",
    event_type="",
    relation="",
    side="",
    event_context=None,
    is_final_goal=False,
    event_id="",
    phase="all",
):
    from grounding_agent import GroundingAgent, GroundingAgentError, GroundingInfrastructureError
    try:
        health = preflight_visual_grounding(
            vision_tool_url=vision_tool_url,
            grounding_backend=grounding_backend,
            qwen_api_timeout=qwen_api_timeout,
        )
        agent = GroundingAgent(
            api_key=api_key,
            qwen_url=url,
            model=model,
            vision_tool_url=vision_tool_url,
            backend=grounding_backend,
            qwen_api_timeout=qwen_api_timeout,
            max_tool_calls=max_tool_calls,
        )
        result = agent.run(
            destination=dest_desc,
            view_paths=view_paths or {},
            view_scales=view_scales or {},
            view_corners=view_corners or {},
            view_sizes=view_sizes or {},
            current_position=current_position,
            heading_deg=heading_deg,
            agent_footprint=agent_footprint,
            artifact_dir=artifact_dir or OUT_DIR,
            entity_kind=entity_kind,
            event_type=event_type,
            relation=relation,
            side=side,
            event_context=event_context,
            is_final_goal=bool(is_final_goal),
            event_id=str(event_id or ""),
            phase=phase,
            skip_preflight=True,
        )
        result.setdefault("service_health", health)
        return result
    except GroundingInfrastructureError:
        raise
    except GroundingAgentError as exc:
        return {
            "dest_present": False,
            "bbox_2d": [0, 0, 0, 0],
            "raw_bbox_2d": [0, 0, 0, 0],
            "candidate_bbox_2d": [0, 0, 0, 0],
            "final_bbox_2d": None,
            "bbox_grounding": {"status": "not_requested"},
            "is_final_goal": bool(is_final_goal),
            "target_point_2d": None,
            "confidence": 0.0,
            "reason": str(exc),
            "selected_candidate_id": None,
            "mask_path": "",
            "provider": "tool-agent",
            "component_scores": {},
            "tool_trace": [],
            "view_paths": {},
            "schema_version": "reach-stage/v1",
            "status": "failed" if phase != "selection" else "failed",
            "all_candidates": [],
            "candidate_sheet_paths": [],
            "artifacts": {},
        }


def qwen_select_target_in_view(
    dest_desc,
    *,
    api_key=DEFAULT_API_KEY,
    url=QWEN_URL,
    model=QWEN_MODEL,
    grounding_backend=GROUNDING_BACKEND,
    vision_tool_url=VISION_TOOL_URL,
    max_tool_calls=MAX_TOOL_CALLS,
    qwen_api_timeout=QWEN_API_TIMEOUT,
    view_paths=None,
    view_scales=None,
    view_corners=None,
    view_sizes=None,
    current_position=None,
    heading_deg=0.0,
    agent_footprint=None,
    artifact_dir=None,
    entity_kind="LANDMARK",
    event_type="",
    event_context=None,
    event_id="",
):
    """Selection-only entry point; it never asks Qwen for a navigation point."""
    return qwen_locate_bbox_in_view(
        dest_desc,
        api_key=api_key, url=url, model=model,
        grounding_backend=grounding_backend,
        vision_tool_url=vision_tool_url,
        max_tool_calls=max_tool_calls,
        qwen_api_timeout=qwen_api_timeout,
        view_paths=view_paths, view_scales=view_scales,
        view_corners=view_corners, view_sizes=view_sizes,
        current_position=current_position, heading_deg=heading_deg,
        agent_footprint=agent_footprint, artifact_dir=artifact_dir,
        entity_kind=entity_kind, event_type=event_type,
        event_context=event_context, event_id=event_id,
        is_final_goal=False,
        phase="selection",
    )


def qwen_plan_navigation_point(
    dest_desc,
    selection,
    *,
    api_key=DEFAULT_API_KEY,
    url=QWEN_URL,
    model=QWEN_MODEL,
    grounding_backend=GROUNDING_BACKEND,
    vision_tool_url=VISION_TOOL_URL,
    qwen_api_timeout=QWEN_API_TIMEOUT,
    view_paths=None,
    artifact_dir=None,
    event_id="",
    event_type="",
    relation="",
    side="",
):
    """Motion-only entry point consuming a selection snapshot."""
    from grounding_agent import GroundingAgent, GroundingAgentError, GroundingInfrastructureError
    try:
        agent = GroundingAgent(
            api_key=api_key, qwen_url=url, model=model,
            vision_tool_url=vision_tool_url, backend=grounding_backend,
            qwen_api_timeout=qwen_api_timeout, max_tool_calls=2,
        )
        result = agent.run_motion(
            destination=str(dest_desc or ""),
            selection=selection,
            view_paths=view_paths or {},
            artifact_dir=artifact_dir or OUT_DIR,
            event_id=str(event_id or ""),
            event_type=event_type, relation=relation, side=side,
        ).to_dict()
        result["event_id"] = str(event_id or "")
        return result
    except GroundingInfrastructureError:
        raise
    except GroundingAgentError as exc:
        return {
            "schema_version": "reach-stage/v1",
            "status": "failed",
            "reason": str(exc),
            "target_point_2d": None,
            "tool_trace": [],
            "artifacts": {},
        }

def _draw_bbox_on_patch(patch, bbox, color=(0,0,255), thickness=3, label=None):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(patch, (x1,y1), (x2,y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = str(label)
        (tw, th), _ = cv2.getTextSize(txt, font, 0.5, 1)
        cv2.rectangle(patch, (x1, max(0, y1- (th+6))), (x1+tw+10, y1), color, -1)
        cv2.putText(patch, txt, (x1+5, max(0, y1-6)), font, 0.5, (255,255,255), 1, cv2.LINE_AA)


def _render_event_views(instr_id, ob, start_pos, heading, out_dir):
    """Render the immutable image context shared by both REACH stages."""
    os.makedirs(out_dir, exist_ok=True)
    view_corners, view_patches, view_paths, render_errors = {}, {}, {}, {}
    for view_id, view_scale in SEARCH_VIEW_SCALES.items():
        corners = generate_view_corners_with_scale(
            start_pos, ob, scale_factor=view_scale, angle_deg=heading
        )
        path = os.path.join(
            out_dir,
            f"{instr_id}_step01_{view_id}_scale{view_scale:g}_search_view_h{heading:.1f}.jpg",
        )
        patch = create_view_image(corners, ob, save_path=path, out_px=FIXED_CROP_SIDE)
        if patch is None:
            render_errors[view_id] = "render failed"
            continue
        draw_pos_on_patch(patch, corners, start_pos)
        cv2.imwrite(path, patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        view_corners[view_id] = np.asarray(corners, dtype=float)
        view_patches[view_id] = patch
        view_paths[view_id] = path
    main_corners = view_corners.get("main")
    main_patch = view_patches.get("main")
    if main_corners is None or main_patch is None:
        raise RuntimeError(f"failed to render main REACH view: {render_errors}")
    return view_corners, view_patches, view_paths, render_errors


def _write_stage_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(Path(path).resolve())


def search_target(
    instr_id, ob, dest_desc, start_pos_latlng, heading_deg, out_dir,
    api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL,
    event_context=None, grounding_backend=GROUNDING_BACKEND,
    vision_tool_url=VISION_TOOL_URL, max_tool_calls=MAX_TOOL_CALLS,
    qwen_api_timeout=QWEN_API_TIMEOUT, entity_kind="LANDMARK", event_type="",
    event_id="",
):
    """Run and persist only the target-selection stage."""
    start_pos = [float(start_pos_latlng[0]), float(start_pos_latlng[1])]
    heading = float(heading_deg or 0.0)
    preflight_visual_grounding(
        vision_tool_url=vision_tool_url,
        grounding_backend=grounding_backend,
        qwen_api_timeout=qwen_api_timeout,
    )
    view_corners, view_patches, view_paths, render_errors = _render_event_views(
        instr_id, ob, start_pos, heading, out_dir
    )
    main_patch = view_patches["main"]
    q = qwen_select_target_in_view(
        str(dest_desc or ""), api_key=api_key, url=url, model=model,
        grounding_backend=grounding_backend, vision_tool_url=vision_tool_url,
        max_tool_calls=max_tool_calls, qwen_api_timeout=qwen_api_timeout,
        view_paths=view_paths, view_scales=dict(SEARCH_VIEW_SCALES),
        view_corners={key: value.tolist() for key, value in view_corners.items()},
        view_sizes={key: [int(value.shape[1]), int(value.shape[0])] for key, value in view_patches.items()},
        current_position=start_pos, heading_deg=heading,
        agent_footprint=_agent_footprint_at_position(ob, start_pos),
        artifact_dir=os.path.join(out_dir, f"{instr_id}_selection_artifacts"),
        entity_kind=entity_kind, event_type=event_type,
        event_context=event_context, event_id=event_id or instr_id,
    )
    selection = TargetSelectionResult.from_dict(q)
    selection.artifacts.update({key: value for key, value in view_paths.items()})
    selection_path = _write_stage_json(
        Path(out_dir) / "selection.json", selection.to_dict()
    )
    selection.artifacts["selection_json"] = selection_path
    selection_path = _write_stage_json(
        Path(out_dir) / "selection.json", selection.to_dict()
    )
    log = {
        "schema_version": "reach-stage/v1",
        "stage": "selection",
        "instr_id": instr_id,
        "event_id": event_id or instr_id,
        "current_position": start_pos,
        "heading_deg": heading,
        "selection": selection.to_dict(),
        "view_paths": view_paths,
        "view_corners": {key: value.tolist() for key, value in view_corners.items()},
        "view_sizes": {key: [int(value.shape[1]), int(value.shape[0])] for key, value in view_patches.items()},
        "render_errors": render_errors,
        "predicted_corners": [view_corners["main"].tolist()],
    }
    _write_stage_json(Path(out_dir) / "selection_log.json", log)
    return {
        "status": selection.status,
        "found": selection.status == "selected",
        "selection": selection.to_dict(),
        "selection_json": selection_path,
        "view_paths": view_paths,
        "view_corners": log["view_corners"],
        "view_sizes": log["view_sizes"],
        "predicted_corners": log["predicted_corners"],
        "search_log": log,
        "search_log_path": str((Path(out_dir) / "selection_log.json").resolve()),
        "main_path": view_paths["main"],
    }


def apply_reach_motion(target_position, heading_deg, event_type, relation, clearance_meters=30.0):
    """Apply the legacy relation policy without involving either model stage."""
    relation_text = str(relation or "").strip().casefold()
    pass_prefixes = ("pass", "fly over", "flyover", "cross", "go through", "travel along")
    if str(event_type or "").upper() == "AVOID":
        return _move_forward_geo(target_position, float(heading_deg) + 180.0, clearance_meters), "reverse_clearance", float(clearance_meters)
    if str(event_type or "").upper() == "REACH" and relation_text.startswith(pass_prefixes):
        return _move_forward_geo(target_position, heading_deg, clearance_meters), "forward_clearance", float(clearance_meters)
    return list(target_position), "none", 0.0


def plan_reach_motion(
    instr_id, ob, dest_desc, start_pos_latlng, heading_deg, selection,
    view_paths, view_corners, out_dir, api_key=DEFAULT_API_KEY, url=QWEN_URL,
    model=QWEN_MODEL, event_type="", relation="", side="", event_id="",
    grounding_backend=GROUNDING_BACKEND, vision_tool_url=VISION_TOOL_URL,
    qwen_api_timeout=QWEN_API_TIMEOUT, clearance_meters=30.0,
):
    """Run only motion planning from a fixed target-selection snapshot."""
    motion = qwen_plan_navigation_point(
        str(dest_desc or ""), selection, api_key=api_key, url=url, model=model,
        grounding_backend=grounding_backend, vision_tool_url=vision_tool_url,
        qwen_api_timeout=qwen_api_timeout, view_paths=view_paths,
        artifact_dir=os.path.join(out_dir, f"{instr_id}_motion_artifacts"),
        event_id=event_id or instr_id, event_type=event_type,
        relation=relation, side=side,
    )
    result = ReachMotionResult.from_dict({
        "schema_version": "reach-stage/v1",
        **motion,
    })
    if result.status == "planned":
        main_corners = np.asarray((view_corners or {}).get("main"), dtype=float)
        target_position = _patch_xy_to_latlng(
            result.target_point_2d[0], result.target_point_2d[1], main_corners, ob
        )
        event_position, offset_mode, offset_meters = apply_reach_motion(
            target_position, heading_deg, event_type, relation, clearance_meters
        )
        result.target_position_latlng = list(target_position)
        result.event_position_latlng = list(event_position)
        result.offset_mode = offset_mode
        result.offset_meters = offset_meters
    motion_path = _write_stage_json(Path(out_dir) / "motion.json", result.to_dict())
    result.artifacts["motion_json"] = motion_path
    _write_stage_json(Path(out_dir) / "motion.json", result.to_dict())
    return result.to_dict()


def search_and_reach_destination(
    instr_id, ob, dest_desc,
    start_pos_latlng, heading_deg,
    out_dir,
    api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL,
    event_context=None,
    grounding_backend=GROUNDING_BACKEND,
    vision_tool_url=VISION_TOOL_URL,
    max_tool_calls=MAX_TOOL_CALLS,
    qwen_api_timeout=QWEN_API_TIMEOUT,
    entity_kind="LANDMARK",
    event_type="",
    relation="",
    side="",
    is_final_goal=False,
):
    start_pos = [float(start_pos_latlng[0]), float(start_pos_latlng[1])]
    pos = start_pos[:]
    heading = float(heading_deg or 0.0)

    steps_log = []
    found = False
    dest_latlng = None

    predicted_corners_seq = []
    zoom_idx = None

    preflight_visual_grounding(
        vision_tool_url=vision_tool_url,
        grounding_backend=grounding_backend,
        qwen_api_timeout=qwen_api_timeout,
    )

    step_n = 1
    view_corners = {}
    view_patches = {}
    view_paths = {}
    render_errors = {}
    for view_id, view_scale in SEARCH_VIEW_SCALES.items():
        corners = generate_view_corners_with_scale(
            start_pos, ob, scale_factor=view_scale, angle_deg=heading
        )
        path = os.path.join(
            out_dir,
            f"{instr_id}_step{step_n:02d}_{view_id}_scale{view_scale:g}_search_view_h{heading:.1f}.jpg",
        )
        patch = create_view_image(corners, ob, save_path=path, out_px=FIXED_CROP_SIDE)
        if patch is None:
            render_errors[view_id] = "render failed"
            continue
        draw_pos_on_patch(patch, corners, start_pos)
        cv2.imwrite(path, patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        view_corners[view_id] = np.asarray(corners, dtype=float)
        view_patches[view_id] = patch
        view_paths[view_id] = path

    main_corners = view_corners.get("main")
    main_patch = view_patches.get("main")
    if main_corners is None or main_patch is None:
        steps_log.append({
            "step": step_n,
            "action": "render_fail",
            "pos": start_pos,
            "view_scales": dict(SEARCH_VIEW_SCALES),
            "render_errors": render_errors,
            "view_paths": view_paths,
        })
    else:
        predicted_corners_seq.append(main_corners.copy())
        q = qwen_locate_bbox_in_view(
            str(dest_desc or ""),
            api_key=api_key,
            url=url,
            model=model,
            grounding_backend=grounding_backend,
            vision_tool_url=vision_tool_url,
            max_tool_calls=max_tool_calls,
            qwen_api_timeout=qwen_api_timeout,
            view_paths=view_paths,
            view_scales=dict(SEARCH_VIEW_SCALES),
            view_corners={key: value.tolist() for key, value in view_corners.items()},
            view_sizes={
                key: [int(value.shape[1]), int(value.shape[0])]
                for key, value in view_patches.items()
            },
            current_position=start_pos,
            heading_deg=heading,
            agent_footprint=_agent_footprint_at_position(ob, start_pos),
            artifact_dir=os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_grounding"),
            entity_kind=entity_kind,
            event_type=event_type,
            relation=relation,
            side=side,
            event_context=event_context,
            is_final_goal=bool(is_final_goal),
        )
        h, w = main_patch.shape[:2]
        step_log = {
            "step": step_n,
            "qwen": q,
            "pos": start_pos,
            "view_scales": dict(SEARCH_VIEW_SCALES),
            "render_errors": render_errors,
            "view_paths": view_paths,
            "w": w,
            "h": h,
            "geometry_context": {
                "current_position": start_pos,
                "heading_deg": heading,
                "view_corners": {key: value.tolist() for key, value in view_corners.items()},
                "agent_footprint": _agent_footprint_at_position(ob, start_pos),
            },
        }
        steps_log.append(step_log)

        if q.get("dest_present"):
            bbox = [float(v) for v in (q.get("candidate_bbox_2d") or q.get("bbox_2d") or [])][:4]
            target_point = q.get("target_point_2d")
            point_is_valid = bool(
                isinstance(target_point, (list, tuple))
                and len(target_point) == 2
                and all(np.isfinite(float(value)) and 0.0 <= float(value) < FIXED_CROP_SIDE for value in target_point)
            )
            if point_is_valid:
                final_bbox = [float(v) for v in (q.get("final_bbox_2d") or [])][:4]
                if bool(is_final_goal) and len(final_bbox) == 4:
                    q["final_bbox_latlng"] = [
                        _patch_xy_to_latlng(u, v, main_corners, ob)
                        for u, v in (
                            (final_bbox[0], final_bbox[1]),
                            (final_bbox[2], final_bbox[1]),
                            (final_bbox[2], final_bbox[3]),
                            (final_bbox[0], final_bbox[3]),
                        )
                    ]
                det_vis_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_det.jpg")
                patch_det = main_patch.copy()
                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    _draw_bbox_on_patch(
                        patch_det,
                        bbox,
                        color=(0, 0, 255),
                        thickness=3,
                        label=f"{float(q.get('confidence', 0.0)):.2f}",
                    )
                cv2.drawMarker(
                    patch_det,
                    tuple(max(0, min(FIXED_CROP_SIDE - 1, int(round(float(value))))) for value in target_point),
                    (255, 0, 255),
                    cv2.MARKER_CROSS,
                    24,
                    3,
                    cv2.LINE_AA,
                )
                cv2.imwrite(det_vis_path, patch_det, [cv2.IMWRITE_JPEG_QUALITY, 95])
                step_log["view_paths"]["detection"] = det_vis_path

                cx_px, cy_px = [float(v) for v in target_point]
                dest_latlng = _patch_xy_to_latlng(cx_px, cy_px, main_corners, ob)
                pos = dest_latlng[:]

                if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                    x1f, y1f, x2f, y2f = bbox
                else:
                    x1f, y1f, x2f, y2f = cx_px - 20.0, cy_px - 20.0, cx_px + 20.0, cy_px + 20.0
                edge = 0.5 * max(x2f - x1f, y2f - y1f)
                cx = 0.5 * (x1f + x2f)
                cy = 0.5 * (y1f + y2f)
                left, right = cx - edge, cx + edge
                top, bottom = cy - edge, cy + edge
                if left < 0:
                    cx -= left
                if right > FIXED_CROP_SIDE - 1:
                    cx -= right - (FIXED_CROP_SIDE - 1)
                if top < 0:
                    cy -= top
                if bottom > FIXED_CROP_SIDE - 1:
                    cy -= bottom - (FIXED_CROP_SIDE - 1)
                edge = min(
                    edge,
                    cx,
                    FIXED_CROP_SIDE - 1 - cx,
                    cy,
                    FIXED_CROP_SIDE - 1 - cy,
                )
                square = [
                    (cx - edge, cy - edge),
                    (cx + edge, cy - edge),
                    (cx + edge, cy + edge),
                    (cx - edge, cy + edge),
                ]
                corners_zoom = [
                    _patch_xy_to_latlng(u, v, main_corners, ob)
                    for u, v in square
                ]
                zoom_path = os.path.join(out_dir, f"{instr_id}_step{step_n:02d}_dest_zoom_hr.jpg")
                create_view_image(corners_zoom, ob, save_path=zoom_path, out_px=FIXED_CROP_SIDE)
                predicted_corners_seq.append(np.asarray(corners_zoom, dtype=float))
                zoom_idx = len(predicted_corners_seq) - 1
                step_log["view_paths"]["zoom"] = zoom_path
                found = True

    search_json_path = os.path.join(out_dir, f"{instr_id}_search.json")
    try:
        out = {
            "instr_id": instr_id,
            "found": bool(found),
            "dest_latlng": dest_latlng,
            "final_pos": pos,
            "heading_deg": heading,
            "is_final_goal": bool(is_final_goal),
            "view_scales": dict(SEARCH_VIEW_SCALES),
            "steps": steps_log
        }
        with open(search_json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] failed to write search JSON: {search_json_path} -> {e}")

    return found, dest_latlng, predicted_corners_seq, zoom_idx

def hook_after_turn_and_crop(row, ob, instr_id, pos, new_heading, scale_factor, out_dir,
                             api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL):
    dest_desc = (row.get("dest") or "").strip()
    print(f"[DEST_DESC] {instr_id}: {dest_desc}")

    found, dest_latlng, predicted_corners_seq, zoom_idx = search_and_reach_destination(
        instr_id=instr_id,
        ob=ob,
        dest_desc=dest_desc,
        start_pos_latlng=pos,
        heading_deg=new_heading,
        out_dir=out_dir,
        api_key=api_key,
        url=url,
        model=model
    )

    if found:
        print(f"[OK][DEST] {instr_id}: destination found, center approx {dest_latlng}")
    else:
        print(f"[INFO][DEST] {instr_id}: destination not found in the simultaneous multi-scale search; search log written.")

    return predicted_corners_seq, zoom_idx

def run_turn_and_crop(anno_dir=ANNO_DIR, dataset_dir=DATASET_DIR, split=SPLIT,
                      results_csv=RESULTS_CSV, out_dir=OUT_DIR,
                      scale_factor=SCALE_FACTOR):
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
                    model=QWEN_MODEL
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

    print(f"\nDone: generated {processed}  entries; skipped {skipped}  entries. Output dir: {out_dir}")

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

def create_view_image(view_corners, ob, save_path=None, out_px=768):
    try:
        map_path = os.path.join(ob['dataset_dir'], 'train_images', f"{ob['map_name']}.tif")
        if not os.path.exists(map_path):
            print(f"[WARN] map file not found: {map_path}")
            return None

        sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if sat_map is None:
            print(f"[WARN] failed to read map: {map_path}")
            return None

        h, w = sat_map.shape[:2]
        lat_min, lng_min = ob['gps_botm_left']
        lat_max, lng_max = ob['gps_top_right']

        src = []
        for lat, lng in view_corners:
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            src.append([x, y])
        src = np.array(src, dtype=np.float32)
        src[:, 0] = np.clip(src[:, 0], 0, w - 1)
        src[:, 1] = np.clip(src[:, 1], 0, h - 1)

        if cv2.contourArea(src.astype(np.float32)) < 1.0:
            print("[WARN] view area too small")
            return None

        out_side = int(768 if (out_px is None or out_px == 'auto') else out_px)

        dst = np.array([[0,0],[out_side-1,0],[out_side-1,out_side-1],[0,out_side-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)

        patch_hr = cv2.warpPerspective(sat_map, M, (out_side, out_side), flags=cv2.INTER_LANCZOS4)
        if patch_hr is None or patch_hr.size == 0:
            print("[WARN] perspective warp returned empty")
            return None

        if save_path:
            cv2.imwrite(save_path, patch_hr, [cv2.IMWRITE_JPEG_QUALITY, 98])

        return patch_hr
    except Exception as e:
        print(f"[ERROR] crop failed: {e}")
        return None

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    run_turn_and_crop()
