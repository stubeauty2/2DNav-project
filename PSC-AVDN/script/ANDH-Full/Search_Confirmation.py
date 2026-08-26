"""Execute a parsed InstructionPlan strictly one event at a time.

TURN changes heading without translation. MOVE advances a short fixed distance.
REACH and AVOID localize only their active visual target through the Qwen +
SAM3 tool chain before applying the event-specific motion. Pass-style motion
is represented by a REACH event whose relation describes how to traverse the
target.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


FILE_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = FILE_DIR.parent
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
for import_path in (FILE_DIR, SCRIPT_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from instruction_schema import EventType, FORMAT_VERSION, InstructionEvent, InstructionPlan
from reach_contracts import TargetSelectionResult


SEARCH_ENGINE_PATH = FILE_DIR / "search_engine.py"
PLANS_PATH = Path(os.getenv(
    "INSTRUCTION_PLANS_PATH",
    str(WORKSPACE_ROOT / "out" / "preds_out_full_sample3_no_thinking" / "instruction_plans.jsonl"),
))
ANNO_DIR = WORKSPACE_ROOT / "datasets" / "sample3"
DATASET_DIR = WORKSPACE_ROOT / "datasets" / "sample3"
SPLIT = "test_unseen_full"
OUT_DIR = (
    WORKSPACE_ROOT / "out" / "preds" / "andh_full_sample3"
    / "sequential_search_output"
)

QWEN_URL = os.getenv(
    "VISION_MODEL_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
QWEN_MODEL = os.getenv("VISION_MODEL_NAME", "qwen3-vl-plus")
QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("API_KEY")
    or os.getenv("API_TOKEN")
    or ""
)
MOTION_ONLY_METERS = float(os.getenv("ROUTE_MOTION_ONLY_METERS", "60"))
EVENT_CLEARANCE_METERS = float(os.getenv("ROUTE_EVENT_CLEARANCE_METERS", "30"))
GROUNDING_BACKEND = os.getenv("GROUNDING_BACKEND", "sam3")
VISION_TOOL_URL = os.getenv("VISION_TOOL_URL", "http://127.0.0.1:8765")
MAX_TOOL_CALLS = int(os.getenv("MAX_TOOL_CALLS", "4"))
QWEN_API_TIMEOUT = float(os.getenv("QWEN_API_TIMEOUT", "600"))


@dataclass
class VisualSearchResult:
    found: bool
    destination_position: Optional[List[float]]
    predicted_corners: List[np.ndarray]
    zoom_index: Optional[int]
    search_log: Dict[str, Any]
    search_log_path: str


@dataclass
class RouteViewHistory:
    corners: List[np.ndarray]
    patches: List[np.ndarray]
    view_ids: List[str]

    @classmethod
    def empty(cls) -> "RouteViewHistory":
        return cls(corners=[], patches=[], view_ids=[])


def _console(stage: str, message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}][{stage}] {message}", flush=True)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_search_engine():
    """Load the ANDH-Full image renderer and visual search lazily."""
    if not SEARCH_ENGINE_PATH.exists():
        raise FileNotFoundError(f"ANDH-Full search engine is missing: {SEARCH_ENGINE_PATH}")
    return _load_module("_psc_full_search_engine", SEARCH_ENGINE_PATH)


def qwen_locate_bbox_in_view(*args, **kwargs):
    """Public proxy for the Qwen + SAM3 grounding call."""
    return _load_search_engine().qwen_locate_bbox_in_view(*args, **kwargs)


def _read_json_if_present(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _attach_view_paths(log: Dict[str, Any]) -> None:
    """Keep existing multi-scale and SAM3 artifacts that are present on disk."""
    for step in log.get("steps", []):
        if not isinstance(step, dict):
            continue
        paths = {
            str(key): str(value)
            for key, value in (step.get("view_paths") or {}).items()
            if value and Path(value).exists()
        }
        qwen = step.get("qwen")
        if isinstance(qwen, dict):
            for key, value in (qwen.get("view_paths") or {}).items():
                if value and Path(value).exists():
                    paths[f"qwen_{key}"] = str(value)
            mask_path = qwen.get("mask_path")
            if mask_path and Path(mask_path).exists():
                paths["qwen_selected_mask"] = str(mask_path)
        step["view_paths"] = paths


def search_and_reach_destination(
    instr_id: str,
    observation: Dict[str, Any],
    destination_description: str,
    start_position: Sequence[float],
    heading_deg: float,
    out_dir: Path,
    *,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    engine=None,
    history: Optional[RouteViewHistory] = None,
    event_context: str = "",
    grounding_backend: str = GROUNDING_BACKEND,
    vision_tool_url: str = VISION_TOOL_URL,
    max_tool_calls: int = MAX_TOOL_CALLS,
    qwen_api_timeout: float = QWEN_API_TIMEOUT,
    entity_kind: str = "LANDMARK",
    event_type: str = "",
    relation: str = "",
    side: str = "",
    is_final_goal: bool = False,
) -> VisualSearchResult:
    """Run one visual search with isolated candidate-selection and waypoint contexts."""
    runtime = engine or _load_search_engine()
    history = history or RouteViewHistory.empty()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found, destination, corners, zoom_index = runtime.search_and_reach_destination(
        instr_id=instr_id,
        ob=observation,
        dest_desc=destination_description,
        start_pos_latlng=list(start_position),
        heading_deg=float(heading_deg),
        out_dir=str(out_dir),
        api_key=api_key,
        url=url,
        model=model,
        event_context=event_context or None,
        grounding_backend=grounding_backend,
        vision_tool_url=vision_tool_url,
        max_tool_calls=int(max_tool_calls),
        qwen_api_timeout=float(qwen_api_timeout),
        entity_kind=entity_kind,
        event_type=event_type,
        relation=relation,
        side=side,
        is_final_goal=bool(is_final_goal),
    )
    log_path = out_dir / f"{instr_id}_search.json"
    search_log = _read_json_if_present(log_path)
    _attach_view_paths(search_log)
    if search_log:
        log_path.write_text(
            json.dumps(_jsonable(search_log), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return VisualSearchResult(
        found=bool(found),
        destination_position=list(destination) if destination is not None else None,
        predicted_corners=[np.asarray(item, dtype=float) for item in (corners or [])],
        zoom_index=int(zoom_index) if zoom_index is not None else None,
        search_log=search_log,
        search_log_path=str(log_path),
    )


def _corners_center(corners: Sequence[Sequence[float]]) -> List[float]:
    return np.mean(np.asarray(corners, dtype=float), axis=0).tolist()


def _move_forward_geo(
    position: Sequence[float], heading_deg: float, meters: float
) -> List[float]:
    latitude, longitude = float(position[0]), float(position[1])
    radians = math.radians(float(heading_deg))
    north_m = float(meters) * math.cos(radians)
    east_m = float(meters) * math.sin(radians)
    return [
        latitude + north_m / 111_320.0,
        longitude + east_m / (111_320.0 * max(0.01, math.cos(math.radians(latitude)))),
    ]


def _translate_footprint(template: np.ndarray, center: Sequence[float]) -> np.ndarray:
    template = np.asarray(template, dtype=float)
    return template + (np.asarray(center, dtype=float) - np.mean(template, axis=0))


def _signed_area(points: Sequence[Sequence[float]]) -> float:
    array = np.asarray(points, dtype=float)
    return 0.5 * float(np.sum(
        array[:, 0] * np.roll(array[:, 1], -1)
        - array[:, 1] * np.roll(array[:, 0], -1)
    ))


def _line_intersection(start, end, clip_start, clip_end):
    segment = end - start
    clip = clip_end - clip_start
    denominator = segment[0] * clip[1] - segment[1] * clip[0]
    if abs(denominator) < 1e-12:
        return end
    delta = clip_start - start
    factor = (delta[0] * clip[1] - delta[1] * clip[0]) / denominator
    return start + factor * segment


def _convex_intersection(subject, clip):
    output = [np.asarray(point, dtype=float) for point in subject]
    clip_points = [np.asarray(point, dtype=float) for point in clip]
    orientation = 1.0 if _signed_area(clip_points) >= 0 else -1.0
    for index, clip_start in enumerate(clip_points):
        clip_end = clip_points[(index + 1) % len(clip_points)]
        input_points = output
        output = []
        if not input_points:
            break

        def inside(point):
            edge = clip_end - clip_start
            relative = point - clip_start
            return orientation * (edge[0] * relative[1] - edge[1] * relative[0]) >= -1e-12

        previous = input_points[-1]
        for current in input_points:
            if inside(current):
                if not inside(previous):
                    output.append(_line_intersection(previous, current, clip_start, clip_end))
                output.append(current)
            elif inside(previous):
                output.append(_line_intersection(previous, current, clip_start, clip_end))
            previous = current
    return output


def polygon_iou(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    intersection = _convex_intersection(left, right)
    intersection_area = abs(_signed_area(intersection)) if len(intersection) >= 3 else 0.0
    left_area = abs(_signed_area(left))
    right_area = abs(_signed_area(right))
    union = left_area + right_area - intersection_area
    return intersection_area / union if union > 0 else 0.0


def _haversine_m(left: Sequence[float], right: Sequence[float]) -> float:
    radius = 6_371_000.0
    lat1, lng1 = math.radians(float(left[0])), math.radians(float(left[1]))
    lat2, lng2 = math.radians(float(right[0])), math.radians(float(right[1]))
    dlat, dlng = lat2 - lat1, lng2 - lng1
    value = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2.0) ** 2
    )
    return radius * 2.0 * math.atan2(
        math.sqrt(value), math.sqrt(max(0.0, 1.0 - value))
    )


def _compact_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(attempt, dict):
        return {}
    return {
        key: attempt.get(key)
        for key in (
            "step", "action", "pos", "view_scales", "render_errors", "w", "h",
            "geometry_context",
            "qwen", "error", "view_paths"
        )
        if key in attempt
    }


def _final_grounding_from_search(
    search_log: Dict[str, Any], event: InstructionEvent
) -> Optional[Dict[str, Any]]:
    for step in reversed(search_log.get("steps", [])):
        qwen = step.get("qwen") if isinstance(step, dict) else None
        if not isinstance(qwen, dict) or not qwen.get("dest_present"):
            continue
        final_bbox = qwen.get("final_bbox_2d")
        if not isinstance(final_bbox, list) or len(final_bbox) != 4:
            return None
        return {
            "event_id": event.event_id,
            "candidate_id": qwen.get("selected_candidate_id"),
            "bbox_2d": list(final_bbox),
            "raw_bbox_2d": list(qwen.get("candidate_bbox_2d") or qwen.get("raw_bbox_2d") or []),
            "candidate_bbox_2d": list(qwen.get("candidate_bbox_2d") or []),
            "bbox_latlng": list(qwen.get("final_bbox_latlng") or []),
            "target_point_2d": qwen.get("target_point_2d"),
            "mask_path": qwen.get("mask_path") or "",
            "bbox_grounding": qwen.get("bbox_grounding") or {},
            "confidence": qwen.get("confidence"),
            "reason": qwen.get("reason") or "",
            "view_id": "main",
            "source_scale": 5,
        }
    return None


def _update_route_history(
    history: RouteViewHistory,
    result: VisualSearchResult,
    observation: Dict[str, Any],
    runtime,
    *,
    heading_deg: float,
) -> None:
    corner_builder = getattr(runtime, "generate_view_corners_with_scale", None)
    cv2_module = getattr(runtime, "cv2", None)
    if not callable(corner_builder) or cv2_module is None:
        return
    for step in result.search_log.get("steps", []):
        if not isinstance(step, dict) or not isinstance(step.get("qwen"), dict):
            continue
        paths = step.get("view_paths") or {}
        main_path = str(paths.get("main") or "")
        position = step.get("pos") or []
        if not main_path or main_path in history.view_ids or len(position) < 2:
            continue
        patch = cv2_module.imread(main_path)
        if patch is None:
            continue
        try:
            corners = corner_builder(
                [float(position[0]), float(position[1])],
                observation,
                scale_factor=float((step.get("view_scales") or {}).get("main", 5.0)),
                angle_deg=float(heading_deg),
            )
        except Exception:
            continue
        history.corners.append(np.asarray(corners, dtype=float))
        history.patches.append(patch)
        history.view_ids.append(main_path)


def _read_selection_replay(path: Path) -> Tuple[TargetSelectionResult, Dict[str, Any]]:
    """Read either selection.json or its richer selection_log.json sibling."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and isinstance(raw.get("selection"), dict):
        selection_raw = raw["selection"]
        context = raw
    else:
        selection_raw = raw
        sibling = path.with_name("selection_log.json")
        context = json.loads(sibling.read_text(encoding="utf-8")) if sibling.exists() else {}
    return TargetSelectionResult.from_dict(selection_raw), context


def _event_target_text(plan: InstructionPlan, event: InstructionEvent) -> str:
    entity = plan.entities[event.entity_ref]
    parts = [entity.description]
    if event.relation:
        parts.append(f"required relation: {event.relation}")
    if event.spatial_constraints:
        parts.append("spatial constraints: " + "; ".join(event.spatial_constraints))
    if event.side:
        parts.append(f"required side relative to travel direction: {event.side}")
    if event.count > 1:
        parts.append(f"ordered instance count: {event.count}")
    return ". ".join(parts)


def _event_context(
    plan: InstructionPlan,
    event: InstructionEvent,
    completed_event_ids: Sequence[str],
    candidate_index: int,
    excluded_positions: Sequence[Sequence[float]],
) -> str:
    entity = plan.entities[event.entity_ref]
    context = {
        "active_event": {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "target": {
                "description": entity.description,
                "kind": entity.kind.value,
                "goal": bool(entity.goal),
            },
            "relation": event.relation or None,
            "spatial_constraints": list(event.spatial_constraints),
            "side_relative_to_image_forward": event.side or None,
            "instance_index": int(candidate_index),
            "instance_count": int(event.count or 1),
        },
        "completed_event_ids": list(completed_event_ids),
        "previous_distinct_instance_count": len(excluded_positions),
        "instruction": (
            "Only the active event may be localized. Historical headings, compass words, "
            "clock directions, raw dialogue, and geographic coordinates are intentionally omitted."
        ),
    }
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def _event_motion_position(
    event: InstructionEvent,
    target_position: Sequence[float],
    heading_deg: float,
) -> List[float]:
    relation = event.relation.strip().casefold()
    pass_prefixes = (
        "pass",
        "fly over",
        "flyover",
        "cross",
        "go through",
        "travel along",
    )
    if event.event_type == EventType.REACH and relation.startswith(pass_prefixes):
        return _move_forward_geo(target_position, heading_deg, EVENT_CLEARANCE_METERS)
    if event.event_type == EventType.AVOID:
        return _move_forward_geo(
            target_position,
            float(heading_deg) + 180.0,
            EVENT_CLEARANCE_METERS,
        )
    return list(target_position)


def _record_pose(
    physical_boxes: List[np.ndarray],
    physical_headings: List[float],
    physical_trajectory: List[Dict[str, Any]],
    template_footprint: np.ndarray,
    *,
    event: InstructionEvent,
    event_index: int,
    position: Sequence[float],
    heading_deg: float,
) -> None:
    physical_boxes.append(_translate_footprint(template_footprint, position))
    physical_headings.append(float(heading_deg))
    physical_trajectory.append({
        "event_index": event_index,
        "event_id": event.event_id,
        "event_type": event.event_type.value,
        "position": list(position),
        "heading_deg": float(heading_deg),
    })


def execute_instruction_plan(
    plan: InstructionPlan,
    observation: Dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    motion_only_meters: float = MOTION_ONLY_METERS,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    grounding_backend: str = GROUNDING_BACKEND,
    vision_tool_url: str = VISION_TOOL_URL,
    max_tool_calls: int = MAX_TOOL_CALLS,
    qwen_api_timeout: float = QWEN_API_TIMEOUT,
    reach_phase: str = "all",
    event_id: Optional[str] = None,
    selection_replay: Optional[Path] = None,
    engine=None,
) -> Dict[str, Any]:
    """Run parsed events in list order and stop at the first failed target."""
    if reach_phase not in {"all", "selection", "motion"}:
        raise ValueError("reach_phase must be one of all, selection, motion")
    if reach_phase != "all" and not event_id:
        raise ValueError("selection and motion phases require --event-id")
    runtime = engine or _load_search_engine()
    route_dir = Path(out_dir or OUT_DIR / plan.plan_id)
    route_dir.mkdir(parents=True, exist_ok=True)
    gt_corners = observation.get("gt_path_corners") or []
    if not gt_corners:
        raise ValueError("observation has no gt_path_corners")

    template_footprint = np.asarray(gt_corners[0], dtype=float)
    position = _corners_center(template_footprint)
    starting_heading = float(
        plan.starting_heading_deg
        if plan.starting_heading_deg is not None
        else observation.get("starting_angle") or 0.0
    ) % 360.0
    body_heading = starting_heading
    travel_heading = starting_heading
    history = RouteViewHistory.empty()

    physical_boxes = [template_footprint.copy()]
    physical_headings = [travel_heading]
    physical_trajectory: List[Dict[str, Any]] = [{
        "event_index": 0,
        "event_id": None,
        "event_type": "START",
        "position": list(position),
        "heading_deg": travel_heading,
    }]
    event_progress: List[Dict[str, Any]] = []
    search_steps: List[Dict[str, Any]] = []
    reasoning: List[Dict[str, Any]] = []
    completed_event_ids: List[str] = []
    terminated = False
    termination_reason = "completed_all_events"
    final_target_grounding: Optional[Dict[str, Any]] = None
    motion_replay: Optional[Tuple[TargetSelectionResult, Dict[str, Any]]] = None
    if reach_phase == "motion":
        replay_path = Path(selection_replay) if selection_replay else (
            route_dir / str(event_id) / "selection.json"
        )
        motion_replay = _read_selection_replay(replay_path)
        replay_context = motion_replay[1]
        replay_position = replay_context.get("current_position")
        replay_heading = replay_context.get("heading_deg")
        if isinstance(replay_position, list) and len(replay_position) >= 2:
            position = [float(replay_position[0]), float(replay_position[1])]
        if replay_heading is not None:
            travel_heading = float(replay_heading)
            body_heading = travel_heading
        physical_boxes[0] = _translate_footprint(template_footprint, position)
        physical_headings[0] = travel_heading
        physical_trajectory[0].update({
            "position": list(position),
            "heading_deg": travel_heading,
        })

    for event_index, event in enumerate(plan.events, start=1):
        if reach_phase == "motion" and event.event_id != event_id:
            continue
        event_record: Dict[str, Any] = {
            "event_index": event_index,
            "event_id": event.event_id,
            "turn": event.turn,
            "type": event.event_type.value,
            "source_text": plan.source_for_turn(event.turn),
            "target_description": (
                _event_target_text(plan, event) if event.entity_ref else None
            ),
            "position_before": list(position),
            "heading_before_deg": travel_heading,
            "visual_searches": [],
        }
        _console(
            "EVENT/START",
            f"route={plan.plan_id} index={event_index}/{len(plan.events)} "
            f"event={event.event_id}:{event.event_type.value}",
        )

        if event.event_type == EventType.TURN:
            if event.direction is None:
                raise ValueError(f"TURN event {event.event_id} has no direction")
            travel_heading = event.direction.resolve(body_heading)
            body_heading = travel_heading
            status = "turned"
        elif event.event_type == EventType.MOVE:
            position = _move_forward_geo(position, travel_heading, motion_only_meters)
            status = "moved"
        else:
            entity = plan.entities[event.entity_ref]
            is_final_goal = bool(entity.goal)
            event_record["is_final_goal"] = is_final_goal
            event_positions: List[List[float]] = []
            status = "located"
            for candidate_index in range(1, max(1, int(event.count or 1)) + 1):
                call_id = f"{plan.plan_id}_{event.event_id}_c{candidate_index}"
                stage_dir = route_dir / str(event.event_id)
                if int(event.count or 1) > 1:
                    stage_dir = stage_dir / f"c{candidate_index}"
                if reach_phase == "motion":
                    selection_result, replay_context = motion_replay  # type: ignore[misc]
                    replay_paths = {
                        str(key): str(value)
                        for key, value in (replay_context.get("view_paths") or {}).items()
                        if value
                    }
                    if not replay_paths:
                        replay_paths = {
                            str(key): str(value)
                            for key, value in (selection_result.artifacts or {}).items()
                            if key in {"main", "narrow", "wide", "minimap"} and value
                        }
                    motion = runtime.plan_reach_motion(
                        call_id,
                        observation,
                        entity.description,
                        position,
                        travel_heading,
                        selection_result.to_dict(),
                        replay_paths,
                        replay_context.get("view_corners") or {},
                        str(stage_dir),
                        api_key=api_key, url=url, model=model,
                        event_type=event.event_type.value,
                        relation=event.relation, side=event.side,
                        event_id=event.event_id,
                        grounding_backend=grounding_backend,
                        vision_tool_url=vision_tool_url,
                        qwen_api_timeout=qwen_api_timeout,
                        clearance_meters=EVENT_CLEARANCE_METERS,
                    )
                    event_record["motion"] = motion
                    target_position = motion.get("target_position_latlng")
                    result = VisualSearchResult(
                        found=motion.get("status") == "planned" and target_position is not None,
                        destination_position=list(target_position) if target_position else None,
                        predicted_corners=[], zoom_index=None,
                        search_log={"stage": "motion", "motion": motion},
                        search_log_path=str((stage_dir / "motion.json").resolve()),
                    )
                elif reach_phase == "selection":
                    selection = runtime.search_target(
                        call_id, observation, entity.description, position, travel_heading,
                        str(stage_dir), api_key=api_key, url=url, model=model,
                        event_context=_event_context(
                            plan, event, completed_event_ids, candidate_index, event_positions,
                        ), grounding_backend=grounding_backend,
                        vision_tool_url=vision_tool_url, max_tool_calls=max_tool_calls,
                        qwen_api_timeout=qwen_api_timeout, entity_kind=entity.kind.value,
                        event_type=event.event_type.value, event_id=event.event_id,
                    )
                    result = VisualSearchResult(
                        found=selection.get("found") is True,
                        destination_position=None,
                        predicted_corners=[np.asarray(item, dtype=float) for item in selection.get("predicted_corners") or []],
                        zoom_index=None,
                        search_log=selection.get("search_log") or {},
                        search_log_path=str(selection.get("search_log_path") or ""),
                    )
                    event_record.setdefault("selection", selection.get("selection") or {})
                elif not callable(getattr(runtime, "search_target", None)):
                    # Compatibility for lightweight engines used by legacy tests.
                    result = search_and_reach_destination(
                        call_id, observation, entity.description, position, travel_heading,
                        route_dir, api_key=api_key, url=url, model=model, engine=runtime,
                        history=history,
                        event_context=_event_context(
                            plan, event, completed_event_ids, candidate_index, event_positions,
                        ), grounding_backend=grounding_backend,
                        vision_tool_url=vision_tool_url, max_tool_calls=max_tool_calls,
                        qwen_api_timeout=qwen_api_timeout, entity_kind=entity.kind.value,
                        event_type=event.event_type.value, relation=event.relation,
                        side=event.side, is_final_goal=is_final_goal,
                    )
                else:
                    selection = runtime.search_target(
                        call_id, observation, entity.description, position, travel_heading,
                        str(stage_dir), api_key=api_key, url=url, model=model,
                        event_context=_event_context(
                            plan, event, completed_event_ids, candidate_index, event_positions,
                        ), grounding_backend=grounding_backend,
                        vision_tool_url=vision_tool_url, max_tool_calls=max_tool_calls,
                        qwen_api_timeout=qwen_api_timeout, entity_kind=entity.kind.value,
                        event_type=event.event_type.value, event_id=event.event_id,
                    )
                    event_record.setdefault("selection", selection.get("selection") or {})
                    motion = runtime.plan_reach_motion(
                        call_id, observation, entity.description, position, travel_heading,
                        selection.get("selection") or {}, selection.get("view_paths") or {},
                        selection.get("view_corners") or {}, str(stage_dir),
                        api_key=api_key, url=url, model=model,
                        event_type=event.event_type.value, relation=event.relation,
                        side=event.side, event_id=event.event_id,
                        grounding_backend=grounding_backend,
                        vision_tool_url=vision_tool_url, qwen_api_timeout=qwen_api_timeout,
                        clearance_meters=EVENT_CLEARANCE_METERS,
                    )
                    event_record["motion"] = motion
                    target_position = motion.get("target_position_latlng")
                    result = VisualSearchResult(
                        found=motion.get("status") == "planned" and target_position is not None,
                        destination_position=list(target_position) if target_position else None,
                        predicted_corners=[np.asarray(item, dtype=float) for item in selection.get("predicted_corners") or []],
                        zoom_index=None,
                        search_log={
                            "stage": "all",
                            "selection": selection.get("search_log") or {},
                            "motion": motion,
                        },
                        search_log_path=str((stage_dir / "motion.json").resolve()),
                    )
                event_record["visual_searches"].extend(
                    _compact_attempt(item)
                    for item in result.search_log.get("steps", [])
                    if isinstance(item, dict)
                )
                event_record.setdefault("search_logs", []).append(result.search_log_path)
                _update_route_history(
                    history,
                    result,
                    observation,
                    runtime,
                    heading_deg=travel_heading,
                )
                if reach_phase == "selection":
                    selection_status = str((event_record.get("selection") or {}).get("status") or "failed")
                    status = "selection_" + selection_status
                    event_record.update({
                        "status": status,
                        "position_after": list(position),
                        "heading_after_deg": travel_heading,
                        "selection_json": str(
                            (stage_dir / "selection.json").resolve()
                        ),
                    })
                    search_steps.append(event_record)
                    reasoning.append({
                        "event_id": event.event_id,
                        "event_type": event.event_type.value,
                        "status": status,
                        "summary": "Selection stage completed without changing route position.",
                    })
                    terminated = True
                    termination_reason = f"selection_phase_{event.event_id}_{selection_status}"
                    break
                candidate = result.destination_position
                duplicate = bool(
                    candidate is not None
                    and any(_haversine_m(candidate, previous) < 1.0 for previous in event_positions)
                )
                if not result.found or candidate is None or duplicate:
                    status = "target_not_found" if not duplicate else "duplicate_target"
                    terminated = True
                    termination_reason = f"event_{event.event_id}_{status}"
                    break
                event_positions.append(list(candidate))
                if is_final_goal:
                    final_target_grounding = _final_grounding_from_search(
                        result.search_log, event
                    )
                if reach_phase in {"all", "motion"} and isinstance(event_record.get("motion"), dict):
                    planned_position = event_record["motion"].get("event_position_latlng")
                    position = list(planned_position) if planned_position else _event_motion_position(
                        event, candidate, travel_heading
                    )
                else:
                    position = _event_motion_position(event, candidate, travel_heading)
            if terminated:
                event_record.update({
                    "status": status,
                    "position_after": list(position),
                    "heading_after_deg": travel_heading,
                })
                search_steps.append(event_record)
                reasoning.append({
                    "event_id": event.event_id,
                    "event_type": event.event_type.value,
                    "status": status,
                    "summary": "Visual localization failed; later events were not executed.",
                })
                _console("EVENT/STOP", f"route={plan.plan_id} reason={termination_reason}")
                break
            event_record["localized_positions"] = event_positions

        completed_event_ids.append(event.event_id)
        _record_pose(
            physical_boxes,
            physical_headings,
            physical_trajectory,
            template_footprint,
            event=event,
            event_index=event_index,
            position=position,
            heading_deg=travel_heading,
        )
        event_record.update({
            "status": status,
            "position_after": list(position),
            "heading_after_deg": travel_heading,
        })
        search_steps.append(event_record)
        event_progress.append({
            "event_index": event_index,
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": "COMPLETED",
            "position": list(position),
            "heading_deg": travel_heading,
        })
        reasoning.append({
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "status": status,
            "summary": (
                "Heading changed without translation."
                if event.event_type == EventType.TURN
                else "Moved a short fixed distance without visual search."
                if event.event_type == EventType.MOVE
                else "Qwen selected the active target through SAM3 and the event pose was applied."
            ),
        })
        _console(
            "EVENT/DONE",
            f"route={plan.plan_id} event={event.event_id} status={status} "
            f"position={position} heading={travel_heading:.1f}",
        )

    goal = np.asarray(gt_corners[-1], dtype=float)
    progress = [polygon_iou(item, goal) for item in physical_boxes]
    trajectory = [_corners_center(item) for item in physical_boxes]
    final_iou = float(progress[-1]) if progress else 0.0
    return {
        "instr_id": plan.plan_id,
        "execution_mode": "sequential-events",
        "reach_phase": reach_phase,
        "debug_event_id": event_id,
        "selection_replay": str(selection_replay) if selection_replay else None,
        "trajectory": trajectory,
        "physical_trajectory": physical_trajectory,
        "path_corners": [
            (box.tolist(), float(heading))
            for box, heading in zip(physical_boxes, physical_headings)
        ],
        "progress": progress,
        "gt_progress": progress[:],
        "gt_path_corners": [np.asarray(item, dtype=float).tolist() for item in gt_corners],
        "reasoning": reasoning,
        "final_iou": final_iou,
        "path_length": len(physical_boxes),
        "success": bool(final_iou > 0.3),
        "search_steps": search_steps,
        "event_progress": event_progress,
        "completed_event_ids": completed_event_ids,
        "terminated": terminated,
        "termination_reason": termination_reason,
        "final_target_grounding": final_target_grounding,
    }


def load_instruction_plans(path: Path = PLANS_PATH) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(raw, dict) or "plan" not in raw:
                raise ValueError(
                    f"instruction plan at line {line_number} is not the flat format; "
                    "old route_programs.jsonl files are unsupported"
                )
            if raw.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"instruction plan at line {line_number} has "
                    f"format_version={raw.get('format_version')!r}, expected={FORMAT_VERSION}"
                )
            plan_id = str(raw.get("plan_id") or "").strip()
            if not plan_id:
                raise ValueError(f"instruction plan at line {line_number} has no plan_id")
            turns = raw.get("turns") if isinstance(raw.get("turns"), list) else []
            try:
                plan = InstructionPlan.from_dict(raw["plan"], plan_id=plan_id, turns=turns)
            except ValueError as exc:
                raise ValueError(
                    f"instruction plan at line {line_number} ({plan_id}) is invalid: {exc}"
                ) from exc
            records[plan_id] = {
                "record": raw,
                "plan": plan,
                "parser_model": str(raw.get("parser_model") or ""),
                "repaired": raw.get("repaired") is True,
            }
    return records


def _selection_bounds(
    max_routes: Optional[int], route_range: Optional[Tuple[int, int]]
) -> Tuple[int, Optional[int]]:
    if max_routes is not None:
        if max_routes <= 0 or route_range is not None:
            raise ValueError("max_routes must be positive and cannot be combined with route_range")
        return 1, max_routes
    if route_range is None:
        return 1, None
    start, end = route_range
    if start <= 0 or end < start:
        raise ValueError("route_range must be a positive inclusive START-END range")
    return start, end


def _aggregate_metrics(predictions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    values = list(predictions.values())
    return {
        "processed_routes": len(values),
        "success_rate": float(np.mean([item["success"] for item in values])) if values else 0.0,
        "avg_iou": float(np.mean([item["final_iou"] for item in values])) if values else 0.0,
        "avg_path_length": (
            float(np.mean([item["path_length"] for item in values])) if values else 0.0
        ),
        "terminated_routes": sum(item.get("terminated") is True for item in values),
    }


def run_instruction_plans(
    anno_dir: Path = ANNO_DIR,
    dataset_dir: Path = DATASET_DIR,
    split: str = SPLIT,
    plans_path: Path = PLANS_PATH,
    out_dir: Optional[Path] = None,
    *,
    motion_only_meters: float = MOTION_ONLY_METERS,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    grounding_backend: str = GROUNDING_BACKEND,
    vision_tool_url: str = VISION_TOOL_URL,
    max_tool_calls: int = MAX_TOOL_CALLS,
    qwen_api_timeout: float = QWEN_API_TIMEOUT,
    reach_phase: str = "all",
    event_id: Optional[str] = None,
    selection_replay: Optional[Path] = None,
    max_routes: Optional[int] = None,
    route_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    selected_start, selected_end = _selection_bounds(max_routes, route_range)
    runtime = _load_search_engine()
    grounding_health: Dict[str, Any] = {}
    preflight = getattr(runtime, "preflight_visual_grounding", None)
    if callable(preflight):
        grounding_health = preflight(
            vision_tool_url=vision_tool_url,
            grounding_backend=grounding_backend,
            qwen_api_timeout=qwen_api_timeout,
        )
    output_dir = Path(out_dir or OUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_records = load_instruction_plans(plans_path)
    environment = runtime.ANDHNavBatch(
        anno_dir=str(anno_dir),
        dataset_dir=str(Path(dataset_dir) / "train_images"),
        splits=[split],
        tokenizer=None,
        max_instr_len=512,
        batch_size=1,
        seed=0,
        full_traj=False,
    )
    loader = runtime.DataLoader(environment, batch_size=1)
    predictions: Dict[str, Dict[str, Any]] = {}
    dataset_ordinal = 0
    for _ in loader:
        dataset_ordinal += 1
        if dataset_ordinal < selected_start:
            continue
        if selected_end is not None and dataset_ordinal > selected_end:
            break
        observations = environment._get_obs(t=0)
        if not observations:
            continue
        observation = dict(observations[0])
        observation["dataset_dir"] = str(dataset_dir)
        plan_id = (
            f"{observation.get('map_name', '')}__"
            f"{observation.get('route_index', dataset_ordinal - 1)}"
        )
        if plan_id not in plan_records:
            raise ValueError(f"selected route {plan_id} has no InstructionPlan in {plans_path}")
        plan = plan_records[plan_id]["plan"]
        route_dir = output_dir / plan_id
        _console("ROUTE/START", f"route={plan_id} events={len(plan.events)}")
        prediction = execute_instruction_plan(
            plan,
            observation,
            out_dir=route_dir,
            motion_only_meters=motion_only_meters,
            api_key=api_key,
            url=url,
            model=model,
            grounding_backend=grounding_backend,
            vision_tool_url=vision_tool_url,
            max_tool_calls=max_tool_calls,
            qwen_api_timeout=qwen_api_timeout,
            reach_phase=reach_phase,
            event_id=event_id,
            selection_replay=selection_replay,
            engine=runtime,
        )
        prediction["grounding"] = {
            "mode": "tool-agent",
            "model": model,
            "backend": grounding_backend,
            "view_scales": {"main": 5.0},
            "max_candidates_per_view": 12,
            "max_tool_calls": int(max_tool_calls),
            "qwen_api_timeout_seconds": float(qwen_api_timeout),
            "service_health": grounding_health,
        }
        predictions[plan_id] = prediction
        route_dir.mkdir(parents=True, exist_ok=True)
        (route_dir / "prediction.json").write_text(
            json.dumps(_jsonable(prediction), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        boxes = [np.asarray(item[0], dtype=float) for item in prediction["path_corners"]]
        runtime.save_traj_boxes_debug_image(
            tif_dataset_dir=str(Path(dataset_dir) / "train_images"),
            map_name=str(observation.get("map_name") or ""),
            ob=observation,
            predicted_boxes=boxes,
            out_path=str(route_dir / "executed_trajectory.jpg"),
            zoom_highlight_idx=None,
        )
        _console(
            "ROUTE/DONE",
            f"route={plan_id} success={prediction['success']} "
            f"final_iou={prediction['final_iou']:.4f} "
            f"status={prediction['termination_reason']}",
        )

    metrics = _aggregate_metrics(predictions)
    if predictions:
        try:
            environment_metrics, _ = environment.eval_metrics(predictions, human_att_eval=False)
        except Exception as exc:
            environment_metrics = {"evaluation_error": str(exc)}
    else:
        environment_metrics = {"evaluation_skipped": "no_predictions"}
    metrics["environment_metrics"] = _jsonable(environment_metrics)
    output_path = output_dir / "turn_and_crop_eval_results_full_traj.json"
    output_path.write_text(
        json.dumps(
            _jsonable({"predictions": predictions, "metrics": metrics}),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _console("RUN/WRITE", f"routes={len(predictions)} output={output_path}")
    return predictions, metrics


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return parsed


def _parse_route_range(value: str) -> Tuple[int, int]:
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if not match:
        raise argparse.ArgumentTypeError("must use START-END")
    start, end = int(match.group(1)), int(match.group(2))
    if start <= 0 or end < start:
        raise argparse.ArgumentTypeError("range must be positive and ordered")
    return start, end


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute parsed navigation events sequentially with Qwen + SAM3 grounding.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-routes", type=_positive_int, default=None)
    selection.add_argument("--route-range", type=_parse_route_range, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--motion-only-meters", type=_positive_float, default=MOTION_ONLY_METERS)
    parser.add_argument(
        "--grounding-backend",
        choices=("sam3", "grounded-sam2"),
        default=GROUNDING_BACKEND,
    )
    parser.add_argument("--vision-tool-url", default=VISION_TOOL_URL)
    parser.add_argument("--max-tool-calls", type=_positive_int, default=MAX_TOOL_CALLS)
    parser.add_argument(
        "--reach-phase",
        choices=("all", "selection", "motion"),
        default="all",
        help="Run both stages, selection only, or motion only from a selection snapshot.",
    )
    parser.add_argument(
        "--event-id",
        default=None,
        help="Target REACH/AVOID event for selection or motion-only debugging.",
    )
    parser.add_argument(
        "--selection-replay",
        type=Path,
        default=None,
        help="Path to selection.json or selection_log.json for --reach-phase motion.",
    )
    parser.add_argument(
        "--qwen-api-timeout",
        type=_positive_float,
        default=QWEN_API_TIMEOUT,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    _, metrics = run_instruction_plans(
        out_dir=args.out_dir,
        max_routes=args.max_routes,
        route_range=args.route_range,
        motion_only_meters=args.motion_only_meters,
        grounding_backend=args.grounding_backend,
        vision_tool_url=args.vision_tool_url,
        max_tool_calls=args.max_tool_calls,
        qwen_api_timeout=args.qwen_api_timeout,
        reach_phase=args.reach_phase,
        event_id=args.event_id,
        selection_replay=args.selection_replay,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
