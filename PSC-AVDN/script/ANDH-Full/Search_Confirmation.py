"""Execute flat InstructionPlans with the original baseline search loop.

The language-model plan is deterministically projected to at most one
``SearchStep`` per INS turn. Visual localization, multi-scale views, semantic
grid history, destination confirmation, fallback motion, and zoom generation
are delegated to the unchanged baseline implementation at runtime.
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

from instruction_schema import (
    FORMAT_VERSION,
    InstructionPlan,
    STATE_EVENT_TYPES,
)
from search_plan_adapter import (
    HeadingState,
    SearchStep,
    project_instruction_plan,
    resolve_step_heading,
)


BASELINE_SEARCH_PATH = (
    WORKSPACE_ROOT / "PSC-AVDN-baseline" / "script" / "ANDH-Full"
    / "Search_Confirmation.py"
)
PLANS_PATH = Path(os.getenv(
    "INSTRUCTION_PLANS_PATH",
    str(WORKSPACE_ROOT / "out" / "preds_out_full_sample2" / "instruction_plans.jsonl"),
))
ANNO_DIR = WORKSPACE_ROOT / "datasets" / "sample2"
DATASET_DIR = WORKSPACE_ROOT / "datasets" / "sample2"
SPLIT = "test_unseen_full"
OUT_DIR = (
    WORKSPACE_ROOT / "out" / "preds" / "andh_full_sample2" / "search_output"
)

QWEN_URL = os.getenv(
    "VISION_MODEL_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
)
QWEN_MODEL = os.getenv("VISION_MODEL_NAME", "qwen-vl-max")
QWEN_API_KEY = (
    os.getenv("QWEN_API_KEY")
    or os.getenv("API_KEY")
    or os.getenv("API_TOKEN")
    or ""
)
SEARCH_SCALE_FACTOR = float(os.getenv("ROUTE_SEARCH_SCALE", "5"))
SEARCH_STEP_METERS = float(os.getenv("ROUTE_SEARCH_STEP_METERS", "120"))
MAX_SEARCH_STEPS = int(os.getenv("ROUTE_MAX_SEARCH_STEPS", "3"))
MOTION_ONLY_METERS = float(os.getenv("ROUTE_MOTION_ONLY_METERS", "60"))


@dataclass
class BaselineSearchResult:
    found: bool
    destination_position: Optional[List[float]]
    predicted_corners: List[np.ndarray]
    zoom_index: Optional[int]
    search_log: Dict[str, Any]
    search_log_path: str


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
def _load_baseline_engine():
    """Load baseline lazily so adapter tests need neither OpenCV nor Torch."""
    if not BASELINE_SEARCH_PATH.exists():
        raise FileNotFoundError(
            f"baseline Search_Confirmation is missing: {BASELINE_SEARCH_PATH}"
        )
    baseline_script = BASELINE_SEARCH_PATH.parents[1]
    baseline_src = baseline_script / "src"
    baseline_full = BASELINE_SEARCH_PATH.parent
    for import_path in (baseline_full, baseline_script, baseline_src):
        if str(import_path) not in sys.path:
            sys.path.insert(0, str(import_path))

    previous_env = sys.modules.get("env")
    previous_util = sys.modules.get("util")
    try:
        baseline_env = _load_module("_psc_baseline_env", baseline_src / "env.py")
        sys.modules["env"] = baseline_env
        baseline_util = _load_module("_psc_baseline_util", baseline_script / "util.py")
        sys.modules["util"] = baseline_util
        return _load_module("_psc_baseline_search", BASELINE_SEARCH_PATH)
    finally:
        if previous_env is None:
            sys.modules.pop("env", None)
        else:
            sys.modules["env"] = previous_env
        if previous_util is None:
            sys.modules.pop("util", None)
        else:
            sys.modules["util"] = previous_util


def qwen_locate_bbox_in_view(*args, **kwargs):
    """Public proxy to the baseline multi-scale grounding call."""
    return _load_baseline_engine().qwen_locate_bbox_in_view(*args, **kwargs)


def _read_json_if_present(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _materialize_context_artifacts(
    runtime,
    observation: Dict[str, Any],
    log: Dict[str, Any],
    *,
    instr_id: str,
    out_dir: Path,
    heading_deg: float,
    destination: Optional[Sequence[float]],
) -> None:
    """Save baseline context views that were model inputs but not persisted."""
    renderer = getattr(runtime, "create_view_image", None)
    corner_builder = getattr(runtime, "generate_view_corners_with_scale", None)
    if not callable(renderer) or not callable(corner_builder):
        return
    attempts = log.get("steps") if isinstance(log.get("steps"), list) else []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        try:
            step_number = int(attempt.get("k", 0)) + 1
            position = [float(value) for value in attempt.get("pos", [])[:2]]
        except (TypeError, ValueError):
            continue
        if len(position) != 2:
            continue
        prefix = out_dir / f"{instr_id}_step{step_number:02d}"
        paths = {
            "main": str(prefix.with_name(
                f"{prefix.name}_search_view_h{heading_deg:.1f}.jpg"
            )),
            "narrow": str(prefix.with_name(f"{prefix.name}_context_narrow_s3.jpg")),
            "wide": str(prefix.with_name(f"{prefix.name}_context_wide_s7.jpg")),
            "minimap": str(prefix.with_name(f"{prefix.name}_minimap.jpg")),
            "detection": str(prefix.with_name(f"{prefix.name}_det.jpg")),
            "confirmation": str(prefix.with_name(f"{prefix.name}_dest_confirm_hr.jpg")),
            "confirmation_detection": str(prefix.with_name(
                f"{prefix.name}_dest_confirm_det.jpg"
            )),
            "zoom": str(prefix.with_name(f"{prefix.name}_dest_zoom_hr.jpg")),
        }
        for scale, key in ((3.0, "narrow"), (7.0, "wide")):
            try:
                corners = corner_builder(
                    position,
                    observation,
                    scale_factor=scale,
                    angle_deg=heading_deg,
                )
                renderer(corners, observation, save_path=paths[key], out_px=768)
            except Exception:
                paths[key] = ""
        attempt["view_paths"] = {
            key: value for key, value in paths.items()
            if value and Path(value).exists()
        }

    if destination is not None and attempts:
        found_attempt = next(
            (
                item for item in attempts
                if isinstance(item, dict) and isinstance(item.get("confirm"), dict)
            ),
            attempts[-1],
        )
        paths = found_attempt.setdefault("view_paths", {})
        step_number = int(found_attempt.get("k", 0)) + 1
        for scale, label in ((3.0, "confirmation_narrow"), (7.0, "confirmation_wide")):
            path = out_dir / f"{instr_id}_step{step_number:02d}_{label}.jpg"
            try:
                corners = corner_builder(
                    destination,
                    observation,
                    scale_factor=scale,
                    angle_deg=heading_deg,
                )
                renderer(corners, observation, save_path=str(path), out_px=768)
            except Exception:
                continue
            if path.exists():
                paths[label] = str(path)


def search_and_reach_destination(
    instr_id: str,
    observation: Dict[str, Any],
    destination_description: str,
    via_description: str,
    start_position: Sequence[float],
    heading_deg: float,
    scale_factor: float,
    out_dir: Path,
    *,
    step_meters: float = SEARCH_STEP_METERS,
    max_steps: int = MAX_SEARCH_STEPS,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    engine=None,
) -> BaselineSearchResult:
    """Call the baseline search unchanged and attach its structured JSON log."""
    runtime = engine or _load_baseline_engine()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    found, destination, corners, zoom_index = runtime.search_and_reach_destination(
        instr_id=instr_id,
        ob=observation,
        dest_desc=destination_description,
        via_desc=via_description or None,
        start_pos_latlng=list(start_position),
        heading_deg=float(heading_deg),
        scale_factor=float(scale_factor),
        out_dir=str(out_dir),
        step_meters=float(step_meters),
        max_steps=int(max_steps),
        api_key=api_key,
        url=url,
        model=model,
    )
    log_path = out_dir / f"{instr_id}_search.json"
    search_log = _read_json_if_present(log_path)
    _materialize_context_artifacts(
        runtime,
        observation,
        search_log,
        instr_id=instr_id,
        out_dir=out_dir,
        heading_deg=float(heading_deg),
        destination=destination,
    )
    if search_log:
        log_path.write_text(
            json.dumps(_jsonable(search_log), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return BaselineSearchResult(
        found=bool(found),
        destination_position=(list(destination) if destination is not None else None),
        predicted_corners=[np.asarray(item, dtype=float) for item in (corners or [])],
        zoom_index=(int(zoom_index) if zoom_index is not None else None),
        search_log=search_log,
        search_log_path=str(log_path),
    )


def _corners_center(corners: Sequence[Sequence[float]]) -> List[float]:
    return np.mean(np.asarray(corners, dtype=float), axis=0).tolist()


def _move_forward_geo(
    position: Sequence[float],
    heading_deg: float,
    meters: float,
) -> List[float]:
    lat, lng = float(position[0]), float(position[1])
    heading = math.radians(float(heading_deg))
    lat += math.cos(heading) * float(meters) / 111_320.0
    longitude_scale = max(1e-9, 111_320.0 * math.cos(math.radians(lat)))
    lng += math.sin(heading) * float(meters) / longitude_scale
    return [lat, lng]


def _translate_footprint(template: np.ndarray, center: Sequence[float]) -> np.ndarray:
    return template + (np.asarray(center, dtype=float) - np.mean(template, axis=0))


def _signed_area(points: Sequence[Sequence[float]]) -> float:
    values = np.asarray(points, dtype=float)
    if len(values) < 3:
        return 0.0
    return 0.5 * float(sum(
        values[index, 0] * values[(index + 1) % len(values), 1]
        - values[(index + 1) % len(values), 0] * values[index, 1]
        for index in range(len(values))
    ))


def _line_intersection(start, end, clip_start, clip_end):
    direction = end - start
    clip_direction = clip_end - clip_start
    denominator = direction[0] * clip_direction[1] - direction[1] * clip_direction[0]
    if abs(denominator) < 1e-15:
        return end
    delta = clip_start - start
    fraction = (delta[0] * clip_direction[1] - delta[1] * clip_direction[0]) / denominator
    return start + fraction * direction


def _convex_intersection(subject, clip):
    output = [np.asarray(item, dtype=float) for item in subject]
    clip_values = [np.asarray(item, dtype=float) for item in clip]
    orientation = 1.0 if _signed_area(clip_values) >= 0.0 else -1.0
    for index, clip_start in enumerate(clip_values):
        clip_end = clip_values[(index + 1) % len(clip_values)]
        input_values = output
        output = []
        if not input_values:
            break

        def inside(point):
            cross = (
                (clip_end[0] - clip_start[0]) * (point[1] - clip_start[1])
                - (clip_end[1] - clip_start[1]) * (point[0] - clip_start[0])
            )
            return orientation * cross >= -1e-15

        previous = input_values[-1]
        for current in input_values:
            current_inside = inside(current)
            previous_inside = inside(previous)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, clip_start, clip_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, clip_start, clip_end)
                )
            previous = current
    return output


def polygon_iou(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
) -> float:
    left_area = abs(_signed_area(left))
    right_area = abs(_signed_area(right))
    intersection_area = abs(_signed_area(_convex_intersection(left, right)))
    union = left_area + right_area - intersection_area
    return intersection_area / union if union > 0.0 else 0.0


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _state_updates(plan: InstructionPlan) -> List[Dict[str, Any]]:
    return [
        {
            "event_id": event.event_id,
            "turn": event.turn,
            "type": event.event_type.value,
            "entity": event.entity_ref,
            "description": event.description,
            "ref": event.ref,
            "completed": event.completed,
        }
        for event in plan.events
        if event.event_type in STATE_EVENT_TYPES
    ]


def _via_text(step: SearchStep) -> str:
    return "; ".join(item.phrase() for item in step.via)


def _compact_attempt(attempt: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(attempt, dict):
        return {}
    return {
        key: attempt.get(key)
        for key in (
            "k", "action", "pos", "w", "h", "qwen", "confirm", "error",
            "view_paths",
        )
        if key in attempt
    }


def execute_instruction_plan(
    plan: InstructionPlan,
    observation: Dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    search_scale: float = SEARCH_SCALE_FACTOR,
    step_meters: float = SEARCH_STEP_METERS,
    max_search_steps: int = MAX_SEARCH_STEPS,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    engine=None,
) -> Dict[str, Any]:
    """Execute projected INS steps and return the baseline evaluation shape."""
    validation_errors = plan.validate()
    if validation_errors:
        raise ValueError("invalid InstructionPlan: " + "; ".join(validation_errors))
    gt_corners = observation.get("gt_path_corners") or []
    if not gt_corners:
        raise ValueError("observation has no gt_path_corners start pose")
    runtime = engine or _load_baseline_engine()
    route_dir = Path(out_dir or OUT_DIR / plan.plan_id)
    route_dir.mkdir(parents=True, exist_ok=True)
    template_footprint = np.asarray(gt_corners[0], dtype=float)
    position = _corners_center(template_footprint)
    starting_heading = float(observation.get("starting_angle") or 0.0)
    heading = HeadingState(starting_heading, starting_heading).normalized()
    steps = project_instruction_plan(plan)
    predicted_boxes: List[np.ndarray] = []
    predicted_headings: List[float] = []
    step_records: List[Dict[str, Any]] = []
    state_updates = _state_updates(plan)
    state_update_by_id = {
        item["event_id"]: item for item in state_updates
    }
    terminated = False
    termination_reason = "completed_all_steps"

    for step in steps:
        heading_before = heading
        heading = resolve_step_heading(step, heading)
        current_view = np.asarray(runtime.generate_view_corners_with_scale(
            position,
            observation,
            scale_factor=search_scale,
            angle_deg=heading_before.body_heading_deg,
        ), dtype=float)
        turned_view = np.asarray(runtime.generate_view_corners_with_scale(
            position,
            observation,
            scale_factor=search_scale,
            angle_deg=heading.travel_heading_deg,
        ), dtype=float)
        predicted_boxes.extend([current_view, turned_view])
        predicted_headings.extend([
            heading_before.body_heading_deg,
            heading.travel_heading_deg,
        ])
        record = step.to_dict()
        record.update({
            "body_heading_before_deg": heading_before.body_heading_deg,
            "body_heading_after_deg": heading.body_heading_deg,
            "resolved_heading_deg": heading.travel_heading_deg,
            "position_before": list(position),
            "state_events": [
                state_update_by_id[event_id]
                for event_id in step.source_event_ids
                if event_id in state_update_by_id
            ],
        })

        if step.motion_only:
            if step.forward:
                position = _move_forward_geo(
                    position,
                    heading.travel_heading_deg,
                    MOTION_ONLY_METERS,
                )
                predicted_boxes.append(
                    _translate_footprint(template_footprint, position)
                )
                predicted_headings.append(heading.travel_heading_deg)
                record["status"] = "motion_only_moved"
            else:
                record["status"] = "turn_only"
            record["position_after"] = list(position)
            step_records.append(record)
            continue

        search_id = f"{plan.plan_id}_step{step.step_index:02d}"
        result = search_and_reach_destination(
            search_id,
            observation,
            step.grounding_query or step.destination_description,
            _via_text(step),
            position,
            heading.travel_heading_deg,
            search_scale,
            route_dir,
            step_meters=step_meters,
            max_steps=max_search_steps,
            api_key=api_key,
            url=url,
            model=model,
            engine=runtime,
        )
        predicted_boxes.extend(result.predicted_corners)
        predicted_headings.extend(
            [heading.travel_heading_deg] * len(result.predicted_corners)
        )
        record.update({
            "status": "found" if result.found else "not_found",
            "position_after": (
                list(result.destination_position)
                if result.destination_position is not None
                else list(position)
            ),
            "search_log_path": result.search_log_path,
            "zoom_index": result.zoom_index,
            "search_attempts": [
                _compact_attempt(item)
                for item in result.search_log.get("steps", [])
            ],
        })
        step_records.append(record)
        if result.found and result.destination_position is not None:
            position = list(result.destination_position)
        else:
            terminated = True
            termination_reason = f"step_{step.step_index}_target_not_found"
            break

    goal = np.asarray(gt_corners[-1], dtype=float)
    progress = [polygon_iou(item, goal) for item in predicted_boxes]
    trajectory = [_corners_center(item) for item in predicted_boxes]
    final_iou = float(progress[-1]) if progress else 0.0
    reasoning = [
        {
            "step_index": item["step_index"],
            "turn": item["turn"],
            "status": item["status"],
            "destination": item.get("destination_description", ""),
            "resolved_heading_deg": item["resolved_heading_deg"],
        }
        for item in step_records
    ]
    return {
        "instr_id": plan.plan_id,
        "trajectory": trajectory,
        "path_corners": [
            (box.tolist(), float(resolved_heading))
            for box, resolved_heading in zip(predicted_boxes, predicted_headings)
        ],
        "progress": progress,
        "gt_progress": progress[:],
        "gt_path_corners": [
            np.asarray(item, dtype=float).tolist() for item in gt_corners
        ],
        "reasoning": reasoning,
        "final_iou": final_iou,
        "path_length": len(predicted_boxes),
        "success": bool(final_iou > 0.3),
        "search_steps": step_records,
        "state_updates": state_updates,
        "terminated": terminated,
        "termination_reason": termination_reason,
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
                raise ValueError(
                    f"instruction plan at line {line_number} has no plan_id"
                )
            turns = raw.get("turns") if isinstance(raw.get("turns"), list) else []
            try:
                plan = InstructionPlan.from_dict(
                    raw["plan"], plan_id=plan_id, turns=turns
                )
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
    max_routes: Optional[int],
    route_range: Optional[Tuple[int, int]],
) -> Tuple[int, Optional[int]]:
    if max_routes is not None:
        if max_routes <= 0 or route_range is not None:
            raise ValueError(
                "max_routes must be positive and cannot be combined with route_range"
            )
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
        "success_rate": (
            float(np.mean([item["success"] for item in values])) if values else 0.0
        ),
        "avg_iou": (
            float(np.mean([item["final_iou"] for item in values])) if values else 0.0
        ),
        "avg_path_length": (
            float(np.mean([item["path_length"] for item in values]))
            if values else 0.0
        ),
        "terminated_routes": sum(
            item.get("terminated") is True for item in values
        ),
    }


def run_instruction_plans(
    anno_dir: Path = ANNO_DIR,
    dataset_dir: Path = DATASET_DIR,
    split: str = SPLIT,
    plans_path: Path = PLANS_PATH,
    out_dir: Path = OUT_DIR,
    *,
    search_scale: float = SEARCH_SCALE_FACTOR,
    step_meters: float = SEARCH_STEP_METERS,
    max_search_steps: int = MAX_SEARCH_STEPS,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    max_routes: Optional[int] = None,
    route_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    selected_start, selected_end = _selection_bounds(max_routes, route_range)
    runtime = _load_baseline_engine()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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
            raise ValueError(
                f"selected route {plan_id} has no InstructionPlan in {plans_path}"
            )
        plan = plan_records[plan_id]["plan"]
        route_dir = out_dir / plan_id
        _console(
            "ROUTE/START",
            f"route={plan_id} steps={len(project_instruction_plan(plan))}",
        )
        prediction = execute_instruction_plan(
            plan,
            observation,
            out_dir=route_dir,
            search_scale=search_scale,
            step_meters=step_meters,
            max_search_steps=max_search_steps,
            api_key=api_key,
            url=url,
            model=model,
            engine=runtime,
        )
        predictions[plan_id] = prediction
        route_dir.mkdir(parents=True, exist_ok=True)
        (route_dir / "prediction.json").write_text(
            json.dumps(_jsonable(prediction), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        boxes = [
            np.asarray(item[0], dtype=float)
            for item in prediction["path_corners"]
        ]
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
            environment_metrics, _ = environment.eval_metrics(
                predictions, human_att_eval=False
            )
        except Exception as exc:
            environment_metrics = {"evaluation_error": str(exc)}
    else:
        environment_metrics = {"evaluation_skipped": "no_predictions"}
    metrics["environment_metrics"] = _jsonable(environment_metrics)
    output_path = out_dir / "turn_and_crop_eval_results_full_traj.json"
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
        description="Execute InstructionPlans with baseline Search/Confirmation.",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-routes", type=_positive_int, default=None)
    selection.add_argument("--route-range", type=_parse_route_range, default=None)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    _, metrics = run_instruction_plans(
        max_routes=args.max_routes,
        route_range=args.route_range,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
