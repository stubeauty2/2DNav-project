"""Execute flat InstructionPlans in legacy or same-heading-window mode.

Legacy mode preserves the one-SearchStep-per-INS behavior.  ``ins-window``
mode keeps parsed event order, splits before a real direction change, reuses
the multi-scale baseline search as an event evidence engine, and commits only
the longest prefix accepted by window-level progress confirmation.
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
    EventType,
    FORMAT_VERSION,
    InstructionEvent,
    InstructionPlan,
    STATE_EVENT_TYPES,
    normalize_angle,
)
from ins_window_runtime import (
    HEADING_CHANGE_TOLERANCE_DEG,
    ConfirmationResult,
    EventCheck,
    EventEvidence,
    ExecutionWindow,
    INSSession,
    TentativeTrace,
    WindowProgressReport,
    aggregate_confirmation,
    apply_event_heading,
    angular_distance,
    build_ins_sessions,
    event_context_text,
    longest_contiguous_prefix,
)
from search_plan_adapter import (
    HeadingState,
    SearchStep,
    project_instruction_plan,
    resolve_step_heading,
)


_EXTERNAL_BASELINE_SEARCH_PATH = (
    WORKSPACE_ROOT / "PSC-AVDN-baseline" / "script" / "ANDH-Full"
    / "Search_Confirmation.py"
)
_IN_TREE_BASELINE_SEARCH_PATH = (
    WORKSPACE_ROOT / "PSC-AVDN" / "script" / "ANDH"
    / "Search_Confirmation.py"
)
BASELINE_SEARCH_PATH = Path(os.getenv(
    "PSC_SEARCH_BASELINE_PATH",
    str(
        _EXTERNAL_BASELINE_SEARCH_PATH
        if _EXTERNAL_BASELINE_SEARCH_PATH.exists()
        else _IN_TREE_BASELINE_SEARCH_PATH
    ),
))
PLANS_PATH = Path(os.getenv(
    "INSTRUCTION_PLANS_PATH",
    str(WORKSPACE_ROOT / "out" / "preds_out_full_sample2" / "instruction_plans.jsonl"),
))
ANNO_DIR = WORKSPACE_ROOT / "datasets" / "sample2"
DATASET_DIR = WORKSPACE_ROOT / "datasets" / "sample2"
SPLIT = "test_unseen_full"
OUT_DIR = (
    WORKSPACE_ROOT / "out" / "preds" / "andh_full_sample2" / "search_output_no_confirmation"
)
WINDOWED_OUT_DIR = (
    WORKSPACE_ROOT / "out" / "preds" / "andh_full_sample2_windowed"
    / "search_output"
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
HEADING_TOLERANCE_DEG = float(os.getenv(
    "ROUTE_HEADING_TOLERANCE_DEG",
    str(HEADING_CHANGE_TOLERANCE_DEG),
))
MAX_WINDOW_RUNS = int(os.getenv("ROUTE_MAX_WINDOW_RUNS", "2"))
EVENT_CLEARANCE_METERS = float(os.getenv("ROUTE_EVENT_CLEARANCE_METERS", "30"))


@dataclass
class BaselineSearchResult:
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
    previous_grid_5x5: Optional[List[List[str]]] = None
    grid_heading_deg: Optional[float] = None

    @classmethod
    def empty(cls) -> "RouteViewHistory":
        return cls(corners=[], patches=[], view_ids=[])

    def prepare_window(self, heading_deg: float) -> None:
        if (
            self.grid_heading_deg is not None
            and angular_distance(self.grid_heading_deg, heading_deg)
            > HEADING_TOLERANCE_DEG
        ):
            self.previous_grid_5x5 = None
        self.grid_heading_deg = float(heading_deg)


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
    history_corners: Optional[Sequence[np.ndarray]] = None,
    history_patches: Optional[Sequence[np.ndarray]] = None,
    previous_grid_5x5: Optional[Sequence[Sequence[str]]] = None,
    event_context: str = "",
    enable_confirmation: bool = True,
) -> BaselineSearchResult:
    """Call the baseline search unchanged and attach its structured JSON log."""
    runtime = engine or _load_baseline_engine()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    search_kwargs = dict(
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
        enable_confirmation=bool(enable_confirmation),
    )
    if (
        history_corners
        or history_patches
        or previous_grid_5x5 is not None
        or event_context
    ):
        search_kwargs.update({
            "history_corners": list(history_corners or []),
            "history_patches": list(history_patches or []),
            "previous_grid_5x5": previous_grid_5x5,
            "event_context": event_context or None,
        })
    found, destination, corners, zoom_index = (
        runtime.search_and_reach_destination(**search_kwargs)
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


def _bbox_values(value: Any) -> Optional[List[float]]:
    if isinstance(value, str):
        parts = re.findall(r"-?\d+(?:\.\d+)?", value)
        values = [float(item) for item in parts[:4]]
    elif isinstance(value, (list, tuple)):
        try:
            values = [float(item) for item in value[:4]]
        except (TypeError, ValueError):
            return None
    else:
        return None
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def _latest_grid(search_log: Dict[str, Any]) -> Optional[List[List[str]]]:
    latest = None
    for attempt in search_log.get("steps", []):
        if not isinstance(attempt, dict):
            continue
        for key in ("qwen", "confirm"):
            value = attempt.get(key)
            grid = value.get("grid_5x5") if isinstance(value, dict) else None
            if isinstance(grid, list) and len(grid) == 5:
                latest = grid
    return latest


def _detection_and_confirmation(
    search_log: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    detection: Dict[str, Any] = {}
    confirmation: Dict[str, Any] = {}
    detection_k: Optional[int] = None
    for attempt in search_log.get("steps", []):
        if not isinstance(attempt, dict):
            continue
        qwen = attempt.get("qwen")
        if isinstance(qwen, dict) and qwen.get("dest_present"):
            detection = qwen
            detection_k = int(attempt.get("k") or 0)
        confirm = attempt.get("confirm")
        if (
            isinstance(confirm, dict)
            and detection_k is not None
            and int(attempt.get("k") or 0) == detection_k
        ):
            confirmation = confirm
    return detection, confirmation


def _observation_view_records(
    search_log: Dict[str, Any],
    *,
    window_id: str,
    run_index: int,
    event_id: str,
    candidate_index: int,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    seen = set()
    for attempt in search_log.get("steps", []):
        if not isinstance(attempt, dict):
            continue
        k = int(attempt.get("k") or 0)
        paths = attempt.get("view_paths")
        if not isinstance(paths, dict):
            continue
        for label, path in paths.items():
            if not path or str(path) in seen:
                continue
            seen.add(str(path))
            records.append({
                "view_id": str(path),
                "window_id": window_id,
                "run_index": run_index,
                "event_id": event_id,
                "candidate_index": candidate_index,
                "search_index": k,
                "view_type": str(label),
                "path": str(path),
                "position": attempt.get("pos"),
            })
    return records


def _update_route_history(
    history: RouteViewHistory,
    result: BaselineSearchResult,
    observation: Dict[str, Any],
    runtime,
    *,
    heading_deg: float,
    scale_factor: float,
) -> None:
    grid = _latest_grid(result.search_log)
    if grid is not None:
        history.previous_grid_5x5 = grid
    corner_builder = getattr(runtime, "generate_view_corners_with_scale", None)
    cv2_module = getattr(runtime, "cv2", None)
    if not callable(corner_builder) or cv2_module is None:
        return
    for attempt in result.search_log.get("steps", []):
        if not isinstance(attempt, dict) or not isinstance(attempt.get("qwen"), dict):
            continue
        paths = attempt.get("view_paths") or {}
        main_path = str(paths.get("main") or "")
        position = attempt.get("pos") or []
        if not main_path or main_path in history.view_ids or len(position) < 2:
            continue
        patch = cv2_module.imread(main_path)
        if patch is None:
            continue
        try:
            corners = corner_builder(
                [float(position[0]), float(position[1])],
                observation,
                scale_factor=float(scale_factor),
                angle_deg=float(heading_deg),
            )
        except Exception:
            continue
        history.corners.append(np.asarray(corners, dtype=float))
        history.patches.append(patch)
        history.view_ids.append(main_path)


def _materialize_event_end_view(
    history: RouteViewHistory,
    runtime,
    observation: Dict[str, Any],
    *,
    route_dir: Path,
    window_id: str,
    run_index: int,
    event_id: str,
    position: Sequence[float],
    heading_deg: float,
    scale_factor: float,
) -> Optional[Dict[str, Any]]:
    corner_builder = getattr(runtime, "generate_view_corners_with_scale", None)
    renderer = getattr(runtime, "create_view_image", None)
    if not callable(corner_builder) or not callable(renderer):
        return None
    path = route_dir / f"{window_id}_r{run_index}_{event_id}_event_end.jpg"
    try:
        corners = np.asarray(corner_builder(
            list(position),
            observation,
            scale_factor=float(scale_factor),
            angle_deg=float(heading_deg),
        ), dtype=float)
        patch = renderer(
            corners,
            observation,
            save_path=str(path),
            out_px=768,
        )
    except Exception:
        return None
    if patch is None:
        return None
    history.corners.append(corners)
    history.patches.append(patch)
    history.view_ids.append(str(path))
    return {
        "view_id": str(path),
        "window_id": window_id,
        "run_index": run_index,
        "event_id": event_id,
        "candidate_index": None,
        "search_index": None,
        "view_type": "event_end",
        "path": str(path),
        "position": list(position),
    }


def _event_description(
    event: InstructionEvent,
    window: ExecutionWindow,
) -> str:
    description = window.event_descriptions.get(event.event_id, "")
    parts = [
        f"Active event {event.event_id} is {event.event_type.value}.",
        f"Localize only this event target: {description or event.entity_ref}.",
    ]
    if event.side:
        parts.append(f"Required relative side: {event.side}.")
    if event.count > 1:
        parts.append(
            f"This call is for one of {event.count} distinct ordered instances."
        )
    return " ".join(parts)


def _event_motion_position(
    event: InstructionEvent,
    candidate_position: Sequence[float],
    heading_deg: float,
) -> List[float]:
    """Convert a localized object into the event's tentative end pose."""
    if event.event_type == EventType.APPROACH:
        return _move_forward_geo(
            candidate_position,
            normalize_angle(float(heading_deg) + 180.0),
            EVENT_CLEARANCE_METERS,
        )
    if event.event_type in {
        EventType.PASS,
        EventType.CROSS,
        EventType.GO_THROUGH,
        EventType.ENTER,
        EventType.EXIT,
        EventType.FOLLOW,
    }:
        return _move_forward_geo(
            candidate_position,
            heading_deg,
            EVENT_CLEARANCE_METERS,
        )
    return list(candidate_position)


def _event_evidence(
    event: InstructionEvent,
    result: BaselineSearchResult,
    *,
    window_id: str,
    candidate_index: int,
) -> EventEvidence:
    detection, confirmation = _detection_and_confirmation(result.search_log)
    bbox = _bbox_values(detection.get("bbox_2d"))
    confirm_present = bool(confirmation.get("dest_present"))
    view_ids = [
        str(path)
        for attempt in result.search_log.get("steps", [])
        if isinstance(attempt, dict)
        for path in (attempt.get("view_paths") or {}).values()
        if path
    ]
    found = bool(
        result.found and result.destination_position is not None and bbox is not None
    )
    explanation_parts = [str(detection.get("reason") or "").strip()]
    if confirmation:
        explanation_parts.append(
            "Confirmation: " + str(confirmation.get("reason") or "").strip()
        )
    return EventEvidence(
        event_id=event.event_id,
        event_type=event.event_type.value,
        candidate_id=f"{window_id}_{event.event_id}_c{candidate_index}",
        candidate_index=candidate_index,
        found=found,
        destination_position=(
            list(result.destination_position)
            if result.destination_position is not None else None
        ),
        bbox_2d=bbox,
        confidence=float(detection.get("confidence") or 0.0),
        confirmation_present=confirm_present,
        confirmation_confidence=float(confirmation.get("confidence") or 0.0),
        evidence_view_ids=list(dict.fromkeys(view_ids)),
        search_log_path=result.search_log_path,
        attempts=[
            _compact_attempt(item)
            for item in result.search_log.get("steps", [])
        ],
        progress_claim="COMPLETED" if found else "NOT_FOUND",
        explanation=" ".join(item for item in explanation_parts if item),
    )


def _side_matches(side: str, bbox: Optional[Sequence[float]]) -> bool:
    side_text = str(side or "").strip().lower()
    if not side_text or bbox is None:
        return True
    center_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
    if "left" in side_text:
        return center_x < 384.0
    if "right" in side_text:
        return center_x > 384.0
    return True


def _run_window_search(
    session: INSSession,
    window: ExecutionWindow,
    observation: Dict[str, Any],
    *,
    run_index: int,
    start_position: Sequence[float],
    confirmed_event_ids: Sequence[str],
    correction: str,
    rejected_candidates: Sequence[str],
    route_dir: Path,
    history: RouteViewHistory,
    search_scale: float,
    step_meters: float,
    max_search_steps: int,
    api_key: str,
    url: str,
    model: str,
    enable_confirmation: bool,
    runtime,
) -> Tuple[WindowProgressReport, List[Dict[str, Any]]]:
    ordered_ids = [event.event_id for event in window.events]
    claimed = list(confirmed_event_ids)
    confirmed_set = set(confirmed_event_ids)
    tentative_position = list(start_position)
    traces: List[TentativeTrace] = []
    evidences: List[EventEvidence] = []
    observation_views: List[Dict[str, Any]] = []
    base_excluded = list(rejected_candidates)
    stop_reason = "WINDOW_COMPLETE"
    run_status = "COMPLETE"

    history.prepare_window(window.travel_heading_deg)
    for event in window.events:
        if event.event_id in confirmed_set:
            continue
        trace_id = f"{window.window_id}_r{run_index}_a{len(traces) + 1}"
        if event.event_type == EventType.TURN:
            claimed.append(event.event_id)
            traces.append(TentativeTrace(
                trace_id=trace_id,
                event_ids=[event.event_id],
                position=list(tentative_position),
                heading_deg=window.travel_heading_deg,
            ))
            continue
        if event.event_type == EventType.MOVE and not event.requires_visual_search:
            has_later_visual = any(
                later.requires_visual_search
                and later.event_id not in confirmed_set
                for later in window.events[
                    list(window.events).index(event) + 1:
                ]
            )
            if not has_later_visual:
                tentative_position = _move_forward_geo(
                    tentative_position,
                    window.travel_heading_deg,
                    MOTION_ONLY_METERS,
                )
            claimed.append(event.event_id)
            traces.append(TentativeTrace(
                trace_id=trace_id,
                event_ids=[event.event_id],
                position=list(tentative_position),
                heading_deg=window.travel_heading_deg,
            ))
            continue
        if not event.requires_visual_search:
            claimed.append(event.event_id)
            traces.append(TentativeTrace(
                trace_id=trace_id,
                event_ids=[event.event_id],
                position=list(tentative_position),
                heading_deg=window.travel_heading_deg,
            ))
            continue

        event_positions: List[List[float]] = []
        # Exclusions are local to this event. Separate entity mentions may
        # intentionally resolve to the same physical object; only count>1
        # requires distinct candidates.
        event_excluded = list(base_excluded)
        required_count = max(1, int(event.count or 1))
        event_complete = True
        for candidate_index in range(1, required_count + 1):
            context = event_context_text(
                session,
                window,
                event,
                confirmed_event_ids=claimed,
                correction=correction,
                excluded_candidates=event_excluded,
            )
            search_id = (
                f"{window.window_id}_r{run_index}_{event.event_id}"
                f"_c{candidate_index}"
            )
            result = search_and_reach_destination(
                search_id,
                observation,
                _event_description(event, window),
                context,
                tentative_position,
                window.travel_heading_deg,
                search_scale,
                route_dir,
                step_meters=step_meters,
                max_steps=max_search_steps,
                api_key=api_key,
                url=url,
                model=model,
                enable_confirmation=enable_confirmation,
                engine=runtime,
                history_corners=history.corners,
                history_patches=history.patches,
                previous_grid_5x5=history.previous_grid_5x5,
                event_context=context,
            )
            evidence = _event_evidence(
                event,
                result,
                window_id=window.window_id,
                candidate_index=candidate_index,
            )
            evidences.append(evidence)
            observation_views.extend(_observation_view_records(
                result.search_log,
                window_id=window.window_id,
                run_index=run_index,
                event_id=event.event_id,
                candidate_index=candidate_index,
            ))
            _update_route_history(
                history,
                result,
                observation,
                runtime,
                heading_deg=window.travel_heading_deg,
                scale_factor=search_scale,
            )
            if not evidence.found or evidence.destination_position is None:
                event_complete = False
                break
            if any(
                _haversine_m(evidence.destination_position, previous) < 3.0
                for previous in event_positions
            ):
                evidence.found = False
                evidence.progress_claim = "NOT_FOUND"
                evidence.explanation += " Candidate duplicates an earlier instance."
                event_complete = False
                break
            event_positions.append(list(evidence.destination_position))
            tentative_position = _event_motion_position(
                event,
                evidence.destination_position,
                window.travel_heading_deg,
            )
            event_excluded.append(
                f"{evidence.candidate_id} bbox={evidence.bbox_2d}"
            )

        if event.event_type == EventType.AVOID:
            event_complete = False
            stop_reason = "NEEDS_REPLAN"
        if not event_complete:
            run_status = "PARTIAL" if len(claimed) > len(confirmed_set) else "BLOCKED"
            if stop_reason != "NEEDS_REPLAN":
                stop_reason = "NOT_FOUND"
            break
        claimed.append(event.event_id)
        end_view = _materialize_event_end_view(
            history,
            runtime,
            observation,
            route_dir=route_dir,
            window_id=window.window_id,
            run_index=run_index,
            event_id=event.event_id,
            position=tentative_position,
            heading_deg=window.travel_heading_deg,
            scale_factor=search_scale,
        )
        if end_view is not None:
            observation_views.append(end_view)
        traces.append(TentativeTrace(
            trace_id=trace_id,
            event_ids=[event.event_id],
            position=list(tentative_position),
            heading_deg=window.travel_heading_deg,
            view_ids=[
                view_id
                for evidence in evidences if evidence.event_id == event.event_id
                for view_id in evidence.evidence_view_ids
            ] + ([end_view["view_id"]] if end_view is not None else []),
        ))

    active = next((event_id for event_id in ordered_ids if event_id not in claimed), None)
    if active is not None and run_status == "COMPLETE":
        run_status = "PARTIAL"
        stop_reason = "SEARCH_BUDGET"
    completed_now = len(claimed) - len(confirmed_event_ids)
    evidence_summary = "; ".join(
        (
            f"{item.event_id}/c{item.candidate_index}:"
            f"search={item.found},confirm={item.confirmation_present},"
            f"confidence={item.confidence:.2f}"
        )
        for item in evidences
    )
    progress_summary = (
        f"Window {window.window_id} run {run_index}: completed {completed_now} "
        f"new events; next={active or 'none'}; stop={stop_reason}. "
        f"Evidence: {evidence_summary or 'geometry-only window'}."
    )
    return WindowProgressReport(
        window_id=window.window_id,
        run_index=run_index,
        run_status=run_status,
        claimed_completed_event_ids=claimed,
        active_event_id=active,
        tentative_trace=traces,
        event_evidence=evidences,
        progress_summary=progress_summary,
        stop_reason=stop_reason,
        correction_applied=correction,
    ), observation_views


def _window_event_checks(
    window: ExecutionWindow,
    report: WindowProgressReport,
    previously_confirmed: Sequence[str],
    start_position: Sequence[float],
) -> List[EventCheck]:
    claimed = list(report.claimed_completed_event_ids)
    ordered = [event.event_id for event in window.events]
    evidence_by_event: Dict[str, List[EventEvidence]] = {}
    for evidence in report.event_evidence:
        evidence_by_event.setdefault(evidence.event_id, []).append(evidence)
    checks: List[EventCheck] = []
    previous_set = set(previously_confirmed)
    for index, event in enumerate(window.events):
        prefix_claimed = claimed[:index + 1] == ordered[:index + 1]
        if event.event_id in previous_set:
            checks.append(EventCheck(
                event_id=event.event_id,
                same_candidate=True,
                order_valid=True,
                motion_valid=True,
                visual_valid=True,
                result="PASS",
                explanation="Accepted by an earlier run of the same window.",
            ))
            continue
        if not event.requires_visual_search:
            valid = event.event_id in claimed and prefix_claimed
            checks.append(EventCheck(
                event_id=event.event_id,
                same_candidate=True,
                order_valid=prefix_claimed,
                motion_valid=valid,
                visual_valid=True,
                result="PASS" if valid else "FAIL",
                explanation=(
                    "Controller trace matches the same-heading window."
                    if valid else "The control event was not completed in order."
                ),
            ))
            continue

        evidences = evidence_by_event.get(event.event_id, [])
        enough = len(evidences) >= max(1, int(event.count or 1))
        found = enough and all(item.found for item in evidences)
        confirmed = enough and all(item.confirmation_present for item in evidences)
        side_valid = enough and all(
            _side_matches(event.side, item.bbox_2d) for item in evidences
        )
        end_position = next(
            (
                trace.position for trace in reversed(report.tentative_trace)
                if event.event_id in trace.event_ids
            ),
            None,
        )
        moved = bool(
            end_position is not None
            and _haversine_m(start_position, end_position) >= 1.0
        )
        relation_requires_motion = event.event_type in {
            EventType.PASS,
            EventType.CROSS,
            EventType.GO_THROUGH,
            EventType.ENTER,
            EventType.EXIT,
            EventType.FOLLOW,
        }
        motion_valid = found and (moved or not relation_requires_motion)
        if event.event_type == EventType.AVOID:
            motion_valid = False
        valid = (
            event.event_id in claimed
            and prefix_claimed
            and found
            and confirmed
            and side_valid
            and motion_valid
        )
        explanation = (
            f"candidates={len(evidences)}/{max(1, event.count)}, "
            f"confirmed={confirmed}, side={side_valid}, motion={motion_valid}."
        )
        checks.append(EventCheck(
            event_id=event.event_id,
            same_candidate=confirmed,
            order_valid=prefix_claimed,
            motion_valid=motion_valid,
            visual_valid=found and confirmed and side_valid,
            result="PASS" if valid else "FAIL",
            explanation=explanation,
        ))
    return checks


def _trace_by_id(
    report: WindowProgressReport,
    trace_id: Optional[str],
) -> Optional[TentativeTrace]:
    return next(
        (trace for trace in report.tentative_trace if trace.trace_id == trace_id),
        None,
    )


def _search_only_commit_decision(
    window: ExecutionWindow,
    report: WindowProgressReport,
    previously_accepted: Sequence[str],
) -> ConfirmationResult:
    """Commit Search's contiguous prefix when Confirmation is disabled.

    ``ConfirmationResult`` is reused internally as the controller decision
    envelope, but this path performs no confirmation checks or model calls.
    """
    ordered_ids = [event.event_id for event in window.events]
    accepted = longest_contiguous_prefix(
        ordered_ids,
        report.claimed_completed_event_ids,
    )
    if list(report.claimed_completed_event_ids) != ordered_ids[:len(accepted)]:
        accepted = list(previously_accepted)

    if accepted == ordered_ids and report.run_status == "COMPLETE":
        verdict = "PASS"
    elif accepted:
        verdict = "PARTIAL"
    else:
        verdict = "REJECT"

    accepted_set = set(accepted)
    commit_trace_id: Optional[str] = None
    for trace in report.tentative_trace:
        if trace.event_ids and set(trace.event_ids).issubset(accepted_set):
            commit_trace_id = trace.trace_id

    confirmed_cursor = window.start_event_index
    if accepted:
        index_by_id = {
            event.event_id: event_index
            for event, event_index in zip(window.events, window.event_indices)
        }
        confirmed_cursor = index_by_id[accepted[-1]] + 1

    if verdict == "PASS":
        correction = ""
        summary = (
            f"Confirmation disabled: committed all {len(accepted)} events "
            f"reported complete by Search in {window.window_id}."
        )
    else:
        next_event = next(
            (event_id for event_id in ordered_ids if event_id not in accepted_set),
            report.active_event_id,
        )
        correction = (
            f"Confirmation disabled; retry Search from {next_event or 'window end'} "
            f"after stop={report.stop_reason}."
        )
        summary = (
            f"Confirmation disabled: committed {len(accepted)} leading events "
            f"reported complete by Search; stop={report.stop_reason}."
        )

    return ConfirmationResult(
        window_id=window.window_id,
        run_index=report.run_index,
        verdict=verdict,
        accepted_event_ids=tuple(accepted),
        confirmed_event_cursor=confirmed_cursor,
        commit_trace_id=commit_trace_id,
        event_checks=tuple(),
        correction=correction,
        progress_summary=summary,
    )


class INSWindowScheduler:
    """Compile parsed INS turns into fixed-travel-heading execution windows."""

    def __init__(self, heading_tolerance_deg: float = HEADING_TOLERANCE_DEG):
        self.heading_tolerance_deg = float(heading_tolerance_deg)

    def build(
        self,
        plan: InstructionPlan,
        heading: HeadingState,
    ) -> List[INSSession]:
        sessions, _ = build_ins_sessions(
            plan,
            heading,
            heading_tolerance_deg=self.heading_tolerance_deg,
        )
        return sessions


class EventSearchAdapter:
    """Run the existing multi-scale search as an event-evidence engine."""

    def __init__(
        self,
        observation: Dict[str, Any],
        route_dir: Path,
        runtime,
        *,
        search_scale: float,
        step_meters: float,
        max_search_steps: int,
        api_key: str,
        url: str,
        model: str,
        enable_confirmation: bool,
    ):
        self.observation = observation
        self.route_dir = route_dir
        self.runtime = runtime
        self.search_scale = search_scale
        self.step_meters = step_meters
        self.max_search_steps = max_search_steps
        self.api_key = api_key
        self.url = url
        self.model = model
        self.enable_confirmation = bool(enable_confirmation)
        self.history = RouteViewHistory.empty()

    def run(
        self,
        session: INSSession,
        window: ExecutionWindow,
        *,
        run_index: int,
        start_position: Sequence[float],
        confirmed_event_ids: Sequence[str],
        correction: str,
        rejected_candidates: Sequence[str],
    ) -> Tuple[WindowProgressReport, List[Dict[str, Any]]]:
        return _run_window_search(
            session,
            window,
            self.observation,
            run_index=run_index,
            start_position=start_position,
            confirmed_event_ids=confirmed_event_ids,
            correction=correction,
            rejected_candidates=rejected_candidates,
            route_dir=self.route_dir,
            history=self.history,
            search_scale=self.search_scale,
            step_meters=self.step_meters,
            max_search_steps=self.max_search_steps,
            api_key=self.api_key,
            url=self.url,
            model=self.model,
            enable_confirmation=self.enable_confirmation,
            runtime=self.runtime,
        )


class ProgressConfirmationAdapter:
    """Validate per-event evidence and accept only a contiguous event prefix."""

    def confirm(
        self,
        window: ExecutionWindow,
        report: WindowProgressReport,
        *,
        previously_confirmed: Sequence[str],
        start_position: Sequence[float],
    ) -> ConfirmationResult:
        checks = _window_event_checks(
            window,
            report,
            previously_confirmed,
            start_position,
        )
        return aggregate_confirmation(window, report, checks)


def execute_instruction_plan_legacy(
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
    enable_confirmation: bool = True,
    engine=None,
) -> Dict[str, Any]:
    """Execute projected INS steps with the unchanged legacy behavior."""
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
            enable_confirmation=enable_confirmation,
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


def execute_instruction_plan_windowed(
    plan: InstructionPlan,
    observation: Dict[str, Any],
    *,
    out_dir: Optional[Path] = None,
    search_scale: float = SEARCH_SCALE_FACTOR,
    step_meters: float = SEARCH_STEP_METERS,
    max_search_steps: int = MAX_SEARCH_STEPS,
    max_window_runs: int = MAX_WINDOW_RUNS,
    heading_tolerance_deg: float = HEADING_TOLERANCE_DEG,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    enable_confirmation: bool = True,
    engine=None,
) -> Dict[str, Any]:
    """Execute one INS at a time through same-heading search windows."""
    validation_errors = plan.validate()
    if validation_errors:
        raise ValueError("invalid InstructionPlan: " + "; ".join(validation_errors))
    gt_corners = observation.get("gt_path_corners") or []
    if not gt_corners:
        raise ValueError("observation has no gt_path_corners start pose")
    if max_window_runs <= 0:
        raise ValueError("max_window_runs must be positive")

    runtime = engine or _load_baseline_engine()
    route_dir = Path(out_dir or WINDOWED_OUT_DIR / plan.plan_id)
    route_dir.mkdir(parents=True, exist_ok=True)
    template_footprint = np.asarray(gt_corners[0], dtype=float)
    position = _corners_center(template_footprint)
    starting_heading = float(observation.get("starting_angle") or 0.0)
    heading = HeadingState(starting_heading, starting_heading).normalized()
    scheduler = INSWindowScheduler(heading_tolerance_deg)
    sessions = scheduler.build(plan, heading)
    search_adapter = EventSearchAdapter(
        observation,
        route_dir,
        runtime,
        search_scale=search_scale,
        step_meters=step_meters,
        max_search_steps=max_search_steps,
        api_key=api_key,
        url=url,
        model=model,
        enable_confirmation=enable_confirmation,
    )
    confirmation_adapter = ProgressConfirmationAdapter()
    execution_windows = [
        window for session in sessions for window in session.windows
    ]
    physical_boxes: List[np.ndarray] = [template_footprint.copy()]
    physical_headings: List[float] = [heading.travel_heading_deg]
    physical_trajectory: List[Dict[str, Any]] = [{
        "trace_id": "start",
        "window_id": None,
        "event_ids": [],
        "position": list(position),
        "heading_deg": heading.travel_heading_deg,
    }]
    search_runs: List[Dict[str, Any]] = []
    confirmation_runs: List[Dict[str, Any]] = []
    event_progress: List[Dict[str, Any]] = []
    observation_views: List[Dict[str, Any]] = []
    search_steps: List[Dict[str, Any]] = []
    reasoning: List[Dict[str, Any]] = []
    terminated = False
    termination_reason = "completed_all_ins_sessions"

    for session in sessions:
        if not session.windows:
            reasoning.append({
                "session_id": session.session_id,
                "turn": session.turn,
                "status": "state_only",
                "summary": "Applied state events without visual search.",
            })
            continue
        for window in session.windows:
            window_start_position = list(position)
            accepted_prefix: List[str] = []
            correction = ""
            rejected_candidates: List[str] = []
            window_complete = False
            last_confirmation: Optional[ConfirmationResult] = None
            last_commit_decision: Optional[ConfirmationResult] = None
            window_run_indices: List[int] = []

            for run_index in range(1, max_window_runs + 1):
                report, views = search_adapter.run(
                    session,
                    window,
                    run_index=run_index,
                    start_position=position,
                    confirmed_event_ids=accepted_prefix,
                    correction=correction,
                    rejected_candidates=rejected_candidates,
                )
                window_run_indices.append(len(search_runs))
                search_runs.append(report.to_dict())
                observation_views.extend(views)
                if enable_confirmation:
                    commit_decision = confirmation_adapter.confirm(
                        window,
                        report,
                        previously_confirmed=accepted_prefix,
                        start_position=position,
                    )
                    confirmation_runs.append(commit_decision.to_dict())
                    last_confirmation = commit_decision
                else:
                    commit_decision = _search_only_commit_decision(
                        window,
                        report,
                        accepted_prefix,
                    )
                last_commit_decision = commit_decision

                previous_accepted = set(accepted_prefix)
                accepted_prefix = list(commit_decision.accepted_event_ids)
                new_accepted = [
                    event_id for event_id in accepted_prefix
                    if event_id not in previous_accepted
                ]
                commit_trace = _trace_by_id(
                    report,
                    commit_decision.commit_trace_id,
                )
                if commit_decision.verdict in {"PASS", "PARTIAL"}:
                    if commit_trace is not None:
                        position = list(commit_trace.position)
                    event_by_id = {
                        event.event_id: event for event in window.events
                    }
                    for event_id in new_accepted:
                        heading = apply_event_heading(event_by_id[event_id], heading)
                        event_trace = next(
                            (
                                trace for trace in report.tentative_trace
                                if event_id in trace.event_ids
                            ),
                            commit_trace,
                        )
                        event_progress.append({
                            "event_id": event_id,
                            "window_id": window.window_id,
                            "turn": window.turn,
                            "status": (
                                "CONFIRMED"
                                if enable_confirmation else "SEARCH_ACCEPTED"
                            ),
                            "run_index": run_index,
                            "position": list(
                                event_trace.position
                                if event_trace is not None else position
                            ),
                            "heading_deg": heading.travel_heading_deg,
                        })
                    accepted_set = set(accepted_prefix)
                    for trace in report.tentative_trace:
                        if (
                            trace.event_ids
                            and set(trace.event_ids).issubset(accepted_set)
                            and any(item in new_accepted for item in trace.event_ids)
                        ):
                            if (
                                _haversine_m(
                                    physical_trajectory[-1]["position"],
                                    trace.position,
                                ) > 0.01
                                or angular_distance(
                                    physical_headings[-1], trace.heading_deg
                                ) > 0.1
                            ):
                                physical_boxes.append(_translate_footprint(
                                    template_footprint,
                                    trace.position,
                                ))
                                physical_headings.append(trace.heading_deg)
                                physical_trajectory.append({
                                    "trace_id": trace.trace_id,
                                    "window_id": window.window_id,
                                    "event_ids": list(trace.event_ids),
                                    "position": list(trace.position),
                                    "heading_deg": trace.heading_deg,
                                })
                if commit_decision.verdict == "PASS":
                    window_complete = True
                    break
                correction = commit_decision.correction
                if enable_confirmation:
                    rejected_candidates.extend(
                        f"{item.candidate_id} bbox={item.bbox_2d}"
                        for item in report.event_evidence
                        if not item.confirmation_present
                    )

            status = (
                "found" if window_complete
                else "partial" if accepted_prefix
                else "not_found"
            )
            search_steps.append({
                **window.to_dict(),
                "step_index": len(search_steps) + 1,
                "source_text": session.instruction_text,
                "source_event_ids": [event.event_id for event in window.events],
                "source_event_types": [
                    event.event_type.value for event in window.events
                ],
                "resolved_heading_deg": window.travel_heading_deg,
                "position_before": window_start_position,
                "position_after": list(position),
                "status": status,
                "run_indices": window_run_indices,
                "confirmation_enabled": bool(enable_confirmation),
                "confirmation": (
                    last_confirmation.to_dict() if last_confirmation else None
                ),
                "commit_decision": (
                    last_commit_decision.to_dict()
                    if last_commit_decision else None
                ),
                "search_attempts": [
                    attempt
                    for run_index_value in window_run_indices
                    for evidence in search_runs[run_index_value]["event_evidence"]
                    for attempt in evidence["attempts"]
                ],
            })
            reasoning.append({
                "window_id": window.window_id,
                "turn": window.turn,
                "status": status,
                "accepted_event_ids": list(accepted_prefix),
                "resolved_heading_deg": window.travel_heading_deg,
                "summary": (
                    last_commit_decision.progress_summary
                    if last_commit_decision else "No commit decision was produced."
                ),
            })
            if not window_complete:
                terminated = True
                if enable_confirmation:
                    termination_reason = (
                        f"{window.window_id}_not_confirmed_after_"
                        f"{max_window_runs}_runs"
                    )
                else:
                    termination_reason = (
                        f"{window.window_id}_not_completed_after_"
                        f"{max_window_runs}_search_runs"
                    )
                break
        if terminated:
            break

    unique_views: List[Dict[str, Any]] = []
    seen_view_ids = set()
    for item in observation_views:
        if item["view_id"] in seen_view_ids:
            continue
        seen_view_ids.add(item["view_id"])
        unique_views.append(item)

    goal = np.asarray(gt_corners[-1], dtype=float)
    progress = [polygon_iou(item, goal) for item in physical_boxes]
    trajectory = [_corners_center(item) for item in physical_boxes]
    final_iou = float(progress[-1]) if progress else 0.0
    return {
        "instr_id": plan.plan_id,
        "execution_mode": "ins-window",
        "confirmation_enabled": bool(enable_confirmation),
        "trajectory": trajectory,
        "physical_trajectory": physical_trajectory,
        "path_corners": [
            (box.tolist(), float(resolved_heading))
            for box, resolved_heading in zip(physical_boxes, physical_headings)
        ],
        "progress": progress,
        "gt_progress": progress[:],
        "gt_path_corners": [
            np.asarray(item, dtype=float).tolist() for item in gt_corners
        ],
        "reasoning": reasoning,
        "final_iou": final_iou,
        "path_length": len(physical_boxes),
        "success": bool(final_iou > 0.3),
        "search_steps": search_steps,
        "ins_sessions": [session.to_dict() for session in sessions],
        "execution_windows": [window.to_dict() for window in execution_windows],
        "search_runs": search_runs,
        "confirmation_runs": confirmation_runs,
        "event_progress": event_progress,
        "observation_views": unique_views,
        "state_updates": _state_updates(plan),
        "terminated": terminated,
        "termination_reason": termination_reason,
    }


def execute_instruction_plan(
    plan: InstructionPlan,
    observation: Dict[str, Any],
    *,
    execution_mode: str = "legacy",
    out_dir: Optional[Path] = None,
    search_scale: float = SEARCH_SCALE_FACTOR,
    step_meters: float = SEARCH_STEP_METERS,
    max_search_steps: int = MAX_SEARCH_STEPS,
    max_window_runs: int = MAX_WINDOW_RUNS,
    heading_tolerance_deg: float = HEADING_TOLERANCE_DEG,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    enable_confirmation: bool = True,
    engine=None,
) -> Dict[str, Any]:
    common = dict(
        out_dir=out_dir,
        search_scale=search_scale,
        step_meters=step_meters,
        max_search_steps=max_search_steps,
        api_key=api_key,
        url=url,
        model=model,
        enable_confirmation=enable_confirmation,
        engine=engine,
    )
    if execution_mode == "legacy":
        return execute_instruction_plan_legacy(plan, observation, **common)
    if execution_mode == "ins-window":
        return execute_instruction_plan_windowed(
            plan,
            observation,
            max_window_runs=max_window_runs,
            heading_tolerance_deg=heading_tolerance_deg,
            **common,
        )
    raise ValueError("execution_mode must be legacy or ins-window")


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
    out_dir: Optional[Path] = None,
    *,
    execution_mode: str = "legacy",
    search_scale: float = SEARCH_SCALE_FACTOR,
    step_meters: float = SEARCH_STEP_METERS,
    max_search_steps: int = MAX_SEARCH_STEPS,
    max_window_runs: int = MAX_WINDOW_RUNS,
    heading_tolerance_deg: float = HEADING_TOLERANCE_DEG,
    api_key: str = QWEN_API_KEY,
    url: str = QWEN_URL,
    model: str = QWEN_MODEL,
    enable_confirmation: bool = True,
    max_routes: Optional[int] = None,
    route_range: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    selected_start, selected_end = _selection_bounds(max_routes, route_range)
    runtime = _load_baseline_engine()
    if execution_mode not in {"legacy", "ins-window"}:
        raise ValueError("execution_mode must be legacy or ins-window")
    out_dir = Path(
        out_dir
        or (WINDOWED_OUT_DIR if execution_mode == "ins-window" else OUT_DIR)
    )
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
        if execution_mode == "ins-window":
            preview_sessions, _ = build_ins_sessions(
                plan,
                HeadingState(
                    float(observation.get("starting_angle") or 0.0),
                    float(observation.get("starting_angle") or 0.0),
                ),
                heading_tolerance_deg=heading_tolerance_deg,
            )
            unit_count = sum(len(item.windows) for item in preview_sessions)
            unit_label = "windows"
        else:
            unit_count = len(project_instruction_plan(plan))
            unit_label = "steps"
        _console(
            "ROUTE/START",
            f"route={plan_id} mode={execution_mode} "
            f"{unit_label}={unit_count}",
        )
        prediction = execute_instruction_plan(
            plan,
            observation,
            execution_mode=execution_mode,
            out_dir=route_dir,
            search_scale=search_scale,
            step_meters=step_meters,
            max_search_steps=max_search_steps,
            max_window_runs=max_window_runs,
            heading_tolerance_deg=heading_tolerance_deg,
            api_key=api_key,
            url=url,
            model=model,
            enable_confirmation=enable_confirmation,
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


def _nonnegative_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
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
        description=(
            "Execute InstructionPlans with legacy or same-heading-window "
            "Search/Confirmation."
        ),
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-routes", type=_positive_int, default=None)
    selection.add_argument("--route-range", type=_parse_route_range, default=None)
    parser.add_argument(
        "--execution-mode",
        choices=("legacy", "ins-window"),
        default="legacy",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-window-runs", type=_positive_int, default=MAX_WINDOW_RUNS)
    parser.add_argument(
        "--confirmation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable the secondary visual and window-level Confirmation "
            "stages (default: enabled). Use --no-confirmation to commit "
            "Search's contiguous completed prefix directly."
        ),
    )
    parser.add_argument(
        "--heading-tolerance-deg",
        type=_nonnegative_float,
        default=HEADING_TOLERANCE_DEG,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    _, metrics = run_instruction_plans(
        execution_mode=args.execution_mode,
        out_dir=args.out_dir,
        max_routes=args.max_routes,
        route_range=args.route_range,
        max_window_runs=args.max_window_runs,
        heading_tolerance_deg=args.heading_tolerance_deg,
        enable_confirmation=args.confirmation,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
