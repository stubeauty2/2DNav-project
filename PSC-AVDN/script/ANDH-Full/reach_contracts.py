"""Versioned contracts for the two independent REACH stages."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


REACH_CONTRACT_VERSION = "reach-stage/v1"
RELATION_POLICY_VERSION = "legacy-relation-policy/v1"


def _dict(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


@dataclass
class TargetSelectionRequest:
    event_id: str
    event_type: str
    target_description: str
    entity_kind: str
    event_context: Dict[str, Any]
    view_paths: Dict[str, str]
    current_position: List[float]
    heading_deg: float
    artifact_dir: str
    schema_version: str = REACH_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetSelectionResult:
    status: str
    selected_candidate_id: Optional[str] = None
    selected_candidate: Optional[Dict[str, Any]] = None
    all_candidates: List[Dict[str, Any]] = field(default_factory=list)
    candidate_sheet_paths: List[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    latency_seconds: float = 0.0
    tool_call_count: int = 0
    model_turn_count: int = 0
    provider: str = "tool-agent"
    schema_version: str = REACH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"selected", "absent", "uncertain", "failed"}:
            raise ValueError(f"invalid target-selection status: {self.status}")
        if self.status == "selected" and not self.selected_candidate_id:
            raise ValueError("selected target requires selected_candidate_id")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TargetSelectionResult":
        raw = _dict(value)
        version = str(raw.get("schema_version") or "")
        if version != REACH_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported selection schema_version={version!r}; "
                f"expected {REACH_CONTRACT_VERSION!r}"
            )
        selected = raw.get("selected_candidate")
        return cls(
            status=str(raw.get("status") or "failed"),
            selected_candidate_id=(
                str(raw["selected_candidate_id"])
                if raw.get("selected_candidate_id") is not None
                else None
            ),
            selected_candidate=_dict(selected) if selected is not None else None,
            all_candidates=[_dict(item) for item in raw.get("all_candidates") or []],
            candidate_sheet_paths=[str(item) for item in raw.get("candidate_sheet_paths") or []],
            confidence=float(raw.get("confidence") or 0.0),
            reason=str(raw.get("reason") or ""),
            tool_trace=[_dict(item) for item in raw.get("tool_trace") or []],
            artifacts={str(key): str(item) for key, item in _dict(raw.get("artifacts")).items()},
            latency_seconds=float(raw.get("latency_seconds") or 0.0),
            tool_call_count=int(raw.get("tool_call_count") or 0),
            model_turn_count=int(raw.get("model_turn_count") or 0),
            provider=str(raw.get("provider") or "tool-agent"),
            schema_version=version,
        )

    @classmethod
    def read_json(cls, path: Path) -> "TargetSelectionResult":
        with Path(path).open("r", encoding="utf-8") as file_obj:
            value = json.load(file_obj)
        if not isinstance(value, dict):
            raise ValueError("selection snapshot must contain a JSON object")
        return cls.from_dict(value)

    def write_json(self, path: Path) -> str:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(output.resolve())


@dataclass
class ReachMotionRequest:
    event_id: str
    event_type: str
    relation: str
    side: str
    target_description: str
    selected_candidate: Dict[str, Any]
    main_path: str
    selected_target_path: str
    current_position: List[float]
    heading_deg: float
    artifact_dir: str
    schema_version: str = REACH_CONTRACT_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReachMotionResult:
    status: str
    target_point_2d: Optional[List[float]] = None
    target_position_latlng: Optional[List[float]] = None
    event_position_latlng: Optional[List[float]] = None
    offset_mode: str = "none"
    offset_meters: float = 0.0
    confidence: float = 0.0
    reason: str = ""
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    latency_seconds: float = 0.0
    tool_call_count: int = 0
    relation_policy_version: str = RELATION_POLICY_VERSION
    schema_version: str = REACH_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.status not in {"planned", "failed"}:
            raise ValueError(f"invalid REACH-motion status: {self.status}")
        if self.offset_mode not in {"none", "forward_clearance", "reverse_clearance"}:
            raise ValueError(f"invalid offset_mode: {self.offset_mode}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReachMotionResult":
        raw = _dict(value)
        version = str(raw.get("schema_version") or "")
        if version != REACH_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported motion schema_version={version!r}; "
                f"expected {REACH_CONTRACT_VERSION!r}"
            )
        return cls(
            status=str(raw.get("status") or "failed"),
            target_point_2d=(list(raw["target_point_2d"]) if raw.get("target_point_2d") else None),
            target_position_latlng=(
                list(raw["target_position_latlng"])
                if raw.get("target_position_latlng")
                else None
            ),
            event_position_latlng=(
                list(raw["event_position_latlng"])
                if raw.get("event_position_latlng")
                else None
            ),
            offset_mode=str(raw.get("offset_mode") or "none"),
            offset_meters=float(raw.get("offset_meters") or 0.0),
            confidence=float(raw.get("confidence") or 0.0),
            reason=str(raw.get("reason") or ""),
            tool_trace=[_dict(item) for item in raw.get("tool_trace") or []],
            artifacts={str(key): str(item) for key, item in _dict(raw.get("artifacts")).items()},
            latency_seconds=float(raw.get("latency_seconds") or 0.0),
            tool_call_count=int(raw.get("tool_call_count") or 0),
            relation_policy_version=str(
                raw.get("relation_policy_version") or RELATION_POLICY_VERSION
            ),
            schema_version=version,
        )

    def write_json(self, path: Path) -> str:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(output.resolve())
