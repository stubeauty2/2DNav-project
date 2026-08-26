"""Shared JSON contracts for the visual-grounding HTTP service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ViewInput:
    view_id: str
    role: str
    image_path: str
    scale: Optional[float] = None


@dataclass
class ConceptInput:
    concept_id: str
    text: str
    role: str = "target"


@dataclass
class Candidate:
    candidate_id: str
    concept_id: str
    concept_text: str
    concept_role: str
    view_id: str
    view_role: str
    bbox_2d: List[float]
    target_point_2d: List[float]
    mask_path: str
    score: float
    mask_quality: float
    area_ratio: float
    provider: str
    component_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    candidate_id: str
    target_view_id: str
    matched_point_2d: Optional[List[float]]
    matched_bbox_2d: Optional[List[float]]
    similarity: float
    geometric_consistency: float
    heatmap_path: str = ""
    provider: str = "dinov3"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def require_mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def require_list(value: Any, name: str, *, minimum: int = 0, maximum: int = 10_000) -> List[Any]:
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        raise ValueError(f"{name} must be a list with {minimum}..{maximum} items")
    return value


def parse_views(value: Any) -> List[ViewInput]:
    result: List[ViewInput] = []
    for index, raw in enumerate(require_list(value, "views", minimum=1, maximum=16)):
        item = require_mapping(raw, f"views[{index}]")
        view_id = str(item.get("view_id") or "").strip()
        role = str(item.get("role") or "").strip().lower()
        path = str(item.get("image_path") or "").strip()
        if not view_id or not role or not path:
            raise ValueError(f"views[{index}] requires view_id, role and image_path")
        raw_scale = item.get("scale")
        try:
            scale = float(raw_scale) if raw_scale is not None else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"views[{index}].scale must be numeric") from exc
        if scale is not None and scale <= 0:
            raise ValueError(f"views[{index}].scale must be positive")
        result.append(ViewInput(view_id=view_id, role=role, image_path=path, scale=scale))
    return result


def parse_concepts(value: Any) -> List[ConceptInput]:
    result: List[ConceptInput] = []
    seen = set()
    for index, raw in enumerate(require_list(value, "concepts", minimum=1, maximum=4)):
        item = require_mapping(raw, f"concepts[{index}]")
        concept_id = str(item.get("concept_id") or f"c{index + 1}").strip()
        text = " ".join(str(item.get("text") or "").split())
        role = str(item.get("role") or "target").strip().lower()
        if not text or role not in {"target", "anchor", "context"}:
            raise ValueError(f"concepts[{index}] has invalid text or role")
        if concept_id in seen:
            raise ValueError(f"duplicate concept_id: {concept_id}")
        seen.add(concept_id)
        result.append(ConceptInput(concept_id=concept_id, text=text[:160], role=role))
    return result
