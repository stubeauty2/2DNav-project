"""Validation and compact data contracts for SAM3 requests."""

from __future__ import annotations

from dataclasses import dataclass, asdict
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


def _mapping(value: Any, name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def parse_views(value: Any) -> List[ViewInput]:
    if not isinstance(value, list) or not 1 <= len(value) <= 8:
        raise ValueError("views must contain 1..8 items")
    result = []
    for index, raw in enumerate(value):
        item = _mapping(raw, f"views[{index}]")
        view_id = str(item.get("view_id") or "").strip()
        role = str(item.get("role") or "").strip().lower()
        image_path = str(item.get("image_path") or "").strip()
        if not view_id or not role or not image_path:
            raise ValueError(f"views[{index}] requires view_id, role and image_path")
        scale = item.get("scale")
        if scale is not None:
            scale = float(scale)
            if scale <= 0:
                raise ValueError(f"views[{index}].scale must be positive")
        result.append(ViewInput(view_id, role, image_path, scale))
    return result


def parse_concepts(value: Any) -> List[ConceptInput]:
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        raise ValueError("concepts must contain 1..4 items")
    result, seen = [], set()
    for index, raw in enumerate(value):
        item = _mapping(raw, f"concepts[{index}]")
        concept_id = str(item.get("concept_id") or f"c{index + 1}").strip()
        text = " ".join(str(item.get("text") or "").split())
        role = str(item.get("role") or "target").strip().lower()
        if not text or role not in {"target", "anchor", "context"} or concept_id in seen:
            raise ValueError(f"concepts[{index}] has invalid or duplicate values")
        seen.add(concept_id)
        result.append(ConceptInput(concept_id, text[:160], role))
    return result


def view_dict(view: ViewInput) -> Dict[str, Any]:
    return asdict(view)

