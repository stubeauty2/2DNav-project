"""Candidate filtering and artifact materialization for the SAM3 service."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .geometry import bbox_from_mask, largest_component, mask_iou


def filter_proposals(proposals: List[Any], *, top_k: int = 12, nms_iou: float = 0.65) -> List[Any]:
    ordered = sorted(proposals, key=lambda item: float(item.score), reverse=True)
    result = []
    for proposal in ordered:
        mask = largest_component(proposal.mask)
        if bbox_from_mask(mask) is None or int(mask.sum()) < 4:
            continue
        if any(mask_iou(mask, kept.mask) >= nms_iou for kept in result):
            continue
        proposal.mask = mask
        result.append(proposal)
        if len(result) >= top_k:
            break
    return result


def materialize_candidates(proposals: List[Any], *, concept: Any, view: Any, artifact_dir: str, start_index: int = 1) -> List[Dict[str, Any]]:
    from PIL import Image
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    result = []
    for offset, proposal in enumerate(proposals, start_index):
        mask = np.asarray(proposal.mask, dtype=bool)
        mask_path = root / f"{concept.concept_id}_{view.view_id}_cand-{offset:03d}_mask.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        box = bbox_from_mask(mask) or [float(v) for v in proposal.bbox]
        candidate_id = f"{concept.concept_id}:{view.view_id}:{offset:03d}"
        result.append({
            "candidate_id": candidate_id,
            "concept_id": concept.concept_id,
            "concept_text": concept.text,
            "concept_role": concept.role,
            "view_id": view.view_id,
            "view_role": view.role,
            "bbox_2d": [float(v) for v in box],
            "mask_path": str(mask_path.resolve()),
            "score": float(proposal.score),
            "mask_quality": float(proposal.score),
            "provider": "sam3",
        })
    return result
