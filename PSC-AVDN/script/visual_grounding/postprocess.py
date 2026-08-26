"""Candidate cleanup, serialization and contact-sheet rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .backends import RawProposal
from .contracts import Candidate, ConceptInput, ViewInput
from .geometry import bbox_from_mask, largest_component, mask_iou, navigation_point


def filter_proposals(
    proposals: Iterable[RawProposal],
    *,
    max_candidates: int = 12,
    nms_iou: float = 0.65,
    roi: Optional[Sequence[float]] = None,
) -> List[RawProposal]:
    cleaned: List[RawProposal] = []
    for proposal in proposals:
        mask = largest_component(proposal.mask)
        if roi is not None and len(roi) == 4:
            x1, y1, x2, y2 = [int(round(float(item))) for item in roi]
            limited = np.zeros_like(mask)
            limited[max(0, y1):max(0, y2), max(0, x1):max(0, x2)] = mask[
                max(0, y1):max(0, y2), max(0, x1):max(0, x2)
            ]
            mask = largest_component(limited)
        box = bbox_from_mask(mask)
        if box is None or int(mask.sum()) < 4:
            continue
        cleaned.append(
            RawProposal(
                mask=mask,
                bbox=box,
                score=float(proposal.score),
                mask_quality=float(proposal.mask_quality),
                component_scores=dict(proposal.component_scores),
            )
        )
    cleaned.sort(key=lambda item: (item.score, item.mask_quality), reverse=True)
    result: List[RawProposal] = []
    for item in cleaned:
        if any(mask_iou(item.mask, kept.mask) >= nms_iou for kept in result):
            continue
        result.append(item)
        if len(result) >= max_candidates:
            break
    return result


def materialize_candidates(
    proposals: Iterable[RawProposal],
    view: ViewInput,
    concept: ConceptInput,
    provider: str,
    artifact_dir: Path,
    entity_kind: str,
    event_type: str,
    start_index: int,
    id_prefix: str = "",
) -> Tuple[List[Candidate], List[np.ndarray]]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    result: List[Candidate] = []
    masks: List[np.ndarray] = []
    for offset, proposal in enumerate(proposals):
        candidate_id = f"{id_prefix + '-' if id_prefix else ''}cand-{start_index + offset:03d}"
        mask_path = artifact_dir / f"{candidate_id}_mask.png"
        cv2.imwrite(str(mask_path), proposal.mask.astype(np.uint8) * 255)
        height, width = proposal.mask.shape[:2]
        result.append(
            Candidate(
                candidate_id=candidate_id,
                concept_id=concept.concept_id,
                concept_text=concept.text,
                concept_role=concept.role,
                view_id=view.view_id,
                view_role=view.role,
                bbox_2d=[float(value) for value in proposal.bbox],
                target_point_2d=navigation_point(proposal.mask, entity_kind, event_type),
                mask_path=str(mask_path.resolve()),
                score=float(proposal.score),
                mask_quality=float(proposal.mask_quality),
                area_ratio=float(proposal.mask.sum() / max(1, width * height)),
                provider=provider,
                component_scores=dict(proposal.component_scores),
            )
        )
        masks.append(proposal.mask)
    return result, masks


def draw_contact_sheet(
    image_path: str,
    candidates: Sequence[Candidate],
    masks: Sequence[np.ndarray],
    output_path: Path,
    title: str,
) -> str:
    from PIL import Image

    base = np.asarray(Image.open(image_path).convert("RGB"))[:, :, ::-1].copy()
    overlay = base.copy()
    palette = [(255, 64, 64), (64, 210, 255), (100, 255, 120), (220, 100, 255)]
    for index, (candidate, mask) in enumerate(zip(candidates, masks)):
        color = palette[index % len(palette)]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 3)
        x1, y1, x2, y2 = [int(round(value)) for value in candidate.bbox_2d]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, candidate.candidate_id, (x1, max(22, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    canvas = cv2.addWeighted(base, 0.72, overlay, 0.28, 0)
    header = np.full((48, canvas.shape[1], 3), 245, np.uint8)
    cv2.putText(header, title[:100], (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    rows = [header, canvas]
    if candidates:
        crop_size = 180
        strip = np.full((crop_size + 36, canvas.shape[1], 3), 245, np.uint8)
        for index, candidate in enumerate(candidates[: max(1, canvas.shape[1] // crop_size)]):
            x1, y1, x2, y2 = [int(round(value)) for value in candidate.bbox_2d]
            pad = max(8, int(max(x2 - x1, y2 - y1) * 0.2))
            crop = base[max(0, y1-pad):min(base.shape[0], y2+pad), max(0, x1-pad):min(base.shape[1], x2+pad)]
            if crop.size:
                crop = cv2.resize(crop, (crop_size - 8, crop_size - 8))
                start = index * crop_size + 4
                strip[4:crop_size-4, start:start+crop_size-8] = crop
                cv2.putText(strip, candidate.candidate_id, (start, crop_size + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)
        rows.append(strip)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), np.vstack(rows))
    return str(output_path.resolve())
