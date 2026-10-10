from pathlib import Path
import cv2
import numpy as np
from . import telemetry as _audit
from ..config import VISION_TOOL_URL
from .grounding.client import VisionToolClient, VisionToolError

VisionToolClient = _audit.vision_client(VisionToolClient)

@_audit.sam3_call
def _sam3_image5(
    patch_main_bgr, target_terms, *, image_path, artifact_dir,
    vision_tool_url=VISION_TOOL_URL,
):
    """Create the experiment's unlabelled SAM3 candidate overlay."""
    if not image_path or not Path(image_path).is_file():
        raise VisionToolError(f"main Search image is unavailable: {image_path}")

    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    concepts = [
        {"concept_id": f"target-{index + 1}", "text": term, "role": "target"}
        for index, term in enumerate(target_terms)
    ]
    response = VisionToolClient(vision_tool_url, timeout=180.0).candidates({
        "backend": "sam3",
        "request_id": Path(image_path).stem,
        "views": [{
            "view_id": "main", "role": "main",
            "image_path": str(Path(image_path).resolve()), "scale": 5,
        }],
        "concepts": concepts,
        # The fixed experiment retained every post-NMS proposal instead of the
        # service's historical top-12 tool-agent limit.
        "max_candidates_per_view": 1000000,
    })

    _audit.event('sam3_service_response', response=response)
    # Reproduce the experiment's exact-mask NMS across all extracted terms.
    candidates = []
    for item in sorted(response.get("candidates") or [], key=lambda value: -float(value.get("score") or 0.0)):
        mask = cv2.imread(str(item.get("mask_path") or ""), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        mask = mask > 0
        area = int(mask.sum())
        if area < 4:
            continue
        duplicate = False
        for kept_item, kept_mask, kept_area in candidates:
            if min(area, kept_area) / max(area, kept_area) < 0.65:
                continue
            intersection = int(np.count_nonzero(mask & kept_mask))
            if intersection / (area + kept_area - intersection) >= 0.65:
                duplicate = True
                break
        if not duplicate:
            candidates.append((item, mask, area))

    overlay = patch_main_bgr.copy()
    candidate_count = 0
    for item, _mask, _area in candidates:
        try:
            x1, y1, x2, y2 = [int(round(float(v))) for v in item["bbox_2d"]]
        except (KeyError, TypeError, ValueError):
            continue
        x1 = max(0, min(overlay.shape[1] - 1, x1))
        x2 = max(0, min(overlay.shape[1] - 1, x2))
        y1 = max(0, min(overlay.shape[0] - 1, y1))
        y2 = max(0, min(overlay.shape[0] - 1, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 3)
        candidate_count += 1

    image5_path = artifact_root / "image5.png"
    if not cv2.imwrite(str(image5_path), overlay):
        raise VisionToolError(f"failed to save SAM3 Image 5: {image5_path}")
    return overlay, {
        "target_terms": list(target_terms),
        "candidate_count": candidate_count,
        "image5_path": str(image5_path.resolve()),
    }

