"""Small mask and bbox helpers kept independent of the SAM3 import."""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    try:
        import cv2
    except ImportError:
        # The navigation environment may not have OpenCV; retain the full
        # mask for the service's dependency-light contract in that case.
        return binary.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return binary.astype(bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.nonzero(np.asarray(mask) > 0)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    a, b = np.asarray(left) > 0, np.asarray(right) > 0
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
