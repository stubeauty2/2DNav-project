"""Mask cleanup, bounding boxes, navigation points and overlap helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=np.uint8) > 0
    if not binary.any():
        return binary
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8)
    if count <= 1:
        return binary
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == index


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def clamp_bbox(box: Sequence[float], width: int, height: int) -> Optional[List[float]]:
    if len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(item) for item in box]
    x1, x2 = sorted((max(0.0, min(x1, width - 1.0)), max(0.0, min(x2, width - 1.0))))
    y1, y2 = sorted((max(0.0, min(y1, height - 1.0)), max(0.0, min(y2, height - 1.0))))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2, y2]


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    left_value = np.asarray(left, dtype=bool)
    right_value = np.asarray(right, dtype=bool)
    union = np.logical_or(left_value, right_value).sum()
    return float(np.logical_and(left_value, right_value).sum() / union) if union else 0.0


def _nearest_mask_point(mask: np.ndarray, point_xy: Tuple[float, float]) -> List[float]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [float(point_xy[0]), float(point_xy[1])]
    distance = (xs - float(point_xy[0])) ** 2 + (ys - float(point_xy[1])) ** 2
    index = int(np.argmin(distance))
    return [float(xs[index]), float(ys[index])]


def navigation_point(mask: np.ndarray, entity_kind: str = "LANDMARK", event_type: str = "") -> List[float]:
    """Return a stable mask-internal pixel for downstream geo projection."""

    value = largest_component(mask)
    height, width = value.shape[:2]
    kind = str(entity_kind or "LANDMARK").upper()
    event = str(event_type or "").upper()
    if kind in {"BOUNDARY", "CORRIDOR"} or event in {"PASS", "CROSS", "GO_THROUGH", "FOLLOW"}:
        center_x = int(round((width - 1) * 0.5))
        # Search from the current pose (image center) towards the forward/top edge.
        for y in range(int(round((height - 1) * 0.5)), -1, -1):
            if value[y, center_x]:
                return [float(center_x), float(y)]
        ys, xs = np.nonzero(value[: max(1, height // 2 + 1)])
        if len(xs):
            index = int(np.argmin(np.abs(xs - center_x)))
            return [float(xs[index]), float(ys[index])]
    if kind == "REGION":
        return _nearest_mask_point(value, ((width - 1) * 0.5, (height - 1) * 0.5))
    moments = cv2.moments(value.astype(np.uint8))
    if moments["m00"]:
        point = [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]
        x, y = int(round(point[0])), int(round(point[1]))
        if 0 <= y < height and 0 <= x < width and value[y, x]:
            return [float(point[0]), float(point[1])]
        return _nearest_mask_point(value, (point[0], point[1]))
    return [float((width - 1) * 0.5), float((height - 1) * 0.5)]


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = [float(item) for item in left]
    rx1, ry1, rx2, ry2 = [float(item) for item in right]
    width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = width * height
    union = max(0.0, (lx2 - lx1) * (ly2 - ly1)) + max(0.0, (rx2 - rx1) * (ry2 - ry1)) - intersection
    return intersection / union if union else 0.0
