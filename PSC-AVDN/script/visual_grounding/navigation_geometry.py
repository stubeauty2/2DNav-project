"""Deterministic geographic measurements for visual-grounding candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


EARTH_METERS_PER_DEGREE = 111_320.0


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def wrap_signed_angle(angle_deg: float) -> float:
    """Normalize an angle to [-180, 180)."""

    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def haversine_m(left: Sequence[float], right: Sequence[float]) -> float:
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


def bearing_deg(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the clockwise-from-north initial bearing."""

    lat1, lat2 = math.radians(float(left[0])), math.radians(float(right[0]))
    dlng = math.radians(float(right[1]) - float(left[1]))
    y = math.sin(dlng) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(dlng)
    )
    return math.degrees(math.atan2(y, x)) % 360.0


def direction_label(relative_bearing_deg: float) -> str:
    labels = (
        "forward", "forward-right", "right", "back-right",
        "behind", "back-left", "left", "forward-left",
    )
    index = int(math.floor((float(relative_bearing_deg) + 22.5) / 45.0)) % 8
    return labels[index]


class ViewGeoTransform:
    """Project between a rendered view and latitude/longitude.

    The four corners are ordered top-left, top-right, bottom-right,
    bottom-left, matching the renderer and Search coordinate conversion.
    """

    def __init__(
        self,
        corners_latlng: Sequence[Sequence[float]],
        width: int,
        height: int,
    ) -> None:
        corners = np.asarray(corners_latlng, dtype=np.float64)
        if corners.shape != (4, 2):
            raise ValueError("view corners must contain four [lat, lng] points")
        self.width = int(width)
        self.height = int(height)
        if self.width < 2 or self.height < 2:
            raise ValueError("view dimensions must be at least 2x2")
        self.reference = corners.mean(axis=0)
        cos_lat = max(1e-6, abs(math.cos(math.radians(float(self.reference[0])))))
        self._meters_per_lng_degree = EARTH_METERS_PER_DEGREE * cos_lat
        metric = np.column_stack([
            (corners[:, 1] - self.reference[1]) * self._meters_per_lng_degree,
            (corners[:, 0] - self.reference[0]) * EARTH_METERS_PER_DEGREE,
        ]).astype(np.float32)
        pixels = np.asarray([
            [0.0, 0.0],
            [self.width - 1.0, 0.0],
            [self.width - 1.0, self.height - 1.0],
            [0.0, self.height - 1.0],
        ], dtype=np.float32)
        self._pixel_to_metric = cv2.getPerspectiveTransform(pixels, metric)
        self._metric_to_pixel = cv2.getPerspectiveTransform(metric, pixels)

    @staticmethod
    def _project(matrix: np.ndarray, point: Sequence[float]) -> List[float]:
        vector = matrix @ np.asarray([float(point[0]), float(point[1]), 1.0])
        denominator = float(vector[2])
        if abs(denominator) < 1e-12:
            raise ValueError("geographic projection is singular")
        return [float(vector[0] / denominator), float(vector[1] / denominator)]

    def pixel_to_latlng(self, point_xy: Sequence[float]) -> List[float]:
        east_m, north_m = self._project(self._pixel_to_metric, point_xy)
        return [
            float(self.reference[0] + north_m / EARTH_METERS_PER_DEGREE),
            float(self.reference[1] + east_m / self._meters_per_lng_degree),
        ]

    def latlng_to_pixel(self, latlng: Sequence[float]) -> List[float]:
        metric = [
            (float(latlng[1]) - self.reference[1]) * self._meters_per_lng_degree,
            (float(latlng[0]) - self.reference[0]) * EARTH_METERS_PER_DEGREE,
        ]
        return self._project(self._metric_to_pixel, metric)


def navigation_vector(
    current_position: Sequence[float],
    target_position: Sequence[float],
    heading_deg: float,
) -> Dict[str, Any]:
    distance = haversine_m(current_position, target_position)
    absolute_bearing = bearing_deg(current_position, target_position) if distance > 1e-6 else float(heading_deg) % 360.0
    relative = wrap_signed_angle(absolute_bearing - float(heading_deg))
    radians = math.radians(relative)
    return {
        "distance_m": _round(distance),
        "bearing_deg": _round(absolute_bearing, 1),
        "relative_bearing_deg": _round(relative, 1),
        "forward_offset_m": _round(distance * math.cos(radians)),
        "right_offset_m": _round(distance * math.sin(radians)),
        "direction_label": direction_label(relative),
    }


def _load_mask(path: str, width: int, height: int) -> Optional[np.ndarray]:
    if not path or not Path(path).is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    if mask.shape[:2] != (height, width):
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def forward_sector_min_distance_px(
    mask: np.ndarray,
    origin_xy: Sequence[float],
    *,
    half_angle_deg: float = 15.0,
) -> Dict[str, Any]:
    """Measure the nearest mask pixel inside an upward image-space sector.

    The sector is centered on the negative image-y axis. Only pixels strictly
    above ``origin_xy`` participate; the two angular boundaries are inclusive.
    This deliberately returns image-space evidence only.
    """

    values = np.asarray(mask)
    if values.ndim != 2:
        raise ValueError("mask must be a two-dimensional array")
    if len(origin_xy) != 2:
        raise ValueError("origin_xy must contain [x, y]")
    origin_x, origin_y = float(origin_xy[0]), float(origin_xy[1])
    angle_limit = float(half_angle_deg)
    if not (math.isfinite(origin_x) and math.isfinite(origin_y)):
        raise ValueError("origin_xy must contain finite values")
    if not math.isfinite(angle_limit) or not 0.0 < angle_limit < 90.0:
        raise ValueError("half_angle_deg must be between 0 and 90 degrees")

    ys, xs = np.nonzero(values > 0)
    if xs.size == 0:
        return {"intersects": False, "minimum_distance_px": None}

    dx = xs.astype(np.float64) - origin_x
    dy_up = origin_y - ys.astype(np.float64)
    above = dy_up > 0.0
    if not np.any(above):
        return {"intersects": False, "minimum_distance_px": None}

    dx = dx[above]
    dy_up = dy_up[above]
    angles = np.degrees(np.arctan2(dx, dy_up))
    inside = np.abs(angles) <= angle_limit + 1e-9
    if not np.any(inside):
        return {"intersects": False, "minimum_distance_px": None}

    distances = np.hypot(dx[inside], dy_up[inside])
    return {
        "intersects": True,
        "minimum_distance_px": round(float(np.min(distances)), 2),
    }


def _point_in_mask(mask: Optional[np.ndarray], point_xy: Sequence[float]) -> Optional[bool]:
    if mask is None:
        return None
    x, y = int(round(float(point_xy[0]))), int(round(float(point_xy[1])))
    if not (0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]):
        return False
    return bool(mask[y, x])


def _footprint_mask(
    transform: ViewGeoTransform,
    footprint_latlng: Sequence[Sequence[float]],
) -> Tuple[Optional[np.ndarray], Optional[float]]:
    if not footprint_latlng:
        return None, None
    try:
        points = np.asarray(
            [transform.latlng_to_pixel(item) for item in footprint_latlng],
            dtype=np.float32,
        )
    except (TypeError, ValueError, IndexError):
        return None, None
    if len(points) < 3:
        return None, None
    canvas = np.zeros((transform.height, transform.width), dtype=np.uint8)
    cv2.fillPoly(canvas, [np.rint(points).astype(np.int32)], 1)
    center = points.mean(axis=0)
    half_width = float(np.max(np.abs(points[:, 0] - center[0])))
    return canvas > 0, max(1.0, half_width)


def _forward_path_analysis(
    mask: Optional[np.ndarray],
    transform: ViewGeoTransform,
    current_pixel: Sequence[float],
    current_position: Sequence[float],
    heading_deg: float,
    corridor_half_width_px: Optional[float],
) -> Dict[str, Any]:
    if mask is None:
        return {"available": False}
    cx, cy = float(current_pixel[0]), float(current_pixel[1])
    half_width = max(1, int(math.ceil(corridor_half_width_px or 1.0)))
    left = max(0, int(math.floor(cx)) - half_width)
    right = min(mask.shape[1], int(math.ceil(cx)) + half_width + 1)
    max_y = min(mask.shape[0] - 1, int(math.floor(cy)))
    if right <= left or max_y < 0:
        return {"available": True, "intersects": False}
    row_hits = np.any(mask[: max_y + 1, left:right], axis=1)
    hit_rows = np.flatnonzero(row_hits)
    if not len(hit_rows):
        return {"available": True, "intersects": False}

    entry_row = int(hit_rows[-1])
    exit_row = entry_row
    gap = 0
    for row in range(entry_row - 1, -1, -1):
        if row_hits[row]:
            exit_row = row
            gap = 0
        else:
            gap += 1
            if gap > 2:
                break

    entry_geo = transform.pixel_to_latlng([cx, entry_row])
    exit_geo = transform.pixel_to_latlng([cx, exit_row])
    entry_vector = navigation_vector(current_position, entry_geo, heading_deg)
    exit_vector = navigation_vector(current_position, exit_geo, heading_deg)
    entry_distance = max(0.0, float(entry_vector["forward_offset_m"] or 0.0))
    exit_distance = max(entry_distance, float(exit_vector["forward_offset_m"] or 0.0))
    return {
        "available": True,
        "intersects": True,
        "corridor_half_width_px": _round(corridor_half_width_px, 1),
        "entry_point_2d": [_round(cx, 1), float(entry_row)],
        "exit_point_2d": [_round(cx, 1), float(exit_row)],
        "entry_distance_m": _round(entry_distance),
        "exit_distance_m": _round(exit_distance),
        "crossing_width_m": _round(max(0.0, exit_distance - entry_distance)),
    }


class NavigationGeometryAnalyzer:
    """Compute candidate facts in one shared geographic frame."""

    def __init__(
        self,
        *,
        current_position: Sequence[float],
        heading_deg: float,
        view_corners: Mapping[str, Sequence[Sequence[float]]],
        view_sizes: Mapping[str, Sequence[int]],
        agent_footprint: Optional[Sequence[Sequence[float]]] = None,
        event_type: str = "",
    ) -> None:
        self.current_position = [float(current_position[0]), float(current_position[1])]
        self.heading_deg = float(heading_deg) % 360.0
        self.event_type = str(event_type or "").upper()
        footprint_values = [] if agent_footprint is None else agent_footprint
        self.agent_footprint = [list(map(float, item)) for item in footprint_values]
        self.transforms: Dict[str, ViewGeoTransform] = {}
        self.current_pixels: Dict[str, List[float]] = {}
        self.footprint_masks: Dict[str, Optional[np.ndarray]] = {}
        self.corridor_half_widths: Dict[str, Optional[float]] = {}
        for view_id, corners in view_corners.items():
            size = list(view_sizes.get(view_id) or [768, 768])
            width, height = int(size[0]), int(size[1])
            transform = ViewGeoTransform(corners, width, height)
            self.transforms[str(view_id)] = transform
            self.current_pixels[str(view_id)] = transform.latlng_to_pixel(self.current_position)
            footprint_mask, half_width = _footprint_mask(transform, self.agent_footprint)
            self.footprint_masks[str(view_id)] = footprint_mask
            self.corridor_half_widths[str(view_id)] = half_width

    def analyze(self, candidate: Mapping[str, Any]) -> Dict[str, Any]:
        view_id = str(candidate.get("view_id") or "")
        transform = self.transforms.get(view_id)
        if transform is None:
            return {"available": False, "reason": f"missing geometry for view {view_id!r}"}
        point = candidate.get("target_point_2d")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return {"available": False, "reason": "candidate has no target point"}
        target_pixel = [float(point[0]), float(point[1])]
        current_pixel = self.current_pixels[view_id]
        target_latlng = transform.pixel_to_latlng(target_pixel)
        vector = navigation_vector(self.current_position, target_latlng, self.heading_deg)
        mask = _load_mask(
            str(candidate.get("mask_path") or ""), transform.width, transform.height
        )
        footprint_mask = self.footprint_masks.get(view_id)
        footprint_overlap = None
        if mask is not None and footprint_mask is not None:
            footprint_overlap = bool(np.logical_and(mask, footprint_mask).any())
        path = _forward_path_analysis(
            mask,
            transform,
            current_pixel,
            self.current_position,
            self.heading_deg,
            self.corridor_half_widths.get(view_id),
        )
        return {
            "available": True,
            "target_latlng": [_round(target_latlng[0], 8), _round(target_latlng[1], 8)],
            **vector,
            "forward_offset_px": _round(float(current_pixel[1]) - target_pixel[1], 1),
            "right_offset_px": _round(target_pixel[0] - float(current_pixel[0]), 1),
            "current_position_2d": [_round(current_pixel[0], 1), _round(current_pixel[1], 1)],
            "current_position_inside_mask": _point_in_mask(mask, current_pixel),
            "mask_intersects_agent_footprint": footprint_overlap,
            "forward_path": path,
        }

    @staticmethod
    def compact(analysis: Mapping[str, Any]) -> Dict[str, Any]:
        if not analysis.get("available"):
            return dict(analysis)
        path = analysis.get("forward_path") or {}
        return {
            key: analysis.get(key)
            for key in (
                "available", "target_latlng", "distance_m", "bearing_deg",
                "relative_bearing_deg", "forward_offset_m", "right_offset_m",
                "direction_label", "forward_offset_px", "right_offset_px",
                "current_position_inside_mask", "mask_intersects_agent_footprint",
            )
        } | {
            "forward_path": {
                key: path.get(key)
                for key in (
                    "available", "intersects", "entry_distance_m",
                    "exit_distance_m", "crossing_width_m",
                )
                if key in path
            }
        }

    def cross_scale_neighbors(
        self,
        candidate: Mapping[str, Any],
        registry: Mapping[str, Mapping[str, Any]],
        analyses: Mapping[str, Mapping[str, Any]],
        *,
        limit_per_view: int = 2,
    ) -> List[Dict[str, Any]]:
        source_id = str(candidate.get("candidate_id") or "")
        source = analyses.get(source_id) or self.analyze(candidate)
        source_geo = source.get("target_latlng")
        if not source.get("available") or not source_geo:
            return []
        grouped: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        for other_id, other in registry.items():
            if other_id == source_id or other.get("view_id") == candidate.get("view_id"):
                continue
            if other.get("concept_id") != candidate.get("concept_id"):
                continue
            other_text = str(other.get("concept_text") or "").strip().lower()
            source_text = str(candidate.get("concept_text") or "").strip().lower()
            if other_text != source_text:
                continue
            if other.get("concept_role") != candidate.get("concept_role"):
                continue
            other_analysis = analyses.get(other_id) or self.analyze(other)
            other_geo = other_analysis.get("target_latlng")
            if not other_analysis.get("available") or not other_geo:
                continue
            separation = haversine_m(source_geo, other_geo)
            bearing_delta = wrap_signed_angle(
                float(other_analysis.get("relative_bearing_deg") or 0.0)
                - float(source.get("relative_bearing_deg") or 0.0)
            )
            item = {
                "candidate_id": other_id,
                "view_id": other.get("view_id"),
                "source_scale": other.get("source_scale"),
                "geographic_separation_m": _round(separation),
                "distance_from_current_delta_m": _round(
                    float(other_analysis.get("distance_m") or 0.0)
                    - float(source.get("distance_m") or 0.0)
                ),
                "relative_bearing_delta_deg": _round(bearing_delta, 1),
                "score": _round(float(other.get("score") or 0.0), 4),
            }
            grouped.setdefault(str(other.get("view_id") or ""), []).append((separation, item))
        result: List[Dict[str, Any]] = []
        for view_id in ("narrow", "main", "wide"):
            ranked = sorted(grouped.get(view_id, []), key=lambda item: item[0])
            result.extend(item for _, item in ranked[:limit_per_view])
        return result

    def pairwise_relations(
        self,
        candidates: Sequence[Mapping[str, Any]],
        analyses: Mapping[str, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for index, left in enumerate(candidates):
            left_id = str(left.get("candidate_id") or "")
            left_analysis = analyses.get(left_id) or self.analyze(left)
            left_geo = left_analysis.get("target_latlng")
            if not left_geo:
                continue
            for right in candidates[index + 1:]:
                right_id = str(right.get("candidate_id") or "")
                right_analysis = analyses.get(right_id) or self.analyze(right)
                right_geo = right_analysis.get("target_latlng")
                if not right_geo:
                    continue
                vector = navigation_vector(left_geo, right_geo, self.heading_deg)
                result.append({
                    "from_candidate_id": left_id,
                    "to_candidate_id": right_id,
                    **vector,
                })
        return result

    def draw_overlays(
        self,
        candidates: Sequence[Mapping[str, Any]],
        analyses: Mapping[str, Mapping[str, Any]],
        artifact_dir: str,
    ) -> List[str]:
        colors = [(0, 255, 255), (255, 160, 40), (90, 240, 90), (255, 90, 220)]
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for candidate in candidates:
            grouped.setdefault(str(candidate.get("view_id") or ""), []).append(candidate)
        paths: List[str] = []
        output_dir = Path(artifact_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for view_id, items in grouped.items():
            source = str(items[0].get("source_image_path") or "")
            image = cv2.imread(source) if source else None
            transform = self.transforms.get(view_id)
            if image is None or transform is None:
                continue
            current = self.current_pixels[view_id]
            cx, cy = int(round(current[0])), int(round(current[1]))
            half_width = int(round(self.corridor_half_widths.get(view_id) or 1.0))
            cv2.arrowedLine(image, (cx, cy), (cx, max(0, cy - 100)), (80, 255, 80), 4, tipLength=0.18)
            cv2.line(image, (max(0, cx - half_width), cy), (max(0, cx - half_width), 0), (80, 255, 80), 1)
            cv2.line(image, (min(image.shape[1] - 1, cx + half_width), cy), (min(image.shape[1] - 1, cx + half_width), 0), (80, 255, 80), 1)
            for index, candidate in enumerate(items):
                color = colors[index % len(colors)]
                bbox = [int(round(float(value))) for value in candidate.get("bbox_2d") or []]
                point = candidate.get("target_point_2d") or []
                if len(bbox) == 4:
                    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                if len(point) == 2:
                    tx, ty = int(round(float(point[0]))), int(round(float(point[1])))
                    cv2.line(image, (cx, cy), (tx, ty), color, 2)
                    cv2.circle(image, (tx, ty), 6, color, -1)
                    analysis = analyses.get(str(candidate.get("candidate_id") or "")) or {}
                    label = (
                        f"{candidate.get('candidate_id')} "
                        f"{analysis.get('distance_m', '?')}m "
                        f"{analysis.get('relative_bearing_deg', '?')}deg"
                    )
                    cv2.putText(image, label, (max(4, tx + 8), max(20, ty - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            path = output_dir / f"navigation_geometry_{view_id}.png"
            cv2.imwrite(str(path), image)
            paths.append(str(path.resolve()))
        return paths


def compact_candidate(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the candidate fields useful to the model, excluding local paths."""

    return {
        key: candidate.get(key)
        for key in (
            "candidate_id", "concept_id", "concept_text", "concept_role",
            "view_id", "view_role", "source_scale", "bbox_2d",
            "target_point_2d", "score", "mask_quality", "area_ratio",
            "navigation_summary",
        )
        if key in candidate
    }
