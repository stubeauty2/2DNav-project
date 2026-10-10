"""Observed crops, candidate-centred previews and shared image rendering."""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import cv2
import numpy as np
from util import _move_forward_geo, generate_view_corners_with_scale
from ..config import FIXED_CROP_SIDE
from .geometry import _haversine_m


@dataclass
class Observation:
    observation_id: str
    stage_id: str
    search_step: int
    position: list
    heading: float
    scale: float
    view_corners: list
    image_path: str
    image: np.ndarray = field(repr=False)
    mode: str = "search"
    acquisition: str = "actual_pose_crop"
    anchor_observation_id: Optional[str] = None
    description: str = "当前位置的视图。"

    def metadata(self):
        # Do not serialize image arrays or API credentials.
        return {
            "observation_id": self.observation_id,
            "stage_id": self.stage_id,
            "search_step": self.search_step,
            "position": list(self.position),
            "heading": self.heading,
            "scale": self.scale,
            "view_corners": [list(p) for p in self.view_corners],
            "image_path": self.image_path,
            "mode": self.mode,
            "acquisition": self.acquisition,
            "anchor_observation_id": self.anchor_observation_id,
        }


def create_view_image(view_corners, ob, save_path=None, out_px=768):
    try:
        map_path = os.path.join(ob['dataset_dir'], 'train_images', f"{ob['map_name']}.tif")
        if not os.path.exists(map_path):
            print(f"[WARN] map file not found: {map_path}")
            return None

        sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
        if sat_map is None:
            print(f"[WARN] failed to read map: {map_path}")
            return None

        h, w = sat_map.shape[:2]
        lat_min, lng_min = ob['gps_botm_left']
        lat_max, lng_max = ob['gps_top_right']

        src = []
        for lat, lng in view_corners:
            x = (lng - lng_min) / (lng_max - lng_min) * w
            y = (lat_max - lat) / (lat_max - lat_min) * h
            src.append([x, y])
        src = np.array(src, dtype=np.float32)
        src[:, 0] = np.clip(src[:, 0], 0, w - 1)
        src[:, 1] = np.clip(src[:, 1], 0, h - 1)

        if cv2.contourArea(src.astype(np.float32)) < 1.0:
            print("[WARN] view area too small")
            return None

        out_side = int(768 if (out_px is None or out_px == 'auto') else out_px)

        dst = np.array([[0,0],[out_side-1,0],[out_side-1,out_side-1],[0,out_side-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)

        patch_hr = cv2.warpPerspective(sat_map, M, (out_side, out_side), flags=cv2.INTER_LANCZOS4)
        if patch_hr is None or patch_hr.size == 0:
            print("[WARN] perspective warp returned empty")
            return None

        if save_path:
            cv2.imwrite(save_path, patch_hr, [cv2.IMWRITE_JPEG_QUALITY, 98])

        return patch_hr
    except Exception as e:
        print(f"[ERROR] crop failed: {e}")
        return None


def render_context_views(pos, ob, heading, *, render, generate_corners):
    """Render the existing scale-3 and scale-7 context, in the original order."""
    detail_corners = generate_corners(pos, ob, scale_factor=3, angle_deg=heading)
    detail = render(detail_corners, ob, save_path=None, out_px=FIXED_CROP_SIDE)
    context_corners = generate_corners(pos, ob, scale_factor=7, angle_deg=heading)
    context = render(context_corners, ob, save_path=None, out_px=FIXED_CROP_SIDE)
    return detail, context


def acquire(ob, position, heading, scale, stage, round_n, name, out_dir,
            acquisition="actual_pose_crop", anchor_id=None, description=None):
    corners = generate_view_corners_with_scale(position, ob, scale_factor=scale,
                                               angle_deg=heading)
    points = np.asarray(corners, dtype=float)
    low, high = np.asarray(ob['gps_botm_left']), np.asarray(ob['gps_top_right'])
    if not np.isfinite(points).all():
        raise ValueError("non-finite crop geometry")
    if acquisition == "map_preview" and (np.any(points <= low) or np.any(points >= high)):
        raise ValueError("crop outside map bounds")
    path = Path(out_dir) / f"{stage}_r{round_n:02d}_{name}.jpg"
    image = create_view_image(corners, ob, save_path=str(path), out_px=FIXED_CROP_SIDE)
    if image is None:
        raise ValueError("crop unavailable")
    return Observation(f"{stage}:r{round_n}:{name}", stage, round_n, list(position),
                       float(heading), float(scale), points.tolist(), str(path.resolve()),
                       image, acquisition=acquisition, anchor_observation_id=anchor_id,
                       description=description or "当前位置的视图。")


def search_preview_poses(current, offset_ratio=0.5, shrink=1.0):
    """Connected T layout; the displacement bearing does not rotate the crop."""
    corners = current.view_corners
    width = _haversine_m(*corners[0], *corners[1])
    height = _haversine_m(*corners[1], *corners[2])
    d = min(width, height) * offset_ratio * shrink
    for name, offset, distance in (("left", -45, d * 2**0.5),
                                    ("forward", 0, d), ("right", 45, d * 2**0.5)):
        yield name, _move_forward_geo(current.position, current.heading + offset, meters=distance), current.heading, offset, distance
