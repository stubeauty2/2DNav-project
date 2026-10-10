"""Geometry used by ANDH-Full, preserving its clipping and projection rules."""
import os
import math
import cv2
import numpy as np
from ..config import FIXED_CROP_SIDE

def _haversine_m(lat1, lng1, lat2, lng2):
    R = 6371000.0
    phi1 = math.radians(lat1);  phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlmb/2.0)**2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c

def _patch_xy_to_latlng(u, v, view_corners, ob):
    map_path = os.path.join(ob['dataset_dir'], 'train_images', f"{ob['map_name']}.tif")
    sat_map = cv2.imread(map_path, cv2.IMREAD_COLOR)
    if sat_map is None:
        raise RuntimeError(f"read map fail: {map_path}")
    H_img, W_img = sat_map.shape[:2]
    lat_min, lng_min = ob['gps_botm_left']
    lat_max, lng_max = ob['gps_top_right']

    src = []
    for lat, lng in view_corners:
        X = (lng - lng_min) / (lng_max - lng_min) * W_img
        Y = (lat_max - lat) / (lat_max - lat_min) * H_img
        src.append([X, Y])
    src = np.array(src, dtype=np.float32)
    src[:, 0] = np.clip(src[:, 0], 0, W_img - 1)
    src[:, 1] = np.clip(src[:, 1], 0, H_img - 1)

    side = int(768)
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype=np.float32)

    H_inv = cv2.getPerspectiveTransform(dst, src)

    vec = H_inv @ np.array([float(u), float(v), 1.0], dtype=np.float32)
    X = float(vec[0] / (vec[2] + 1e-12))
    Y = float(vec[1] / (vec[2] + 1e-12))

    lng = lng_min + (X / W_img) * (lng_max - lng_min)
    lat = lat_max - (Y / H_img) * (lat_max - lat_min)
    return [lat, lng]

def _minimal_square_uv_from_bbox(bbox, side=FIXED_CROP_SIDE, pad_px=0):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    edge = max(w, h) * 0.5
    edge += float(pad_px)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    def _clamp_square(cx, cy, half, S):
        left  = cx - half; right = cx + half
        top   = cy - half; bot   = cy + half

        if left < 0:      cx += -left
        if right > S-1:   cx -= (right - (S-1))
        if top  < 0:      cy += -top
        if bot  > S-1:    cy -= (bot - (S-1))

        half = min(half, cx, S-1-cx, cy, S-1-cy)

        left  = cx - half; right = cx + half
        top   = cy - half; bot   = cy + half
        return cx, cy, half, (left, top, right, bot)

    cx, cy, edge, (l, t, r, b) = _clamp_square(
        cx, cy, edge, float(side)
    )
    return [(l, t), (r, t), (r, b), (l, b)]


def project_bbox(bbox, observation, ob):
    x1, y1, x2, y2 = bbox
    polygon = [_patch_xy_to_latlng(x, y, observation.view_corners, ob)
               for x, y in ((x1,y1), (x2,y1), (x2,y2), (x1,y2))]
    center = _patch_xy_to_latlng((x1+x2)/2, (y1+y2)/2, observation.view_corners, ob)
    return polygon, center


def polygon_iou(a, b):
    # A shared local frame avoids precision loss at geographic coordinate magnitudes.
    origin = np.mean(np.asarray(list(a) + list(b), dtype=float), axis=0)
    pa = np.float32(local_xy(a, origin))
    pb = np.float32(local_xy(b, origin))
    overlap, _ = cv2.intersectConvexConvex(pa, pb)
    union = abs(cv2.contourArea(pa)) + abs(cv2.contourArea(pb)) - overlap
    return float(overlap / union) if union > 0 else 0.0


def local_xy(points, origin, heading=0):
    """Stateless local metric coordinates: right and forward relative to heading."""
    points = np.asarray(points, dtype=float)
    delta = points - np.asarray(origin, dtype=float)
    delta[:, 1] = (delta[:, 1] + 180) % 360 - 180
    meters = math.pi / 180 * 6371000
    east = delta[:, 1] * meters * math.cos(math.radians(float(origin[0])))
    north = delta[:, 0] * meters
    h = math.radians(heading)
    return np.column_stack((east * math.cos(h) - north * math.sin(h),
                            north * math.cos(h) + east * math.sin(h)))


def relative_pose(reference, current):
    """Current center in the reference heading frame; right/clockwise positive."""
    lat0, lon0 = map(math.radians, reference.position)
    lat1, lon1 = map(math.radians, current.position)
    radius = 6371000.0
    delta_lon = (lon1 - lon0 + math.pi) % (2 * math.pi) - math.pi
    east = radius * delta_lon * math.cos((lat0 + lat1) / 2)
    north = radius * (lat1 - lat0)
    heading = math.radians(reference.heading)
    return {
        "reference_observation_id": reference.observation_id,
        "subject_observation_id": current.observation_id,
        "forward_m": round(north * math.cos(heading) + east * math.sin(heading), 2),
        "right_m": round(east * math.cos(heading) - north * math.sin(heading), 2),
        "clockwise_heading_delta_deg": round(
            (current.heading - reference.heading + 180) % 360 - 180, 2),
    }
