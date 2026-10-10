import math
import os
import cv2
import numpy as np

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


def _latlng_to_patch_xy(points, view_corners, ob):
    """Project map points into the exact clipped crop used by create_view_image."""
    sat = cv2.imread(os.path.join(ob['dataset_dir'], 'train_images', str(ob['map_name'])+'.tif'))
    if sat is None:
        raise RuntimeError('Cannot read map for proposal projection')
    height, width = sat.shape[:2]
    lat_min, lng_min = ob['gps_botm_left']; lat_max, lng_max = ob['gps_top_right']
    def xy(seq):
        return np.array([[(lng-lng_min)/(lng_max-lng_min)*width,
                          (lat_max-lat)/(lat_max-lat_min)*height] for lat,lng in seq], dtype=np.float32)
    src = xy(view_corners)
    src[:,0] = np.clip(src[:,0], 0, width-1); src[:,1] = np.clip(src[:,1], 0, height-1)
    matrix = cv2.getPerspectiveTransform(src, np.float32([[0,0],[767,0],[767,767],[0,767]]))
    return cv2.perspectiveTransform(xy(points).reshape(-1,1,2), matrix).reshape(-1,2).tolist()


# Initial R0/R1 settings, fixed before the sample2 run. Signals trigger inspection,
# never automatic rejection; no ground-truth coordinates are accepted here.
REVIEW_THRESHOLDS = dict(expansion_ratio=2.0, search_coverage=0.8,
    normalized_shift=0.5, low_overlap=0.2, edge_margin_px=3, low_confidence=0.65)


def confirmation_signals(search_polygon, confirmation_polygon, search_bbox, confirmation_bbox,
                         confidence=1.0, thresholds=None):
    from shapely.geometry import Polygon
    limits = dict(REVIEW_THRESHOLDS, **(thresholds or {}))
    lat0, lng0 = np.mean(np.asarray(search_polygon, dtype=float), axis=0)
    def meters(poly):
        return [(math.radians(lng-lng0)*6371000*math.cos(math.radians(lat0)),
                 math.radians(lat-lat0)*6371000) for lat,lng in poly]
    a = Polygon(meters(search_polygon)).convex_hull
    b = Polygon(meters(confirmation_polygon)).convex_hull
    if a.area <= 0 or b.area <= 0:
        raise ValueError('Degenerate proposal polygon')
    intersection = a.intersection(b).area
    ratio = b.area/a.area; coverage = intersection/a.area
    overlap = intersection/a.union(b).area
    points = meters(search_polygon)
    diagonal = max(math.dist(x,y) for x in points for y in points)
    shift = a.centroid.distance(b.centroid)
    def edges(box):
        return [name for name, gap in zip(('left','top','right','bottom'),
            (box[0],box[1],767-box[2],767-box[3])) if gap <= limits['edge_margin_px']]
    signals = dict(area_ratio=ratio, search_coverage=coverage, map_union_iou=overlap,
        center_shift_m=shift, normalized_shift=shift/max(diagonal,1e-9),
        search_edges=edges(search_bbox), confirmation_edges=edges(confirmation_bbox),
        confidence=float(confidence), thresholds=limits)
    reasons = []
    if ratio >= limits['expansion_ratio'] and coverage >= limits['search_coverage']:
        reasons.append('expanded_while_covering_search')
    if signals['normalized_shift'] >= limits['normalized_shift'] and overlap < limits['low_overlap']:
        reasons.append('shifted_with_low_overlap')
    if signals['confirmation_edges'] and ratio >= limits['expansion_ratio']:
        reasons.append('expanded_to_confirmation_border')
    if float(confidence) < limits['low_confidence']:
        reasons.append('low_initial_confirmation_confidence')
    signals['trigger_reasons'] = reasons
    return signals

