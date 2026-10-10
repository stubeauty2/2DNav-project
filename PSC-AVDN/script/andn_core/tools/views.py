import os
import cv2
import numpy as np
from util import generate_view_corners_with_scale, draw_pos_on_patch, make_minimap_crop_basemap

def _draw_bbox_on_patch(patch, bbox, color=(0,0,255), thickness=3, label=None):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cv2.rectangle(patch, (x1,y1), (x2,y2), color, thickness)
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = str(label)
        (tw, th), _ = cv2.getTextSize(txt, font, 0.5, 1)
        cv2.rectangle(patch, (x1, max(0, y1- (th+6))), (x1+tw+10, y1), color, -1)
        cv2.putText(patch, txt, (x1+5, max(0, y1-6)), font, 0.5, (255,255,255), 1, cv2.LINE_AA)

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

