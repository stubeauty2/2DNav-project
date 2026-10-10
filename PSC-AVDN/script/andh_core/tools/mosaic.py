"""Per-request image composition from observed crops; no persistent map state."""
from pathlib import Path
import cv2
import numpy as np
from .geometry import local_xy


def save_view(view_id, image, description, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Cannot save image: {path}")
    return {"view_id": view_id, "image": image, "description": description,
            "selectable": False, "image_path": str(path.resolve())}


def build_mosaic(observations, anchor, path, long_side=1024):
    quads = [local_xy(o.view_corners, anchor.position, anchor.heading) * [1, -1]
             for o in observations]
    points = np.vstack(quads)
    low, high = points.min(0), points.max(0)
    scale = (long_side - 64) / max(float((high - low).max()), 1)
    size = np.ceil((high - low) * scale + 64).astype(int)
    width, height = map(int, size)
    yy, xx = np.indices((height, width))
    canvas = np.repeat(np.where((xx // 16 + yy // 16) % 2, 240, 225)[..., None], 3, axis=2).astype(np.uint8)
    projected = []
    for obs, quad in zip(observations, quads):
        target = np.float32((quad - low) * scale + 32)
        h, w = obs.image.shape[:2]
        source = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        transform = cv2.getPerspectiveTransform(source, target)
        warped = cv2.warpPerspective(obs.image, transform, (width, height))
        mask = cv2.warpPerspective(np.full((h,w),255,np.uint8), transform,
                                   (width,height), flags=cv2.INTER_NEAREST)
        canvas[mask > 0] = warped[mask > 0]
        projected.append(target)
    colors = [(0,170,0),(220,100,20),(20,150,210),(180,30,180)]
    for i, (obs, quad) in enumerate(zip(observations, projected)):
        color = colors[i % len(colors)]
        cv2.polylines(canvas, [np.int32(quad)], True, color, 2, cv2.LINE_AA)
        name = obs.observation_id.rsplit(':',1)[-1]
        label = {'current':'C','left':'L','forward':'F','right':'R'}.get(name,name)
        center = np.int32(quad.mean(0))
        cv2.putText(canvas, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2, cv2.LINE_AA)
    agent = tuple(np.int32((np.array([0.,0.]) - low) * scale + 32))
    cv2.circle(canvas, agent, 5, (0,0,0), -1)
    cv2.arrowedLine(canvas, agent, (agent[0],agent[1]-22), (0,0,0), 2)
    view = save_view(anchor.observation_id + ':mosaic', canvas,
                     '本轮视图的空间拼接图，标签对应各原始视图。', path)
    view['source_views'] = { {'current':'C','left':'L','forward':'F','right':'R'}.get(o.observation_id.rsplit(':',1)[-1], str(i)): o.observation_id + ':rgb'
                            for i,o in enumerate(observations)}
    return view
