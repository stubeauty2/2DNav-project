"""East/north geometry; normalized image coordinates always refer to a view."""
import math
import numpy as np


def project(matrix, points):
    p = np.c_[np.asarray(points, dtype=float), np.ones(len(points))] @ np.asarray(matrix).T
    if np.any(abs(p[:, 2]) < 1e-12):
        raise ValueError('Singular projection')
    return (p[:, :2] / p[:, 2:]).tolist()

def rect_corners(center, heading, width, height):
    h = math.radians(heading)
    f = np.array([math.sin(h), math.cos(h)]) * height / 2
    r = np.array([math.cos(h), -math.sin(h)]) * width / 2
    c = np.asarray(center)
    return [p.tolist() for p in (c+f-r, c+f+r, c-f+r, c-f-r)]

def box_points(box):
    a,b,c,d = box
    return [[a,b],[c,b],[c,d],[a,d]]

def pixels(view, normalized):
    return (np.asarray(normalized) * (np.asarray(view['image_size'])-1) / 1000).tolist()

def world(view, normalized):
    return project(view['pixel_to_world'], pixels(view, normalized))

def normalized(view, xy):
    p = project(np.linalg.inv(view['pixel_to_world']), xy)
    return (np.asarray(p) / (np.asarray(view['image_size'])-1)*1000).tolist()

def bearing(start, end, fallback=0):
    d = np.asarray(end)-start
    return math.degrees(math.atan2(d[0], d[1])) % 360 if np.linalg.norm(d)>1e-9 else fallback

def offset(position, heading, turn_deg, step_scale, height_m):
    h = (heading+turn_deg)%360
    distance = step_scale*height_m
    return [position[0]+distance*math.sin(math.radians(h)),
            position[1]+distance*math.cos(math.radians(h))], h

def square_bbox(box, side=768):
    # Exact legacy enclosing-square shift/shrink rule, in source-view pixels.
    x1,y1,x2,y2 = box
    cx,cy = (x1+x2)/2,(y1+y2)/2
    half = max(1.0,x2-x1,y2-y1)/2
    cx = max(half, min(side-1-half, cx))
    cy = max(half, min(side-1-half, cy))
    half = min(half,cx,side-1-cx,cy,side-1-cy)
    return box_points([cx-half,cy-half,cx+half,cy+half])


def homography(src, dst):
    a, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        a.extend([[x, y, 1, 0, 0, 0, -u*x, -u*y], [0, 0, 0, x, y, 1, -v*x, -v*y]])
        b.extend([u, v])
    return np.r_[np.linalg.solve(a, b), 1].reshape(3, 3)

def initialization_bearing(start, end):
    dx, dy = end[0] - start[0], end[1] - start[1]
    if math.hypot(dx, dy) < 1e-8:
        return None
    return math.degrees(math.atan2(dx, dy)) % 360
