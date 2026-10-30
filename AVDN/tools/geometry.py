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


# Public candidate and landmark geometry; never selects or confirms targets.

def candidate_context(center, episode, position, heading):
    start = episode['agent_input']['initial_pose']['xy_m']
    prior = episode['initialization']['direction']
    original_heading = episode['agent_input']['initial_pose']['heading_deg']
    forward = right = None
    if original_heading is not None:
        angle = math.radians(original_heading)
        east, north = center[0]-start[0], center[1]-start[1]
        forward = round(east*math.sin(angle)+north*math.cos(angle), 2)
        right = round(east*math.cos(angle)-north*math.sin(angle), 2)
    initial_distance = math.dist(start, center)
    current_distance = math.dist(position, center)
    initial_bearing = bearing(start, center) if initial_distance > 1e-8 else None
    current_bearing = bearing(position, center) if current_distance > 1e-8 else None
    original_turn = clock = None
    if initial_bearing is not None and original_heading is not None:
        relative = (initial_bearing-original_heading)%360
        original_turn = round((relative+180)%360-180, 2)
        if original_turn==180:original_turn=-180.
        minutes = int(math.floor(relative*2+.5))%720
        clock = f'{minutes//60 or 12}:{minutes%60:02d}'
    return dict(
        source='Candidate projection, public initial pose and executed pose only',
        distance_from_initial_m=round(initial_distance, 2),
        initial_forward_m=forward, initial_right_m=right,
        bearing_from_initial_deg=round(initial_bearing, 2) if initial_bearing is not None else None,
        turn_from_initial_heading_deg=original_turn,
        clock_from_initial_pose=clock,
        turn_from_initial_prior_deg=round((initial_bearing-prior+180)%360-180, 2)
            if initial_bearing is not None and prior is not None else None,
        distance_from_current_m=round(current_distance, 2),
        turn_from_current_heading_deg=round((current_bearing-heading+180)%360-180, 2)
            if current_bearing is not None else None,
        note='initial_forward/right use the original pose heading, not the exploration prior. Signed geometry does not identify a road or target.')
