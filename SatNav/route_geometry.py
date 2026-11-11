"""Public language-derived geometry, never annotation waypoints or goal truth."""
import math


def numeric_route(diagnostics):
    stages = diagnostics.get('stages', []) if diagnostics else []
    if not stages:
        return None
    east = north = length = 0.
    points = []
    for stage in stages:
        heading, distance = stage.get('heading_deg'), stage.get('distance_m')
        if heading is None or distance is None:
            return None
        if any(isinstance(x, bool) or not isinstance(x, (float,int)) or not math.isfinite(x)
               for x in (heading,distance)) or distance < 0 or not 0<=heading<360:
            raise ValueError('Invalid language stage geometry')
        east += distance * math.sin(math.radians(heading))
        north += distance * math.cos(math.radians(heading))
        length += distance
        points.append([east,north])
    if length <= 0:
        return None
    return dict(source='model-extracted instruction distances and relative turns; NOT ground truth',
                endpoint_xy_m=[east,north], route_length_m=length,
                endpoint_distance_m=math.hypot(east,north),
                final_heading_deg=stages[-1]['heading_deg'],
                uncertainty='Conditional on correct language extraction and straight segments; verify target visually')


def guidance(route, position, heading):
    if route is None:
        return None
    dx,dy = (route['endpoint_xy_m'][i]-position[i] for i in (0,1))
    distance = math.hypot(dx,dy)
    bearing = math.degrees(math.atan2(dx,dy)) % 360 if distance>1e-9 else heading
    return dict(route, remaining_distance_m=distance, bearing_deg=bearing,
                relative_turn_deg=(bearing-heading+180)%360-180,
                instruction='Search near this language-predicted stopping area. This is a camera endpoint, not an object center. Do not confirm target identity from geometry alone.')


def final_ray(diagnostics):
    """Exact known prefix followed by one explicitly directed, unknown-length leg."""
    stages = diagnostics.get('stages', []) if diagnostics else []
    if len(stages) < 2 or stages[-1].get('distance_m') is not None:
        return None
    if any(s.get('heading_deg') is None for s in stages):
        return None
    prefix = numeric_route({'stages': stages[:-1]})
    if prefix is None:
        return None
    heading = stages[-1]['heading_deg']
    if isinstance(heading, bool) or not isinstance(heading, (int, float)) or not math.isfinite(heading) or not 0 <= heading < 360:
        raise ValueError('Invalid final ray heading')
    return dict(anchor_xy_m=prefix['endpoint_xy_m'], heading_deg=heading,
                axis=[math.sin(math.radians(heading)), math.cos(math.radians(heading))],
                source='Explicit instruction prefix and final heading; final distance UNKNOWN')


def project_ray(ray, point):
    along = max(0., sum((point[i]-ray['anchor_xy_m'][i])*ray['axis'][i] for i in (0, 1)))
    projected = [ray['anchor_xy_m'][i]+along*ray['axis'][i] for i in (0, 1)]
    return projected, math.dist(point, projected)
