"""Public arithmetic for comparing proposals; never ranks or confirms targets."""
import math
from .geometry import bearing


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
    return dict(
        source='Candidate projection, public initial pose and executed pose only',
        distance_from_initial_m=round(initial_distance, 2),
        initial_forward_m=forward, initial_right_m=right,
        bearing_from_initial_deg=round(initial_bearing, 2) if initial_bearing is not None else None,
        turn_from_initial_prior_deg=round((initial_bearing-prior+180)%360-180, 2)
            if initial_bearing is not None and prior is not None else None,
        distance_from_current_m=round(current_distance, 2),
        turn_from_current_heading_deg=round((current_bearing-heading+180)%360-180, 2)
            if current_bearing is not None else None,
        note='initial_forward/right use the original pose heading, not the exploration prior. Signed geometry does not identify a road or target.')
