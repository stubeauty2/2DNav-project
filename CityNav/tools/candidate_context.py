"""Public arithmetic for comparing proposals; never ranks or confirms targets."""
import math
from .geometry import bearing


def candidate_context(center, episode, position, heading):
    start = episode['agent_input']['initial_pose']['xy_m']
    prior = episode['initialization']['direction']
    initial_distance = math.dist(start, center)
    current_distance = math.dist(position, center)
    initial_bearing = bearing(start, center) if initial_distance > 1e-8 else None
    current_bearing = bearing(position, center) if current_distance > 1e-8 else None
    result = dict(
        source='Candidate projection, public initial pose and executed pose only',
        distance_from_initial_m=round(initial_distance, 2),
        bearing_from_initial_deg=round(initial_bearing, 2) if initial_bearing is not None else None,
        turn_from_initial_prior_deg=round((initial_bearing-prior+180)%360-180, 2)
            if initial_bearing is not None and prior is not None else None,
        distance_from_current_m=round(current_distance, 2),
        turn_from_current_heading_deg=round((current_bearing-heading+180)%360-180, 2)
            if current_bearing is not None else None,
        note='The initial direction is uncertain. These are geometric relations, not target evidence.')
    groups = {}
    for landmark in episode['agent_input']['named_landmarks']:
        name = landmark['name']
        distance = math.dist(center, landmark['xy_m'])
        box = landmark.get('bbox_xy_m')
        extent_distance = math.hypot(max(box[0]-center[0], center[0]-box[2], 0),
                                     max(box[1]-center[1], center[1]-box[3], 0)) if box else None
        ref = dict(landmark_id=landmark['id'], distance_to_center_m=round(distance, 2),
                   distance_to_extent_m=round(extent_distance, 2) if extent_distance is not None else None)
        groups.setdefault(name, []).append(ref)
    result['public_reference_distances'] = [dict(name=name,
        nearest_center=min(refs, key=lambda r:r['distance_to_center_m']),
        nearest_extent=min((r for r in refs if r['distance_to_extent_m'] is not None),
                           key=lambda r:r['distance_to_extent_m'], default=None))
        for name, refs in sorted(groups.items())]
    result['reference_note'] = ('Distances are from this candidate, not the agent. Extents are catalog bounding '
        'rectangles, not exact building/road outlines. A zero extent distance is not identity confirmation.')
    return result
