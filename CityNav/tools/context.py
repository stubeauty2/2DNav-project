"""Public arithmetic for comparing proposals; never ranks or confirms targets."""
import math
from .geometry import bearing
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from .geometry import bearing, normalized, project, box_points


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
    result = dict(
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


def heading_of_view(view):
    c=np.asarray(view['world_corners_xy_m'],dtype=float)
    return bearing((c[2]+c[3])/2,(c[0]+c[1])/2)

def overlaps(a,b):
    """Separating-axis test for two convex polygons (inclusive boundaries)."""
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    for poly in (a,b):
        for p,q in zip(poly,np.roll(poly,-1,axis=0)):
            edge=q-p;axis=np.array([-edge[1],edge[0]])
            if np.linalg.norm(axis)<1e-12:continue
            x,y=a@axis,b@axis
            if x.max()<y.min()-1e-8 or y.max()<x.min()-1e-8:return False
    return True

def in_view(view, xy):
    p=normalized(view,[xy])[0]
    return all(-1e-8<=v<=1000+1e-8 for v in p)

def polygon_parts(shape):
    if shape.geom_type=='Polygon':
        return [shape] if shape.area>0.5 else []
    return [p for child in getattr(shape,'geoms',[]) for p in polygon_parts(child)]

def unseen_extent_hints(box, map_region, observed_region, position, heading, main_height):
    if not box:
        return dict(extent_observed_fraction=None,unobserved_extent_probes=[])
    region=Polygon(box_points(box)).intersection(map_region)
    if region.area<1e-8:
        return dict(extent_observed_fraction=None,unobserved_extent_probes=[])
    unseen=region.difference(observed_region)
    probes=[]
    for part in sorted(polygon_parts(unseen),key=lambda p:(-p.area,p.bounds))[:2]:
        point=part.representative_point()
        xy=[point.x,point.y]
        distance=float(np.linalg.norm(np.asarray(xy)-position))
        if distance<1e-6:continue
        h=bearing(position,xy)
        probes.append(dict(xy_m=xy,turn_deg=round((h-heading+180)%360-180,4),
                           step_scale=round(distance/main_height,8),
                           unobserved_component_area_m2=round(part.area,2)))
    return dict(extent_observed_fraction=round(max(0.,min(1.,1-unseen.area/region.area)),4),
                unobserved_extent_probes=probes)

def navigation_context(landmarks, position, heading, current_view, observed_views, specification, trajectory):
    corners=project(np.linalg.inv(specification['world_to_raster']),
                    box_points([0,0,*specification['image_size']]))
    map_region=Polygon(corners)
    # Include the current view delivered with this context. This is footprint
    # arithmetic only: no unobserved RGB is loaded to create the suggestions.
    observed_region=unary_union([Polygon(v['world_corners_xy_m']) for v in
                                 observed_views+[current_view]]).intersection(map_region)
    refs=[]
    for landmark in landmarks:
        xy=landmark['xy_m']
        delta=np.asarray(xy,dtype=float)-position
        distance=float(np.linalg.norm(delta))
        absolute=bearing(position,xy) if distance>1e-8 else None
        relative=(absolute-heading+180)%360-180 if absolute is not None else None
        pixel=project(specification['world_to_raster'],[xy])[0]
        within_map=all(0<=v<n for v,n in zip(pixel,specification['image_size']))
        visible=within_map and in_view(current_view,xy)
        box=landmark['bbox_xy_m']
        nearest=None
        if box:
            lo=np.array(box[:2]);hi=np.array(box[2:])
            nearest=float(np.linalg.norm(np.maximum(np.maximum(lo-position,np.asarray(position)-hi),0)))
        refs.append(dict(landmark_id=landmark['id'],name=landmark['name'],
            distance_to_center_m=round(distance,2),distance_to_extent_m=round(nearest,2) if nearest is not None else None,
            bearing_deg=round(absolute,2) if absolute is not None else None,
            turn_from_current_heading_deg=round(relative,2) if relative is not None else None,
            center_in_current_view=bool(visible),
            center_image_point=[round(v,1) for v in normalized(current_view,[xy])[0]] if visible else None,
            extent_overlaps_current_view=bool(overlaps(box_points(box),current_view['world_corners_xy_m'])) if box else bool(visible),
            center_previously_observed=bool(within_map and any(in_view(v,xy) for v in observed_views)),
            center_is_valid_exploration_waypoint=within_map,
            **unseen_extent_hints(box,map_region,observed_region,position,heading,current_view['height_m'])))
    return dict(source='public landmark catalog, public map calibration and executed trajectory only',
        current_main_view_id=current_view['view_id'],main_image_up_heading_deg=round(heading_of_view(current_view),2),
        map_corners_xy_m=corners,landmarks=refs,last_executed_moves=trajectory[-3:],
        note='Coverage and probes use only public reference extents and observed footprints. Probes are optional exploration positions, not detections or target predictions. No remote RGB or target coordinates are used.')
