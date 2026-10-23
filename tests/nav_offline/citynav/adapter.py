import hashlib
import json
from pathlib import Path
from .geometry import city_heading


def named_catalog(objects):
    """Public named objects only. Never expose descriptions or target labels."""
    catalog = []
    for key, obj in sorted(objects.items()):
        if not obj.get('name'):
            continue
        contour = obj.get('contour', [])
        bbox = ([min(p[0] for p in contour), min(p[1] for p in contour),
                 max(p[0] for p in contour), max(p[1] for p in contour)] if contour else None)
        catalog.append({'id': 'lm-'+str(key), 'name': obj['name'], 'type': obj['object_type'],
                        'xy_m': obj['position'][:2], 'bbox_xy_m': bbox})
    return catalog


def load_episodes(annotation, objects_path, landmarks_path):
    rows = json.loads(Path(annotation).read_text(encoding='utf-8'))
    objects = json.loads(Path(objects_path).read_text(encoding='utf-8'))
    landmarks = json.loads(Path(landmarks_path).read_text(encoding='utf-8'))
    for i, row in enumerate(rows):
        if len(row['object_ids']) != 1 or len(row['descriptions']) != 1:
            raise ValueError('Multi-goal CityNav episode needs an explicit adapter; refusing silent first-goal selection')
        map_name = f"{row['area']}_block_{row['block']}"
        p = row['trajectory'][0]
        raw = row['descriptions'][0]
        ident = 'city-' + hashlib.sha256((Path(annotation).name+':'+str(i)).encode()).hexdigest()[:16]
        private = {'dataset': 'CityNav', 'source_row': i, 'map_name': map_name,
                   'start_xy_m': p[:2], 'initial_heading_deg': city_heading(p),
                   'goal_xy_m': objects[map_name][str(row['object_ids'][0])]['position'][:2],
                   'goal_definition': 'CityRefer target object position, not human marker'}
        catalog = named_catalog(landmarks.get(map_name, {}))
        for setting in ('landmark_assisted',):
            episode_id = ident+'-'+setting
            agent = {'dataset': 'CityNav', 'setting': setting, 'instruction_raw': raw,
                     'turns': [{'turn_id': 't0', 'role': 'INS', 'text': raw}],
                     'initial_pose': {'xy_m': p[:2], 'heading_deg': city_heading(p)},
                     'coordinate_frame': 'map world x=east y=north meters',
                     'named_landmarks': catalog if setting == 'landmark_assisted' else [], 'views': []}
            yield episode_id, agent, dict(private, episode_id=episode_id), {'map_name': map_name, 'pose': p}


def render_initial(context, image_dir, output, side_m=100):
    import tifffile
    from .views import crop, heading_corners
    folder = Path(image_dir)
    with tifffile.TiffFile(folder/(context['map_name']+'.tif')) as tif:
        tags = tif.pages[0].tags
        sx, sy, _ = tags['ModelPixelScaleTag'].value
        ix, iy, _, wx, wy, _ = tags['ModelTiepointTag'].value
    pose = context['pose']
    world = heading_corners(pose[:2], city_heading(pose), side_m)
    raster = [[(x-wx)/sx+ix, (wy-y)/sy+iy] for x, y in world]
    view = crop(folder/(context['map_name']+'.png'), raster, world, output)
    view['fov_side_m'] = side_m
    view['observation_policy'] = 'fixed 100m default square, heading-up; altitude/pitch ignored in this 2D initialization experiment'
    return view
