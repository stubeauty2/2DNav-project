"""Offline preparation. Public inputs and evaluator truth are kept separate."""
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform


def local_xy(position, origin):
    """Local ground meters using start-latitude Web Mercator scale."""
    x, y = transform('EPSG:4326', 'EPSG:3857',
                     [origin[0], position[0]], [origin[1], position[1]])
    scale = math.cos(math.radians(origin[1]))
    return [(x[1] - x[0]) * scale, (y[1] - y[0]) * scale]


def public_input(episode):
    text = episode['instruction']['instruction_text']
    return dict(dataset='SatNav', setting='visual', instruction_raw=text,
                turns=[dict(turn_id='t0', role='INS', text=text)],
                initial_pose=dict(xy_m=[0., 0.], heading_deg=float(episode['start_rotation']) % 360),
                coordinate_frame='local east/north ground meters centered on initial pose',
                named_landmarks=[], views=[])


def raster_spec(image, origin):
    with rasterio.open(image) as raster:
        if raster.crs.to_epsg() != 3857:
            raise ValueError('Expected EPSG:3857 scene; do not silently assume projection')
        x, y = transform('EPSG:4326', raster.crs, [origin[0]], [origin[1]])
        scale = 1 / math.cos(math.radians(origin[1]))
        inverse = ~raster.transform
        projected_to_pixel = np.array([[inverse.a, inverse.b, inverse.c],
                                       [inverse.d, inverse.e, inverse.f], [0, 0, 1]])
        local_to_projected = np.array([[scale, 0, x[0]], [0, scale, y[0]], [0, 0, 1]])
        return dict(image_path=str(Path(image).resolve()), image_size=[raster.width, raster.height],
                    world_to_raster=(projected_to_pixel @ local_to_projected).tolist(),
                    main_width_m=100., main_height_m=100.)


def load_episodes(root, split):
    if split not in ('val_seen', 'val_unseen', 'train'):
        raise ValueError('Unknown split')
    root = Path(root)
    relative = 'train' if split == 'train' else 'eval/' + split
    data = json.loads((root / 'episodes' / relative / 'all_episodes.json').read_text(encoding='utf-8'))
    for episode in data if isinstance(data, list) else data['episodes']:
        if episode['trajectory_type'] != 'LandmarkSet':
            continue
        ident = f"satnav-{split}-{episode['scene_id']}-{episode['episode_id']}"
        public = public_input(episode)
        private = dict(episode_id=ident, split=split, scene_id=episode['scene_id'],
                       trajectory_id=episode['trajectory_id'],
                       goal_xy_m=local_xy(episode['goals'][0]['position'], episode['start_position']),
                       success_radius_m=30.)
        image = root / 'scenes_release/scenes' / (episode['scene_id'] + '.tif')
        yield ident, public, private, dict(image=image, origin=episode['start_position'])
