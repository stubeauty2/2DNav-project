"""Initial crops with an explicit pixel/world transform and validity mask.

No corner clipping: outside-map pixels remain black and invalid. This differs
from the historical AVDN renderer, which distorts crops by clipping each corner.
"""
import math
import numpy as np
from PIL import Image




def heading_corners(xy, heading, side):
    h = math.radians(heading or 0)
    forward = np.array([math.sin(h), math.cos(h)]) * side/2
    right = np.array([math.cos(h), -math.sin(h)]) * side/2
    p = np.asarray(xy)
    return np.array([p+forward-right, p+forward+right, p-forward+right, p-forward-right]).tolist()


def crop(image_path, raster_corners, world_corners, output, size=512):
    corners = [(0, 0), (size-1, 0), (size-1, size-1), (0, size-1)]
    to_source = homography(corners, raster_corners)
    coeffs = tuple(to_source.flatten()[:8])
    # Local source rasters are trusted large dataset assets, not external uploads.
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(image_path) as raw:
            image = raw.convert('RGB')
        view = image.transform((size, size), Image.Transform.PERSPECTIVE, coeffs,
                               resample=Image.Resampling.BICUBIC)
        valid = Image.new('L', image.size, 255).transform((size, size), Image.Transform.PERSPECTIVE,
                                                       coeffs, resample=Image.Resampling.NEAREST)
        output.parent.mkdir(parents=True, exist_ok=True)
        view.save(output)
        mask_path = output.with_name(output.stem + '.valid.png')
        valid.save(mask_path)
        fraction = float(np.mean(np.asarray(valid) > 0))
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    return {'view_id': 'initial', 'image_path': str(output.resolve()),
            'valid_mask_path': str(mask_path.resolve()), 'valid_fraction': fraction,
            'image_size': [size, size], 'world_corners_xy_m': world_corners,
            'pixel_to_world': homography(corners, world_corners).tolist(),
            'valid_mask_scope': 'raster bounds only; point-cloud holes and occlusion are not measured',
            'source': 'initial_observation', 'outside_map_pixels': 'black, invalid'}

from AVDN.tools.geometry import homography
