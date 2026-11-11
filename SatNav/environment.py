"""Windowed GeoTIFF access with the existing project view/tool interface.

Uses the project's 768px continuous-action protocol, not native SatSim actions.
"""
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
from AVDN.tools.views import Environment as BaseEnvironment


class RasterImage:
    def __init__(self, path):
        self.dataset = rasterio.open(path)
        self.size = (self.dataset.width, self.dataset.height)
        if self.dataset.count < 3:
            raise ValueError('Scene must provide RGB bands')

    def crop(self, box):
        left, top, right, bottom = box
        rgb = self.dataset.read([1, 2, 3], window=Window(left, top, right-left, bottom-top))
        if rgb.dtype != np.uint8:
            raise ValueError('Expected uint8 RGB scene')
        return Image.fromarray(np.moveaxis(rgb, 0, -1), 'RGB')

    def close(self):
        self.dataset.close()


class Environment(BaseEnvironment):
    def __init__(self, specification, output):
        self.spec = specification
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.image = RasterImage(specification['image_path'])
        if list(self.image.size) != specification['image_size']:
            self.image.close()
            raise ValueError('Raster dimensions changed')
        self.views = {}
