"""Offline raster fixture shared by current navigation tests."""
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'PSC-AVDN/script'
sys.path[:0] = [str(SCRIPT), str(ROOT / 'src')]


class NavigationFixture:
    def setUp(self):
        (ROOT / 'tests/tmp').mkdir(exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'tests/tmp')
        self.root = Path(self.tmp.name)
        (self.root / 'train_images').mkdir()
        raster = np.empty((2400,2400,3), np.uint8)
        raster[:,:,0] = np.arange(2400,dtype=np.uint8)[None,:]
        raster[:,:,1] = np.arange(2400,dtype=np.uint8)[:,None]
        raster[:,:,2] = raster[:,:,0] + raster[:,:,1]
        assert cv2.imwrite(str(self.root / 'train_images/map.tif'), raster)
        self.ob = dict(dataset_dir=str(self.root), map_name='map', map_size=[2400,2400],
                       gps_botm_left=[0,0], gps_top_right=[.1,.1],
                       gt_path_corners=[[[.049,.049],[.049,.051],[.051,.051],[.051,.049]]])
        self.out = self.root / 'out'
        self.out.mkdir()

    def tearDown(self):
        self.tmp.cleanup()
