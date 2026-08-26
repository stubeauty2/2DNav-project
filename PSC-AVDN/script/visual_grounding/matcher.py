"""DINOv3 dense patch matcher with visual diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from .contracts import MatchResult


class DinoV3Matcher:
    def __init__(self, repo: str, checkpoint: str, device: str = "cuda", image_size: int = 768):
        self.repo = str(Path(repo).expanduser().resolve())
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = device
        self.image_size = int(image_size // 16 * 16)
        self.model = None
        self.torch = None

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from torchvision.transforms import v2

        if not Path(self.repo).is_dir():
            raise FileNotFoundError(f"DINOv3 repository not found: {self.repo}")
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {self.checkpoint}")
        # Import the backbone directly. Loading the repository's broad
        # ``hubconf.py`` also imports optional segmentation/evaluation stacks
        # (torchmetrics, mmcv) that dense feature matching does not need.
        if self.repo not in sys.path:
            sys.path.insert(0, self.repo)
        from dinov3.hub.backbones import dinov3_vitb16

        model = dinov3_vitb16(weights=self.checkpoint).to(self.device).eval()
        self.model = model
        self.torch = torch
        self.transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize((self.image_size, self.image_size)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def _features(self, image: Image.Image):
        self.load()
        tensor = self.transform(np.asarray(image.convert("RGB")).copy()).unsqueeze(0).to(self.device)
        with self.torch.inference_mode():
            tokens = self.model.forward_features(tensor)["x_norm_patchtokens"][0]
        side = int(round(tokens.shape[0] ** 0.5))
        return tokens[: side * side].reshape(side, side, -1)

    def match(
        self,
        candidate: Dict,
        source_image_path: str,
        target_view_id: str,
        target_image_path: str,
        artifact_dir: Path,
        expected_point: Optional[Sequence[float]] = None,
    ) -> MatchResult:
        source_image = Image.open(source_image_path).convert("RGB")
        target_image = Image.open(target_image_path).convert("RGB")
        source_tokens = self._features(source_image)
        target_tokens = self._features(target_image)
        grid_h, grid_w = source_tokens.shape[:2]
        mask = cv2.imread(str(candidate["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"candidate mask not found: {candidate['mask_path']}")
        mask_grid = cv2.resize(mask, (grid_w, grid_h), interpolation=cv2.INTER_AREA) > 64
        if not mask_grid.any():
            x1, y1, x2, y2 = [float(item) for item in candidate["bbox_2d"]]
            mask_grid[
                int(y1 / source_image.height * grid_h):max(1, int(np.ceil(y2 / source_image.height * grid_h))),
                int(x1 / source_image.width * grid_w):max(1, int(np.ceil(x2 / source_image.width * grid_w))),
            ] = True
        reference = source_tokens[self.torch.from_numpy(mask_grid).to(self.device)].mean(dim=0)
        reference = self.torch.nn.functional.normalize(reference, dim=0)
        target_norm = self.torch.nn.functional.normalize(target_tokens, dim=-1)
        similarity = self.torch.einsum("hwc,c->hw", target_norm, reference).float().cpu().numpy()
        row, col = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
        point = [
            float((col + 0.5) / similarity.shape[1] * target_image.width),
            float((row + 0.5) / similarity.shape[0] * target_image.height),
        ]
        x1, y1, x2, y2 = [float(item) for item in candidate["bbox_2d"]]
        bw = (x2 - x1) / source_image.width * target_image.width
        bh = (y2 - y1) / source_image.height * target_image.height
        box = [point[0] - bw / 2, point[1] - bh / 2, point[0] + bw / 2, point[1] + bh / 2]
        geometry = 1.0
        if expected_point is not None and len(expected_point) == 2:
            diagonal = max(1.0, float(np.hypot(target_image.width, target_image.height)))
            distance = float(np.hypot(point[0] - float(expected_point[0]), point[1] - float(expected_point[1])))
            geometry = max(0.0, 1.0 - distance / (0.5 * diagonal))
        normalized = cv2.normalize(similarity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.applyColorMap(cv2.resize(normalized, target_image.size), cv2.COLORMAP_TURBO)
        base = np.asarray(target_image)[:, :, ::-1]
        visual = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
        cv2.drawMarker(visual, (int(point[0]), int(point[1])), (255, 255, 255), cv2.MARKER_CROSS, 30, 3)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = artifact_dir / f"{candidate['candidate_id']}_to_{target_view_id}_dinov3.png"
        cv2.imwrite(str(heatmap_path), visual)
        return MatchResult(
            candidate_id=str(candidate["candidate_id"]),
            target_view_id=target_view_id,
            matched_point_2d=point,
            matched_bbox_2d=[float(value) for value in box],
            similarity=float(similarity[row, col]),
            geometric_consistency=float(geometry),
            heatmap_path=str(heatmap_path.resolve()),
        )
