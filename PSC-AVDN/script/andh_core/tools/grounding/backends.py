"""Lazy SAM3 backend; importing this module does not import torch or sam3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


@dataclass
class RawProposal:
    mask: np.ndarray
    bbox: List[float]
    score: float


class Sam3Backend:
    name = "sam3"

    def __init__(self, checkpoint: str, device: str = "cuda", score_threshold: float = 0.08):
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.device = device
        self.score_threshold = float(score_threshold)
        self.processor = None
        self.loaded = False

    def load(self) -> None:
        if self.loaded:
            return
        if not Path(self.checkpoint).is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {self.checkpoint}")
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model(
            checkpoint_path=self.checkpoint,
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
            compile=False,
        )
        self.processor = Sam3Processor(model, device=self.device, confidence_threshold=self.score_threshold)
        self.loaded = True

    def propose(self, image: Any, text: str) -> List[RawProposal]:
        self.load()
        state = self.processor.set_image(image.convert("RGB"))
        state = self.processor.set_text_prompt(prompt=text, state=state)
        masks, boxes, scores = state.get("masks"), state.get("boxes"), state.get("scores")
        if masks is None or boxes is None or scores is None:
            return []
        masks = masks.detach().float().cpu().numpy()
        boxes = boxes.detach().float().cpu().numpy()
        scores = scores.detach().float().cpu().numpy()
        result = []
        for mask, box, score in zip(masks, boxes, scores):
            value = np.squeeze(mask) > 0.0
            if value.any():
                value_score = float(np.squeeze(score))
                result.append(RawProposal(value, [float(v) for v in np.asarray(box).reshape(-1)[:4]], value_score))
        return result

