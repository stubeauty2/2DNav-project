"""Lazy visual-model backends used by the standalone service.

Heavy imports intentionally live inside ``load`` so importing the navigation
code never imports the service's Python/Torch environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class RawProposal:
    mask: np.ndarray
    bbox: List[float]
    score: float
    mask_quality: float
    component_scores: Dict[str, float]


class CandidateBackend:
    name = "base"

    def __init__(self, device: str = "cuda", score_threshold: float = 0.08):
        self.device = device
        self.score_threshold = float(score_threshold)
        self.loaded = False
        self.weights: Dict[str, str] = {}

    def load(self) -> None:
        raise NotImplementedError

    def propose(self, image: Any, text: str) -> List[RawProposal]:
        raise NotImplementedError


class Sam3Backend(CandidateBackend):
    name = "sam3"

    def __init__(self, checkpoint: str, **kwargs: Any):
        super().__init__(**kwargs)
        self.checkpoint = str(Path(checkpoint).expanduser().resolve())
        self.processor = None

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
        self.processor = Sam3Processor(
            model,
            device=self.device,
            confidence_threshold=self.score_threshold,
        )
        self.weights = {"sam3": self.checkpoint}
        self.loaded = True

    def propose(self, image: Any, text: str) -> List[RawProposal]:
        self.load()
        state = self.processor.set_image(image.convert("RGB"))
        state = self.processor.set_text_prompt(prompt=text, state=state)
        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")
        if masks is None or boxes is None or scores is None:
            return []
        masks_np = masks.detach().float().cpu().numpy()
        boxes_np = boxes.detach().float().cpu().numpy()
        scores_np = scores.detach().float().cpu().numpy()
        result: List[RawProposal] = []
        for mask, box, score in zip(masks_np, boxes_np, scores_np):
            value = np.squeeze(mask) > 0.0
            if not value.any():
                continue
            numeric_score = float(np.squeeze(score))
            result.append(
                RawProposal(
                    mask=value,
                    bbox=[float(item) for item in np.asarray(box).reshape(-1)[:4]],
                    score=numeric_score,
                    mask_quality=numeric_score,
                    component_scores={"sam3": numeric_score},
                )
            )
        return result


class GroundedSam2Backend(CandidateBackend):
    name = "grounded-sam2"

    def __init__(
        self,
        grounding_model: str = "IDEA-Research/grounding-dino-base",
        sam2_model: str = "facebook/sam2.1-hiera-small",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.grounding_model_id = grounding_model
        self.sam2_model_id = sam2_model
        self.grounding_processor = None
        self.grounding_model = None
        self.sam_processor = None
        self.sam_model = None

    def load(self) -> None:
        if self.loaded:
            return
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        # Grounding DINO's text/vision fusion contains operations that remain
        # float32 in Transformers; forcing the whole model to fp16 creates
        # mixed-dtype matmul failures. The base+small pair still fits the 8 GB
        # target in float32.
        dtype = torch.float32
        self.dtype = dtype
        self.grounding_processor = AutoProcessor.from_pretrained(self.grounding_model_id)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.grounding_model_id, dtype=dtype
        ).to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(self.sam2_model_id)
        self.sam_model = Sam2Model.from_pretrained(
            self.sam2_model_id, dtype=dtype
        ).to(self.device).eval()
        grounding_revision = getattr(self.grounding_model.config, "_commit_hash", None)
        sam_revision = getattr(self.sam_model.config, "_commit_hash", None)
        self.weights = {
            "grounding_dino": self.grounding_model_id + (f"@{grounding_revision}" if grounding_revision else ""),
            "sam2": self.sam2_model_id + (f"@{sam_revision}" if sam_revision else ""),
        }
        self.loaded = True

    def propose(self, image: Any, text: str) -> List[RawProposal]:
        self.load()
        import torch

        rgb = image.convert("RGB")
        query = text.strip().rstrip(".") + "."
        inputs = self.grounding_processor(images=rgb, text=query, return_tensors="pt")
        inputs = {
            key: value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.grounding_model(**inputs)
        detections = self.grounding_processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.score_threshold,
            text_threshold=self.score_threshold,
            target_sizes=[(rgb.height, rgb.width)],
        )[0]
        boxes = detections.get("boxes", torch.empty((0, 4))).detach().cpu()
        scores = detections.get("scores", torch.empty((0,))).detach().cpu()
        if not len(boxes):
            return []

        sam_inputs = self.sam_processor(
            images=rgb,
            input_boxes=[boxes.tolist()],
            return_tensors="pt",
        )
        sam_inputs = {
            key: value.to(self.device, dtype=self.dtype) if value.is_floating_point() else value.to(self.device)
            for key, value in sam_inputs.items()
        }
        with torch.inference_mode():
            sam_outputs = self.sam_model(**sam_inputs, multimask_output=False)
        processed = self.sam_processor.post_process_masks(
            sam_outputs.pred_masks.cpu(),
            sam_inputs["original_sizes"].cpu(),
        )[0]
        iou_scores = sam_outputs.iou_scores.detach().float().cpu().reshape(-1)
        result: List[RawProposal] = []
        for index, (box, score) in enumerate(zip(boxes, scores)):
            mask = processed[index].detach().float().cpu().numpy().squeeze() > 0.0
            quality = float(iou_scores[index]) if index < len(iou_scores) else 0.0
            detection = float(score)
            result.append(
                RawProposal(
                    mask=mask,
                    bbox=[float(item) for item in box.tolist()],
                    score=detection,
                    mask_quality=quality,
                    component_scores={"grounding_dino": detection, "sam2_iou": quality},
                )
            )
        return result


def create_backend(name: str, device: str, score_threshold: float, sam3_checkpoint: str) -> CandidateBackend:
    if name == "sam3":
        return Sam3Backend(
            checkpoint=sam3_checkpoint,
            device=device,
            score_threshold=score_threshold,
        )
    if name == "grounded-sam2":
        return GroundedSam2Backend(device=device, score_threshold=score_threshold)
    raise ValueError(f"unsupported grounding backend: {name}")
