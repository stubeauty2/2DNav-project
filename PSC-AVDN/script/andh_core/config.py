"""Existing model and image defaults; navigation policy remains in navigation.py."""
import os
import math
from dataclasses import dataclass

FIXED_CROP_SIDE = 768
DEFAULT_API_KEY = os.getenv("API_KEY", "")
QWEN_URL = os.getenv("QWEN_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-vl-max")
SEARCH_QWEN_TIMEOUT = 600


@dataclass(frozen=True)
class NavigationOptions:
    history_size: int = 0
    use_relative_pose: bool = False
    enable_trace: bool = False
    strict_confirmation: bool = False
    model_retries: int = 1
    candidate_count: int = 3
    preview_offset_ratio: float = 0.5
    max_steps: int = 3
    step_meters: float = 120.0

    def __post_init__(self):
        if not 0 <= self.history_size <= 2:
            raise ValueError("history_size must be 0, 1, or 2 for this pilot")
        if not 1 <= self.candidate_count <= 6 or not 0 < self.preview_offset_ratio <= 1:
            raise ValueError("candidate_count must be 1..6 and preview_offset_ratio in (0, 1]")
        if self.max_steps < 1 or not 0 <= self.model_retries <= 3:
            raise ValueError("max_steps must be positive and model_retries must be 0..3")
        if not math.isfinite(self.step_meters) or self.step_meters <= 0:
            raise ValueError("step_meters must be finite and positive")

    @classmethod
    def from_env(cls):
        def flag(name):
            value = os.getenv(name, "0").strip().lower()
            if value not in ("0", "1", "false", "true"):
                raise ValueError(f"{name} must be 0/1/false/true")
            return value in ("1", "true")
        return cls(
            history_size=int(os.getenv("ANDH_HISTORY_SIZE", "0")),
            use_relative_pose=flag("ANDH_RELATIVE_POSE"),
            enable_trace=flag("ANDH_TRACE"),
            candidate_count=int(os.getenv("ANDH_CANDIDATE_COUNT", "3")),
            preview_offset_ratio=float(os.getenv("ANDH_PREVIEW_OFFSET_RATIO", "0.5")),
            strict_confirmation=flag("ANDH_STRICT_CONFIRMATION"),
            model_retries=int(os.getenv("ANDH_MODEL_RETRIES", "1")),
            max_steps=int(os.getenv("ANDH_MAX_STEPS", "3")),
            step_meters=float(os.getenv("ANDH_STEP_METERS", "120")),
        )
