"""Frozen SAM3 defaults; only the output directory is isolated."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

RESULTS_CSV = str(PROJECT_ROOT / "out" / "parse" / "preds_out_Parse_test_unseen_full" / "parsing_results_full.csv")

ANNO_DIR = str(PROJECT_ROOT / "datasets" / "FULL")

DATASET_DIR = str(PROJECT_ROOT / "datasets" / "AVDN")

SPLIT       = "test_unseen_full"

OUT_DIR = str(PROJECT_ROOT / "out" / "andn_core" / "current")

SCALE_FACTOR       = 5

FIXED_CROP_SIDE    = 768

DEFAULT_API_KEY = os.getenv("API_KEY", "")

QWEN_URL        = os.getenv("QWEN_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")

QWEN_MODEL      = os.getenv("QWEN_MODEL", "qwen-vl-max")

VISION_TOOL_URL = os.getenv("VISION_TOOL_URL", "http://127.0.0.1:8765")

SEARCH_QWEN_TIMEOUT = 600

