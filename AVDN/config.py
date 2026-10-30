"""Public runtime defaults. Credentials are read only from process environment."""
import os
MODEL = 'qwen3.5-flash'
VISUAL_MODEL = 'qwen3.8-flash'  # Evidence tool only; never initialization, decision or review.
ENABLE_THINKING = False
MAX_ROUNDS = 12
MAX_TOOLS_PER_ROUND = 4
MAX_REQUESTS = 48
MAX_OUTPUT_TOKENS = 2048
SAM3_URL = os.getenv('SAM3_URL') or os.getenv('VISION_TOOL_URL')
DINO_URL = os.getenv('DINO_URL')
# Raw tool in adaptive/direct modes; legacy skill mode uses DINO and SAM3.
LOCATE_ANYTHING_URL = os.getenv('LOCATE_ANYTHING_URL')

API_URL = os.getenv("QWEN_URL") or "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
