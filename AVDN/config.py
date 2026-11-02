"""Public runtime defaults. Credentials are read only from process environment."""
import os
MODEL = 'qwen3.8-flash'
ENABLE_THINKING = False
MAX_ROUNDS = 12
MAX_TOOLS_PER_ROUND = 2
MAX_REQUESTS = 48
MAX_OUTPUT_TOKENS = 2048
# User removed the spending cap on 2026-10-14; accounting remains enabled.
BUDGET_CNY = None
SAM3_URL = os.getenv('SAM3_URL') or os.getenv('VISION_TOOL_URL')
DINO_URL = os.getenv('DINO_URL')
LOCATE_ANYTHING_URL = os.getenv('LOCATE_ANYTHING_URL')
