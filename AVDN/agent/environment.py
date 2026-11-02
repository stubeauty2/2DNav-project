"""Compatibility import; implementation lives in AVDN/tools/views.py."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module(".tools.views", "AVDN")
