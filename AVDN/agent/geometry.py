"""Compatibility import; implementation lives in AVDN/tools/geometry.py."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module(".tools.geometry", "AVDN")
