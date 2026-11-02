"""Compatibility import; implementation lives in AVDN/navigation.py."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module(".navigation", "AVDN")
