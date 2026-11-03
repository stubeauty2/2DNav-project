"""Compatibility import; implementation lives in CityNav/tools/navigation_context.py."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module(".tools.navigation_context", "CityNav")
