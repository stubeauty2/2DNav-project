"""Compatibility import; implementation lives in CityNav/tools/sam3.py."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module(".tools.sam3", "CityNav")
