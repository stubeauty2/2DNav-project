"""Compatibility import; implementation lives in CityNav/tools/llm.py."""
import sys
from importlib import import_module
sys.modules[__name__] = import_module(".tools.llm", "CityNav")
