"""Legacy imports for the implementation in andh_core.tools.grounding."""
from importlib import import_module
import sys

# Keep old module imports and monkeypatch targets bound to the same objects.
for _name in ("backends", "client", "contracts", "geometry", "postprocess"):
    _module = import_module("andh_core.tools.grounding." + _name)
    sys.modules[__name__ + "." + _name] = _module
    globals()[_name] = _module
