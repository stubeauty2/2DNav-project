"""Local visual-grounding service and lightweight client contracts.

The package deliberately keeps its top-level import dependency-free so the
Python 3.9 navigation process can import the HTTP client without importing the
Python 3.12/CUDA model stack used by the service.
"""

__all__ = ["VisionToolClient", "VisionToolError"]


def __getattr__(name):
    if name in __all__:
        from .client import VisionToolClient, VisionToolError
        return {"VisionToolClient": VisionToolClient, "VisionToolError": VisionToolError}[name]
    raise AttributeError(name)
