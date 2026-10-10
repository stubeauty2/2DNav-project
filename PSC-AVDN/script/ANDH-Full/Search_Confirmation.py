"""Compatibility entry point. The implementation lives in andh_core.runner."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from andh_core import runner as _runner

if __name__ == "__main__":
    _runner.main()
else:
    # Imports and monkeypatches through the old name reach the same module.
    sys.modules[__name__] = _runner
