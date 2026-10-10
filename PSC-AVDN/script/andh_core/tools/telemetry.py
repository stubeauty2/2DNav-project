"""Optional local JSONL diagnostics; never modifies the localization result."""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path


def append_event(path, event):
    if path is None:
        return
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False) + "\n")
    except Exception as exc:
        warnings.warn(f"Could not write navigation trace: {exc}", RuntimeWarning)
