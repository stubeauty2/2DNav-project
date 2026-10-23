"""Atomic artifacts and complete online-package fingerprints."""
import hashlib
import json
import os
import time
from pathlib import Path

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(str(path)+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    replace_retry(temp, path)

def replace_retry(temp, path):
    # Windows readers/indexers can briefly hold the destination without delete sharing.
    for attempt in range(20):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(min(0.02 * (attempt + 1), 0.2))

def write_jsonl(path, rows):
    temp = Path(str(path)+'.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
    replace_retry(temp, path)
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def runtime_hash():
    package = Path(__file__).resolve().parents[1]
    return digest({p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(package.rglob('*.py'))})
