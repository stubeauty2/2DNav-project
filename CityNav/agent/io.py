import hashlib
import json
from pathlib import Path
from ..initialization.cli import save_json, read_jsonl, write_jsonl

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def runtime_hash():
    base = Path(__file__).parent.parent
    paths = list(Path(__file__).parent.glob('*.py')) + list((base/'tools').glob('*.py'))
    paths += [base/'navigation.py', base/'config.py']
    paths += [Path(__file__).parent.parent/'initialization'/n for n in ('client.py','budget.py','views.py','cli.py')]
    return digest({str(p.relative_to(Path(__file__).parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(paths)})

def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

