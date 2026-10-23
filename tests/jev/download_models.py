"""Fetch public, revision-pinned OneJev weights; no model execution or secrets."""
import argparse
import hashlib
import json
from pathlib import Path
import time

MODELS = {
    '0.8B': ('OmniJev/OneJev1.6-0.8B', '0acd65660ac22ca42f40b8b6a6da3a039590b412'),
    '4B': ('OmniJev/OneJev1.6-4B', '7d55a87ce87ff77e25c4a43eb637dc3a52342977'),
}
ROOT = Path(__file__).resolve().parents[2] / 'out/jev_study_20261022'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=list(MODELS), required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    repo, revision = MODELS[args.size]
    folder = ROOT / 'models' / repo.split('/')[-1]
    start = time.time()
    snapshot_download(repo, revision=revision, local_dir=str(folder), max_workers=3)
    inventory = {}
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or '.cache' in p.parts:
            continue
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        inventory[p.relative_to(folder).as_posix()] = {'bytes': p.stat().st_size, 'sha256': h.hexdigest()}
    manifest = dict(repository=repo, revision=revision, local_path=str(folder), files=inventory,
                    elapsed_seconds=time.time()-start, completed=True)
    dest = ROOT / 'models' / (repo.split('/')[-1] + '.manifest.json')
    dest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'manifest': str(dest), 'files': len(inventory), 'completed': True}), flush=True)

if __name__ == '__main__':
    main()
