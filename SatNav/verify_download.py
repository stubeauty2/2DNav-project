"""Independently verify downloaded bytes against both release manifests."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / 'datasets/SatNav'
results = []
for group, folder in [('annotations', root), ('scenes', root / 'scenes_release')]:
    for line in (folder / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        path = folder / name
        actual = None
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            actual = digest.hexdigest()
        result = dict(group=group, path=name, expected=expected, actual=actual,
                      verified=actual == expected,
                      bytes=path.stat().st_size if path.exists() else None)
        results.append(result)
        print(group, name, 'PASS' if result['verified'] else 'FAIL', flush=True)
(root / 'final_verification.json').write_text(json.dumps(results, indent=2))
print('VERIFIED', sum(x['verified'] for x in results), '/', len(results), flush=True)
