"""Summarize tool/API validity separately from navigation performance."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def audit(directory):
    calls = []
    missing_sam3 = []
    land = []
    for path in sorted(directory.glob('*_search.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        for i, step in enumerate(value.get('steps', [])):
            for kind in ('qwen', 'confirm'):
                if kind in step:
                    call = step[kind]
                    calls.append(dict(file=path.name, step=i, kind=kind,
                                      fallback=str(call.get('reason', '')).startswith('fallback:'),
                                      reason=call.get('reason', '')))
                    if kind == 'qwen' and not call.get('sam3_image5'):
                        missing_sam3.append(dict(file=path.name, step=i))
    for path in sorted(directory.glob('*_analysis.json')):
        value = json.loads(path.read_text(encoding='utf-8'))
        if value.get('used_view') == 'absolute':
            land.append(dict(file=path.name, reason=value.get('reason', ''),
                             clock=value.get('clock')))
    manifest = json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
    package = ROOT/'PSC-AVDN/script/andn_core'
    source_changed = [name for name, digest in manifest['source_hashes'].items()
                      if hashlib.sha256((package/name).read_bytes()).hexdigest() != digest]
    result = dict(search_calls=sum(c['kind']=='qwen' for c in calls),
                  confirmation_calls=sum(c['kind']=='confirm' for c in calls),
                  search_without_sam3=missing_sam3,
                  localization_fallbacks=[c for c in calls if c['fallback']],
                  land_calls=land, source_changed=source_changed,
                  parse_csv_unchanged=hashlib.sha256(Path(manifest['parse_csv']).read_bytes()).hexdigest()==manifest['parse_sha256'])
    (directory/'runtime_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]).resolve())
