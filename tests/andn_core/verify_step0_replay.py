"""Check STEP 0 against frozen baseline, disabled trace, and archived wire data."""
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'out/step0_observability_20261007'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    baseline=read(ROOT/'out/sam3_core_restructure_20261006/replay_before/audit.json')
    disabled=read(OUTPUT/'final_off/replay_after/audit.json')
    directory=OUTPUT/'verified/replay_after'
    enabled=read(directory/'audit.json')
    for key in ('requests','result','stages'):
        assert baseline[key]==disabled[key]==enabled[key],key
    assert not (OUTPUT/'final_off/replay_after/trace').exists()
    trace=next((directory/'trace').iterdir())
    manifest=read(trace/'manifest.json')
    for relative,sha in manifest['source_sha256'].items():
        assert digest((trace/'snapshot'/relative).read_bytes())==sha,relative
        assert digest((ROOT/relative).read_bytes())==sha,relative
    assert digest((trace/'snapshot/parsing_results.csv').read_bytes())==manifest['input_files']['parse_csv']['sha256']
    calls=[read(p) for p in sorted((trace/'calls').glob('*.json')) if not p.name.endswith('.request.json')]
    assert len(calls)==75
    image_count=0
    for call, reference in zip(calls,baseline['requests']):
        payload=call['request']
        assets={a['path']:a for a in call['images']}
        for message in payload['messages']:
            if isinstance(message['content'],list):
                for block in message['content']:
                    if block['type']=='image_url':
                        relative=block['image_url']['url'].removeprefix('asset:')
                        data=(trace/relative).read_bytes()
                        assert digest(data)==assets[relative]['sha256']
                        block['image_url']['url']=assets[relative]['data_url_prefix']+','+base64.b64encode(data).decode()
                        image_count+=1
        assert digest(json.dumps(payload,ensure_ascii=False,sort_keys=True).encode('utf-8'))==call['request_payload_sha256']==reference['payload_sha256']
    summary=read(trace/'summary.json')
    assert summary['status']=='completed' and summary['recording_errors']==0
    assert image_count==summary['counts']['images']==323
    events=[json.loads(line) for line in (trace/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    counts=Counter(e['event'] for e in events)
    assert counts['trajectory_result']==20 and counts['search_bbox']==49 and counts['confirmation_bbox']==26
    original=read(OUTPUT/'andh_core_before_sha256.json')
    pkg=ROOT/'PSC-AVDN/script/andh_core'
    actual={str(p.relative_to(pkg)):digest(p.read_bytes()) for p in pkg.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert original==actual
    result=dict(status='passed', baseline_ref='57b9ecb31c2b64967d6dea9aaebdbce92c8f02a8',
                trajectories=20, matching_request_payloads=75, reconstructed_image_occurrences=image_count,
                trace_on_off_and_original_equal=['requests','stages','results'],
                source_snapshots_match=True, andh_core_unchanged=True, recording_errors=0,
                metrics=enabled['result']['metrics'], trace_directory=str(trace), event_counts=dict(counts),
                limitations=['Offline fixed-response replay, not a new online experiment.',
                             'SAM3 images and Land replies use historical fixtures in full replay; no GPU inference.',
                             'Actual Land HTTP and SAM3 client transport equality/error logging tested separately in test_telemetry.py.'])
    (OUTPUT/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
