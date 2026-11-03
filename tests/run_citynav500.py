"""Public-only initialization then the archived navigation batch driver."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
OUT=Path('C:/Levir/2DNav/2DNav-project/out/23_citynav500_decision35/20261103')


def main():
    from CityNav.initialization.cli import run
    from CityNav.initialization.client import DEFAULT_URL
    switch_path=OUT/'initialization/model_switch.json'
    switched=switch_path.exists()
    input_path=OUT/('initialization/remaining_flash_inputs.jsonl' if switched else 'initialization/agent_inputs.jsonl')
    init_output=OUT/('initialization/run_flash' if switched else 'initialization/run')
    args=SimpleNamespace(inputs=str(input_path),out=str(init_output),
        model='qwen3.8-flash' if switched else 'qwen3.8-max',url=DEFAULT_URL,protocol='auto',workers=4,timeout=180,
        thinking='on',thinking_budget=2048,max_output_tokens=2048,max_cost_cny=None,budget_ledger=None)
    run(args)
    predictions={r['episode_id']:r for r in map(json.loads,(init_output/'predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    if switched:
        switch=json.loads(switch_path.read_text(encoding='utf-8'))
        for eid in switch['retained_ids']:
            record=json.loads((OUT/'initialization/run/records'/(eid+'.json')).read_text(encoding='utf-8'))
            assert record['status']=='ok' and eid not in predictions
            predictions[eid]=record
    (OUT/'initialization/combined_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in predictions.values()),encoding='utf-8')
    failed=[eid for eid,r in predictions.items() if r['status']!='ok']
    if failed:
        (OUT/'initialization_failures.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
        raise RuntimeError(f'{len(failed)} initialization failures require inspection; no samples replaced')
    public=[json.loads(s) for s in (OUT/'citynav/public_uninitialized.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(public)==500 and len(predictions)==500
    for row in public:row['initialization']=predictions[row['episode_id']]['parsed']
    (OUT/'citynav/episodes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in public),encoding='utf-8')
    command=[sys.executable,'-u','-B',str(SOURCE/'tests/run_frozen_agent_batch.py'),
        '--dataset','CityNav','--source',str(SOURCE),'--root',str(OUT),
        '--budget',str(OUT/'navigation_budget.json'),'--model','qwen3.8-max',
        '--thinking-budget','2048','--workers','4','--sam3-url','http://127.0.0.1:8765',
        '--dino-url','http://127.0.0.1:8766','--observation-grounder','sam3',
        '--observation-query-mode','category']
    subprocess.run(command,cwd=SOURCE,check=True)


if __name__=='__main__':main()
