"""Hold a user-entered API key only in memory, preflight, then run the frozen study."""
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys
import time

WORK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORK))
from AVDN.tools.model import call_model

def main():
    key = getpass.getpass('API key (hidden): ')
    if not key.strip():
        raise RuntimeError('Empty credential')
    urls = ['https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation',
            'https://maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation']
    results = []
    selected = None
    for url in urls:
        good = True
        for model in ('qwen3.8-flash', 'qwen3.8-max'):
            try:
                response = call_model(url, model, key,
                    [{'role': 'user', 'content': 'Reply with the single word OK.'}],
                    timeout=90, enable_thinking=False, max_tokens=64, temperature=0)
                row = dict(endpoint=url, model=model, status=response['status'],
                           text=response.get('text'), usage=response.get('usage'))
                good = good and response['status'] == 200
            except Exception as exc:
                row = dict(endpoint=url, model=model, error_type=type(exc).__name__)
                good = False
            results.append(row)
            print(json.dumps(row), flush=True)
        if good:
            selected = url
            break
    folder = WORK/'out/jev_study_20261022/online_preflight'
    folder.mkdir(parents=True, exist_ok=True)
    (folder/'results.json').write_text(json.dumps(dict(selected_endpoint=selected, results=results),indent=2), encoding='utf-8')
    if selected is None:
        print('No usable endpoint; credential not persisted.', flush=True)
        return
    print('PREFLIGHT_OK. Waiting for RUN after local services and frozen plan are ready.', flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            time.sleep(1)
            continue
        if line.strip() == 'RUN':
            break
    env = os.environ.copy()
    env['DASHSCOPE_API_KEY'] = key
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    script = str(WORK/'tests/jev/online_experiment.py')
    subprocess.run([sys.executable, '-B', script, 'prepare', '--endpoint', selected],
                   cwd=WORK, env=env, check=True)
    subprocess.run([sys.executable, '-B', script, 'run', '--max-workers', '2'],
                   cwd=WORK, env=env, check=True)
    subprocess.run([sys.executable, '-B', script, 'analyze'], cwd=WORK, check=True)

if __name__ == '__main__':
    main()
