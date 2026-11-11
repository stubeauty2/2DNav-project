"""Resumable online initialization with public-only request journals."""
import json
import os
from pathlib import Path
from AVDN.agent.io import read_jsonl, write_jsonl, save_json, load_json, digest
from AVDN.initialization.client import call_model, DEFAULT_URL
from AVDN.initialization.budget import RequestBudget
from .prompts import SYSTEM
from .parser import parse


def initialize(root, model='qwen3.8-flash', url=DEFAULT_URL):
    root = Path(root)
    key = os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('API_KEY')
    if not key:
        raise ValueError('Configure DASHSCOPE_API_KEY or API_KEY in process environment')
    budget = RequestBudget(root / 'initialization_budget.json', None, save_json)
    episodes, outcomes = [], []
    for entry in read_jsonl(root / 'inputs.jsonl'):
        agent = entry['agent_input']
        messages = [dict(role='system', content=SYSTEM),
                    dict(role='user', content=json.dumps(agent, ensure_ascii=False))]
        signature = digest(dict(messages=messages, model=model, url=url))
        path = root / 'initialization' / (entry['episode_id'] + '.json')
        if path.exists():
            record = load_json(path)
            if record['signature'] != signature:
                raise ValueError('Changed initialization input; use a new experiment')
            if record['status'] == 'pending':
                raise ValueError('Uncertain pending API request; do not automatically reissue')
        else:
            reservation = budget.reserve(model, messages, 2048)
            record = dict(signature=signature, status='pending', messages=messages, model=model)
            save_json(path, record)
            try:
                response = call_model(url, model, key, messages, timeout=90,
                                      enable_thinking=False, max_tokens=2048,
                                      response_format={'type': 'json_object'})
                budget.settle(reservation, response.get('usage'))
                record.update(status='ok' if response['status'] == 200 else 'api_error', response=response)
            except Exception as exc:
                budget.settle(reservation, None)
                record.update(status='api_error', error=type(exc).__name__)
            save_json(path, record)
        outcome = dict(episode_id=entry['episode_id'], status=record['status'])
        if record['status'] == 'ok':
            try:
                result, diagnostics = parse(record['response']['text'], agent['initial_pose']['heading_deg'],agent['instruction_raw'])
                episodes.append(dict(entry, initialization=result))
                outcome.update(initialization=result, diagnostics=diagnostics)
            except (ValueError, TypeError, KeyError) as exc:
                outcome.update(status='parse_error', error=str(exc))
        outcomes.append(outcome)
        write_jsonl(root / 'initialization_results.jsonl', outcomes)
        write_jsonl(root / 'episodes.jsonl', episodes)
        print(entry['episode_id'], outcome['status'], flush=True)
    return outcomes
