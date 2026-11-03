"""Resume budget-interrupted format repairs using saved first responses.

Without --execute this only validates and prepares the job manifest. It never
reads evaluator data or repeats a completed repair, including a failed repair.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from .budget import RequestBudget
from .cli import model_messages, read_jsonl, save_json, write_jsonl
from .client import call_model
from .contracts import exploration_seed, grounding_query, parse_model_output, resolve_direction
from .prompts import system_prompt


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, inputs):
    source = Path(source)
    manifest = read(source/'run_manifest.json')
    if manifest['status'] != 'completed':
        raise ValueError('The source run must be terminal before selecting repairs')
    if digest(inputs) != manifest['agent_inputs_sha256']:
        raise ValueError('Use the unchanged original public input file')
    if hashlib.sha256(system_prompt().encode()).hexdigest() != manifest['system_prompt_sha256']:
        raise ValueError('Prompt differs from the source run')
    for name, expected in manifest['runtime_source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Runtime differs from source: '+name)
    public = {r['episode_id']:r['agent_input'] for r in read_jsonl(inputs)}
    jobs = []
    for record in read_jsonl(source/'predictions.jsonl'):
        if record.get('error') != 'BudgetExceeded' or not record.get('usage'):
            continue
        if record.get('repair_attempts') != 1 or record.get('repair_usage'):
            raise ValueError('Record is not an unexecuted first format repair')
        ident = record['episode_id']
        answer = source/'calls'/(ident+'.answer.json')
        feedback = source/'calls'/(ident+'.repair.json')
        messages, _ = model_messages(public[ident])
        messages += [{'role':'assistant', 'content':read(answer)['content']},
                     {'role':'user', 'content':read(feedback)['feedback']}]
        jobs.append({'record':record, 'agent':public[ident], 'messages':messages,
                     'answer_sha256':digest(answer), 'feedback_sha256':digest(feedback)})
    return manifest, jobs


def run(args):
    manifest, jobs = prepare(args.source_run, args.inputs)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    signature = {'source_run':str(Path(args.source_run).resolve()),
                 'source_manifest_sha256':digest(Path(args.source_run)/'run_manifest.json'),
                 'source_predictions_sha256':digest(Path(args.source_run)/'predictions.jsonl'),
                 'repair_runtime_sha256':digest(__file__),
                 'budget_ledger':str(Path(args.budget_ledger).resolve()),
                 'max_cost_cny':args.max_cost_cny,
                 'jobs':[{k:v for k,v in j.items() if k.endswith('_sha256')} |
                         {'episode_id':j['record']['episode_id']} for j in jobs]}
    job_path = out/'repair_manifest.json'
    previous = read(job_path) if job_path.exists() else None
    if previous and previous['signature'] != signature:
        raise ValueError('Repair output belongs to a different source or runtime')
    state = {'signature':signature, 'episodes':len(jobs), 'gt_read':False,
             'status':'prepared', 'initial_calls_repeated':0}
    if previous:
        state = previous
    else:
        save_json(job_path, state)
    if not args.execute:
        return state
    key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('API_KEY') or os.getenv('QWEN_API_KEY')
    if not key:
        raise ValueError('Configure an API key in the process environment')
    budget = RequestBudget(args.budget_ledger, args.max_cost_cny, save_json)
    state.update(status='running', pid=os.getpid())
    save_json(job_path, state)
    results = []
    for job in jobs:
        original = job['record']; ident = original['episode_id']
        target = out/'records'/(ident+'.json')
        if target.exists():
            results.append(read(target))
            continue
        started = out/'calls'/(ident+'.started.json')
        if started.exists():
            raise ValueError('An interrupted repair requires inspection before retry: '+ident)
        record = dict(original)
        record.pop('error', None); record.pop('validation_detail', None)
        record['deferred_repair_source'] = str(Path(args.source_run).resolve())
        begin = time.monotonic(); response = None; reservation = None
        try:
            reservation = budget.reserve(manifest['model'], job['messages'],
                                         manifest['inference_parameters']['max_tokens'])
            save_json(started, {'episode_id':ident, 'first_answer_reused':True})
            response = call_model(manifest['endpoint'], manifest['model'], key,
                                  job['messages'], args.timeout, manifest['protocol'],
                                  **manifest['inference_parameters'])
            record['repair_usage'] = response.get('usage')
            if response['status'] != 200:
                raise ValueError('Correction HTTP '+str(response['status']))
            save_json(out/'calls'/(ident+'.corrected_answer.json'), {'content':response['text']})
            parsed, diagnostics = parse_model_output(response['text'], job['agent'], require_plan=False)
            record.update(status='ok', parsed=parsed, diagnostics=diagnostics,
                          grounding_query=grounding_query(parsed))
            record['resolved_direction'] = resolve_direction(parsed, job['agent'])
            record['resolved_direction']['source'] = diagnostics['direction_source']
            save_json(out/'labels'/(ident+'.json'), parsed)
        except Exception as exc:
            record.update(status='error', error=type(exc).__name__)
            if isinstance(exc, ValueError):
                record['validation_detail'] = str(exc)[:600]
        finally:
            if reservation is not None:
                budget.settle(reservation, response.get('usage') if response else None)
        record['exploration_seed'] = exploration_seed(record.get('parsed'), job['agent'], record.get('diagnostics'))
        record['elapsed_seconds'] = round(original['elapsed_seconds']+time.monotonic()-begin, 3)
        save_json(target, record)
        results.append(record)
        write_jsonl(out/'predictions.jsonl', results)
    write_jsonl(out/'predictions.jsonl', results)
    state.update(status='completed', completed=len(results), ok=sum(r['status']=='ok' for r in results))
    save_json(job_path, state)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--budget-ledger', required=True)
    parser.add_argument('--max-cost-cny', type=float, default=10)
    parser.add_argument('--timeout', type=float, default=60)
    parser.add_argument('--execute', action='store_true', help='Send repairs; omission is strictly offline')
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False))
