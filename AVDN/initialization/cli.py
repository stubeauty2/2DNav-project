"""Prepare safe inputs, run initialization, then evaluate predictions offline."""
import argparse
import base64
import hashlib
import json
import os
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .contracts import resolve_direction, validate_agent_input, exploration_seed, grounding_query, parse_model_output, SCHEMA
from .geometry import angular_error, bearing
from .prompts import system_prompt

DATASET = 'AVDN'


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


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from ..adapter import load_episodes, render_initial
    entries = list(load_episodes(args.annotation))
    ids = sorted({x[2]['source_row'] for x in entries})
    if args.limit:
        ids = random.Random(args.seed).sample(ids, min(args.limit, len(ids)))
    selected = set(ids)
    public, private = [], []
    view_cache = {}
    for ident, agent, gt, context in entries:
        row_id = gt['source_row']
        if row_id not in selected:
            continue
        if args.image_dir:
            if row_id not in view_cache:
                view_cache[row_id] = render_initial(context, args.image_dir, out/'images'/f'{row_id}.png')
            agent['views'] = [view_cache[row_id]]
        public.append({'episode_id': ident, 'agent_input': agent})
        private.append(gt)
    write_jsonl(out/'agent_inputs.jsonl', public)
    write_jsonl(out/'evaluator_only.jsonl', private)
    save_json(out/'schema.json', SCHEMA)
    (out/'system_prompt.txt').write_text(system_prompt(), encoding='utf-8')
    save_json(out/'manifest.json', {'dataset': DATASET, 'annotation': str(Path(args.annotation).resolve()),
              'annotation_sha256': hashlib.sha256(Path(args.annotation).read_bytes()).hexdigest(),
              'seed': args.seed, 'sampled_source_rows': sorted(selected), 'episodes': len(public),
              'input_regime': 'initial RGB + text' if args.image_dir else 'text/pose only, no RGB',
              'evaluation_isolation': 'run reads agent_inputs.jsonl only; evaluate reads evaluator_only.jsonl',
              'initialization_only': True})
    print(json.dumps({'episodes': len(public), 'out': str(out.resolve())}))


def model_messages(agent, direction_mode='staged'):
    if agent['dataset'] != DATASET:
        raise ValueError(f'This entry point accepts {DATASET} inputs only')
    validate_agent_input(agent)
    safe = json.loads(json.dumps(agent))
    images = []
    for view in safe['views']:
        p = Path(view.pop('image_path'))
        view.pop('valid_mask_path', None)
        view['asset_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        images.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()}})
    messages = [{'role': 'system', 'content': system_prompt(direction_mode)},
                {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(safe, ensure_ascii=False)}]+images}]
    return messages, safe


def run(args):
    from .client import call_model, DEFAULT_MODEL, DEFAULT_URL
    key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('API_KEY') or os.getenv('QWEN_API_KEY')
    if not key:
        raise SystemExit('Set API_KEY or QWEN_API_KEY in process environment. No key is stored by this program.')
    model = args.model or os.getenv('QWEN_MODEL') or DEFAULT_MODEL
    url = args.url or os.getenv('QWEN_URL') or DEFAULT_URL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    inputs = read_jsonl(args.inputs)
    budget = None
    if getattr(args, 'max_cost_cny', None) is not None:
        from .budget import RequestBudget
        budget = RequestBudget(getattr(args,'budget_ledger',None) or out/'budget.json',
                               args.max_cost_cny, save_json)
    if len({r['episode_id'] for r in inputs}) != len(inputs):
        raise ValueError('Duplicate input ids')

    prompt_text = system_prompt(args.direction_mode) if DATASET == 'AVDN' else system_prompt()
    inference_parameters = {'enable_thinking':getattr(args,'thinking','on')=='on',
                            'thinking_budget':getattr(args,'thinking_budget',50),
                            'max_tokens':getattr(args,'max_output_tokens',4096)}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [Path(__file__), Path(__file__).with_name('contracts.py'),
                               Path(__file__).with_name('client.py'),Path(__file__).with_name('budget.py')]}
    signature = {'model': model, 'endpoint': url, 'protocol': args.protocol,
                 'direction_mode': getattr(args, 'direction_mode', None),
                 'agent_inputs_sha256': hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
                 'system_prompt_sha256': hashlib.sha256(prompt_text.encode()).hexdigest(),
                 'runtime_source_sha256': source_hashes}
    signature['inference_parameters'] = inference_parameters
    signature['budget_limit_cny'] = getattr(args,'max_cost_cny',None)
    signature['budget_ledger'] = str(Path(args.budget_ledger).resolve()) if getattr(args,'budget_ledger',None) else None
    existing = {}
    manifest_path = out/'run_manifest.json'
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding='utf-8'))
        if prior.get('signature') != signature:
            raise ValueError('Output directory belongs to a different prompt/input/runtime; use a new directory')
        if (out/'predictions.jsonl').exists():
            existing = {r['episode_id']:r for r in read_jsonl(out/'predictions.jsonl') if r['status']=='ok'}
        for record_path in (out/'records').glob('*.json'):
            record = json.loads(record_path.read_text(encoding='utf-8'))
            if record['status'] == 'ok':
                existing[record['episode_id']] = record
    elif (out/'predictions.jsonl').exists():
        raise ValueError('Refusing to overwrite an unversioned historical run')
    manifest = dict(signature, signature=signature, episodes=len(inputs), gt_read_by_run=False,
                    status='running', pid=os.getpid(), resumed_successes=len(existing),
                    native_parameters=dict(inference_parameters,incremental_output=True))
    save_json(manifest_path,manifest)
    (out/'system_prompt.txt').write_text(prompt_text,encoding='utf-8')

    def request(messages):
        import requests
        for attempt in range(3):
            reservation = budget.reserve(model,messages,inference_parameters['max_tokens']) if budget else None
            response = None
            try:
                response = call_model(url, model, key, messages, args.timeout, args.protocol,
                                      **inference_parameters)
                if response['status'] not in [429,500,502,503,504] or attempt==2:
                    return response
            except (requests.Timeout, requests.ConnectionError):
                if attempt==2: raise
            finally:
                if reservation is not None:
                    budget.settle(reservation, response.get('usage') if response else None)
            time.sleep(2**attempt)

    def one(row):
        ident, agent = row['episode_id'], row['agent_input']
        begin = time.monotonic()
        record = {'episode_id': ident, 'dataset': agent['dataset'], 'setting': agent['setting'],
                  'model': model, 'status': 'error'}
        try:
            messages, safe = model_messages(agent, args.direction_mode)
            save_json(out/'calls'/f'{ident}.input.json', {'model': model, 'endpoint': url,
                      'agent_input': safe, 'system_prompt': messages[0]['content']})
            response = request(messages)
            record['http_status'] = response['status']
            if response['status'] != 200:
                record['error'] = 'HTTP request failed; response body deliberately not persisted'
                return record
            raw = response['text']
            record['usage'] = response['usage']
            # Store model answer, not provider reasoning_content or authentication headers.
            save_json(out/'calls'/f'{ident}.answer.json', {'content': raw})
            from jsonschema.exceptions import ValidationError
            try:
                parsed, diagnostics = parse_model_output(raw, agent, require_plan=args.direction_mode == 'staged')
            except (ValueError, ValidationError) as validation:
                reason = str(validation).split('\n')[0][:500]
                feedback = ('Output failed validation: '+reason+
                            '. Return the corrected three-field answer and the requested compact plan. '
                            'Use null for unsupported directions. No additional labels or long reasoning.')
                save_json(out/'calls'/f'{ident}.repair.json', {'validation_error':reason,'feedback':feedback})
                record['repair_attempts'] = 1
                repaired = request(messages+[
                    {'role':'assistant','content':raw}, {'role':'user','content':feedback}])
                if repaired['status'] != 200:
                    raise ValueError('Correction request HTTP '+str(repaired['status']))
                record['repair_usage'] = repaired['usage']
                save_json(out/'calls'/f'{ident}.corrected_answer.json', {'content':repaired['text']})
                parsed, diagnostics = parse_model_output(repaired['text'], agent, require_plan=args.direction_mode == 'staged')
            record['resolved_direction'] = resolve_direction(parsed, agent)
            record['resolved_direction']['source'] = diagnostics['direction_source']
            record['diagnostics'] = diagnostics
            record['parsed'] = parsed
            record['grounding_query'] = grounding_query(parsed)
            save_json(out/'labels'/f'{ident}.json', parsed)
            record['status'] = 'ok'
        except Exception as exc:
            record['error'] = type(exc).__name__
            # Validation messages contain schema/input data, not request headers.
            if isinstance(exc, ValueError):
                record['validation_detail'] = str(exc)[:600]
        finally:
            record['exploration_seed'] = exploration_seed(record.get('parsed'), agent, record.get('diagnostics'))
            record['elapsed_seconds'] = round(time.monotonic()-begin, 3)
            save_json(out/'records'/f'{ident}.json', record)
        return record

    # Persist in completion order, then restore deterministic input order at the end.
    pending = [r for r in inputs if r['episode_id'] not in existing]
    results = dict(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one,row):row['episode_id'] for row in pending}
        for future in as_completed(futures):
            record = future.result()
            results[record['episode_id']] = record
            write_jsonl(out/'predictions.jsonl',list(results.values()))
            progress = {'completed':len(results),'total':len(inputs),
                        'ok':sum(r['status']=='ok' for r in results.values()),
                        'errors':sum(r['status']!='ok' for r in results.values()),'pid':os.getpid()}
            save_json(out/'progress.json',progress)
            print(record['episode_id'],record['status'],len(results),'/',len(inputs),flush=True)
    write_jsonl(out/'predictions.jsonl',[results[r['episode_id']] for r in inputs])
    manifest.update(status='completed',completed=len(results),
                    ok=sum(r['status']=='ok' for r in results.values()))
    save_json(manifest_path,manifest)



def evaluate(args):
    truths = read_jsonl(args.truth)
    if any(row['dataset'] != DATASET for row in truths):
        raise ValueError(f'This evaluator accepts {DATASET} ground truth only')
    preds = read_jsonl(args.predictions)
    if len({r['episode_id'] for r in preds}) != len(preds):
        raise ValueError('Duplicate prediction ids')
    indexed = {r['episode_id']: r for r in preds}
    unknown_ids = set(indexed) - {r['episode_id'] for r in truths}
    if unknown_ids:
        raise ValueError('Predictions have ids outside evaluation manifest')
    groups = defaultdict(list)
    details = []
    for gt in truths:
        pred = indexed.get(gt['episode_id'], {})
        goal_bearing = bearing(gt['start_xy_m'], gt['goal_xy_m'])
        resolved = pred.get('resolved_direction', {})
        heading = resolved.get('heading_deg') if pred.get('status') == 'ok' else None
        baseline = gt.get('initial_heading_deg')
        e = angular_error(heading, goal_bearing) if heading is not None and goal_bearing is not None else None
        b = angular_error(baseline, goal_bearing) if baseline is not None and goal_bearing is not None else None
        seed = pred.get('exploration_seed', {}).get('heading_deg')
        seed_error = angular_error(seed, goal_bearing) if seed is not None and goal_bearing is not None else None
        setting = pred.get('setting') or ('landmark_assisted' if gt['episode_id'].endswith('landmark_assisted') else 'visual')
        rec = {'episode_id': gt['episode_id'], 'goal_bearing_deg': goal_bearing,
               'predicted_heading_deg': heading, 'error_deg': e, 'initial_heading_error_deg': b,
               'exploration_seed_error_deg': seed_error,
               'status': pred.get('status', 'missing'), 'source': resolved.get('source'),
               'sector_contains_goal': e <= resolved['half_width_deg'] if e is not None and resolved.get('half_width_deg') is not None else None,
               'half_width_deg': resolved.get('half_width_deg')}
        groups[gt['dataset']+'/'+setting].append(rec)
        details.append(rec)

    def stats(records, field):
        eligible = [r for r in records if r['goal_bearing_deg'] is not None]
        values = [r[field] for r in eligible if r[field] is not None]
        n = len(eligible)
        return {'eligible_count': n, 'undefined_bearing_excluded': len(records)-n,
                'resolved_count': len(values), 'unresolved_or_failed': n-len(values),
                'resolution_rate': len(values)/n if n else None,
                'median_error_on_resolved_deg': statistics.median(values) if values else None,
                'within_deg_all_eligible': {str(t): sum(v <= t for v in values)/n if n else None for t in (30,45,60,90)},
                'within_deg_resolved_only': {str(t): sum(v <= t for v in values)/len(values) if values else None for t in (30,45,60,90)}}
    summary = {}
    for name, records in groups.items():
        paired = [r for r in records if r['error_deg'] is not None and r['initial_heading_error_deg'] is not None]
        widths = [r['half_width_deg'] for r in records if r['error_deg'] is not None and r['half_width_deg'] is not None]
        summary[name] = {'total': len(records), 'model': stats(records, 'error_deg'),
                         'exploration_policy_seed': stats(records, 'exploration_seed_error_deg'),
                         'initial_heading_baseline': stats(records, 'initial_heading_error_deg'),
                         'paired_resolved_count': len(paired),
                         'paired_mean_improvement_deg': statistics.mean(r['initial_heading_error_deg']-r['error_deg'] for r in paired) if paired else None,
                         'mean_sector_half_width_deg': statistics.mean(widths) if widths else None}
    save_json(args.out, {'summary': summary, 'records': details,
                        'limitations': ['Initialization bearing only, not navigation SR.',
                                        'Unknown, failed and missing predictions stay in eligible denominator.',
                                        'Zero start-goal displacement has no bearing and is excluded explicitly.',
                                        'Landmark centers are coarse geographic priors; target offsets remain unknown.']})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--annotation', required=True)
    prep.add_argument('--image-dir')
    prep.add_argument('--out', required=True)
    prep.add_argument('--limit', type=int, default=8, help='Number of AVDN source episodes')
    prep.add_argument('--seed', type=int, default=13)
    prep.set_defaults(func=prepare)
    runp = sub.add_parser('run')
    runp.add_argument('--inputs', required=True)
    runp.add_argument('--out', required=True)
    runp.add_argument('--direction-mode', choices=['direct','staged'], default='staged')
    runp.add_argument('--model')
    runp.add_argument('--url')
    runp.add_argument('--protocol', choices=['auto','native','compatible'], default='auto')
    runp.add_argument('--workers', type=int, default=1)
    runp.add_argument('--timeout', type=int, default=180)
    runp.add_argument('--thinking', choices=['on','off'], default='off')
    runp.add_argument('--thinking-budget', type=int, default=50)
    runp.add_argument('--max-output-tokens', type=int, default=2048)
    runp.add_argument('--max-cost-cny', type=float, default=10.0)
    runp.add_argument('--budget-ledger', help='Share one ledger across datasets to enforce one round cap')
    runp.set_defaults(func=run)
    ev = sub.add_parser('evaluate')
    ev.add_argument('--truth', required=True)
    ev.add_argument('--predictions', required=True)
    ev.add_argument('--out', required=True)
    ev.set_defaults(func=evaluate)
    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
