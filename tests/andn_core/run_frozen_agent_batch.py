"""Run independent public episodes with a frozen dataset package in worker processes.

The parent writes the manifest/source once. Workers never read evaluator_only.jsonl.
Concurrency changes scheduling, not per-episode navigation or resource limits.
"""
import argparse
import concurrent.futures
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import sys


def package_modules(dataset, source):
    source = Path(source).resolve()
    sys.path.insert(0, str(source))
    names = ('agent.io', 'agent.contracts', 'config', 'navigation', 'tools.views',
             'tools.llm', 'tools.sam3', 'tools.dino', 'tools.locate_anything')
    current = (source / dataset / 'loop.py').exists()
    targets = {'agent.io':'tools.runtime', 'agent.contracts':'loop', 'navigation':'loop',
               'tools.llm':'tools.model', 'tools.sam3':'tools.grounding',
               'tools.dino':'tools.grounding', 'tools.locate_anything':'tools.grounding'} if current else {}
    modules = {name: importlib.import_module(dataset + '.' + targets.get(name,name)) for name in names}
    for module in modules.values():
        if not Path(module.__file__).resolve().is_relative_to(source / dataset):
            raise ValueError('A different dataset package is already imported')
    return modules


def run_episode(job):
    dataset, source, folder, entry, options = job
    m = package_modules(dataset, source)
    io, cfg = m['agent.io'], m['config']
    m['agent.contracts'].public_episode(entry)
    directory = Path(folder) / 'episodes' / entry['episode_id']
    result_path = directory / 'result.json'
    if result_path.exists() and io.load_json(result_path)['completed']:
        return {'episode_id': entry['episode_id'], 'cached': True}
    key = os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('API_KEY')
    if not key:
        raise ValueError('Set process DASHSCOPE_API_KEY')
    # Historical frozen packages retain their original accounting API.
    legacy = hasattr(m['tools.llm'], 'AgentBudget')
    model_args = (options['model'], key)
    if legacy:
        budget = m['tools.llm'].AgentBudget(options['budget_path'], cfg.BUDGET_CNY, io.save_json)
        model_args = (budget, *model_args)
    env = m['tools.views'].Environment(entry['environment'], directory / 'views')
    try:
        model = m['tools.llm'].Model(directory / 'requests', *model_args,
            max_requests=cfg.MAX_REQUESTS, enable_thinking=options['thinking_budget'] > 0,
            thinking_budget=options['thinking_budget'])
        sam = m['tools.sam3'].SAM3(options['sam3_url'])
        grounders = {'dino': m['tools.dino'].GroundingDINO(options['dino_url']),
                    'locate_anything': m['tools.locate_anything'].LocateAnything(options['locate_anything_url'])}
        result = m['navigation'].Agent(entry, env, model, sam, directory,
            grounders=grounders, observation_grounder=options['observation_grounder'],
            observation_query_mode=options['observation_query_mode']).run()
        return {k: result[k] for k in ('episode_id', 'termination', 'completed', 'model_requests')}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['AVDN', 'CityNav'], required=True)
    parser.add_argument('--source', required=True, help='Frozen source directory containing dataset package')
    parser.add_argument('--root', required=True)
    parser.add_argument('--budget', help='Only for historical frozen packages that require a ledger')
    parser.add_argument('--model', required=True)
    parser.add_argument('--thinking-budget', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sam3-url')
    parser.add_argument('--dino-url')
    parser.add_argument('--locate-anything-url')
    parser.add_argument('--observation-grounder', choices=['sam3', 'dino', 'locate_anything'])
    parser.add_argument('--observation-query-mode', choices=['full', 'category'], default='full')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or args.limit < 0 or args.thinking_budget < 0:
        parser.error('Use 1..8 workers and nonnegative limits')
    m = package_modules(args.dataset, args.source)
    io, cfg = m['agent.io'], m['config']
    folder = Path(args.root).resolve() / args.dataset.lower()
    episodes = io.read_jsonl(folder / 'episodes.jsonl')
    if len({e['episode_id'] for e in episodes}) != len(episodes):
        raise ValueError('Duplicate episode IDs would cause concurrent journal writes')
    for entry in episodes:
        m['agent.contracts'].public_episode(entry)
    options = dict(model=args.model, thinking_budget=args.thinking_budget,
        sam3_url=args.sam3_url,
        dino_url=args.dino_url, locate_anything_url=args.locate_anything_url,
        observation_grounder=args.observation_grounder, observation_query_mode=args.observation_query_mode)
    legacy = hasattr(m['tools.llm'], 'AgentBudget')
    if legacy:
        if not args.budget:
            parser.error('This historical frozen package requires --budget; use its saved ledger')
        options['budget_path'] = str(Path(args.budget).resolve())
    elif args.budget:
        parser.error('Current packages no longer use --budget')
    settings = dict(**options, runtime_sha256=io.runtime_hash(), enable_thinking=args.thinking_budget > 0,
        public_sha256=io.digest(episodes),
        max_rounds=cfg.MAX_ROUNDS, max_tools_per_round=cfg.MAX_TOOLS_PER_ROUND,
        max_requests=cfg.MAX_REQUESTS, max_output_tokens=cfg.MAX_OUTPUT_TOKENS,
        source_directory=str(Path(args.source).resolve()), workers=args.workers,
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if legacy:
        settings['budget_cny'] = cfg.BUDGET_CNY
    manifest = folder / 'run_manifest.json'
    if manifest.exists() and io.load_json(manifest) != settings:
        raise ValueError('Frozen run settings differ; use a new experiment record')
    io.save_json(manifest, settings)
    package = Path(args.source).resolve() / args.dataset
    for source in package.rglob('*.py'):
        dest = folder / 'source' / args.dataset / source.relative_to(package)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    shutil.copyfile(__file__, folder / 'source' / Path(__file__).name)
    prompt_module = '.prompts' if (Path(args.source)/args.dataset/'loop.py').exists() else '.agent.prompts'
    prompts = importlib.import_module(args.dataset + prompt_module)
    (folder / 'system_prompt.txt').write_text(prompts.SYSTEM, encoding='utf-8')
    selected = episodes[:args.limit] if args.limit else episodes
    if args.prepare_only:
        print(json.dumps({'prepared': len(selected), 'runtime_sha256': settings['runtime_sha256']}))
        return
    jobs = [(args.dataset, str(Path(args.source).resolve()), str(folder), e, options) for e in selected]
    failures = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_episode, job): job[3]['episode_id'] for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            ident = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                # No automatic restart of uncertain/in-flight API requests.
                result = {'episode_id': ident, 'worker_error': type(exc).__name__}
                failures.append(result)
            print(json.dumps(result), flush=True)
    io.save_json(folder / 'batch_driver_result.json', dict(selected=len(selected), worker_errors=failures))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
