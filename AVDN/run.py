"""AVDN online entry: initial instruction analysis followed by navigation.

python -m AVDN.run --inputs out/experiment/episodes.jsonl --root out/new_run
Input may contain validated initialization labels or omit them to run analysis.
No annotation reader or evaluator is imported here.
"""
import argparse
import copy
import json
import os
from pathlib import Path

if __package__ in (None, ''):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = 'AVDN'

from . import analysis
from .loop import Agent, public_episode
from .tools.views import Environment
from .tools.model import Model, Halt
from .tools.grounding import SAM3, GroundingDINO, LocateAnything
from .tools.runtime import read_jsonl, write_jsonl, save_json, load_json, runtime_hash, digest
from .config import (MODEL, API_URL, MAX_ROUNDS, MAX_TOOLS_PER_ROUND, MAX_REQUESTS,
                     SAM3_URL, DINO_URL, LOCATE_ANYTHING_URL)

DATASET = 'AVDN'


def validate_input(entry):
    required = {'episode_id','agent_input','environment'}
    if set(entry) not in (required, required | {'initialization'}):
        raise ValueError('Expected public episode_id, agent_input, environment and optional initialization')
    ident = entry['episode_id']
    if not isinstance(ident,str) or not ident or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in ident):
        raise ValueError('episode_id must be a safe alphanumeric filename')
    if entry['agent_input']['dataset'] != DATASET:
        raise ValueError('Wrong dataset for this entry point')
    # Validate all geometry/input fields before rendering or making any request.
    labels = entry.get('initialization')
    if labels is None:
        labels = dict(direction=None,final_target='pending initialization',auxiliary=[])
    probe = dict(entry, initialization=labels)
    public_episode(probe)


def run_episode(entry, directory, *, key, model=MODEL, analysis_model=None, url=API_URL,
                thinking_budget=0, direction_mode='staged', analysis_only=False,
                max_rounds=MAX_ROUNDS, max_tools=MAX_TOOLS_PER_ROUND, max_requests=MAX_REQUESTS,
                sam3_url=SAM3_URL, dino_url=DINO_URL, locate_anything_url=LOCATE_ANYTHING_URL,
                observation_grounder=None, observation_query_mode='full'):
    """Public-data orchestration API; Model/analysis can be replaced in tests."""
    validate_input(entry)
    episode = copy.deepcopy(entry)
    directory = Path(directory)
    env = Environment(episode['environment'], directory/'views')
    try:
        if episode.get('initialization') is None:
            source = copy.deepcopy(episode['agent_input'])
            if not source['views']:
                pose = source['initial_pose']
                source['views'] = [env.main(pose['xy_m'], pose['heading_deg'] or 0., 'initial')]
            episode['initialization'] = analysis.analyze(source, directory/'analysis', key,
                model=analysis_model or model, url=url, direction_mode=direction_mode,
                thinking_budget=thinking_budget)
        public_episode(episode)
        save_json(directory/'public_episode.json', episode)
        if analysis_only:
            return dict(episode_id=episode['episode_id'], status='analyzed',
                        initialization=episode['initialization'])
        llm = Model(directory/'requests', model, key, url=url, max_requests=max_requests,
                    enable_thinking=thinking_budget>0, thinking_budget=thinking_budget)
        return Agent(episode, env, llm, SAM3(sam3_url), directory,
            max_rounds=max_rounds, max_tools=max_tools,
            grounders=dict(dino=GroundingDINO(dino_url),locate_anything=LocateAnything(locate_anything_url)),
            observation_grounder=observation_grounder,observation_query_mode=observation_query_mode).run()
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Output experiment root; creates '+DATASET.lower()+'/')
    parser.add_argument('--inputs', help='Public episodes JSONL; defaults to root/'+DATASET.lower()+'/episodes.jsonl')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--model',default=MODEL)
    parser.add_argument('--analysis-model',help='Defaults to the navigation model')
    parser.add_argument('--url',default=API_URL,help='Shared endpoint for analysis and loop')
    parser.add_argument('--thinking-budget',type=int,default=0)
    parser.add_argument('--direction-mode',choices=['staged','direct'],default='staged')
    parser.add_argument('--analysis-only',action='store_true')
    parser.add_argument('--max-rounds',type=int,default=MAX_ROUNDS)
    parser.add_argument('--max-tools',type=int,default=MAX_TOOLS_PER_ROUND)
    parser.add_argument('--max-requests',type=int,default=MAX_REQUESTS)
    parser.add_argument('--sam3-url',default=SAM3_URL)
    parser.add_argument('--dino-url',default=DINO_URL)
    parser.add_argument('--locate-anything-url',default=LOCATE_ANYTHING_URL)
    parser.add_argument('--observation-grounder',choices=['sam3','dino','locate_anything'])
    parser.add_argument('--observation-query-mode',choices=['full','category'],default='full')
    args = parser.parse_args(argv)
    if args.limit<0 or args.thinking_budget<0 or args.max_tools<0 or min(args.max_rounds,args.max_requests)<=0:
        parser.error('Limits must be nonnegative; rounds and requests must be positive')
    folder = Path(args.root)/DATASET.lower()
    source = Path(args.inputs) if args.inputs else folder/'episodes.jsonl'
    rows = read_jsonl(source)
    for entry in rows:validate_input(entry)
    if len({r['episode_id'] for r in rows})!=len(rows):raise ValueError('Duplicate episode IDs')
    if args.limit:rows=rows[:args.limit]
    key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('API_KEY') or os.getenv('QWEN_API_KEY')
    if not key:parser.error('Set DASHSCOPE_API_KEY in the process environment')
    options = vars(args).copy()
    for field in ('root','inputs','limit'):options.pop(field)
    settings = dict(schema='online-agent-v1',dataset=DATASET,runtime_sha256=runtime_hash(),
                    public_sha256=digest(rows),options=options)
    manifest = folder/'run_manifest.json'
    if manifest.exists() and load_json(manifest)!=settings:
        raise ValueError('Runtime/input/settings changed; choose a new experiment root')
    if not manifest.exists() and (folder/'episodes').exists():
        raise ValueError('Refusing to reuse unversioned episode outputs')
    save_json(manifest, settings)
    if source.resolve() != (folder/'episodes.jsonl').resolve():
        write_jsonl(folder/'episodes.jsonl', rows)
    package = Path(__file__).resolve().parent
    for p in package.rglob('*.py'):
        dest = folder/'source'/DATASET/p.relative_to(package)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    for entry in rows:
        directory = folder/'episodes'/entry['episode_id']
        result = directory/'result.json'
        if not args.analysis_only and result.exists() and load_json(result).get('completed'):
            print(entry['episode_id']+' cached',flush=True)
            continue
        try:
            outcome = run_episode(entry,directory,key=key,**options)
        except Halt as exc:
            outcome = dict(episode_id=entry['episode_id'],status='blocked',reason=str(exc))
            save_json(directory/'analysis_failure.json',outcome)
        print(json.dumps(outcome,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
