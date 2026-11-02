"""Dataset-local entrypoint. Run only reads public episodes; evaluate is offline."""
import argparse
import os
from pathlib import Path
from .io import read_jsonl, load_json, save_json, runtime_hash, digest
DATASET='AVDN'
from ..config import (MODEL as DEFAULT_MODEL, MAX_ROUNDS, MAX_TOOLS_PER_ROUND,
    MAX_REQUESTS, MAX_OUTPUT_TOKENS, BUDGET_CNY, SAM3_URL, DINO_URL, LOCATE_ANYTHING_URL)

def run(args):
    from ..tools.llm import Model, AgentBudget
    from ..tools.views import Environment
    from ..navigation import Agent
    from ..tools.sam3 import SAM3
    from ..tools.dino import GroundingDINO
    from ..tools.locate_anything import LocateAnything
    from .contracts import public_episode
    folder=Path(args.root)/DATASET.lower()
    episodes=read_jsonl(folder/'episodes.jsonl')
    for e in episodes:public_episode(e)
    if args.limit:episodes=episodes[:args.limit]
    key=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('API_KEY')
    if not key:raise ValueError('Set DASHSCOPE_API_KEY in the process environment')
    settings=dict(runtime_sha256=runtime_hash(),model=args.model,enable_thinking=args.thinking_budget>0,thinking_budget=args.thinking_budget,
        max_rounds=MAX_ROUNDS,max_tools_per_round=MAX_TOOLS_PER_ROUND,max_requests=MAX_REQUESTS,
        max_output_tokens=MAX_OUTPUT_TOKENS,budget_path=str(Path(args.budget).resolve()),
        budget_cny=BUDGET_CNY,sam3_url=args.sam3_url,
        dino_url=args.dino_url,locate_anything_url=args.locate_anything_url,
        observation_grounder=args.observation_grounder,observation_query_mode=args.observation_query_mode,public_sha256=digest(read_jsonl(folder/'episodes.jsonl')))
    manifest=folder/'run_manifest.json'
    if manifest.exists() and load_json(manifest)!=settings:
        raise ValueError('Runtime/input changed: create a new experiment record; cannot mix prompts or code')
    save_json(manifest,settings)
    (folder/'source').mkdir(exist_ok=True)
    package=Path(__file__).resolve().parents[1]
    for p in package.rglob('*.py'):
        relative=p.relative_to(package)
        dest=folder/'source'/DATASET/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    from .prompts import SYSTEM
    (folder/'system_prompt.txt').write_text(SYSTEM,encoding='utf-8')
    budget=AgentBudget(args.budget,BUDGET_CNY,save_json)
    for entry in episodes:
        directory=folder/'episodes'/entry['episode_id']
        result_path=directory/'result.json'
        if result_path.exists() and load_json(result_path)['completed']:
            print(entry['episode_id']+' cached',flush=True);continue
        env=Environment(entry['environment'],directory/'views')
        try:
            model=Model(directory/'requests',budget,args.model,key,max_requests=MAX_REQUESTS,
                enable_thinking=args.thinking_budget>0,thinking_budget=args.thinking_budget)
            sam=SAM3(args.sam3_url)
            grounders=dict(dino=GroundingDINO(args.dino_url),locate_anything=LocateAnything(args.locate_anything_url))
            result=Agent(entry,env,model,sam,directory,grounders=grounders,observation_grounder=args.observation_grounder,observation_query_mode=args.observation_query_mode).run()
            print(__import__('json').dumps(result,ensure_ascii=False),flush=True)
            if result['termination']=='budget_limit':break
        finally:env.close()
    print(__import__('json').dumps(dict(dataset=DATASET,budget=budget.read())),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','run','evaluate'):
        p=sub.add_parser(command);p.add_argument('--root',required=True)
        if command=='prepare':p.add_argument('--count',type=int,default=10)
        if command=='run':
            p.add_argument('--limit',type=int,default=0);p.add_argument('--model',default=DEFAULT_MODEL)
            p.add_argument('--budget',required=True);p.add_argument('--thinking-budget',type=int,default=0);p.add_argument('--sam3-url',default=SAM3_URL)
            p.add_argument('--dino-url',default=DINO_URL)
            p.add_argument('--observation-grounder',choices=['sam3','dino','locate_anything'])
            p.add_argument('--observation-query-mode',choices=['full','category'],default='full')
            p.add_argument('--locate-anything-url',default=LOCATE_ANYTHING_URL)
    args=parser.parse_args()
    if getattr(args,'thinking_budget',0)<0:parser.error('--thinking-budget must be nonnegative')
    if args.command=='run':run(args)
    elif args.command=='prepare':
        from .prepare import prepare
        print(prepare(args.root,args.count))
    else:
        from .evaluate import evaluate
        print(evaluate(Path(args.root)/DATASET.lower()))

if __name__=='__main__':main()

