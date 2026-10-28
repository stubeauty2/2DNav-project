"""Fresh paired samples; evaluator-only GT never enters the navigation worker.

prepare freezes public inputs and before/after runtimes. visible starts both arms
at the first historical MAIN view containing the GT center (an explicitly oracle
selected observation, NOT a navigation success-rate experiment). navigation starts
at the original pose. Sampling never filters on prior success or replaces failures.
"""
import argparse
import getpass
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

WORK=next(p for p in Path(__file__).resolve().parents if (p/'AVDN/loop.py').exists() and (p/'tests').is_dir())
ROOT=WORK/'out/16_adaptive_agent/20261027'
AUDIT=WORK/'out/15_target_visibility_analysis/20261026/audit.json'
ENDPOINT='https://maas.qianwenaiapi.com/api/v1/services/aigc/multimodal-generation/generation'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def prepare(args):
    folder=ROOT/args.batch
    if folder.exists():raise RuntimeError('Choose a new batch; frozen samples are write-once')
    audit=read(AUDIT);excluded={r['episode_id'] for r in audit['rows'] if r['group']=='latest'}
    for p in ROOT.glob('*/plan.json'):
        excluded.update(j['episode_id'] for j in read(p)['jobs'])
    rng=random.Random(args.seed);selected=[];pool_sizes={}
    for ds in ('AVDN','CityNav'):
        pool=sorted([r for r in audit['rows'] if r['group']=='historical' and r['dataset']==ds
            and r['episode_id'] not in excluded and (args.scope!='visible' or r['main_center_covered'])],
            key=lambda r:r['episode_id'])
        pool_sizes[ds]=len(pool)
        selected.extend(rng.sample(pool,args.per_dataset))
    jobs=[];gt={};sources={}
    for arm,base in [('before',ROOT/'before'),('after',WORK)]:
        for ds in ('AVDN','CityNav'):
            for p in sorted((base/ds).rglob('*.py')):
                target=folder/'runtime'/arm/p.relative_to(base)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
                sources[target.relative_to(folder).as_posix()]=sha(target)
    for row in selected:
        old=Path(row['folder']);ident=row['episode_id'];ds=row['dataset']
        ep=next(json.loads(line) for line in (old.parents[1]/'episodes.jsonl').read_text(encoding='utf-8').splitlines()
                if json.loads(line)['episode_id']==ident)
        checkpoint=None
        if args.scope=='visible':
            obs=next(v for v in row['observations'] if v['source']=='main' and v['navigation_delivered'] and v['center'])
            view=next(v for v in read(old/'trace.json')['views'] if v['view_id']==obs['view_id'])
            corners=view['world_corners_xy_m']
            checkpoint=dict(xy_m=[sum(p[i] for p in corners)/len(corners) for i in range(2)],
                heading_deg=view['image_up_heading_deg'])
        public=folder/'public'/ds/(ident+'.json');save(public,dict(episode=ep,checkpoint=checkpoint))
        jobs.append(dict(dataset=ds,episode_id=ident,public_input=public.relative_to(folder).as_posix(),
            public_sha256=sha(public),historical_folder=str(old),historical_view_id=obs['view_id'] if checkpoint else None))
        gt[ident]=row['goal']
    save(folder/'evaluator_only.json',gt)
    save(folder/'plan.json',dict(seed=args.seed,scope=args.scope,pool_sizes=pool_sizes,jobs=jobs,sources=sources,
        audit_sha256=sha(AUDIT),excluded_ids=sorted(excluded),created=time.time(),
        decision_model='qwen3.5-flash',visual_model=args.visual_model,url=ENDPOINT,
        thinking_budget=getattr(args,'thinking_budget',0),
        budgets=dict(max_rounds=4 if args.scope=='visible' else 12,max_tools=6,max_requests=32 if args.scope=='visible' else 48),
        comparison='Matched public inputs, 3.5 decision/fusion/review, budgets and services. After changes tool/skill orchestration and optional visual evidence together; not a single-component ablation.',
        limitations='visible is GT-conditioned checkpoint localization with no historical decision memory; original initial pose and instruction retained. It does not estimate natural navigation SR or semantic visibility.',
        harness_sha256=sha(__file__)))
    shutil.copyfile(__file__,folder/'harness.py')
    print('FROZEN',args.batch,[(j['dataset'],j['episode_id']) for j in jobs],flush=True)


def worker(args):
    # This entry point reads ONLY the public job plus hash-checked runtime. No GT.
    folder=ROOT/args.batch;plan=read(folder/'plan.json')
    job=next(j for j in plan['jobs'] if j['episode_id']==args.episode)
    if sha(folder/job['public_input'])!=job['public_sha256']:raise RuntimeError('Public input changed')
    runtime=folder/'runtime'/args.arm
    for rel,h in plan['sources'].items():
        if rel.startswith('runtime/'+args.arm+'/') and sha(folder/rel)!=h:raise RuntimeError('Runtime changed')
    sys.path.insert(0,str(runtime))
    ds=job['dataset'];run=importlib.import_module(ds+'.run')
    if not Path(run.__file__).is_relative_to(runtime):raise RuntimeError('Wrong runtime import')
    public=read(folder/job['public_input']);ep=public['episode'];out=folder/'runs'/args.arm/args.episode
    if (out/'attempt.json').exists():raise RuntimeError('Existing attempt; do not silently retry uncertain API calls')
    save(out/'attempt.json',dict(started=time.time(),arm=args.arm,public_sha256=job['public_sha256']))
    run.validate_input(ep)
    env=run.Environment(ep['environment'],out/'views')
    try:
        thinking=plan.get('thinking_budget',0)
        options=dict(url=plan['url'],max_requests=plan['budgets']['max_requests'],enable_thinking=thinking>0,
            thinking_budget=thinking or 1024,tool_protocol='structured_v2')
        if args.arm=='after':options['visual_model']=plan['visual_model']
        model=run.Model(out/'requests',plan['decision_model'],os.environ['DASHSCOPE_API_KEY'],**options)
        grounders=dict(dino=run.GroundingDINO('http://127.0.0.1:8766'))
        # LocateAnything remains available to the runtime, but this initial matched
        # comparison omits its endpoint; compare it in a separately frozen batch.
        agent=run.Agent(ep,env,model,run.SAM3('http://127.0.0.1:8765'),out,
            max_rounds=plan['budgets']['max_rounds'],max_tools=plan['budgets']['max_tools'],
            grounders=grounders,grounding_mode='skill' if args.arm=='before' else 'adaptive',max_observation_tools=10,
            jev=run.JevEvidence('http://127.0.0.1:8770',model='imajev-4b',max_questions=24))
        if public['checkpoint']:
            agent.position=list(public['checkpoint']['xy_m']);agent.heading=public['checkpoint']['heading_deg']
            agent.trajectory=[dict(xy_m=agent.position[:],heading_deg=agent.heading,source='frozen_visible_checkpoint')]
        result=agent.run();save(out/'exit.json',dict(termination=result['termination'],finished=time.time()))
        print(args.arm,args.episode,result['termination'],result['model_requests'],flush=True)
    finally:env.close()


def run_batch(args):
    folder=ROOT/args.batch;plan=read(folder/'plan.json')
    if sha(__file__)!=plan['harness_sha256']:raise RuntimeError('Use frozen batch harness')
    # Refuse the entire batch before sending a request if it contains any attempts.
    if any((folder/'runs'/a/j['episode_id']/'attempt.json').exists() for j in plan['jobs'] for a in ('before','after')):
        raise RuntimeError('Batch already attempted; inspect journals instead of rerunning')
    import requests
    for name,port in [('dino',8766),('sam3',8765),('jev',8770)]:
        response=requests.get(f'http://127.0.0.1:{port}/health',timeout=5);response.raise_for_status()
        state=response.json()
        if name in ('dino','sam3') and not (state.get('ready') and state.get('device','').startswith('cuda')):
            raise RuntimeError('CUDA backend not ready: '+name)
        if name=='jev' and state.get('model')!='imajev-4b':raise RuntimeError('Unexpected Jev model')
        save(folder/('health-'+name+'.json'),state)
    key=os.getenv('DASHSCOPE_API_KEY') or getpass.getpass('API key (hidden, memory only): ')
    sys.path.insert(0,str(WORK))
    from AVDN.tools.model import call_model
    for name in dict.fromkeys([plan['decision_model'],plan['visual_model']]):
        if not name:continue
        thinking=plan.get('thinking_budget',0)
        result=call_model(plan['url'],name,key,[dict(role='user',content='Reply OK.')],timeout=60,
            enable_thinking=thinking>0,thinking_budget=min(thinking,128),max_tokens=256 if thinking else 32)
        save(folder/('preflight-'+name+'.json'),dict(model=name,status=result['status'],requested_thinking=thinking>0))
        if result['status']!=200:raise RuntimeError('Model preflight failed: '+name)
    env=dict(os.environ,DASHSCOPE_API_KEY=key,PYTHONDONTWRITEBYTECODE='1')
    for index,job in enumerate(plan['jobs']):
        for arm in (('before','after') if index%2==0 else ('after','before')):
            command=[sys.executable,'-B',str(folder/'harness.py'),'worker','--batch',args.batch,'--arm',arm,'--episode',job['episode_id']]
            done=subprocess.run(command,cwd=WORK,env=env)
            if done.returncode:raise RuntimeError('Worker failed; preserve every attempt and diagnose')
    report(args)


def report(args):
    import math
    from shapely.geometry import Polygon,Point
    folder=ROOT/args.batch;plan=read(folder/'plan.json');gt=read(folder/'evaluator_only.json');rows=[]
    sys.path.insert(0,str(WORK))
    from AVDN.tools.geometry import project,square_bbox
    def score(c,g,views):
        if c is None:return None,False
        if g['dataset']=='CityNav':
            value=math.dist(c['center_xy_m'],g['goal_xy_m']);return value,value<=20
        points=c.get('evaluation_polygon_xy_m')
        if points is None:points=project(views[c['view_id']]['pixel_to_world'],square_bbox(c['bbox_pixels']))
        a=Polygon(points);b=Polygon(g['evaluation_polygon_xy_m']);value=a.intersection(b).area/a.union(b).area
        return value,value>.4
    for job in plan['jobs']:
        for arm in ('before','after'):
            out=folder/'runs'/arm/job['episode_id']
            if not (out/'result.json').exists():continue
            result=read(out/'result.json');trace=read(out/'trace.json');views={v['view_id']:v for v in trace['views']}
            g=gt[job['episode_id']];metric,success=score(result['final'],g,views)
            proposals=[e for e in trace['events'] if e['kind']=='candidate' and e['target']=='final']
            reviews=[];target=None
            for event in trace['events']:
                if event['kind']=='candidate':target=event['target']
                elif event['kind']=='review_result' and target=='final':reviews.append(event)
            requests=[read(p) for p in sorted((out/'requests').glob('*.json'))]
            delivered={a['view_id'] for r in requests if r['status']=='ok' and r['context']['mode']=='decision' for a in r['assets']}
            visible=any(v['source']=='main' and ident in delivered and Polygon(v['world_corners_xy_m']).covers(Point(g['goal_xy_m']))
                for ident,v in views.items())
            rows.append(dict(dataset=job['dataset'],episode_id=job['episode_id'],arm=arm,metric=metric,success=success,
                target_center_in_main=visible,
                correct_proposal=any(score(e['candidate'],g,views)[1] for e in proposals),
                correct_proposal_rejected=sum(score(e['candidate'],g,views)[1] and e['answer']['verdict']!='confirm' for e in reviews),
                termination=result['termination'],requests=result['model_requests'],
                api_errors=sum(r['status']!='ok' for r in requests),
                calls=[dict(kind=e['kind'],name=e['input'].get('tool',e['input'].get('skill')),status=e['output']['status'])
                    for e in trace['events'] if e['kind'] in ('tool','skill')]))
    summary=[]
    for ds in ('AVDN','CityNav'):
        for arm in ('before','after'):
            rr=[r for r in rows if r['dataset']==ds and r['arm']==arm]
            summary.append(dict(dataset=ds,arm=arm,n=len(rr),success=sum(r['success'] for r in rr),
                visible=sum(r['target_center_in_main'] for r in rr),
                visible_success=sum(r['target_center_in_main'] and r['success'] for r in rr),
                proposed=sum(r['correct_proposal'] for r in rr),correct_proposal_rejected=sum(r['correct_proposal_rejected'] for r in rr)))
    save(folder/'evaluation.json',dict(scope=plan['scope'],summary=summary,rows=rows))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run','worker','report'])
    parser.add_argument('--batch',required=True)
    parser.add_argument('--scope',choices=['visible','navigation'],default='visible')
    parser.add_argument('--seed',type=int,default=20261027)
    parser.add_argument('--per-dataset',type=int,default=3)
    parser.add_argument('--visual-model',default='qwen3.8-flash')
    parser.add_argument('--thinking-budget',type=int,default=0,help='Same optional reasoning budget for both arms; default preserves earlier non-thinking experiments')
    parser.add_argument('--arm',choices=['before','after']);parser.add_argument('--episode')
    args=parser.parse_args()
    if not args.batch or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in args.batch):parser.error('Safe batch name required')
    if args.per_dataset<1:parser.error('Positive sample count required')
    if args.thinking_budget<0:parser.error('Reasoning budget must be nonnegative')
    dict(prepare=prepare,run=run_batch,worker=worker,report=report)[args.command](args)


if __name__=='__main__':main()
