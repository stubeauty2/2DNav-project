"""Paired direct/staged bearing diagnostic. Evaluation alone reads goal labels."""
import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from .cli import read_jsonl, write_jsonl, save_json
from .geometry import bearing, angular_error


def prepare(args):
    from ..adapter import load_episodes, render_initial
    entries = list(load_episodes(args.annotation))
    groups = {'multi_turn':[], 'single_turn':[]}
    excluded = set(args.exclude_rows)
    for entry in entries:
        if entry[2]['source_row'] in excluded:
            continue
        n = sum(t['role']=='INS' for t in entry[1]['turns'])
        groups['multi_turn' if n>1 else 'single_turn'].append(entry)
    rng = random.Random(args.seed)
    chosen = []
    for group, count in [('multi_turn',args.multi), ('single_turn',args.single)]:
        chosen.extend((group,e) for e in rng.sample(groups[group],min(count,len(groups[group]))))
    out = Path(args.out); out.mkdir(parents=True,exist_ok=True)
    public, private, selection = [], [], []
    for group,(ident,agent,gt,context) in chosen:
        if args.image_dir:
            agent['views']=[render_initial(context,args.image_dir,out/'images'/f'{ident}.png')]
        public.append({'episode_id':ident,'agent_input':agent})
        private.append(gt)
        selection.append({'episode_id':ident,'source_row':gt['source_row'],'group':group})
    write_jsonl(out/'agent_inputs.jsonl',public)
    write_jsonl(out/'evaluator_only.jsonl',private)
    save_json(out/'selection.json',{'seed':args.seed,'excluded_source_rows':sorted(excluded),
              'annotation_sha256':hashlib.sha256(Path(args.annotation).read_bytes()).hexdigest(),
              'selection_basis':'instruction turn count only; no GT geometry used for sampling',
              'rows':selection})
    print(json.dumps({'episodes':len(public),'out':str(out)}))


def evaluate(args):
    root = Path(args.root)
    truths = read_jsonl(root/'evaluator_only.jsonl')
    selection = json.loads((root/'selection.json').read_text(encoding='utf-8'))
    group_by_id = {r['episode_id']:r['group'] for r in selection['rows']}
    def index(path):
        rows=read_jsonl(path)
        if len({r['episode_id'] for r in rows})!=len(rows):
            raise ValueError('Duplicate prediction id')
        if {r['episode_id'] for r in rows}-{r['episode_id'] for r in truths}:
            raise ValueError('Prediction outside selected sample')
        return {r['episode_id']:r for r in rows}
    direct=index(args.direct); staged=index(args.staged)
    details=[]
    for gt in truths:
        ident=gt['episode_id']; d=direct.get(ident,{});s=staged.get(ident,{})
        goal=bearing(gt['start_xy_m'],gt['goal_xy_m'])
        plan=s.get('diagnostics',{}) if s.get('status')=='ok' else {}
        methods={'direct_model':d.get('parsed',{}).get('direction') if d.get('status')=='ok' else None,
                 'staged_model':s.get('parsed',{}).get('direction') if s.get('status')=='ok' else None,
                 'first_stage':plan.get('first_stage_heading_deg'),
                 'last_stage':plan.get('last_stage_heading_deg'),
                 'equal_vector':plan.get('equal_weight_heading_deg'),
                 'weighted_vector':plan.get('weighted_heading_deg'),
                 'initial_heading':gt.get('initial_heading_deg')}
        details.append({'episode_id':ident,'group':group_by_id[ident],
                        'goal_bearing_deg':goal,'headings_deg':methods,
                        'errors_deg':{k:angular_error(v,goal) if v is not None and goal is not None else None for k,v in methods.items()},
                        'stage_headings_deg':plan.get('stage_headings_deg'),
                        'stage_weights':plan.get('stage_weights'),
                        'direct_status':d.get('status','missing'),'staged_status':s.get('status','missing')})
    summary={}
    for group in ['all','multi_turn','single_turn']:
        rows=[r for r in details if group=='all' or r['group']==group]
        eligible=[r for r in rows if r['goal_bearing_deg'] is not None]
        n=len(eligible); metrics={}
        for method in details[0]['headings_deg'] if details else []:
            values=[r['errors_deg'][method] for r in eligible if r['errors_deg'][method] is not None]
            metrics[method]={'total':len(rows),'eligible':n,'resolved':len(values),
                'median_error_resolved_deg':statistics.median(values) if values else None,
                'within_deg_all':{str(t):sum(v<=t for v in values)/n if n else None for t in (30,45,60,90)}}
        paired={}
        for test,base in [('staged_model','direct_model'),('weighted_vector','first_stage'),('equal_vector','first_stage'),('weighted_vector','direct_model')]:
            values=[(r['errors_deg'][test],r['errors_deg'][base]) for r in eligible
                    if r['errors_deg'][test] is not None and r['errors_deg'][base] is not None]
            paired[test+'_vs_'+base]={'count':len(values),
                'mean_error_reduction_deg':statistics.mean(b-a for a,b in values) if values else None,
                'better':sum(a<b-1e-6 for a,b in values),'worse':sum(a>b+1e-6 for a,b in values),
                'tied':sum(abs(a-b)<=1e-6 for a,b in values)}
        summary[group]={'methods':metrics,'paired':paired}
    save_json(args.out,{'summary':summary,'records':details,'selection':selection,
              'limitations':['Small single-run direction diagnostic, not navigation SR.',
                             'Stage weights and later-heading chaining are model assumptions, not observed distances/poses.',
                             'First-stage baseline uses the same newly parsed plan, not a replay of the original andn_core parser.',
                             'Unknown/failed predictions remain in all-eligible denominators.',
                             'Direction agreement does not establish correct final target description.']})
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--annotation',required=True)
    prep.add_argument('--image-dir');prep.add_argument('--out',required=True)
    prep.add_argument('--multi',type=int,default=12);prep.add_argument('--single',type=int,default=4)
    prep.add_argument('--seed',type=int,default=73);prep.add_argument('--exclude-rows',type=int,nargs='*',default=[])
    prep.set_defaults(func=prepare)
    ev=sub.add_parser('evaluate');ev.add_argument('--root',required=True)
    ev.add_argument('--direct',required=True);ev.add_argument('--staged',required=True);ev.add_argument('--out',required=True)
    ev.set_defaults(func=evaluate)
    args=p.parse_args();args.func(args)


if __name__=='__main__':
    main()
