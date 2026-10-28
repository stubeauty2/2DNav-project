"""Offline evaluator report; never imported by navigation workers.

Projects evaluator-only targets and final predictions onto each arm's first main
view. This display does not alter online artifacts or provide feedback to agents.
"""
import argparse
import base64
from collections import Counter
import html
import io
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

WORK=Path(__file__).resolve().parents[2]
ROOT=WORK/'out/16_adaptive_agent/20261027'


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def esc(value):return html.escape(str(value))


def project(view, points):
    inverse=np.linalg.inv(np.asarray(view['pixel_to_world']))
    points=np.asarray(points)
    mapped=np.c_[points,np.ones(len(points))]@inverse.T
    return [(float(p[0]/p[2]),float(p[1]/p[2])) for p in mapped]


def render_view(folder, job, arm, goal, row, output):
    run=folder/'runs'/arm/job['episode_id']
    trace=read(run/'trace.json');result=read(run/'result.json')
    view=next(v for v in trace['views'] if v['source']=='main')
    image=Image.open(view['image_path']).convert('RGB');draw=ImageDraw.Draw(image)
    if goal['dataset']=='AVDN':
        points=project(view,goal['evaluation_polygon_xy_m'])
        draw.line(points+[points[0]],fill='#35ff76',width=4)
    center=project(view,[goal['goal_xy_m']])[0]
    draw.ellipse([center[0]-7,center[1]-7,center[0]+7,center[1]+7],outline='#35ff76',width=3)
    final=result.get('final')
    if final:
        polygon=final.get('evaluation_polygon_xy_m',final['polygon_xy_m']) if goal['dataset']=='AVDN' else final['polygon_xy_m']
        points=project(view,polygon)
        draw.line(points+[points[0]],fill='#ff57dc',width=4)
        center=project(view,[final['center_xy_m']])[0]
        draw.line([(center[0]-8,center[1]),(center[0]+8,center[1])],fill='#ff57dc',width=3)
        draw.line([(center[0],center[1]-8),(center[0],center[1]+8)],fill='#ff57dc',width=3)
    image.save(output)
    thumbnail=io.BytesIO();image.save(thumbnail,format='JPEG',quality=86)
    encoded=base64.b64encode(thumbnail.getvalue()).decode('ascii')
    calls=Counter(e['input'].get('tool',e['input'].get('skill')) for e in trace['events'] if e['kind'] in ('tool','skill'))
    cached=sum(bool(e['output'].get('cache_hit')) for e in trace['events'] if e['kind'] in ('tool','skill'))
    decisions=[e['action'] for e in trace['events'] if e['kind']=='decision']
    reviews=[e['answer'] for e in trace['events'] if e['kind']=='review_result']
    status='SUCCESS' if row['success'] else 'FAIL'
    return f'''<article><h3>{arm} · {status}</h3><p>{esc(row['termination'])} · metric {esc(row['metric'])} · {row['requests']} requests</p>
<img loading="lazy" src="data:image/jpeg;base64,{encoded}" alt="First main view with evaluator target and final prediction">
<p>Tools: {esc(dict(calls))}; cache hits: {cached}</p>
<details><summary>Final decision and review</summary><pre>{esc(json.dumps(dict(decision=decisions[-1] if decisions else None,review=reviews[-1] if reviews else None),ensure_ascii=False,indent=2))}</pre></details></article>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batches',nargs='+',required=True)
    parser.add_argument('--output',default='audit_report',help='Report directory name under the experiment root')
    args=parser.parse_args()
    if not re.fullmatch('[a-z0-9_-]+',args.output):parser.error('Safe report directory name required')
    output=ROOT/args.output;output.mkdir(exist_ok=True)
    sections=[];summary=[]
    audit=read(ROOT/'manual_audit.json') if (ROOT/'manual_audit.json').exists() else {'cases':[]}
    notes={(c['batch'],c['episode_id']):c['finding'] for c in audit['cases']}
    for batch in args.batches:
        folder=ROOT/batch;plan=read(folder/'plan.json');evaluation=read(folder/'evaluation.json')
        goals=read(folder/'evaluator_only.json')
        summary.extend(dict(batch=batch,thinking_budget=plan.get('thinking_budget',0),**s) for s in evaluation['summary'])
        for job in plan['jobs']:
            episode=read(folder/job['public_input'])['episode']
            instruction=episode['agent_input']['instruction_raw'];arms=[]
            for arm in ('before','after'):
                row=next(r for r in evaluation['rows'] if r['episode_id']==job['episode_id'] and r['arm']==arm)
                target=output/(batch+'-'+arm+'-'+job['episode_id']+'.png')
                arms.append(render_view(folder,job,arm,goals[job['episode_id']],row,target))
            note=notes.get((batch,job['episode_id']))
            annotation=f'<p><strong>Offline image/log audit:</strong> {esc(note)}</p>' if note else ''
            sections.append(f'<section><h2>{esc(batch)} · {esc(job["episode_id"])}</h2><p>{esc(instruction)}</p>'+annotation+'<div class="pair">'+''.join(arms)+'</div></section>')
    table='<table><tr><th>Batch</th><th>Reasoning budget (both arms)</th><th>Dataset</th><th>Arm</th><th>Final metric successes</th><th>Metric-passing proposals</th><th>Metric-passing proposals rejected</th></tr>'
    for s in summary:table+=f'<tr><td>{s["batch"]}</td><td>{s["thinking_budget"]}</td><td>{s["dataset"]}</td><td>{s["arm"]}</td><td>{s["success"]}/{s["n"]}</td><td>{s["proposed"]}</td><td>{s["correct_proposal_rejected"]}</td></tr>'
    table+='</table>'
    document='''<!doctype html><meta charset="utf-8"><title>Adaptive grounding paired audit</title>
<style>body{max-width:1500px;margin:32px auto;padding:0 24px;background:#10141c;color:#e8edf6;font:16px/1.55 system-ui}h1,h2,h3{line-height:1.25}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}article,section{border:1px solid #374152;padding:16px;margin:22px 0}article{margin:0;min-width:0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{border-collapse:collapse;width:100%}th,td{padding:8px;border:1px solid #374152;text-align:left}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>Adaptive grounding: paired visible-checkpoint audit</h1>
<p>Green: evaluator target. Magenta: final prediction (AVDN evaluation square; CityNav object box and center). Every image is the first main observation; a later prediction can fall outside it. Target overlays are generated offline and never sent to the decision model or tools.</p>
<p>Metric success means AVDN square IoU &gt; 0.4 or CityNav center distance &le; 20 m. Nearby wrong objects can pass a distance threshold. Read the offline image audits separately from model-generated decision/review claims.</p>
<p>These are seeded, disjoint, small batches at GT-center-visible historical checkpoints. They do not estimate full navigation success or semantic visibility. Different batches test different frozen after versions; do not pool them as a single fixed-policy evaluation. Failed cases are retained. Decision, fusion and review use Qwen 3.5; only the optional visual evidence helper uses 3.8.</p>'''+table+''.join(sections)
    (output/'index.html').write_text(document,encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(output/'index.html')


if __name__=='__main__':main()
