"""Offline annotation audit; ground truth is used only for evaluation."""
import json, math, collections, hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'datasets'/'SatNav'
report={}
for split,path in [('train','episodes/train/all_episodes.json'),('val_seen','episodes/eval/val_seen/all_episodes.json'),('val_unseen','episodes/eval/val_unseen/all_episodes.json')]:
 d=json.loads((root/path).read_text(encoding='utf-8')); es=d if isinstance(d,list) else d['episodes']; ls=[e for e in es if e['trajectory_type']=='LandmarkSet']; unique={(e['scene_id'],str(e['trajectory_id'])):e for e in ls}; errors=[]
 for e in unique.values():
  x,y,_=e['start_position']; gx,gy,_=e['goals'][0]['position']; bearing=math.degrees(math.atan2((gx-x)*math.cos(math.radians(y)),gy-y))%360; errors.append(abs((bearing-e['start_rotation']+180)%360-180))
 errors.sort()
 report[split]={'episodes':len(ls),'unique_scene_trajectory_ids':len(unique),'difficulty':dict(collections.Counter(str(e['instruction'].get('difficulty_level')) for e in ls)),'subtypes':dict(collections.Counter(e.get('trajectory_subtype') for e in ls)),'initial_heading_goal_error_deg':{'median':errors[len(errors)//2],'mean':sum(errors)/len(errors),'within_30_fraction':sum(v<=30 for v in errors)/len(errors),'within_45_fraction':sum(v<=45 for v in errors)/len(errors)},'examples':[{'scene':e['scene_id'],'episode_id':e['episode_id'],'instruction':e['instruction']} for e in ls[::max(1,len(ls)//12)]]}
checks=[]
for line in (root/'SHA256SUMS').read_text().splitlines():
 expected,name=line.split(maxsplit=1); p=root/name; actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None; checks.append({'path':name,'verified':actual==expected})
report['checksums']=checks
(root/'annotation_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
for s in ['train','val_seen','val_unseen']:print(s,json.dumps({k:v for k,v in report[s].items() if k!='examples'}))
print('Verified',sum(c['verified'] for c in checks),'/',len(checks))
