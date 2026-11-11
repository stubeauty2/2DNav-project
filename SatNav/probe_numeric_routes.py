"""Conservative text-only numeric-route probe, not a production language parser."""
import re,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'datasets'/'SatNav'
DIST=re.compile(r'\b(\d+(?:\.\d+)?)\s*(?:-\s*)?(?:meters?|metres?|m)\b',re.I)
TURN=re.compile(r'\b(?:(\d+(?:\.\d+)?)\s*[- ]?degrees?\s+(left|right)(?:ward)?|(left|right)\s+(\d+(?:\.\d+)?)\s*degrees?)\b',re.I)
def predict(text,heading):
 events=[(m.start(),'move',float(m[1])) for m in DIST.finditer(text)]
 events += [(m.start(),'turn',float(m[1] or m[4])*(-1 if (m[2] or m[3]).lower()=='left' else 1)) for m in TURN.finditer(text)]
 events.sort()
 if not events or [e[1] for e in events] not in [['move','turn','move'],['move','turn','move','turn','move']]:return None
 x=y=0
 for _,kind,value in events:
  if kind=='turn':heading+=value
  else:x+=value*math.sin(math.radians(heading));y+=value*math.cos(math.radians(heading))
 return math.degrees(math.atan2(x,y))%360 if math.hypot(x,y)>1e-6 else None
report={}
for split in ['val_seen','val_unseen']:
 data=json.loads((ROOT/f'episodes/eval/{split}/all_episodes.json').read_text(encoding='utf-8'));es=data if isinstance(data,list) else data['episodes'];ls=[e for e in es if e['trajectory_type']=='LandmarkSet'];rows=[]
 for e in ls:
  t=e['instruction']['instruction_text'];p=predict(t,e['start_rotation'])
  if p is None:continue
  x,y,_=e['start_position'];gx,gy,_=e['goals'][0]['position'];g=math.degrees(math.atan2((gx-x)*math.cos(math.radians(y)),gy-y))%360;err=abs((p-g+180)%360-180)
  rows.append({'scene_id':e['scene_id'],'episode_id':e['episode_id'],'text':t,'prediction':p,'error_deg':err})
 report[split]={'total':len(ls),'parsed':len(rows),'coverage':len(rows)/len(ls),'median_error_deg':statistics.median(r['error_deg'] for r in rows),'within_15_fraction':sum(r['error_deg']<=15 for r in rows)/len(rows),'rows':rows}
 print(split,json.dumps({k:v for k,v in report[split].items() if k!='rows'}))
(ROOT/'numeric_route_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
