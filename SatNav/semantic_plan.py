"""Extract public instruction phases; no map annotations or evaluator imports."""
import json
import copy
import math
import os
from pathlib import Path
from AVDN.agent.io import read_jsonl, write_jsonl, save_json, load_json, digest
from AVDN.initialization.client import call_model, DEFAULT_URL
from AVDN.initialization.budget import RequestBudget

REGIONS = {f'{y}_{x}': [u,v] for y,v in [('upper',200),('middle',500),('lower',800)]
           for x,u in [('left',200),('center',500),('right',800)]}
FIELDS = {'landmark','distance_m','arrival_region','turn_direction','turn_deg','align_region'}
SYSTEM = '''Extract an ordered flight plan from the provided UAV instruction only.
Return JSON with exactly {"legs": [...], "unsupported_reason": null|string}.
Each leg has exactly:
{"landmark": "visual description of the object reached by THIS forward leg",
 "distance_m": number|null, "arrival_region": string|null,
 "turn_direction": "left"|"right"|null, "turn_deg": number|null,
 "align_region": string|null}.
There is one leg per FORWARD traversal, ending at a landmark or final stopping cue.
The turn fields describe the turn AFTER reaching this leg's landmark, before the
next forward traversal. Keep the last leg's turn fields null. Do not create a
separate leg for an in-place rotation. Include the final destination as the last leg.
distance_m is present only for an explicitly stated distance of that forward leg.
turn_deg is the positive magnitude only when explicitly stated; never infer it
from quadrant words. Preserve a qualitative turn direction when its angle is unknown.
arrival_region is the object's image region BEFORE turning; align_region is its
image region AFTER turning. Use only these normalized labels:
upper_left, upper_center, upper_right, middle_left, middle_center, middle_right,
lower_left, lower_center, lower_right. Map top/bottom/mid/low synonyms accordingly.
Do not mistake a regional condition for the object's visual identity. A description
can retain relationships like "pool in the backyard", but exclude turn instructions.
No numeric distance, turn, position or waypoint may be invented. Missing stays null.
If a phase or turn cannot be represented unambiguously, return legs=[] and explain
unsupported_reason. Never use outside knowledge or coordinates.
Example: "Forward until red roof is lower right. Turn right until upper middle.
Forward 90m to blue pool at lower left. Stop." gives two legs:
[{"landmark":"red roof","distance_m":null,"arrival_region":"lower_right",
  "turn_direction":"right","turn_deg":null,"align_region":"upper_center"},
 {"landmark":"blue pool","distance_m":90,"arrival_region":"lower_left",
  "turn_direction":null,"turn_deg":null,"align_region":null}].'''


def validate_plan(value, *, allow_qualitative=False):
    if not isinstance(value,dict) or set(value)!={'legs','unsupported_reason'}:
        raise ValueError('Invalid semantic plan fields')
    legs=value['legs'];reason=value['unsupported_reason']
    if not isinstance(legs,list) or len(legs)>8:
        raise ValueError('Invalid leg count')
    if reason is not None:
        if not isinstance(reason,str) or not reason.strip() or legs:
            raise ValueError('Unsupported plans must have a reason and no legs')
        return value
    if not legs:raise ValueError('Supported plans require legs')
    for index,leg in enumerate(legs):
        if not isinstance(leg,dict) or set(leg)!=FIELDS:
            raise ValueError('Invalid leg fields')
        if not isinstance(leg['landmark'],str) or not leg['landmark'].strip() or len(leg['landmark'])>1000:
            raise ValueError('Missing/oversized landmark')
        for field in ['distance_m','turn_deg']:
            number=leg[field]
            if number is not None and (isinstance(number,bool) or not isinstance(number,(int,float))
                                       or not math.isfinite(number) or number<0):
                raise ValueError('Invalid '+field)
        if leg['turn_deg'] is not None and leg['turn_deg']>360:raise ValueError('Invalid turn')
        for field in ['arrival_region','align_region']:
            if leg[field] is not None and leg[field] not in REGIONS:raise ValueError('Invalid region')
        if leg['turn_direction'] not in (None,'left','right'):raise ValueError('Invalid turn direction')
        turning=leg['turn_deg'] is not None or leg['align_region'] is not None
        if turning and leg['turn_direction'] is None:raise ValueError('Turn requires direction')
        if leg['align_region']=='middle_center':raise ValueError('Center cannot determine a turn bearing')
        if index==len(legs)-1 and (turning or leg['turn_direction'] is not None):
            raise ValueError('Final leg cannot turn')
        if index<len(legs)-1 and leg['turn_direction'] is not None and not turning and not allow_qualitative:
            raise ValueError('Qualitative turn requires an alignment region')
    return value


def parse_plan(text, *, allow_qualitative=False):
    """Normalize syntax only; missing geometric quantities remain missing."""
    value=json.loads(text)
    if isinstance(value,list):value={'legs':value,'unsupported_reason':None}
    value=copy.deepcopy(value)
    if isinstance(value,dict) and isinstance(value.get('legs'),list):
        for leg in value['legs']:
            if not isinstance(leg,dict):continue
            for field in ('arrival_region','align_region'):
                region=leg.get(field)
                if isinstance(region,str):
                    parts=region.split('_')
                    if len(parts)==2 and parts[0] in ('upper','middle','lower') and parts[1]=='middle':
                        leg[field]=parts[0]+'_center'
    return validate_plan(value,allow_qualitative=allow_qualitative)


def region_bearing(region):
    x,y=REGIONS[region]
    if x==500 and y==500:raise ValueError('Center has no bearing')
    return math.degrees(math.atan2(x-500,500-y)) % 360


def aligned_heading(camera_xy, landmark_xy, region):
    dx,dy=landmark_xy[0]-camera_xy[0],landmark_xy[1]-camera_xy[1]
    if math.hypot(dx,dy)<1e-6:raise ValueError('Landmark at camera center cannot define rotation')
    return (math.degrees(math.atan2(dx,dy))-region_bearing(region)) % 360


def prepare_plans(root, model='qwen3.8-flash', *, allow_qualitative=False):
    root=Path(root)
    key=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('API_KEY')
    if not key:raise ValueError('Configure API key in environment')
    budget=RequestBudget(root/'semantic_plan_budget.json',None,save_json)
    outcomes=[]
    for episode in read_jsonl(root/'episodes.jsonl'):
        # Whitelist the instruction only. No environment, annotation, GT or coordinates.
        messages=[dict(role='system',content=SYSTEM),dict(role='user',content=episode['agent_input']['instruction_raw'])]
        signature=digest(dict(messages=messages,model=model,url=DEFAULT_URL))
        path=root/'semantic_plan_requests'/(episode['episode_id']+'.json')
        if path.exists():
            record=load_json(path)
            if record['signature']!=signature:raise ValueError('Plan input changed; use a fresh run directory')
            if record['status']=='pending':raise ValueError('Uncertain request; do not auto reissue')
        else:
            reservation=budget.reserve(model,messages,2048)
            record=dict(signature=signature,status='pending',messages=messages,model=model,reservation=reservation)
            save_json(path,record)
            try:
                response=call_model(DEFAULT_URL,model,key,messages,timeout=180,connect_timeout=90,
                                    enable_thinking=False,max_tokens=2048,response_format={'type':'json_object'})
                budget.settle(reservation,response.get('usage'))
                record.update(status='ok' if response['status']==200 else 'api_error',response=response)
            except Exception as exc:
                budget.settle(reservation,None)
                record.update(status='api_error',error=type(exc).__name__)
            save_json(path,record)
        row=dict(episode_id=episode['episode_id'],status=record['status'])
        if record['status']=='ok':
            try:row['plan']=parse_plan(record['response']['text'],allow_qualitative=allow_qualitative)
            except (ValueError,TypeError,KeyError) as exc:row.update(status='parse_error',error=str(exc))
        outcomes.append(row);write_jsonl(root/'semantic_plans.jsonl',outcomes)
        print(episode['episode_id'],row['status'],flush=True)
        if row['status']=='api_error':raise RuntimeError('Plan API failed; remaining episodes not attempted')
    return outcomes


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--allow-qualitative',action='store_true')
    args=parser.parse_args()
    prepare_plans(args.root,allow_qualitative=args.allow_qualitative)
