"""Three-field labels. Route/reference plans are separate diagnostic data."""
import json
import math
import re
from .tools.geometry import initialization_bearing as bearing
import base64
import hashlib
from pathlib import Path


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)

SCHEMA = obj({
    'direction': {'type':['number','null'], 'minimum':0, 'exclusiveMaximum':360},
    'final_target': {'type':'string', 'minLength':1},
    'auxiliary': {'type':'array','items':{'type':'string'}}
})

def validate_result(result, agent):
    from jsonschema import Draft202012Validator
    Draft202012Validator(SCHEMA).validate(result)
    h = result['direction']
    if h is not None and not math.isfinite(h):
        raise ValueError('Direction must be finite or null')
    if not result['final_target'].strip():
        raise ValueError('Final target cannot be empty')
    return result

def parse_blocks(raw):
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n',1)[1].rsplit('```',1)[0].strip()
    answers = re.findall(r'<answer>\s*(.*?)\s*</answer>', text, re.S)
    plans = re.findall(r'<plan>\s*(.*?)\s*</plan>', text, re.S)
    # XML wrappers are optional transport decoration; accept complete JSON at EOF.
    # Incomplete JSON remains an error and still goes through normal repair.
    for tag, blocks in [('answer',answers),('plan',plans)]:
        if '<'+tag+'>' in text and not blocks:
            fragment=text.split('<'+tag+'>',1)[1].strip()
            try:
                value,end=json.JSONDecoder().raw_decode(fragment)
            except json.JSONDecodeError:
                continue
            trailing=fragment[end:].strip()
            if not trailing or (tag=='plan' and trailing.startswith('<answer>')):
                blocks.append(json.dumps(value))
    if text.count('<answer>')>1 or text.count('<plan>')>1:
        raise ValueError('Duplicate answer or plan blocks')
    if len(answers)>1 or len(plans)>1:
        raise ValueError('Duplicate answer or plan blocks')
    if '<answer>' in text and not answers:
        raise ValueError('Unclosed answer block')
    if '<plan>' in text and not plans:
        raise ValueError('Unclosed plan block')
    if plans and not answers:
        text = re.sub(r'<plan>\s*.*?\s*</plan>', '', text, count=1, flags=re.S).strip()
    return json.loads(answers[0] if answers else text), json.loads(plans[0]) if plans else None




LEGACY_PLAN_SCHEMA = obj({
    'stages': {'type':'array','maxItems':12,'items':obj({
        'heading_deg':{'type':['number','null'],'minimum':0,'exclusiveMaximum':360},
        'weight':{'type':'number','exclusiveMinimum':0},
        'target':{'type':'string'}
    })},
    'assumptions':{'type':'array','items':{'type':'string'}}
})

PLAN_SCHEMA = obj({
    'stages': {'type':'array','maxItems':12,'items':obj({
        'move_dir':{'type':'string'},
        'weight':{'type':'number','minimum':0},
        'target':{'type':'string'}
    })},
    'assumptions':{'type':'array','items':{'type':'string'}}
})


def stage_heading(move_dir, previous):
    """Match the original parser: clock is relative; compass degrees are absolute."""
    value = str(move_dir).strip()
    compass={'N':0,'NORTH':0,'NE':45,'NORTHEAST':45,'E':90,'EAST':90,
             'SE':135,'SOUTHEAST':135,'S':180,'SOUTH':180,'SW':225,'SOUTHWEST':225,
             'W':270,'WEST':270,'NW':315,'NORTHWEST':315}
    normalized=re.sub(r'[\s_-]','',value.upper())
    if normalized in compass:return float(compass[normalized])
    if value.lower() == 'land':
        return None
    absolute = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:deg|°)',value,re.I)
    if absolute:
        angle=float(absolute.group(1))
        if not 0<=angle<360: raise ValueError('Absolute direction outside [0,360)')
        return angle
    clock = re.fullmatch(r'(\d{1,2}):(\d{2})',value)
    if clock:
        hour,minute=map(int,clock.groups())
        if not 1<=hour<=12 or not 0<=minute<60:
            raise ValueError('Invalid relative clock direction')
        return (previous+(hour%12)*30+minute*0.5)%360 if previous is not None else None
    raise ValueError('move_dir must be a clock, absolute deg, or Land')


def compose_stages(plan, initial_heading=None):
    from jsonschema import Draft202012Validator
    legacy = bool(plan.get('stages')) and 'heading_deg' in plan['stages'][0]
    Draft202012Validator(LEGACY_PLAN_SCHEMA if legacy else PLAN_SCHEMA).validate(plan)
    headings, weights = [], []
    previous = initial_heading
    for stage in plan['stages']:
        h = stage['heading_deg'] if legacy else stage_heading(stage['move_dir'],previous)
        weight = stage['weight']
        if not math.isfinite(weight) or (h is not None and not math.isfinite(h)):
            raise ValueError('Stage values must be finite or null heading')
        headings.append(h)
        weights.append(weight)
        previous=h
    def combine(ws):
        if not headings or any(h is None and w>0 for h,w in zip(headings,ws)):
            return None
        total = sum(ws)
        if total<=0:return None
        # Normalize first to avoid scale-dependent cancellation tolerance.
        east = sum((w/total)*math.sin(math.radians(h)) for w,h in zip(ws,headings) if w>0)
        north = sum((w/total)*math.cos(math.radians(h)) for w,h in zip(ws,headings) if w>0)
        if math.hypot(east,north)<1e-6:
            return None
        heading = math.degrees(math.atan2(east,north)) % 360
        # Tiny negative east residuals can round modulo to exactly 360.0.
        return 0.0 if heading == 360.0 else heading
    return {'stage_headings_deg':headings, 'stage_weights':weights,
            'first_stage_heading_deg':next((h for h,w in zip(headings,weights) if w>0),None),
            'last_stage_heading_deg':headings[-1] if headings else None,
            'equal_weight_heading_deg':combine([int(w>0) for w in weights]),
            'weighted_heading_deg':combine(weights),
            'reference_assumption':'Legacy absolute headings or original-parser clock/compass conversion; later camera alignment is inferred, not observed'}

def parse_model_output(raw, agent, require_plan=False):
    result, plan = parse_blocks(raw)
    validate_result(result, agent)
    if require_plan and plan is None:
        raise ValueError('Staged mode requires a compact plan block')
    diagnostics = {'plan':plan, 'direction_source':'model_staged' if plan is not None else 'model_direct'}
    if plan is not None:
        diagnostics.update(compose_stages(plan,agent['initial_pose']['heading_deg']))
        if not plan['stages'] or 'move_dir' in plan['stages'][0]:
            diagnostics['raw_model_direction']=result['direction']
            heading=diagnostics['weighted_heading_deg']
            source='stage_vector'
            if heading is None:
                heading=result['direction']
                source='model_unresolved_plan'
            if heading is None:
                heading=diagnostics['first_stage_heading_deg']
                source='first_stage_prior'
            if heading is None:
                heading=agent['initial_pose']['heading_deg']
                source='initial_heading_prior'
            result=dict(result,direction=heading)
            diagnostics['direction_source']=source
    return result, diagnostics


def validate_agent_input(agent):
    allowed = {'dataset', 'setting', 'instruction_raw', 'turns', 'initial_pose',
               'coordinate_frame', 'named_landmarks', 'views'}
    if set(agent) != allowed:
        raise ValueError('Agent input contains unexpected or missing fields')
    if set(agent['initial_pose']) != {'xy_m', 'heading_deg'}:
        raise ValueError('Only initial xy and heading are agent-visible')
    if agent['setting'] == 'visual' and agent['named_landmarks']:
        raise ValueError('Visual-only input cannot contain geographic landmarks')
    for lm in agent['named_landmarks']:
        if set(lm) != {'id','name','type','xy_m','bbox_xy_m'}:
            raise ValueError('Landmarks must use public geometry/name whitelist')
    for turn in agent['turns']:
        if set(turn) != {'turn_id','role','text'}:
            raise ValueError('Turn metadata outside whitelist')


DATASET = 'AVDN'
def model_messages(agent, direction_mode='staged'):
    from .prompts import initialization_prompt
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
    messages = [{'role': 'system', 'content': initialization_prompt(direction_mode)},
                {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps(safe, ensure_ascii=False)}]+images}]
    return messages, safe

def analyze(agent, directory, key, *, model, url, direction_mode='staged',
            protocol='auto', thinking_budget=0, max_tokens=2048, timeout=180):
    """Analyze one public initial observation; never read annotations or GT.

    Requests are journalled before sending. An interrupted pending request is
    never automatically repeated. One invalid-format repair is allowed, using a
    distinct request record. Existing compatible requests are replayed.
    """
    from jsonschema.exceptions import ValidationError
    from .tools.model import call_model, Halt
    from .tools.runtime import save_json, load_json, digest
    from .prompts import ANALYSIS_REPAIR
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(thinking_budget, bool) or thinking_budget < 0:
        raise ValueError('thinking_budget must be nonnegative')
    messages, safe = model_messages(agent, direction_mode) if DATASET == 'AVDN' else model_messages(agent)
    parameters = dict(enable_thinking=thinking_budget > 0, thinking_budget=thinking_budget,
                      max_tokens=max_tokens)
    for attempt in range(2):
        signature = digest(dict(messages=messages, model=model, endpoint=url, protocol=protocol,
                                parameters=parameters))
        path = directory/f'{attempt:03d}.json'
        if path.exists():
            record = load_json(path)
            if record['signature'] != signature:
                raise Halt('analysis_resume_input_mismatch')
            if record['status'] == 'pending':
                raise Halt('analysis_uncertain_request_pending')
        else:
            record = dict(signature=signature, status='pending', model=model, endpoint=url,
                          protocol=protocol, parameters=parameters, agent_input=safe,
                          system_prompt=messages[0]['content'], attempt=attempt)
            save_json(path, record)
            try:
                response = call_model(url, model, key, messages, timeout, protocol, **parameters)
                record.update(response=response, status='ok' if response['status']==200 else 'api_error')
            except Exception as exc:
                record.update(status='api_error', error=type(exc).__name__)
            save_json(path, record)
        if record['status'] != 'ok':
            raise Halt('analysis_api_error')
        raw = record['response']['text']
        try:
            labels, diagnostics = parse_model_output(raw, agent,
                require_plan=DATASET == 'AVDN' and direction_mode == 'staged')
            result = dict(status='ok', parsed=labels, diagnostics=diagnostics,
                          resolved_direction=dict(heading_deg=labels['direction'],
                              source=diagnostics['direction_source']),
                          request_count=attempt+1)
            save_json(directory/'result.json', result)
            return labels
        except (ValueError, KeyError, TypeError, ValidationError) as exc:
            if attempt == 1:
                raise Halt('analysis_invalid_format') from exc
            reason = str(exc).split('\n')[0][:500]
            messages += [dict(role='assistant',content=raw),
                         dict(role='user',content=ANALYSIS_REPAIR.format(error=reason))]
    raise AssertionError('unreachable')
