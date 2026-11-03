"""Three-field labels. Route/reference plans are separate diagnostic data."""
import json
import math
import re
from .geometry import bearing

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

def resolve_direction(result, agent):
    validate_result(result, agent)
    return {'heading_deg':result['direction'], 'source':'model' if result['direction'] is not None else 'unresolved',
            'resolved':result['direction'] is not None}

def exploration_seed(result, agent, diagnostics=None):
    if result is not None:
        resolved = resolve_direction(result, agent)
        if resolved['resolved']:
            return dict(resolved,role='goal_prior')
    stage_heading = (diagnostics or {}).get('first_stage_heading_deg')
    heading = stage_heading if stage_heading is not None else agent['initial_pose']['heading_deg']
    return {'heading_deg':heading, 'source':'first_route_stage' if stage_heading is not None else 'initial_heading_metadata',
            'role':'exploration_only', 'resolved':heading is not None}

def grounding_query(result):
    # The complete locating description is already one string.
    return result['final_target']

PLAN_SCHEMA = obj({'landmark_id':{'type':['string','null']}})

def parse_model_output(raw, agent, require_plan=False):
    from jsonschema import Draft202012Validator
    result, plan = parse_blocks(raw)
    validate_result(result, agent)
    plan = plan or {'landmark_id':None}
    Draft202012Validator(PLAN_SCHEMA).validate(plan)
    landmark_id = plan['landmark_id']
    diagnostics = {'plan':plan, 'raw_model_direction':result['direction'], 'direction_source':'model_visual'}
    if landmark_id is not None:
        allowed = {lm['id']:lm for lm in agent['named_landmarks']}
        if agent['setting'] != 'landmark_assisted' or landmark_id not in allowed:
            raise ValueError('Selected landmark is absent from allowed geographic input')
        result = dict(result, direction=bearing(agent['initial_pose']['xy_m'],allowed[landmark_id]['xy_m']))
        diagnostics['direction_source']='landmark_center'
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

