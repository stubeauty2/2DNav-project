"""Validate extracted stages and compose exact numeric displacement when possible."""
import json
import math
import re
from jsonschema.exceptions import ValidationError
from AVDN.initialization.contracts import validate_result


def correct_single_turn_sign(value, instruction):
    """Repair only an unambiguous single explicit turn with the same magnitude."""
    stages=value.get('stages',[])
    if not instruction or not isinstance(stages,list) or len(stages)!=2 or not all(isinstance(s,dict) for s in stages) or stages[0].get('turn_before_deg')!=0:
        return []
    predicted=stages[1].get('turn_before_deg')
    if isinstance(predicted,bool) or not isinstance(predicted,(int,float)) or not math.isfinite(predicted):
        return []
    number=r'(\d+(?:\.\d+)?)'
    unit=r'(?:degrees?|deg|°)'
    patterns=[(number+r'[\s-]*'+unit+r'[\s-]+(?:to\s+the\s+)?(left|right)\b',False),
              (r'\b(left|right)(?:\s+turn)?\s+(?:(?:by|of|through)\s+)?'+number+r'[\s-]*'+unit+r'(?=$|[\s,.;:-])',True)]
    turns=[]
    for pattern,reverse in patterns:
        for match in re.finditer(pattern,instruction,re.I):
            amount,side=(match.group(2),match.group(1)) if reverse else match.groups()
            turns.append(float(amount)*(1 if side.lower()=='right' else -1))
    if len(turns)!=1 or not math.isclose(abs(predicted),abs(turns[0])) or predicted==turns[0]:
        return []
    stages[1]['turn_before_deg']=turns[0]
    return [dict(field='stages[1].turn_before_deg',model_value=predicted,corrected_value=turns[0],
                 basis='Single explicit left/right turn in public instruction, matching magnitude')]


def parse(text, initial_heading=None, instruction=None):
    value = json.loads(text)
    corrections=correct_single_turn_sign(value,instruction)
    relative = value.get('stages', [])
    if relative and 'turn_before_deg' in relative[0]:
        heading = initial_heading
        absolute = []
        for stage in relative:
            if set(stage) != {'turn_before_deg', 'distance_m'}:
                raise ValueError('Invalid relative stage fields')
            turn = stage['turn_before_deg']
            if turn is not None and (isinstance(turn, bool) or not isinstance(turn, (int, float)) or not math.isfinite(turn) or abs(turn)>360):
                raise ValueError('Invalid relative turn')
            heading = (heading + turn) % 360 if heading is not None and turn is not None else None
            absolute.append(dict(heading_deg=heading, distance_m=stage['distance_m']))
        value['stages'] = absolute
    required = {'stages', 'estimated_direction', 'final_target', 'auxiliary', 'assumptions', 'stop_condition'}
    if set(value) != required:
        raise ValueError('Initialization fields differ from contract')
    if not isinstance(value['stages'], list) or len(value['stages']) > 20:
        raise ValueError('Invalid stages')
    def number(x, angle=False):
        if x is None:
            return
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            raise ValueError('Expected finite number or null')
        if x < 0 or (angle and x >= 360):
            raise ValueError('Number out of range')
    number(value['estimated_direction'], True)
    for stage in value['stages']:
        if set(stage) != {'heading_deg', 'distance_m'}:
            raise ValueError('Invalid stage fields')
        number(stage['heading_deg'], True)
        number(stage['distance_m'])
    if not isinstance(value['assumptions'], list) or not all(isinstance(x, str) for x in value['assumptions']):
        raise ValueError('Invalid assumptions')
    if not isinstance(value['stop_condition'], str) or not value['stop_condition'].strip():
        raise ValueError('Missing stopping condition')
    heading = value['estimated_direction']
    source = 'language_estimate' if heading is not None else 'unresolved'
    stages = value['stages']
    if stages and all(s['heading_deg'] is not None and s['distance_m'] is not None for s in stages):
        east = sum(s['distance_m'] * math.sin(math.radians(s['heading_deg'])) for s in stages)
        north = sum(s['distance_m'] * math.cos(math.radians(s['heading_deg'])) for s in stages)
        heading = math.degrees(math.atan2(east, north)) % 360 if math.hypot(east, north) > 1e-6 else None
        source = 'numeric_vector' if heading is not None else 'zero_displacement'
    elif stages and all(s['heading_deg'] is not None for s in stages):
        # Any positive combination lies in this cone when its width is <180deg.
        # No guessed distances and no GT are required to select its midpoint.
        angles = sorted(s['heading_deg'] for s in stages)
        gaps = [(angles[(i+1) % len(angles)]-a) % 360 for i,a in enumerate(angles)]
        if len(set(angles)) == 1:
            heading, source = angles[0], 'heading_cone'
        else:
            i = max(range(len(gaps)), key=gaps.__getitem__)
            width = 360-gaps[i]
            if width < 180:
                heading = (angles[(i+1) % len(angles)]+width/2) % 360
                source = 'heading_cone'
    result = dict(direction=heading, final_target=value['final_target'], auxiliary=value['auxiliary'])
    try:
        validate_result(result, {})
    except ValidationError as exc:
        raise ValueError('Invalid initialization labels: ' + exc.message) from exc
    diagnostics=dict(value, direction_source=source, extracted_stages=relative)
    if corrections:diagnostics['language_consistency_corrections']=corrections
    return result, diagnostics
