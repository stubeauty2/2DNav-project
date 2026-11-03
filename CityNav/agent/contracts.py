import math
from ..initialization.contracts import validate_agent_input, validate_result

def public_episode(episode):
    if set(episode)!={'episode_id','agent_input','initialization','environment'}:
        raise ValueError('Unexpected episode fields: navigation accepts public input only')
    validate_agent_input(episode['agent_input'])
    validate_result(episode['initialization'],episode['agent_input'])
    spec=episode['environment']
    if set(spec)!={'image_path','image_size','world_to_raster','main_width_m','main_height_m'}:
        raise ValueError('Environment must contain only public raster calibration')
    return episode

def number(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError('Coordinate or angle must be a finite number')
    return value

def validate_action(a, review=False):
    if not review and a.get('action')=='review':
        raise ValueError('This is the DECISION stage. Submit locate with target final or auxiliary, view_id, bbox and reason to propose an object. Only the separate reviewer may return review/verdict.')
    if not review and a.get('action') in ('tool','zoom','sam3','dino','locate_anything') and 'view_id' not in a:
        raise ValueError('Missing view_id: every tool request must include the source image view_id from the displayed views, together with action, tool, query or bbox, and reason.')
    if (not review and a.get('action')=='explore' and a.get('mode')=='offset'
            and a.get('point') is None):
        # Offset movement uses current pose; these optional empty/source labels
        # cannot change the numeric turn or distance already supplied.
        a={k:v for k,v in a.items() if k!='point' and not (k=='view_id' and isinstance(v,str))}
    # An explicit tool name as action is an unambiguous spelling alias. Never
    # infer missing coordinates, accept conflicting tools, or relax other fields.
    if not review and a.get('action') in ('zoom','sam3','dino','locate_anything'):
        name=a['action']
        if a.get('tool',name)!=name:
            raise ValueError('Conflicting action/tool names')
        a=dict(a,action='tool',tool=name)
    # Landmark coordinates are independent of any image. A redundant source-view
    # label is harmless metadata; keep the raw reply in the request journal, but
    # do not let it block an otherwise valid global landmark waypoint.
    if (not review and a.get('action')=='explore' and a.get('mode')=='landmark'
            and isinstance(a.get('view_id'),str)):
        a={k:v for k,v in a.items() if k!='view_id'}
    if review and a.get('verdict') in ('reject','uncertain'):
        # A non-confirming verdict never executes its illustrative box. Retain it
        # in the raw journal, without using it as a fallback localization.
        a={k:v for k,v in a.items() if k not in ('view_id','bbox')}
    if review:
        if a.get('action')!='review' or a.get('verdict') not in ('confirm','reject','uncertain'):
            raise ValueError('Review requires action review, verdict confirm/reject/uncertain')
        keys={'action','verdict','reason'}
        if a['verdict']=='confirm':
            keys|={'view_id','bbox'}
    elif a.get('action')=='explore':
        mode=a.get('mode')
        fields={'point':{'view_id','point'},'offset':{'turn_deg','step_scale'},'landmark':{'landmark_id'}}
        if mode not in fields:
            raise ValueError('explore mode must be point, offset or landmark')
        keys={'action','mode','reason'}|fields[mode]
        if mode=='offset':
            number(a.get('turn_deg'));number(a.get('step_scale'))
            if a['step_scale']<=0:raise ValueError('step_scale must be positive')
    elif a.get('action')=='locate':
        keys={'action','target','view_id','bbox','reason'}
        if a.get('target') not in ('final','auxiliary'):raise ValueError('Unknown target role')
    elif a.get('action')=='tool':
        if a.get('tool') not in ('zoom','sam3','dino','locate_anything'):raise ValueError('Unknown tool')
        keys={'action','tool','view_id','reason'}|({'bbox'} if a['tool']=='zoom' else {'query'})
    elif a.get('action')=='give_up':
        keys={'action','reason'}
    else:
        raise ValueError('Unknown action')
    if set(a)!=keys:raise ValueError('Action fields must be exactly '+str(sorted(keys)))
    for k in ('point','bbox'):
        if k in a:
            if not isinstance(a[k],list) or len(a[k])!=(2 if k=='point' else 4):
                raise ValueError('Invalid '+k+' length')
            for v in a[k]:number(v)
    for k in ('view_id','landmark_id','query'):
        if k in a and (not isinstance(a[k],str) or not a[k].strip()):
            raise ValueError(k+' must be a nonempty string')
    return a

