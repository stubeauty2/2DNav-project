"""Bounded multimodal questions about observed RGB; never navigation or review."""
import json
import copy
from .grounding_skill import valid_box, region_quality
from .runtime import save_json


SYSTEM = '''Answer a specific visual evidence question about the supplied aerial RGB.
The question and region labels are hypotheses, not facts. Inspect the image yourself.
Regions use the source view's [0,1000] xyxy frame. Compare the specified regions,
including their visible surroundings. Distinguish target objects from reference
objects. For ordinals, report the counting origin/direction and ambiguity; for
relations, state which visible reference supports them. Do not infer a named place
from appearance or invisible facade details from a roof. A supplied public landmark
is geometry, not proof of object identity. Say unknown when resolution is insufficient.
When public_reference is supplied, check directional claims in the question against
the original instruction, original pose and source-view geometry. Image-up need not
equal the original heading; reorienting a view does not reset the instruction's
reference frame. If the language leaves the reference ambiguous, report ambiguity
rather than silently choosing a frame. Region geometry projects the requested ROI
center, not a verified object or destination. Keep geometric checks separate from
visible appearance; report a mistaken premise without selecting the destination.
region_visibility describes raster coverage only. Black map padding is unobserved,
not evidence of an empty scene. For partial regions, assess only the visible portion
and explicitly leave the rest unknown; do not infer a complete object from padding.
No navigation action, final verdict or destination selection is allowed. Provide
observable facts and differences so the navigation agent can make its own decision.
Return ONE JSON object with these fields:
{"observations":[{"region_id":"A","support":["visible facts"],
"contradictions":[],"unobservable":[]}],"comparison":"differences/relations visible in RGB",
"uncertainty":["unresolved details"],"suggested_check":"specific missing observation, or none",
"proposals":[{"id":"rgb-1","bbox":[100,200,300,400],"category":"building",
"evidence":"directly visible category/shape/layout","uncertainty":[]}]}.
Include each supplied region exactly once; if regions=[], observations=[]. Never
invent region IDs or cite detector agreement as visual support. Keep the answer concise.
If requested regions are wrong, or the question asks about visible instances, propose
up to eight plausible objects with tight boxes in the ORIGINAL source view frame.
Include multiple alternatives when ambiguous. These are fallible RGB proposals,
not a final target verdict. Use proposals=[] when none can be grounded confidently.
'''


def validate_request(question, regions):
    if not isinstance(question, str) or not question.strip() or len(question)>1200:
        raise ValueError('visual_question requires a concrete question of 1..1200 characters')
    if not isinstance(regions, list) or len(regions)>4:
        raise ValueError('visual_question accepts up to four regions in the selected view')
    seen=set()
    for region in regions:
        if not isinstance(region,dict) or set(region)!={'id','bbox'}:
            raise ValueError('Each visual region requires id and bbox only')
        ident=region['id']
        if not isinstance(ident,str) or not ident.strip() or len(ident)>40 or ident in seen:
            raise ValueError('Visual region IDs must be unique nonempty short strings')
        if not valid_box(region['bbox']):raise ValueError('Invalid visual region bbox')
        seen.add(ident)


def reference_context(view, regions, episode):
    """Public task and fixed source geometry; independent of later camera moves."""
    from .geometry import world, candidate_context
    source=episode['agent_input'];pose=source['initial_pose']
    projected=[]
    for region in regions:
        x1,y1,x2,y2=region['bbox']
        center=world(view,[[(x1+x2)/2,(y1+y2)/2]])[0]
        geometry=candidate_context(center,episode,pose['xy_m'],pose['heading_deg'] or 0.)
        keys=('distance_from_initial_m','initial_forward_m','initial_right_m',
              'bearing_from_initial_deg','turn_from_initial_heading_deg','clock_from_initial_pose')
        projected.append(dict(region_id=region['id'],center_xy_m=center,
                              **{k:geometry[k] for k in keys}))
    return dict(instruction_raw=source['instruction_raw'],
        original_pose=dict(xy_m=list(pose['xy_m']),heading_deg=pose['heading_deg']),
        source_view_id=view['view_id'],region_geometry=projected,
        note='Only public starting pose and requested ROI projections. Coordinates are east/north meters; headings clockwise from north. ROI centers are not verified object centers. Original pose remains fixed for old views and zooms; use the separately labelled source image orientation. Unknown initial heading leaves relative directions unknown. No target annotations or prior review judgments supplied.')


def region_visibility(view, regions):
    """Evidence ROIs may cross padding; they are not executable locate boxes."""
    coverage=[]
    for region in regions:
        quality=region_quality(view,region['bbox'])
        if quality.get('valid_area_fraction')==0:
            raise ValueError('Visual region '+region['id']+' contains no valid raster pixels; choose an observed region')
        coverage.append(dict(region_id=region['id'],**quality))
    return coverage


def validate_answer(answer, regions):
    base={'observations','comparison','uncertainty','suggested_check'}
    if not isinstance(answer,dict) or set(answer) not in (base,base|{'proposals'}):
        raise ValueError('Invalid visual evidence fields')
    def strings(value):
        return isinstance(value,list) and all(isinstance(x,str) for x in value)
    if not all(isinstance(answer[k],str) for k in ('comparison','suggested_check')) or not strings(answer['uncertainty']):
        raise ValueError('Invalid visual evidence text')
    obs=answer['observations']
    if not isinstance(obs,list):raise ValueError('Invalid observations')
    for item in obs:
        if (not isinstance(item,dict) or set(item)!={'region_id','support','contradictions','unobservable'}
                or not isinstance(item['region_id'],str)
                or not all(strings(item[k]) for k in ('support','contradictions','unobservable'))):
            raise ValueError('Invalid region observation')
    if sorted(o['region_id'] for o in obs)!=sorted(r['id'] for r in regions):
        raise ValueError('Visual evidence must address exactly the supplied regions')
    proposals=answer.get('proposals',[])
    if not isinstance(proposals,list) or len(proposals)>8:raise ValueError('Invalid RGB proposals')
    seen=set()
    for p in proposals:
        if not isinstance(p,dict) or set(p)!={'id','bbox','category','evidence','uncertainty'}:
            raise ValueError('Invalid RGB proposal fields')
        if not all(isinstance(p[k],str) and p[k].strip() and len(p[k])<=800 for k in ('id','category','evidence')):
            raise ValueError('Invalid RGB proposal text')
        if p['id'] in seen or not valid_box(p['bbox']) or not strings(p['uncertainty']):
            raise ValueError('Invalid RGB proposal identity or coordinates')
        seen.add(p['id'])
    return answer


def recover_answer(answer, regions, view_id):
    """Preserve bound evidence without inferring missing visual judgments."""
    answer=copy.deepcopy(answer);notes=[]
    if not isinstance(answer,dict):raise ValueError('Invalid visual evidence fields')
    # A common response nests the summary object as the last observations item.
    if set(answer)=={'observations'} and isinstance(answer['observations'],list) and answer['observations']:
        tail=answer['observations'][-1]
        summary={'comparison','uncertainty','suggested_check'}
        if isinstance(tail,dict) and set(tail) in (summary,summary|{'proposals'}):
            answer.update(answer['observations'].pop())
            notes.append('Moved the uniquely nested summary object to the response root; text unchanged.')
    if 'source_view_id' in answer:
        if answer.pop('source_view_id')!=view_id:raise ValueError('Visual evidence source view mismatch')
        notes.append('Removed redundant matching source_view_id.')
    expected={r['id'] for r in regions};seen=set();observations=[];unrequested=[]
    bound_text=False
    if not isinstance(answer.get('observations'),list):raise ValueError('Invalid observations')
    for item in answer['observations']:
        if not isinstance(item,dict) or not isinstance(item.get('region_id'),str):
            raise ValueError('Invalid region observation')
        ident=item['region_id']
        if ident in seen:raise ValueError('Duplicate visual evidence region')
        seen.add(ident)
        allowed={'region_id','support','contradictions','unobservable'}
        if set(item)-allowed:raise ValueError('Invalid region observation fields')
        missing=[field for field in ('support','contradictions','unobservable') if field not in item]
        for field in ('support','contradictions','unobservable'):
            if field not in item:item[field]=[]
            if not isinstance(item[field],list) or not all(isinstance(x,str) for x in item[field]):
                raise ValueError('Invalid region observation text')
        if ident not in expected:
            unrequested.append(ident)
            continue
        bound_text=bound_text or any(text.strip() for field in ('support','contradictions','unobservable') for text in item[field])
        if missing:
            item['unobservable'].append('Model omitted these assessment fields: '+', '.join(missing)+'. Missing assessments do not establish absence of support or contradiction.')
            notes.append('Incomplete assessment for requested region '+ident+'.')
        observations.append(item)
    missing_regions=[r['id'] for r in regions if r['id'] not in seen]
    for ident in missing_regions:
        observations.append(dict(region_id=ident,support=[],contradictions=[],
            unobservable=['The model did not assess this requested region.']))
    if missing_regions:notes.append('Requested regions not assessed: '+', '.join(missing_regions)+'.')
    if unrequested:notes.append('Discarded observations with unrequested region IDs: '+', '.join(unrequested)+'. Their claims were not rebound to requested regions.')
    answer['observations']=observations
    missing_summary=[field for field in ('comparison','uncertainty','suggested_check') if field not in answer]
    if missing_summary:
        # Keep actual regional observations without inventing a global comparison.
        # Synthetic missing-region notes alone do not constitute visual evidence.
        comparison=answer.get('comparison')
        proposals=answer.get('proposals')
        if not (bound_text or isinstance(comparison,str) and comparison.strip()
                or isinstance(proposals,list) and proposals):
            raise ValueError('Incomplete visual response contains no retained evidence')
        note='Model omitted summary fields: '+', '.join(missing_summary)+'. No missing comparison, uncertainty assessment, or suggested check was inferred.'
        for field in missing_summary:answer[field]=[] if field=='uncertainty' else ''
        if isinstance(answer['uncertainty'],list):answer['uncertainty'].append(note)
        notes.append(note)
    # Present fields, region bindings, proposal fields and coordinates remain strict.
    return validate_answer(answer,regions),notes,missing_regions,unrequested


class VisualEvidence:
    def __init__(self, model):
        self.model=model

    def inspect(self, view, question, regions, output, *, public_reference=None):
        validate_request(question,regions)
        coverage=region_visibility(view,regions)
        context=dict(mode='visual_evidence',question=question,regions=regions,
                     region_visibility=coverage,
                     source_view_id=view['view_id'])
        if public_reference is not None:context['public_reference']=copy.deepcopy(public_reference)
        record=self.model.request(context,[view])
        result=dict(backend='visual_question',status='error',view_id=view['view_id'],
                    question=question,regions=regions,model=record.get('model'),
                    region_visibility=coverage,
                    request_index=self.model.cursor-1,
                    note='Fallible RGB evidence, not confirmation. Missing evidence is not absence.')
        if record['status']=='ok':
            try:
                answer,notes,missing,unrequested=recover_answer(json.loads(record['response']['text']),regions,view['view_id'])
                result.update(status='partial' if notes else 'ok',**answer)
                if notes:result.update(normalization_notes=notes,missing_regions=missing,unrequested_region_ids=unrequested)
            except (ValueError,KeyError,TypeError):
                result['error']='Invalid visual evidence response; use RGB or a different question/view'
        else:result['error']='Visual evidence API unavailable'
        save_json(output,result)
        return result
