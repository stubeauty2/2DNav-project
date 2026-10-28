"""Bounded multimodal questions about observed RGB; never navigation or review."""
import json
from .grounding_skill import valid_box
from .runtime import save_json


SYSTEM = '''Answer a specific visual evidence question about the supplied aerial RGB.
The question and region labels are hypotheses, not facts. Inspect the image yourself.
Regions use the source view's [0,1000] xyxy frame. Compare the specified regions,
including their visible surroundings. Distinguish target objects from reference
objects. For ordinals, report the counting origin/direction and ambiguity; for
relations, state which visible reference supports them. Do not infer a named place
from appearance or invisible facade details from a roof. A supplied public landmark
is geometry, not proof of object identity. Say unknown when resolution is insufficient.
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


class VisualEvidence:
    def __init__(self, model):
        self.model=model

    def inspect(self, view, question, regions, output):
        validate_request(question,regions)
        context=dict(mode='visual_evidence',question=question,regions=regions,
                     source_view_id=view['view_id'])
        record=self.model.request(context,[view])
        result=dict(backend='visual_question',status='error',view_id=view['view_id'],
                    question=question,regions=regions,model=record.get('model'),
                    request_index=self.model.cursor-1,
                    note='Fallible RGB evidence, not confirmation. Missing evidence is not absence.')
        if record['status']=='ok':
            try:
                answer=validate_answer(json.loads(record['response']['text']),regions)
                result.update(status='ok',**answer)
            except (ValueError,KeyError,TypeError):
                result['error']='Invalid visual evidence response; use RGB or a different question/view'
        else:result['error']='Visual evidence API unavailable'
        save_json(output,result)
        return result
