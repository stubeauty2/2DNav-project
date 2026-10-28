"""Typed, local visual evidence. This module cannot confirm or move a target."""
import base64
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import time

import requests
from PIL import Image

TEMPLATE_VERSION = 'visual-evidence-v2-object-bound'
COLORS = ['red','blue','green','white','black','gray','tan','brown','yellow','mixed','other','unknown']


def normalize_action(action):
    """Move explicit per-question reasons to the action; infer no visual fields."""
    if not isinstance(action,dict) or action.get('action')!='tool' or action.get('tool')!='jev':
        return action
    questions=action.get('questions')
    if not isinstance(questions,list) or not questions or any(not isinstance(q,dict) for q in questions):
        return action
    reasons=[q['reason'] for q in questions if isinstance(q.get('reason'),str) and q['reason'].strip()]
    if any('reason' in q and (not isinstance(q['reason'],str) or not q['reason'].strip()) for q in questions):
        raise ValueError('Question reason must be a nonempty string')
    result=dict(action,questions=[{k:v for k,v in q.items() if k!='reason'} for q in questions])
    if 'reason' not in result and len(reasons)==len(questions):result['reason']='; '.join(reasons)
    return result


def validate_questions(questions):
    if not isinstance(questions,list) or not 1<=len(questions)<=3:
        raise ValueError('Jev requires one to three questions')
    seen=set()
    for q in questions:
        if not isinstance(q,dict):raise ValueError('Invalid Jev question')
        kind=q.get('kind')
        if kind not in ('color','count','presence','shape'):raise ValueError('Unknown Jev question kind')
        keys={'id','kind','object'}
        if ('bbox' in q)==('object_ref' in q):raise ValueError('Specify exactly one bbox or object_ref')
        keys.add('bbox' if 'bbox' in q else 'object_ref')
        if set(q)!=keys:raise ValueError('Unexpected Jev question fields')
        for k in ('id','object'):
            if not isinstance(q[k],str) or not q[k].strip() or len(q[k])>80:
                raise ValueError('Invalid Jev '+k)
        if q['id'] in seen:raise ValueError('Duplicate Jev question ID')
        seen.add(q['id'])
        if 'object_ref' in q:
            if not isinstance(q['object_ref'],str) or not q['object_ref']:raise ValueError('Invalid object reference')
        else:
            b=q['bbox']
            if not isinstance(b,list) or len(b)!=4 or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in b):
                raise ValueError('Invalid Jev bbox')
            if not (0<=b[0]<b[2]<=1000 and 0<=b[1]<b[3]<=1000):raise ValueError('Jev bbox outside view')
    return questions


def question_template(q):
    obj=q['object']
    if q['kind']=='color':
        choices=COLORS
        text=f'What is the dominant visible surface color of the {obj} itself, excluding surrounding vegetation, shadows and background? Do not substitute roof color for unobservable wall color. Choose unknown if the object or requested surface cannot be resolved.'
    elif q['kind']=='shape':
        choices=['rectangular','L-shaped','U-shaped','circular','irregular','multiple','unknown']
        text=f'What is the visible outline shape of the {obj} itself? Exclude neighboring objects. Choose multiple if distinct instances prevent one shape answer, or unknown if unresolved.'
    elif q['kind']=='count':
        choices=[str(i) for i in range(11)]+['more_than_10','unknown']
        text=f'How many visible {obj} instances can be independently recognized in this image? Count instances whose centers lie inside the image. Do not infer hidden objects.'
    else:
        choices=['present','absent','unknown']
        text=f'Is any {obj} visibly recognizable in this image?'
    return dict(type='choice',instructions=text,criteria={x:x for x in choices})


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class JevEvidence:
    def __init__(self,url,model='OneJev1.6-4B',max_questions=24,transport=None):
        if isinstance(max_questions,bool) or not isinstance(max_questions,int) or max_questions<=0:
            raise ValueError('Positive Jev question budget required')
        self.url=url.rstrip('/') if url else None
        self.model,self.max_questions,self.transport=model,max_questions,transport
        self.used=0;self.cache={};self.health=None

    def available(self):
        if self.transport:
            self.health=dict(status='ok',model=self.model,service_manifest_sha256='test-transport')
            return True
        if not self.url:return False
        try:
            r=requests.get(self.url+'/health',timeout=3);r.raise_for_status();h=r.json()
            if h.get('status')!='ok' or h.get('model')!=self.model or not h.get('service_manifest_sha256'):
                return False
            self.health=h;return True
        except (requests.RequestException,ValueError):return False

    def inspect(self,view,questions,output):
        # Resolved questions include only validated normalized boxes, not labels.
        validate_questions(questions)
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        result=dict(backend='jev',status='unavailable',view_id=view['view_id'],answers=[],
                    note='Visual evidence only; unavailable/error is not evidence of absence.')
        if self.health is None:return result
        if self.used+len(questions)>self.max_questions:
            return dict(result,status='budget_exhausted')
        with Image.open(view.get('unmarked_image_path',view['image_path'])) as im:
            source=im.convert('RGB')
        source_hash=hashlib.sha256(source.tobytes()).hexdigest()
        request_spec=dict(view_id=view['view_id'],image_sha256=source_hash,image_size=list(source.size),
            questions=questions,model=self.model,service_identity=self.health['service_manifest_sha256'],
            template=TEMPLATE_VERSION,debias=1)
        sig=signature(request_spec)
        if output.exists():
            saved=json.loads(output.read_text(encoding='utf-8'))
            if saved.get('signature')!=sig:raise ValueError('Jev replay input changed; use a new experiment')
            self.used+=len(questions)
            return saved
        if sig in self.cache:
            saved=dict(self.cache[sig],cache_hit=True)
            self.used+=len(questions)
            output.write_text(json.dumps(saved,ensure_ascii=False,indent=2),encoding='utf-8')
            return saved
        self.used+=len(questions)
        result.update(status='ok',signature=sig,input=request_spec,raw=[],cache_hit=False,
            calibration='uncalibrated',template_version=TEMPLATE_VERSION)
        start=time.perf_counter()
        for q in questions:
            b=q['bbox'];w,h=source.size
            bounds=[math.floor(b[0]*w/1000),math.floor(b[1]*h/1000),math.ceil(b[2]*w/1000),math.ceil(b[3]*h/1000)]
            crop=source.crop(bounds);buf=BytesIO();crop.save(buf,format='PNG');data=buf.getvalue()
            template=question_template(q)
            record=dict(question_id=q['id'],kind=q['kind'],object=q['object'],view_id=view['view_id'],bbox=b,
                image_sha256=hashlib.sha256(data).hexdigest(),source_pixels=bounds,source_size=list(source.size),
                scope='Visible instances whose centers are inside this ROI; no cross-view counting.')
            try:
                payload=dict(model=self.model,state={'image':'<image:1>'},questions={'visual_fact':template},
                    media=[dict(type='image',data='data:image/png;base64,'+base64.b64encode(data).decode())])
                if self.transport:raw=self.transport(payload)
                else:
                    response=requests.post(self.url+'/v1/systemone',params={'debias':1},json=payload,timeout=90)
                    response.raise_for_status();raw=response.json()
                if raw.get('model')!=self.model:raise ValueError('Wrong Jev response model')
                answer=raw['answers']['visual_fact'];probs=answer['probabilities']
                if set(probs)!=set(template['criteria']):raise ValueError('Wrong answer options')
                if any(isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1 for p in probs.values()):
                    raise ValueError('Invalid probabilities')
                if abs(sum(probs.values())-1)>1e-3 or answer['choice'] not in probs:raise ValueError('Invalid distribution')
                if probs[answer['choice']]<max(probs.values())-1e-6:raise ValueError('Choice differs from argmax')
                record.update(status='ok',choice=answer['choice'],probabilities=probs,
                    calibration='uncalibrated',evidence_id='jev-'+sig[:12]+'-'+str(len(result['answers'])))
                result['raw'].append(raw)
                artifact=output.with_name(output.stem+'-q'+str(len(result['answers']))+'.png')
                artifact.write_bytes(data)
            except (requests.RequestException,ValueError,KeyError,TypeError) as exc:
                record.update(status='error',error=type(exc).__name__)
                result['status']='partial_error'
            result['answers'].append(record)
        result['elapsed_s']=time.perf_counter()-start
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        self.cache[sig]=result
        return result


def public_evidence(result):
    public={k:result[k] for k in ('backend','status','view_id','note','calibration','template_version','cache_hit','elapsed_s') if k in result}
    public['answers']=[{k:v for k,v in a.items() if k not in ('source_pixels','source_size','image_sha256')}
                       for a in result.get('answers',[])]
    return public


def evidence_metadata(public, raw):
    """Expose original crop size and same-call presence binding, not a verdict."""
    source = {a['question_id']:a for a in raw.get('answers', [])}
    for answer in public.get('answers', []):
        detail = source[answer['question_id']]
        if 'source_pixels' in detail:
            x1,y1,x2,y2 = detail['source_pixels']
            answer['roi_pixels'] = [x2-x1,y2-y1]
        if answer['kind'] == 'color':
            presence = next((a for a in public['answers'] if a['kind']=='presence'
                and a['object']==answer['object'] and a['view_id']==answer['view_id']
                and a['bbox']==answer['bbox'] and a.get('status')=='ok'), None)
            answer['same_roi_presence'] = presence['choice'] if presence else 'not_checked_in_this_call'
            if presence and presence['choice'] in ('absent','unknown'):
                answer['attribute_applicability'] = 'not established: object presence absent or unknown'
    return public
