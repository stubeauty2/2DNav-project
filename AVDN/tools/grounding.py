"""Grounding service protocol: proposals are evidence, never navigation decisions."""
import json
import re
import math
import shutil
import hashlib
from pathlib import Path
import requests


class Grounder:
    backend='unknown'
    def __init__(self,url=None,transport=None):
        self.url=url.rstrip('/') if url else None
        self.transport=transport

    def available(self):
        if self.transport:return True
        if not self.url:return False
        try:
            response=requests.get(self.url+'/health',timeout=3)
            state=response.json()
            return response.ok and not state.get('error') and state.get('backend',self.backend)==self.backend
        except (requests.RequestException,ValueError):return False

    def candidates(self,view,query,output):
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(backend=self.backend,request_id=output.stem,
            views=[dict(view_id=view['view_id'],role='search',image_path=view['image_path'],scale=5)],
            concepts=[dict(concept_id='query',text=query,role='target')],max_candidates_per_view=24)
        signature=hashlib.sha256(json.dumps(dict(payload=payload,url=self.url,
            image_sha256=hashlib.sha256(Path(view['image_path']).read_bytes()).hexdigest()),
            sort_keys=True).encode()).hexdigest()
        if output.exists():
            saved=json.loads(output.read_text(encoding='utf-8'))
            if saved.get('signature')!=signature:
                raise ValueError('Tool replay input changed; use a new experiment')
            return saved
        result=dict(status='unavailable',backend=self.backend,input=payload,candidates=[],
                    error='Service is not configured; this is not evidence of target absence')
        if self.url or self.transport:
            try:
                if self.transport:raw=self.transport(payload)
                else:
                    response=requests.post(self.url+'/v1/candidates',json=payload,timeout=180)
                    response.raise_for_status();raw=response.json()
                if raw.get('error'):raise ValueError('Service returned an error')
                size=view.get('image_size',[768,768])
                extent=raw.get('coordinate_extent',[size[0]-1,size[1]-1])
                if len(extent)!=2 or any(not math.isfinite(x) or x<=0 for x in extent):raise ValueError('Invalid coordinate extent')
                candidates=[]
                for i,c in enumerate(raw.get('candidates',[])):
                    if c.get('view_id',view['view_id'])!=view['view_id']:raise ValueError('Response references another view')
                    b=c['bbox_2d']
                    if len(b)!=4 or any(not math.isfinite(x) for x in b):raise ValueError('Invalid bbox')
                    box=[b[j]/extent[j%2]*1000 for j in range(4)]
                    if not(0<=box[0]<box[2]<=1000 and 0<=box[1]<box[3]<=1000):raise ValueError('Out-of-image detector box')
                    score=c.get('score')
                    if score is not None and (not math.isfinite(score) or not 0<=score<=1):raise ValueError('Invalid score')
                    mask_path=c.get('mask_path')
                    if mask_path:
                        dest=output.with_name(output.stem+'-mask-'+str(i)+'.png')
                        shutil.copyfile(mask_path,dest);mask_path=str(dest.resolve())
                    candidates.append(dict(candidate_id=c.get('candidate_id',str(i)),view_id=view['view_id'],
                        bbox=box,score=score,mask_path=mask_path,label=c.get('label',query)))
                result=dict(status='ok',backend=self.backend,input=payload,candidates=candidates,raw=raw)
            except (requests.RequestException,ValueError,KeyError,TypeError,OSError) as exc:
                result=dict(status='error',backend=self.backend,input=payload,candidates=[],error=type(exc).__name__,
                            note='Tool errors are not evidence of absence')
        result['signature']=signature
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
        return result


def observation_query(target, mode='full'):
    """Optional category-only proposal query; selection still uses the full instruction."""
    if mode == 'full':
        return target
    if mode != 'category':
        raise ValueError('Unknown observation query mode')
    match = re.search(r'\b(parking lot|parking area|warehouse|building|house|car|van|bus|field|court|tree|roof|road|bridge|pond|lake|pool)\b', target, re.I)
    return match.group().lower() if match else target



class SAM3(Grounder):
    backend="sam3"



class GroundingDINO(Grounder):
    backend='dino'



class LocateAnything(Grounder):
    backend='locate_anything'
