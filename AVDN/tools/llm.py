"""Durable request journal + a round-wide budget. Secrets never enter traces."""
import base64
import hashlib
import json
from pathlib import Path
from ..initialization.budget import RequestBudget, BudgetExceeded
from ..initialization.client import call_model, DEFAULT_URL
from ..agent.io import save_json, load_json, digest
from ..agent.prompts import SYSTEM, REVIEW_SYSTEM
from ..config import MAX_REQUESTS, MAX_OUTPUT_TOKENS, ENABLE_THINKING

class Halt(RuntimeError):
    pass

class AgentBudget(RequestBudget):
    def reserve(self, model, messages, max_tokens):
        # Parent reserves 1024/image; add 3072 conservative tokens for each 768 image.
        count=sum(p.get('type')=='image_url' for m in messages if isinstance(m['content'],list) for p in m['content'])
        enlarged=messages+[dict(role='system',content='x'*(3072*count))]
        return super().reserve(model,enlarged,max_tokens)

def model_context(value):
    # All model-facing image coordinates use [0,1000]. Keep internal pixel geometry
    # in the trace/evaluator, not alongside normalized coordinates in the prompt.
    if isinstance(value,dict):
        if 'backend' in value and 'candidates' in value and 'status' in value:
            # Preserve every validated proposal. Omit transport duplicates and local
            # artifacts; the unmodified service response remains in the tool journal.
            value={k:v for k,v in value.items() if k in (
                'backend','status','candidates','error','note','rejected_candidates')}
        return {k:model_context(v) for k,v in value.items()
                if k not in ('bbox_pixels','pixel_to_world','world_to_raster','mask_path')}
    if isinstance(value,list):return [model_context(v) for v in value]
    return value


def safe_view(view):
    return model_context({k:v for k,v in view.items() if k not in ('image_path','valid_mask_path','unmarked_image_path')})

class Model:
    def __init__(self, directory, budget, model, key, max_requests=MAX_REQUESTS, url=DEFAULT_URL,
                 enable_thinking=ENABLE_THINKING, thinking_budget=1024,
                 decision_prompt=None, review_prompt=None, request_timeout=90, connect_timeout=20):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.budget,self.model,self.key,self.url=budget,model,key,url
        self.request_timeout,self.connect_timeout=request_timeout,connect_timeout
        self.max_requests=max_requests
        self.enable_thinking=bool(enable_thinking)
        if self.enable_thinking and (isinstance(thinking_budget,bool) or not isinstance(thinking_budget,int) or thinking_budget<=0):
            raise ValueError('thinking_budget must be a positive integer')
        self.thinking_budget=thinking_budget if self.enable_thinking else 0
        self.reserved_output_tokens=MAX_OUTPUT_TOKENS+self.thinking_budget
        self.cursor=0
        self.successful_views=set()
        self.decision_prompt=SYSTEM if decision_prompt is None else decision_prompt
        self.review_prompt=REVIEW_SYSTEM if review_prompt is None else review_prompt

    def request(self, context, views):
        if self.cursor>=self.max_requests:
            raise Halt('model_request_limit')
        request_model = 'qwen3.7-plus' if context.get('mode') == 'decision' else self.model
        safe=model_context(dict(context,views=[safe_view(v) for v in views]))
        if context.get('mode')=='review':
            # Preserve full navigation memory in traces/decisions; review the current
            # object without anchoring on earlier guesses about different candidates.
            safe['memory']={'confirmed_auxiliary':safe.get('memory',{}).get('confirmed_auxiliary',[])}
        system=self.review_prompt if context.get('mode')=='review' else self.decision_prompt
        assets=[dict(view_id=v['view_id'],image_path=v['image_path'],
                     sha256=hashlib.sha256(Path(v['image_path']).read_bytes()).hexdigest()) for v in views]
        signature=digest(dict(system=system,context=safe,assets=assets,model=request_model,
            request_format='labelled-views-v6-json-object',temperature=0,
            enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,response_format={'type':'json_object'}))
        path=self.directory/f'{self.cursor:03d}.json'
        self.cursor+=1
        if path.exists():
            record=load_json(path)
            if record['signature']!=signature:
                raise Halt('resume_input_mismatch')
            if record['status']=='pending':
                raise Halt('uncertain_request_pending_reservation')
            if record['status']=='ok':
                self.successful_views.update(v['view_id'] for v in views)
            return record
        content=[dict(type='text',text=json.dumps(safe,ensure_ascii=False))]
        for index,view in enumerate(views,1):
            label=dict(image_number=index,view_id=view['view_id'],purpose=view['source'],
                image_coordinates='[0,1000]; x right, y down',
                image_up_heading_deg=safe_view(view)['image_up_heading_deg'],parent_view_id=view.get('parent_id'),
                annotation=view.get('annotation'))
            if context.get('candidate'):
                candidate=context['candidate']
                projection=candidate.get('projection_in_review',{})
                if projection.get('view_id')==view['view_id']:
                    label['current_unconfirmed_proposal_bbox']=projection['bbox']
                elif candidate.get('view_id')==view['view_id']:
                    label['current_unconfirmed_proposal_bbox']=candidate['bbox']
            content.append(dict(type='text',text='The NEXT image is '+json.dumps(label)))
            content.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+
                base64.b64encode(Path(view['image_path']).read_bytes()).decode())))
        messages=[dict(role='system',content=system),dict(role='user',content=content)]
        try:
            reservation=self.budget.reserve(request_model,messages,self.reserved_output_tokens)
        except BudgetExceeded as exc:
            self.cursor-=1
            reason = 'pricing_configuration_limit' if 'price' in str(exc).lower() else 'budget_limit'
            raise Halt(reason) from exc
        record=dict(signature=signature,status='pending',reservation=reservation,model=request_model,
                    system_prompt=system,budget_path=str(self.budget.path.resolve()),
                    reservation_cny=self.budget.read()['reservations'][reservation]['cny'],
                    enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,
                    reserved_output_tokens=self.reserved_output_tokens,max_tokens=MAX_OUTPUT_TOKENS,temperature=0,
                    response_format={'type':'json_object'},
                    request_format='labelled-views-v6-json-object',context=safe,assets=assets)
        save_json(path,record)  # Crash here conservatively retains reservation; never auto reissue.
        try:
            response=call_model(self.url,request_model,self.key,messages,timeout=self.request_timeout,connect_timeout=self.connect_timeout,enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,
                max_tokens=MAX_OUTPUT_TOKENS,temperature=0,response_format={'type':'json_object'})
            self.budget.settle(reservation,response.get('usage'))
            record.update(response=response,status='ok' if response['status']==200 else 'api_error')
        except Exception as exc:
            # The provider may have processed a timed-out request. Charge its full upper bound.
            self.budget.settle(reservation,None)
            record.update(status='api_error',error=type(exc).__name__)
        save_json(path,record)
        if record['status']=='ok':
            self.successful_views.update(v['view_id'] for v in views)
        return record

def parse_action(text):
    text=text.strip()
    if text.startswith('```') and text.endswith('```'):
        text=text.split('\n',1)[1].rsplit('```',1)[0].strip()
    obj=json.loads(text)
    if not isinstance(obj,dict) or not isinstance(obj.get('reason'),str) or not obj['reason'].strip():
        raise ValueError('Return one JSON action with a nonempty reason')
    if len(obj['reason'])>2000:
        raise ValueError('Keep reason concise')
    return obj

