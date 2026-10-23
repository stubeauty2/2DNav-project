"""Native DashScope SSE and compatible chat transport; never logs headers."""
import json
import time
import requests
from ..config import MODEL as DEFAULT_MODEL, API_URL as DEFAULT_URL
import base64
import hashlib
from pathlib import Path
from .runtime import save_json, load_json, digest
from ..prompts import SYSTEM, REVIEW_SYSTEM
from ..config import MAX_REQUESTS, MAX_OUTPUT_TOKENS, ENABLE_THINKING


def native_messages(messages):
    converted = []
    for message in messages:
        content = message['content']
        if isinstance(content, str):
            parts = [{'text': content}]
        else:
            parts = []
            for item in content:
                if item['type'] == 'text':
                    parts.append({'text': item['text']})
                elif item['type'] == 'image_url':
                    parts.append({'image': item['image_url']['url']})
        converted.append({'role': message['role'], 'content': parts})
    return converted


def extract_text(body):
    choices = body.get('output', body).get('choices', [])
    if not choices:
        return ''
    content = choices[0].get('message', {}).get('content', '')
    if isinstance(content, str):
        return content
    # Native thinking fields are deliberately not persisted as output evidence.
    return ''.join(part.get('text', '') for part in content)


def consume_native(lines, deadline=None):
    texts, usage = [], None
    for line in lines:
        if deadline is not None and time.monotonic() >= deadline:
            raise requests.Timeout('Native SSE exceeded total response deadline')
        if not line or not line.startswith('data:'):
            continue
        value = line[5:].strip()
        if not value or value == '[DONE]':
            continue
        body = json.loads(value)
        if body.get('code') and not body.get('output'):
            raise ValueError('Native API error code: '+str(body['code']))
        texts.append(extract_text(body))
        if body.get('usage'):
            usage = body['usage']
    return ''.join(texts), usage


def call_model(url, model, key, messages, timeout, protocol='auto', enable_thinking=True,
               thinking_budget=50, max_tokens=4096, temperature=None, response_format=None):
    deadline = time.monotonic() + timeout
    native = protocol == 'native' or (protocol == 'auto' and '/multimodal-generation/' in url)
    headers = {'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'}
    if native:
        headers['X-DashScope-SSE'] = 'enable'
        parameters = {'enable_thinking': enable_thinking, 'incremental_output': True,
                      'max_tokens': max_tokens}
        if enable_thinking:
            parameters['thinking_budget'] = thinking_budget
        if temperature is not None:
            parameters['temperature'] = temperature
        if response_format is not None:
            parameters['response_format'] = response_format
        payload = {'model': model, 'input': {'messages': native_messages(messages)},
                   'parameters': parameters}
    else:
        payload = {'model': model, 'messages': messages, 'temperature': 0,
                   'max_tokens': max_tokens, 'stream': False, 'enable_thinking': enable_thinking}
        if enable_thinking:
            payload['thinking_budget'] = thinking_budget
        if response_format is not None:
            payload['response_format'] = response_format
    with requests.post(url, headers=headers, json=payload, timeout=(20,timeout), stream=native) as response:
        if response.status_code != 200:
            return {'status': response.status_code, 'text': None, 'usage': None}
        if native and 'event-stream' in response.headers.get('Content-Type',''):
            response.encoding = 'utf-8'
            # Small chunks let heartbeat-only streams reach the deadline check.
            text, usage = consume_native(response.iter_lines(chunk_size=1, decode_unicode=True), deadline)
        else:
            body = response.json()
            if body.get('code') and not body.get('output'):
                raise ValueError('Native API error code: '+str(body['code']))
            text, usage = extract_text(body), body.get('usage')
        if not text.strip():
            raise ValueError('Model returned no final answer text')
        return {'status': response.status_code, 'text': text, 'usage': usage}



class Halt(RuntimeError):
    pass

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
    def __init__(self, directory, model, key, max_requests=MAX_REQUESTS, url=DEFAULT_URL,
                 enable_thinking=ENABLE_THINKING, thinking_budget=1024):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.model,self.key,self.url=model,key,url
        self.max_requests=max_requests
        self.enable_thinking=bool(enable_thinking)
        if self.enable_thinking and (isinstance(thinking_budget,bool) or not isinstance(thinking_budget,int) or thinking_budget<=0):
            raise ValueError('thinking_budget must be a positive integer')
        self.thinking_budget=thinking_budget if self.enable_thinking else 0
        self.cursor=0
        self.successful_views=set()

    def request(self, context, views):
        if self.cursor>=self.max_requests:
            raise Halt('model_request_limit')
        safe=model_context(dict(context,views=[safe_view(v) for v in views]))
        if context.get('mode')=='review':
            # Preserve full navigation memory in traces/decisions; review the current
            # object without anchoring on earlier guesses about different candidates.
            safe['memory']={'confirmed_auxiliary':safe.get('memory',{}).get('confirmed_auxiliary',[])}
        system=REVIEW_SYSTEM if context.get('mode')=='review' else SYSTEM
        assets=[dict(view_id=v['view_id'],image_path=v['image_path'],
                     sha256=hashlib.sha256(Path(v['image_path']).read_bytes()).hexdigest()) for v in views]
        signature=digest(dict(system=system,context=safe,assets=assets,model=self.model,endpoint=self.url,
            request_format='labelled-views-v6-json-object',temperature=0,
            enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,response_format={'type':'json_object'}))
        path=self.directory/f'{self.cursor:03d}.json'
        self.cursor+=1
        if path.exists():
            record=load_json(path)
            if record['signature']!=signature:
                raise Halt('resume_input_mismatch')
            if record['status']=='pending':
                raise Halt('uncertain_request_pending')
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
        record=dict(signature=signature,status='pending',model=self.model,
                    system_prompt=system,
                    enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,
                    max_tokens=MAX_OUTPUT_TOKENS,temperature=0,
                    response_format={'type':'json_object'},
                    request_format='labelled-views-v6-json-object',context=safe,assets=assets)
        save_json(path,record)  # Persist before sending; never auto reissue an uncertain request.
        try:
            response=call_model(self.url,self.model,self.key,messages,timeout=90,enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,
                max_tokens=MAX_OUTPUT_TOKENS,temperature=0,response_format={'type':'json_object'})
            record.update(response=response,status='ok' if response['status']==200 else 'api_error')
        except Exception as exc:
            # Preserve failures in the journal; do not automatically repeat uncertain calls.
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
