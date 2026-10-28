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
                'backend','status','candidates','error','note','rejected_candidates','query','view_id',
                'sources','unresolved','unassessed_sources','attribution_issues','fusion_requests','skill_version','evidence_id','cache_hit')}
        return {k:model_context(v) for k,v in value.items()
                if k not in ('bbox_pixels','pixel_to_world','world_to_raster','mask_path')}
    if isinstance(value,list):return [model_context(v) for v in value]
    return value


def safe_view(view):
    return model_context({k:v for k,v in view.items() if k not in ('image_path','valid_mask_path','unmarked_image_path')})

class Model:
    def __init__(self, directory, model, key, max_requests=MAX_REQUESTS, url=DEFAULT_URL,
                 enable_thinking=ENABLE_THINKING, thinking_budget=1024, tool_protocol='legacy', visual_model=None):
        self.directory=Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        if tool_protocol not in ('legacy','structured_v1','structured_v2'):raise ValueError('Unknown tool protocol')
        self.tool_protocol=tool_protocol
        self.visual_model=visual_model
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
            if context.get('grounding_mode') in ('skill','adaptive'):
                history=safe.pop('history',{})
                safe['review_evidence']=dict(
                    current_observation=history.get('current_observation'),
                    previous_review=history.get('latest_review'),
                    note='Previous review is a fallible claim; independently verify the current candidate from RGB.')
                if context.get('grounding_mode')=='adaptive':
                    # Old rejection prose can be copied onto a different candidate.
                    # Recovery uses it in decision context; independent review does not.
                    safe['review_evidence'].pop('previous_review',None)
                    safe['review_evidence']['note']='Inspect this marked candidate independently. Tool observations are fallible and may concern other regions; only facts bound to this candidate support its identity.'
        skill_decision=context.get('mode')=='decision' and context.get('grounding_mode')=='skill'
        adaptive=context.get('grounding_mode')=='adaptive'
        system=REVIEW_SYSTEM if context.get('mode')=='review' else SYSTEM
        if adaptive and context.get('mode')=='review':
            from .adaptive_policy import REVIEW_IDENTITY
            system += REVIEW_IDENTITY
        if skill_decision:
            from ..prompts import SKILL_SYSTEM
            system=SKILL_SYSTEM
        if context.get('mode')=='decision' and context.get('grounding_mode')=='adaptive':
            from .adaptive_policy import DECISION_SYSTEM
            system=DECISION_SYSTEM
            if context.get('evidence_focus'):
                from .adaptive_policy import RECOVERY_SYSTEM
                system=RECOVERY_SYSTEM
        request_model=self.model
        output_tokens=MAX_OUTPUT_TOKENS
        if context.get('mode')=='visual_evidence':
            if not self.visual_model:raise ValueError('Visual evidence model is disabled')
            from .visual_evidence import SYSTEM as VISUAL_SYSTEM
            system=VISUAL_SYSTEM
            request_model=self.visual_model
            output_tokens=4096
        if context.get('mode')=='grounding_fusion':
            from .grounding_skill import FUSION_SYSTEM
            system=FUSION_SYSTEM
            output_tokens=6144
        if not adaptive and context.get('mode')=='decision' and 'jev' in context.get('available_evidence_tools',[]):
            from ..prompts import JEV_TOOL_SYSTEM
            system += JEV_TOOL_SYSTEM
        if not adaptive and context.get('mode')=='decision' and self.tool_protocol in ('structured_v1','structured_v2'):
            from .memory import presentation
            from ..prompts import TOOL_SYSTEM
            if skill_decision:
                from ..prompts import SKILL_TOOL_SYSTEM as TOOL_SYSTEM
            if self.tool_protocol=='structured_v2':
                from .memory import presentation_v2
                from ..prompts import TOOL_SYSTEM_V2
                if skill_decision:
                    from ..prompts import SKILL_TOOL_SYSTEM_V2 as TOOL_SYSTEM_V2
                safe=presentation_v2(safe)
                system += TOOL_SYSTEM + TOOL_SYSTEM_V2
            else:
                safe=presentation(safe)
                system += TOOL_SYSTEM
        if context.get('mode')=='decision' and context.get('grounding_mode')=='skill':
            from .grounding_skill import DECISION_SYSTEM
            system += DECISION_SYSTEM
        if adaptive and context.get('mode') in ('decision','review'):
            from .memory import presentation_adaptive
            safe=presentation_adaptive(safe)
        request_format='labelled-views-v7-adaptive' if adaptive else 'labelled-views-v6-json-object'
        assets=[dict(view_id=v['view_id'],image_path=v['image_path'],
                     sha256=hashlib.sha256(Path(v['image_path']).read_bytes()).hexdigest()) for v in views]
        signature=digest(dict(system=system,context=safe,assets=assets,model=request_model,endpoint=self.url,
            request_format=request_format,temperature=0,max_output_tokens=output_tokens,
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
        if adaptive and context.get('mode')=='decision':
            review=safe.get('history',{}).get('latest_review')
            focus=dict(latest_feedback=safe.get('latest_feedback',[]),
                requested_destination=safe.get('initialization',{}).get('final_target'),
                target_note='Fallible initialization; verify the requested noun/group against instruction_raw. Do not substitute its reference object.',
                evidence_focus=safe.get('evidence_focus'),
                latest_review=review,
                tools_callable_now=safe.get('tools_callable_now',[]),
                skills_callable_now=safe.get('skills_callable_now',[]),
                optional_tools_remaining=safe.get('optional_tools_remaining'),
                model_requests_remaining=safe.get('model_requests_remaining'))
            content.append(dict(type='text',text='CURRENT execution state: '+json.dumps(focus,ensure_ascii=False)+
                '\nReturn the next valid action. Address the latest error/review conflict; do not repeat a rejected action unchanged.'))
        if adaptive and context.get('mode')=='review':
            task=dict(instruction_raw=safe.get('instruction_raw'),
                requested_destination=safe.get('initialization',{}).get('final_target'),
                proposal_target=safe.get('candidate',{}).get('target'),
                proposal_source_view=safe.get('candidate',{}).get('view_id'),
                proposal_source_bbox=safe.get('candidate',{}).get('bbox'),
                projection_in_review=safe.get('candidate',{}).get('projection_in_review'))
            content.append(dict(type='text',text='INDEPENDENT IDENTITY CHECK: '+json.dumps(task,ensure_ascii=False)+
                '\nFirst name the requested object/group in reason. Check that the marked proposal is that object, not merely its reference. Verify the full instruction; initialization and tool opinions are fallible.'))
        messages=[dict(role='system',content=system),dict(role='user',content=content)]
        record=dict(signature=signature,status='pending',model=request_model,
                    system_prompt=system,
                    enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,
                    max_tokens=output_tokens,temperature=0,
                    response_format={'type':'json_object'},
                    request_format=request_format,context=safe,assets=assets)
        save_json(path,record)  # Persist before sending; never auto reissue an uncertain request.
        try:
            response=call_model(self.url,request_model,self.key,messages,timeout=90,enable_thinking=self.enable_thinking,thinking_budget=self.thinking_budget,
                max_tokens=output_tokens,temperature=0,response_format={'type':'json_object'})
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
    from .jev_evidence import normalize_action
    obj=normalize_action(json.loads(text))
    if not isinstance(obj,dict) or not isinstance(obj.get('reason'),str) or not obj['reason'].strip():
        raise ValueError('Return one JSON action with a nonempty reason')
    if len(obj['reason'])>2000:
        raise ValueError('Keep reason concise')
    return obj
