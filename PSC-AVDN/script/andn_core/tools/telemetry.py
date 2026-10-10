"""STEP 0: local, best-effort observations; never choose or change an action.

Model bodies and request arguments are unchanged. RequestsProxy observes only
the owning function's transport, without patching the shared requests module.
"""
import base64
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from time import perf_counter
from types import FunctionType

_run = ContextVar('andn_trace_run', default=None)
_call = ContextVar('andn_trace_call', default=None)


def _jsonable(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, default=_jsonable))


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _safe(function):
    """A full disk or serialization failure must not change navigation."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        recorder = _run.get()
        if recorder is None:
            return None
        try:
            return function(recorder, *args, **kwargs)
        except Exception as exc:
            recorder.errors += 1
            if recorder.errors == 1:
                print('[WARN][TRACE] recording incomplete: ' + type(exc).__name__)
            return None
    return wrapped


class Recorder:
    def __init__(self, out_dir, parameters):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        self.root = Path(out_dir).resolve() / 'trace' / (stamp + '-' + str(os.getpid()))
        self.root.mkdir(parents=True, exist_ok=False)
        self.context = {}
        self.errors = 0
        self.counter = 0
        self.secrets = {v for k, v in os.environ.items()
                        if any(w in k.upper() for w in ('API_KEY', 'API_TOKEN')) and v}
        if parameters.get('api_key'):
            self.secrets.add(parameters['api_key'])
        self.started = perf_counter()
        self.counts = {'model_calls': 0, 'images': 0, 'search': 0, 'confirmation': 0,
                       'land': 0, 'sam3_calls': 0, 'model_fallbacks': 0, 'sam3_errors': 0}

    def clean(self, value):
        if isinstance(value, dict):
            return {k: ('[REDACTED]' if any(x in k.lower() for x in
                       ('api_key', 'api_token', 'authorization', 'password')) else self.clean(v))
                    for k, v in value.items()}
        if isinstance(value, list):
            return [self.clean(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, '[REDACTED]')
            value = re.sub(r'(?i)(bearer\s+)\S+', r'\1[REDACTED]', value)
            value = re.sub(r'sk-[A-Za-z0-9_.-]{12,}', '[REDACTED]', value)
            value = re.sub(r'(?i)([?&](?:api_key|token|key)=)[^&\s]+', r'\1[REDACTED]', value)
        return value

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.clean(_copy(value)), ensure_ascii=False, indent=2), encoding='utf-8')

    def source_bytes(self, data):
        # Preserve source bytes/line endings and dynamic Bearer templates.
        # Only actual known credentials and literal sk-* keys are scrubbed here.
        text = data.decode('utf-8-sig')
        cleaned = text
        for secret in self.secrets:
            cleaned = cleaned.replace(secret, '[REDACTED]')
        cleaned = re.sub(r'sk-[A-Za-z0-9_.-]{12,}', '[REDACTED]', cleaned)
        return data if cleaned == text else cleaned.encode('utf-8')

    def emit(self, kind, data):
        record = self.clean(_copy(dict(event=kind, since_start_s=perf_counter()-self.started,
                                       context=self.context, **data)))
        with (self.root/'events.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False)+'\n')

    def asset(self, data_url, index):
        prefix, encoded = data_url.split(',', 1)
        data = base64.b64decode(encoded, validate=True)
        digest = _digest(data)
        suffix = '.png' if 'image/png' in prefix else '.jpg' if 'image/jpeg' in prefix else '.bin'
        relative = 'images/' + digest + suffix
        path = self.root / relative
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        return dict(image_id=_call.get()['call_id']+':image'+str(index), path=relative,
                    sha256=digest, data_url_prefix=prefix, bytes=len(data))


@_safe
def context(recorder, **values):
    recorder.context = _copy(values)


@_safe
def event(recorder, kind, **data):
    recorder.emit(kind, data)


@_safe
def move(recorder, position, reason):
    previous = recorder.context.get('position')
    recorder.emit('position_assignment', {'from': previous, 'to': position, 'reason': reason,
                                         'meaning': 'baseline position update; not an evaluation crop'})
    recorder.context['position'] = _copy(position)


@_safe
def bbox(recorder, kind, box, corners, ob, **extra):
    from .geometry import _patch_xy_to_latlng
    polygon = None
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = box
        if x2 > x1 and y2 > y1:
            polygon = [_patch_xy_to_latlng(x, y, corners, ob)
                       for x, y in ((x1,y1),(x2,y1),(x2,y2),(x1,y2))]
    recorder.emit(kind, dict(bbox_2d=box, view_corners_latlng=corners,
                            map_polygon_latlng=polygon, pixel_grid=[768,768], **extra))


@_safe
def parsed(recorder, value):
    if _call.get() is not None:
        _call.get()['structured_reply'] = _copy(value)


@_safe
def _request(recorder, url, kwargs):
    call = _call.get()
    if call is None:
        return
    payload = kwargs.get('json')
    call['request_payload_sha256'] = _digest(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode('utf-8'))
    archived = _copy(payload)
    assets = []
    for message in archived.get('messages', []):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for block in content:
            if block.get('type') == 'image_url':
                original = block['image_url']['url']
                if original.startswith('data:'):
                    asset = recorder.asset(original, len(assets)+1)
                    descriptions = call.get('input_images', [])
                    if len(assets) < len(descriptions):
                        asset.update(descriptions[len(assets)])
                    assets.append(asset)
                    block['image_url']['url'] = 'asset:' + asset['path']
    call.update(request=archived, url=url, timeout=kwargs.get('timeout'), images=assets)
    recorder.counts['images'] += len(assets)
    recorder.write('calls/'+call['call_id']+'.request.json', call)
    recorder.emit('model_request', dict(call_id=call['call_id'], kind=call['kind'],
                                       image_count=len(assets), request_payload_sha256=call['request_payload_sha256']))


@_safe
def _response(recorder, response):
    call = _call.get()
    if call is not None:
        call['http'] = dict(status=getattr(response, 'status_code', None),
                            content_type=response.headers.get('Content-Type'),
                            body=getattr(response, 'text', None))


class _Response:
    def __init__(self, response):
        self.response = response

    def __getattr__(self, name):
        return getattr(self.response, name)

    def json(self, *args, **kwargs):
        value = self.response.json(*args, **kwargs)
        _decoded_response(value)
        return value


@_safe
def _decoded_response(recorder, value):
    if _call.get() is not None:
        _call.get()['response_json'] = _copy(value)


class RequestsProxy:
    """Forward precisely the original arguments and return/raise normally."""
    def __init__(self, transport):
        self.transport = transport

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def post(self, url, **kwargs):
        if _run.get() is None or _call.get() is None:
            return self.transport.post(url, **kwargs)
        _request(url, kwargs)
        response = self.transport.post(url, **kwargs)
        _response(response)
        return _Response(response)


def model_call(function=None, *, kind=None):
    if function is None:
        return lambda fn: model_call(fn, kind=kind)
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        recorder = _run.get()
        if recorder is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        parameters = bound.arguments
        if parameters.get('api_key'):
            recorder.secrets.add(parameters['api_key'])
        role = kind or ('search' if parameters.get('restrict_top_half', True) else 'confirmation')
        recorder.counter += 1
        recorder.counts['model_calls'] += 1
        recorder.counts[role] = recorder.counts.get(role, 0) + 1
        images = []
        for name, image_role in (('patch_bgr','main'), ('patch_main_bgr','main'),
                                 ('patch_s4_bgr','detail'), ('patch_s8_bgr','context'),
                                 ('minimap_bgr','minimap'), ('segmentation_image','sam3')):
            image = parameters.get(name)
            if image is not None:
                images.append(dict(role=image_role, width=int(image.shape[1]), height=int(image.shape[0]),
                                   parameter=name, localization_allowed=image_role=='main' and role!='land'))
        if parameters.get('candidate_count'):
            for item in images:
                item['view_id'] = 'search_' + item['role'] if item['role'] in ('main', 'detail', 'context') else item['role']
        if parameters.get('confirmation_views') is not None:
            images = [dict(role=v['role'], candidate_id=v['candidate_id'], view_id=v['view_id'],
                           width=int(v['image'].shape[1]), height=int(v['image'].shape[0]),
                           view_corners_latlng=_copy(v['corners']),
                           parameter='confirmation_views', localization_allowed=v['role']=='main')
                      for v in parameters['confirmation_views']]
        if parameters.get('search_context') is not None:
            images=[dict(role=v['role'],view_id=v['view_id'],width=int(v['image'].shape[1]),
                height=int(v['image'].shape[0]),parameter='search_context',
                localization_allowed=v['localization_allowed'],view_corners_latlng=_copy(v.get('corners')))
                for v in parameters['search_context']['views']]
        call = dict(call_id='call%05d'%recorder.counter, kind=role, context=_copy(recorder.context), input_images=images)
        token = _call.set(call)
        started = perf_counter()
        try:
            result = function(*args, **kwargs)
            _model_result(result)
            return result
        except BaseException as exc:
            _model_exception(type(exc).__name__, str(exc))
            raise
        finally:
            _finish_call(perf_counter()-started)
            _call.reset(token)
    return wrapped


@_safe
def _model_result(recorder, result):
    call = _call.get()
    call['result'] = _copy(result)
    reason = result.get('reason', '') if isinstance(result, dict) else result[1]
    fallback = str(reason).startswith('fallback:')
    issue = 'baseline_fallback' if fallback else None
    if 'structured_reply' in call and not call['structured_reply']:
        issue = issue or 'empty_or_unparseable_reply'
    structured = call.get('structured_reply')
    if isinstance(result, dict) and result.get('validation_errors'):
        issue = issue or 'candidate_protocol_validation'
    if isinstance(structured, dict) and structured.get('dest_present'):
        box = structured.get('bbox_2d')
        try:
            x1,y1,x2,y2 = [float(v) for v in box]
            valid = x2 > x1 and y2 > y1
        except (TypeError, ValueError):
            valid = False
        if not valid:
            issue = issue or 'invalid_model_bbox'
    call.update(fallback=fallback, issue=issue)
    recorder.counts['model_fallbacks'] += int(fallback)


@_safe
def _model_exception(recorder, name, message):
    _call.get()['exception'] = dict(type=name, message=message)


@_safe
def _finish_call(recorder, elapsed):
    call = _call.get()
    call['elapsed_s'] = elapsed
    recorder.write('calls/'+call['call_id']+'.json', call)
    recorder.emit('model_result', dict(call_id=call['call_id'], kind=call['kind'], elapsed_s=elapsed,
                                      issue=call.get('issue'), exception=call.get('exception')))


def external_model(function, kind):
    """Observe unchanged shared Land code without modifying util or its globals."""
    namespace = dict(function.__globals__)
    namespace['requests'] = RequestsProxy(namespace['requests'])
    clone = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    observed = model_call(clone, kind=kind)

    @wraps(function)
    def wrapped(*args, **kwargs):
        return (observed if _run.get() is not None else function)(*args, **kwargs)
    return wrapped


def sam3_call(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        if _run.get() is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        parameters = {k:v for k,v in bound.arguments.items() if k != 'patch_main_bgr'}
        _run.get().counts['sam3_calls'] += 1
        event('sam3_request', parameters=parameters)
        started = perf_counter()
        try:
            result = function(*args, **kwargs)
            event('sam3_result', metadata=result[1], elapsed_s=perf_counter()-started)
            return result
        except BaseException as exc:
            _run.get().counts['sam3_errors'] += 1
            event('sam3_error', error_type=type(exc).__name__, error=str(exc), elapsed_s=perf_counter()-started)
            raise
    return wrapped


def vision_client(client_class):
    """Observe the actual SAM3 payload while retaining its original client."""
    class ObservedClient(client_class):
        def candidates(self, payload):
            event('sam3_service_request', url=self.base_url+'/v1/candidates',
                  timeout=self.timeout, payload=payload)
            return super().candidates(payload)
    return ObservedClient


@_safe
def _start_run(recorder, parameters):
    from ..config import PROJECT_ROOT
    package = Path(__file__).resolve().parents[1]
    sources = list(package.rglob('*.py')) + [PROJECT_ROOT/'PSC-AVDN/script/util.py', PROJECT_ROOT/'src/env.py']
    hashes = {}
    for source in sources:
        relative = source.relative_to(PROJECT_ROOT).as_posix()
        data = source.read_bytes()
        hashes[relative] = _digest(data)
        target = recorder.root/'snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(recorder.source_bytes(data))
    inputs = {}
    for key, source in [('parse_csv', Path(parameters['results_csv'])),
                        ('annotations', Path(parameters['anno_dir'])/(parameters['split']+'_data.json'))]:
        inputs[key] = dict(path=str(source.resolve()), exists=source.exists())
        if source.is_file():
            inputs[key]['sha256'] = _digest(source.read_bytes())
            if key == 'parse_csv':
                target = recorder.root/'snapshot'/'parsing_results.csv'
                data = source.read_bytes()
                source_text = data.decode('utf-8-sig')
                clean_text = recorder.clean(source_text)
                target.write_bytes(data if clean_text == source_text else clean_text.encode('utf-8'))
    try:
        git_ref = subprocess.check_output(['git','rev-parse','HEAD'],cwd=PROJECT_ROOT,stderr=subprocess.DEVNULL,timeout=5).decode().strip()
    except Exception:
        git_ref = None
    recorder.write('manifest.json', dict(schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
        parameters=parameters, source_sha256=hashes, input_files=inputs, git_ref=git_ref,
        runtime=dict(python=sys.version, executable=sys.executable,
                     packages={name:getattr(sys.modules.get(name),'__version__',None)
                               for name in ('numpy','cv2','torch','requests','pandas','shapely')}),
        policy=('STEP 2 C1: current-round candidates, independent confirmation previews, no text memory'
                if parameters.get('candidate_count') else 'STEP 0: no prompt, retry, movement, or fallback changes'),
        image_archive='Exact decoded data-URL bytes; reconstruct with recorded prefix and base64.',
        sam3_checkpoint='Not queried by trace; supplied by the separately managed SAM3 service.'))
    event('run_start', parameters=parameters)


@_safe
def _finish_run(recorder, status):
    recorder.write('summary.json', dict(status=status, counts=recorder.counts,
                                        elapsed_s=perf_counter()-recorder.started, recording_errors=recorder.errors))


def navigation_run(function):
    signature = inspect.signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        enabled = kwargs.pop('trace', True)
        if not enabled or _run.get() is not None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs); bound.apply_defaults()
        try:
            recorder = Recorder(bound.arguments['out_dir'], bound.arguments)
        except Exception as exc:
            print('[WARN][TRACE] disabled: '+type(exc).__name__)
            return function(*args, **kwargs)
        token = _run.set(recorder)
        status = 'failed'
        try:
            _start_run(bound.arguments)
            result = function(*args, **kwargs)
            status = 'completed'
            return result
        finally:
            _finish_run(status)
            _run.reset(token)
    return wrapped
