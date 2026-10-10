"""Bounded multi-image requests with source-bound, strictly validated results."""
import json
import math
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from util import data_url_from_image_array, _parse_json_loose
import requests
from util import _extract_content
from .telemetry import append_event
from ..config import SEARCH_QWEN_TIMEOUT

class ModelCallError(RuntimeError):
    def __init__(self, reason, raw=None):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw

class ModelClient:
    def __init__(self, *, api_key, url, model, transport=None):
        self.api_key = api_key
        self.url = url
        self.model = model
        self.transport = transport if transport is not None else requests

    def complete(self, messages, *, timeout=120, temperature=0):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {"model": self.model, "temperature": temperature,
                   "stream": False, "messages": messages}
        try:
            response = self.transport.post(self.url, json=payload, headers=headers, timeout=timeout)
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "application/json" not in content_type:
                raise ModelCallError("fallback: non-json response", response.text[:200])
            result = response.json()
        except ModelCallError:
            raise
        except Exception as exc:
            raise ModelCallError(f"fallback: request/json error: {exc}") from exc
        if isinstance(result, dict) and "error" in result:
            raise ModelCallError(f"fallback: api error {result.get('error')}", result)
        return (_extract_content(result) or "").strip()

PROMPTS = Path(__file__).resolve().parents[1] / 'prompts'


def rgb_view(obs, description=None):
    return {"view_id": obs.observation_id + ':rgb', "image": obs.image,
            "description": description or obs.description, "selectable": True, "image_path": obs.image_path}


def messages(mode, destination, views, via=None, memory=None, extra=None):
    context = {"destination": destination, "via": via or None,
               "selectable_view_ids": [v['view_id'] for v in views if v['selectable']]}
    if memory is not None:
        context['memory'] = memory
    if extra:
        context.update(extra)
    content = [{"type": "text", "text": json.dumps(context, ensure_ascii=False)}]
    for view in views:
        h, w = view['image'].shape[:2]
        caption = f"view_id: {view['view_id']}\nselectable: {str(view['selectable']).lower()}\nimage_size: {w}x{h}\n{view['description']}"
        content.extend([{"type": "text", "text": caption},
                        {"type": "image_url", "image_url": {"url": data_url_from_image_array(view['image'])}}])
    if mode not in ('candidate_proposal', 'candidate_confirmation'):
        raise ValueError('unsupported request mode')
    search, confirmation = (PROMPTS / 'candidates.md').read_text(encoding='utf-8').split('<!-- CONFIRMATION -->', 1)
    prompt = search.replace('<!-- SEARCH -->', '').strip() if mode == 'candidate_proposal' else confirmation.strip()
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": content}]


def validate(raw, mode, views, extra=None):
    result = dict(raw) if isinstance(raw, dict) else _parse_json_loose(raw)
    if not isinstance(result, dict):
        raise ValueError('response must contain a JSON object')
    if mode not in ('candidate_proposal', 'candidate_confirmation'):
        raise ValueError('unsupported response mode')
    return validate_candidates(result, mode, views, extra or {})


def _validate_localization(result, views, *, has_box):
    """Check source-bound coordinates and evidence without a second mode protocol."""
    supplied = {v['view_id']: v for v in views}
    allowed = {v['view_id']: v for v in views if v['selectable']}
    # Models may cite observation IDs instead
    # of its RGB view ID. Only accept an unambiguous alias supplied in this call.
    aliases = {name[:-4]: name for name in supplied if name.endswith(':rgb')
               and name[:-4] not in supplied}
    def canonical(name):
        return aliases.get(name, name)
    evidence = result.get('evidence_view_ids', [])
    if not isinstance(evidence, list) or any(not isinstance(i, str) or canonical(i) not in supplied for i in evidence):
        raise ValueError('unknown evidence ID')
    result['evidence_view_ids'] = [canonical(i) for i in evidence]
    for key in ('support', 'conflicts'):
        value = result.get(key, [])
        if not isinstance(value, list) or len(value) > 8 or any(not isinstance(s, str) or len(s) > 500 for s in value):
            raise ValueError('invalid evidence text')
    if has_box:
        target = canonical(result.get('target_view_id'))
        if not isinstance(target, str) or target not in allowed:
            raise ValueError('invalid source view ID')
        result['target_view_id'] = target
        b = result.get('bbox_2d')
        if not isinstance(b, list) or len(b) != 4 or any(type(x) not in (int, float) or not math.isfinite(x) for x in b):
            raise ValueError('invalid bbox numbers')
        h, w = allowed[target]['image'].shape[:2]
        if not (0 <= b[0] < b[2] <= w-1 and 0 <= b[1] < b[3] <= h-1):
            raise ValueError('bbox outside image or empty')
    elif result.get('target_view_id') is not None or result.get('bbox_2d') is not None:
        raise ValueError('negative decision must not have target or bbox')
    return result


def infer(client, mode, destination, views, *, via=None, memory=None,
          extra=None, retries=1, trace_path=None, timeout=SEARCH_QWEN_TIMEOUT, audit=None):
    request = messages(mode, destination, views, via, memory, extra)
    request_record = {"mode": mode, "system_prompt": request[0]['content'],
                      "text_input": [part['text'] for part in request[1]['content'] if part['type'] == 'text'],
                      "views": [{k:v.get(k) for k in ('view_id','description','selectable','image_path')} for v in views]}
    if audit is not None:
        audit.update(request_record)
        audit['attempts'] = []
    append_event(trace_path, {"event": "model_request", **request_record})
    for attempt in range(retries + 1):
        started = perf_counter()
        raw = None
        try:
            raw = client.complete(request, timeout=timeout, temperature=0)
            result = validate(raw, mode, views, extra)
            status = 'returned'
        except ModelCallError as exc:
            result = {"decision": "tool_error", "error": type(exc).__name__}
            status = 'tool_error'
        except (ValueError, TypeError, KeyError) as exc:
            result = {"decision": "parse_error", "error": str(exc),
                      "raw_prefix": str(raw or '')[:500]}
            status = 'parse_error'
        if audit is not None:
            audit['attempts'].append({"attempt": attempt, "status": status,
                                      "response_text": raw, "result": deepcopy(result)})
        append_event(trace_path, {"event": "multiview_call", "mode": mode, "attempt": attempt,
                                  "status": status, "result": result,
                                  "input_view_ids": [v['view_id'] for v in views],
                                  "image_count": len(views), "elapsed_s": round(perf_counter()-started, 3)})
        if status == 'returned':
            break
    return result


def validate_candidates(result, mode, views, extra):
    for key in ('support', 'conflicts'):
        value = result.get(key, [])
        if not isinstance(value, list) or len(value)>8 or any(not isinstance(x,str) or len(x)>500 for x in value):
            raise ValueError('invalid evidence text')
    if mode == 'candidate_proposal':
        slots = result.get('candidates')
        if not isinstance(slots, list) or len(slots) != extra.get('candidate_count', 3):
            raise ValueError('candidate slots must match candidate_count; use null for empty slots')
        normalized = []
        invalid = []
        for slot, candidate in enumerate(slots):
            if candidate is None:
                normalized.append(None)
                continue
            try:
                if not isinstance(candidate, dict):
                    raise ValueError('invalid candidate')
                checked = _validate_localization({**candidate,
                                    'target_view_id':candidate.get('source_view_id')}, views, has_box=True)
            except (ValueError, TypeError, KeyError) as exc:
                invalid.append({'slot': slot, 'error': str(exc)})
                normalized.append(None)
                continue
            normalized.append({k:checked.get(k,[]) for k in ('bbox_2d','support','conflicts','evidence_view_ids')})
            normalized[-1]['source_view_id'] = checked['target_view_id']
        if invalid and not any(c is not None for c in normalized):
            raise ValueError('all proposed candidates invalid: ' + str(invalid))
        return {'decision':'proposed', 'candidates':normalized, 'invalid_candidates': invalid,
                'grid_5x5_by_view': result.get('grid_5x5_by_view', {})}
    decision = result.get('decision')
    if decision not in ('confirmed', 'continue_search', 'no_match'):
        raise ValueError('invalid candidate confirmation decision')
    mapping = extra.get('candidate_views', {})
    cid = result.get('candidate_id')
    if decision == 'confirmed':
        if not isinstance(cid, str) or cid not in mapping:
            raise ValueError('unknown candidate ID')
        checked = _validate_localization(dict(result), views, has_box=True)
        if checked['target_view_id'] != mapping[cid]:
            raise ValueError('confirmation bbox must belong to the selected candidate view')
        result.update(target_view_id=checked['target_view_id'], evidence_view_ids=checked['evidence_view_ids'])
    else:
        if cid is not None:
            raise ValueError('unconfirmed result must have null candidate_id')
        _validate_localization(dict(result), views, has_box=False)
    if decision == 'continue_search' and not extra.get('can_continue_search', False):
        raise ValueError('search budget exhausted')
    assessments = result.get('candidate_assessments')
    if not isinstance(assessments, list):
        raise ValueError('candidate_assessments must be a list')
    seen = set()
    for item in assessments:
        if not isinstance(item, dict):
            raise ValueError('invalid candidate assessment')
        candidate_id = item.get('candidate_id')
        if not isinstance(candidate_id, str) or candidate_id not in mapping or candidate_id in seen:
            raise ValueError('unknown or duplicate assessed candidate')
        seen.add(candidate_id)
        status = item.get('status')
        if status not in ('uncertain', 'rejected', 'confirmed'):
            raise ValueError('invalid assessment status')
        if (status == 'confirmed') != (decision == 'confirmed' and candidate_id == cid):
            raise ValueError('assessment conflicts with final decision')
        if not isinstance(item.get('reason'), str) or not item['reason'].strip():
            raise ValueError('assessment requires a reason')
        unresolved = item.get('unresolved', [])
        if not isinstance(unresolved, list) or any(not isinstance(x, str) for x in unresolved):
            raise ValueError('invalid unresolved evidence')
    if seen != set(mapping):
        raise ValueError('assess every supplied candidate')
    return result
