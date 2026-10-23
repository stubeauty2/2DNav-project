"""OneJev native System One review client; probabilities are not domain-calibrated."""
import base64
import hashlib
import json
import math
import time
from pathlib import Path

import requests

from .model import Halt, model_context, safe_view
from .runtime import digest, load_json, save_json


DEFAULT_MIN_PROBABILITY = 0.7
REVIEW_QUESTION = """Verify only the CURRENT proposed target in the two overhead RGB images.
Image 1 is the parent main view; image 2 is the local review view. The yellow
UNCONFIRMED outline and the candidate coordinates identify the proposal, not proof.
Use the full instruction, current target role, visible shape/layout, spatial relations,
and plausible alternatives. For an auxiliary target check its own conditions; do not
apply final-target-only attributes to it. For a pair/group inspect the entire group;
for an ordinal target verify the counting origin, direction and complete visible row.
White building may mean white walls, not a white roof. Doors, windows, wall color,
stairs and people may be unobservable from above: do not invent them or reject solely
because they are invisible. A strong unique visible layout/shape match can support
confirmation despite such unobservable attributes. Reject only on a genuine visible
contradiction such as wrong relation, shape or explicitly specified roof color.
If decisive evidence is missing or multiple alternatives remain, choose uncertain.
Category/detector score or public reference rectangle overlap alone cannot confirm
identity. Classify the existing proposal; do not find another object or refine its box."""
REVIEW_CRITERIA = {
    'confirm': 'The current proposal is uniquely supported by the visible identifying evidence, without a visible contradiction.',
    'reject': 'A genuine visible contradiction shows that the current proposal is not the requested object.',
    'uncertain': 'The visible evidence is insufficient, ambiguous, incomplete for the required group/ordinal relation, or contradictory evidence cannot be resolved.',
}
PRIVATE_KEYS = {'goal_xy_m', 'goal_polygon_xy_m', 'evaluation_polygon_xy_m',
                'gt_path_corners', 'evaluator_only', 'ground_truth', 'source_row',
                'target_coordinates', 'success', 'iou', 'prediction_distance_m'}


def _public_only(value):
    if isinstance(value, dict):
        if PRIVATE_KEYS.intersection(value) or any(str(k).startswith('gt_') for k in value):
            raise ValueError('Evaluator information is not allowed in Jev input')
        for child in value.values():
            _public_only(child)
    elif isinstance(value, list):
        for child in value:
            _public_only(child)


def build_review_payload(context, views, model):
    """Return (wire payload, journal payload without base64, exact image assets)."""
    _public_only(context)
    if context.get('mode') != 'review' or not isinstance(context.get('candidate'), dict):
        raise ValueError('Jev requires a public review context with a candidate')
    if len(views) != 2 or len({v['view_id'] for v in views}) != 2:
        raise ValueError('Review requires the same parent and local images as MLLM review')
    safe = model_context(dict(context, views=[safe_view(v) for v in views]))
    safe['memory'] = {'confirmed_auxiliary': safe.get('memory', {}).get('confirmed_auxiliary', [])}
    assets, media, references = [], [], []
    for i, view in enumerate(views, 1):
        path = Path(view['image_path'])
        content = path.read_bytes()
        asset = dict(view_id=view['view_id'], image_path=str(path.resolve()),
                     sha256=hashlib.sha256(content).hexdigest())
        assets.append(asset)
        media.append(dict(type='image', data='data:image/png;base64,'+base64.b64encode(content).decode('ascii')))
        references.append(dict(image='<image:'+str(i)+'>', view_id=view['view_id'],
                               purpose=view.get('source'), annotation=view.get('annotation')))
    payload = dict(model=model, state=dict(context=safe, images=references),
                   questions={'review': dict(type='choice', instructions=REVIEW_QUESTION,
                                             criteria=dict(REVIEW_CRITERIA))}, media=media)
    journal = dict(payload, media=[dict(type='image', **a) for a in assets])
    return payload, journal, assets


def validate_answer(raw, expected_model, min_probability=DEFAULT_MIN_PROBABILITY):
    if not isinstance(raw, dict) or raw.get('model') != expected_model:
        raise ValueError('Jev response model identity differs from the expected service')
    answers = raw.get('answers')
    if not isinstance(answers, dict) or set(answers) != {'review'}:
        raise ValueError('Expected exactly one review answer')
    answer = answers['review']
    if not isinstance(answer, dict) or answer.get('type') != 'choice':
        raise ValueError('Expected a typed choice answer')
    probabilities = answer.get('probabilities')
    if not isinstance(probabilities, dict) or set(probabilities) != set(REVIEW_CRITERIA):
        raise ValueError('Unexpected review options')
    values = list(probabilities.values())
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or
           not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError('Invalid review probabilities')
    # Native server rounds each probability to six decimals. Do not renormalize.
    if not math.isclose(sum(values), 1., abs_tol=3e-6):
        raise ValueError('Review probabilities are not normalized')
    choice = answer.get('choice')
    peak = max(values)
    if choice not in probabilities or probabilities[choice] < peak-1e-6:
        raise ValueError('Choice does not match the maximal probability')
    confidence = answer.get('confidence')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence):
        raise ValueError('Invalid native choice confidence')
    expected_confidence = (peak/sum(values)-1/3)/(2/3)
    if not 0 <= confidence <= 1 or not math.isclose(confidence, expected_confidence, abs_tol=4e-6):
        raise ValueError('Native choice confidence does not match probabilities')
    return dict(choice=choice, verdict=choice if peak >= min_probability else 'uncertain',
                max_probability=peak, confidence=confidence, probabilities=probabilities,
                abstained=peak < min_probability)


class Jev:
    """No MLLM fallback; service failures return uncertainty with explicit status."""
    def __init__(self, url, expected_model, min_probability=DEFAULT_MIN_PROBABILITY,
                 service_manifest=None, timeout=180):
        if not expected_model or expected_model in ('jev-latest', 'jev-preview', 'qev-latest'):
            raise ValueError('Specify an exact Jev service model identity, not an alias')
        if isinstance(min_probability, bool) or not isinstance(min_probability, (int, float)) or not math.isfinite(min_probability) or not 0 <= min_probability <= 1:
            raise ValueError('Jev minimum probability must be finite and within [0,1]')
        self.url = url.rstrip('/') if url else None
        self.expected_model, self.min_probability = expected_model, min_probability
        self.timeout = timeout
        self.manifest = None
        if service_manifest is not None:
            if isinstance(service_manifest, dict):
                self.manifest = dict(content=service_manifest, sha256=digest(service_manifest))
            else:
                path = Path(service_manifest)
                self.manifest = dict(path=str(path.resolve()), content=load_json(path),
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if not isinstance(self.manifest['content'], dict):
                raise ValueError('Frozen service manifest must be a JSON object')
            served_model = self.manifest['content'].get('served_model')
            if served_model != expected_model:
                raise ValueError('Frozen service manifest model differs from expected Jev model')
        self.cursor = 0

    def verify(self, context, views, output):
        wire, journal, assets = build_review_payload(context, views, self.expected_model)
        output = Path(output)
        signature = digest(dict(protocol='onejev-native-review-v1', input=journal, endpoint=self.url,
                                min_probability=self.min_probability, service_manifest=self.manifest))
        self.cursor += 1
        if output.exists():
            saved = load_json(output)
            if saved.get('signature') != signature:
                raise Halt('jev_resume_input_mismatch')
            if saved.get('status') == 'pending':
                raise Halt('jev_uncertain_request_pending')
            if saved.get('status') == 'identity_error':
                raise Halt(saved.get('error', 'jev_service_identity_changed'))
            return saved
        record = dict(signature=signature, status='pending', backend='jev', model=self.expected_model,
                      endpoint=self.url, input=journal, assets=assets, service_manifest=self.manifest,
                      min_probability=self.min_probability, verdict='uncertain', confidence=None,
                      probabilities=None, max_probability=None, latency_ms=None,
                      calibration_note='No AVDN/CityNav calibration; threshold was declared before evaluation')
        save_json(output, record)
        start = time.perf_counter()
        try:
            if not self.url:
                raise ValueError('Jev service URL is not configured')
            health_response = requests.get(self.url+'/health', timeout=5)
            health_response.raise_for_status()
            health = health_response.json()
            if not isinstance(health, dict):
                raise ValueError('Invalid Jev health object')
            json.dumps(health, allow_nan=False)
            record['health'] = health
            if health.get('status') != 'ok':
                raise ValueError('Jev service is not healthy')
            if health.get('model') != self.expected_model:
                raise Halt('jev_model_identity_mismatch')
            if self.manifest is not None:
                content = self.manifest['content']
                if health.get('service_manifest_sha256') != digest(content):
                    raise Halt('jev_service_manifest_mismatch')
                if health.get('checkpoint_revision') != content.get('checkpoint_revision'):
                    raise Halt('jev_checkpoint_revision_mismatch')
            identity = {k: health.get(k) for k in ('model', 'fork_mode', 'prompt_version',
                                                  'checkpoint_revision', 'service_manifest_sha256')}
            identity_path = output.parent/'jev_service_identity.json'
            if identity_path.exists() and load_json(identity_path) != identity:
                raise Halt('jev_service_identity_changed')
            if not identity_path.exists():
                save_json(identity_path, identity)
            response = requests.post(self.url+'/v1/systemone', json=wire, timeout=self.timeout,
                                     headers={'x-typesafe-request-id': output.stem})
            record['http_status'] = response.status_code
            record['raw_text'] = response.text
            response.raise_for_status()
            raw = response.json()
            if isinstance(raw, dict) and raw.get('model') is not None and raw['model'] != self.expected_model:
                raise Halt('jev_response_model_identity_mismatch')
            parsed = validate_answer(raw, self.expected_model, self.min_probability)
            json.dumps(raw, allow_nan=False)
            record.update(parsed, status='ok', raw=raw)
        except Halt as exc:
            record.update(status='identity_error', error=str(exc),
                          note='Configuration identity mismatch; no MLLM fallback')
            raise
        except (requests.RequestException, ValueError, KeyError, TypeError, OSError) as exc:
            record.update(status='error', error=type(exc).__name__, error_detail=str(exc),
                          note='Service or protocol failure is not evidence against the candidate; no MLLM fallback')
        finally:
            record['latency_ms'] = round((time.perf_counter()-start)*1000, 3)
            save_json(output, record)
        return record


def review_action(record, candidate):
    """Only code restores the existing box; OneJev never produces coordinates."""
    verdict = record['verdict'] if record.get('status') == 'ok' else 'uncertain'
    reason = ('OneJev typed review: '+verdict+'; status='+record['status']+
              '; max_probability='+str(record.get('max_probability'))+
              '; minimum_probability='+str(record['min_probability'])+
              '; no box refinement; see Jev request journal for evidence input and probabilities')
    action = dict(action='review', verdict=verdict, reason=reason)
    if verdict == 'confirm':
        action.update(view_id=candidate['view_id'], bbox=list(candidate['bbox']))
    return action
