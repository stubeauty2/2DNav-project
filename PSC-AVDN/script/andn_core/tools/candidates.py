"""STEP 2 C1 protocol: current-round candidates and view-bound localization.

Search evidence is archived by navigation, never included in Confirmation input.
This module does not move the agent or retain cross-round memory.
"""
import math

from ..config import FIXED_CROP_SIDE
from util import data_url_from_image_array


def valid_box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        if any(isinstance(v, bool) for v in value):
            return None
        box = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in box):
        return None
    box = [max(0.0, min(v, FIXED_CROP_SIDE - 1.0)) for v in box]
    return box if box[2] > box[0] and box[3] > box[1] else None


def _confidence(value):
    try:
        number = float(value)
        return max(0.0, min(number, 1.0)) if math.isfinite(number) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def confirmation_crops(candidate, ob, anchor, heading, scale_factor, out_dir, prefix):
    """Preview each center without assigning agent position or drawing a Search box."""
    import os
    import cv2
    from .views import generate_view_corners_with_scale, create_view_image, draw_pos_on_patch
    views = []
    labels = {'main': '以候选中心裁剪的确认主图，可定位。',
              'detail': '同中心、同朝向的细节视图。',
              'context': '同中心、同朝向的广域视图。'}
    for role, scale in (('main', scale_factor), ('detail', 3), ('context', 7)):
        view_id = candidate['candidate_id'] + '_confirm_' + role
        corners = generate_view_corners_with_scale(candidate['map_center'], ob, scale_factor=scale, angle_deg=heading)
        path = os.path.join(out_dir, prefix + '_' + view_id + '.jpg')
        patch = create_view_image(corners, ob, save_path=path, out_px=FIXED_CROP_SIDE)
        if patch is None:
            if role == 'main':
                return []
            continue
        if role == 'main':
            draw_pos_on_patch(patch, corners, anchor)
            cv2.imwrite(path, patch, [cv2.IMWRITE_JPEG_QUALITY, 95])
        views.append(dict(candidate_id=candidate['candidate_id'], view_id=view_id,
                          role=role, image=patch, corners=corners, path=path,
                          description='候选 %s · %s：%s' % (candidate['candidate_id'], view_id, labels[role])))
    return views


def square_uv(bbox, side=FIXED_CROP_SIDE):
    """Original minimal-square conversion, including its image-edge adjustment."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    half = max(max(1.0, x2 - x1), max(1.0, y2 - y1)) * 0.5
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    left, right, top, bot = cx - half, cx + half, cy - half, cy + half
    if left < 0: cx += -left
    if right > side - 1: cx -= right - (side - 1)
    if top < 0: cy += -top
    if bot > side - 1: cy -= bot - (side - 1)
    half = min(half, cx, side - 1 - cx, cy, side - 1 - cy)
    return [(cx-half, cy-half), (cx+half, cy-half), (cx+half, cy+half), (cx-half, cy+half)]


def normalize_review(obj, selected_id, main_view_id):
    """Invalid replies are operational failures, never semantic rejections."""
    if not isinstance(obj, dict) or obj.get('action') not in ('keep','revise','reject'):
        raise ValueError('Invalid review action')
    if obj.get('selected_candidate_id') != selected_id or obj.get('view_id') != main_view_id:
        raise ValueError('Review cannot change candidate or localization view')
    if not isinstance(obj.get('reason'), str) or not obj['reason'].strip():
        raise ValueError('Review requires an evidence explanation')
    confidence = obj.get('confidence')
    if isinstance(confidence,bool) or not isinstance(confidence,(float,int)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('Invalid review confidence')
    box = obj.get('bbox_2d')
    if obj['action']=='revise':
        checked = valid_box(box)
        if checked is None or checked != list(box):
            raise ValueError('Invalid review box')
        box = [max(1.0,min(v,FIXED_CROP_SIDE-2.0)) for v in checked]
        if box[2]<=box[0] or box[3]<=box[1]:raise ValueError('Degenerate review box')
    elif box is not None:
        raise ValueError('keep/reject requires null bbox')
    return dict(action=obj['action'], selected_candidate_id=selected_id, view_id=main_view_id,
                bbox_2d=box, confidence=float(confidence), reason=obj['reason'])


def verify_confirmation(dest_desc, candidate, initial, views, ob, out_dir, prefix,
                        api_key, url, model, vision_tool_url, mode, budget_available):
    """At most one evidence review; exceptions preserve the initial decision."""
    import os
    import cv2
    import numpy as np
    from copy import deepcopy
    from .geometry import _patch_xy_to_latlng, _latlng_to_patch_xy, confirmation_signals
    from .sam3 import _sam3_image5
    from .category_terms import extract_category_terms
    from .model import review_candidate_with_evidence
    from . import telemetry as audit
    from .views import _draw_bbox_on_patch
    record = dict(initial_confirmation=deepcopy(initial), attempted=False, applied='initial')
    try:
        selected_views = [v for v in views if v['candidate_id']==candidate['candidate_id']]
        main = next(v for v in selected_views if v['role']=='main')
        def polygon(box,corners):
            x1,y1,x2,y2=box
            return [_patch_xy_to_latlng(u,v,corners,ob) for u,v in [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]]
        search_poly=polygon(candidate['bbox_2d'],candidate['source_corners'])
        confirm_poly=polygon(initial['bbox_2d'],main['corners'])
        signals=confirmation_signals(search_poly,confirm_poly,candidate['bbox_2d'],initial['bbox_2d'],initial.get('confidence',0))
        record['signals']=signals
        record['search_polygon']=search_poly;record['initial_polygon']=confirm_poly
        audit.event('confirmation_review_signals', candidate_id=candidate['candidate_id'], signals=signals)
        if not signals['trigger_reasons'] or mode=='observe' or not budget_available:
            record['status']='not_triggered' if not signals['trigger_reasons'] else 'observe_only' if mode=='observe' else 'budget_exhausted'
            return initial,record
        # Consume the single stage budget even if the tool or model later fails.
        record['attempted']=True
        artifact=os.path.join(out_dir,prefix+'_verification');os.makedirs(artifact,exist_ok=True)
        comparison=main['image'].copy()
        projected=_latlng_to_patch_xy(search_poly,main['corners'],ob)
        cv2.polylines(comparison,[np.rint(projected).astype(np.int32)],True,(255,0,255),3)
        _draw_bbox_on_patch(comparison,initial['bbox_2d'],color=(0,165,255),label='Initial Confirmation')
        cv2.putText(comparison,'Search proposal: purple',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,255),2)
        comparison_path=os.path.join(artifact,'proposal_comparison.png');cv2.imwrite(comparison_path,comparison)
        overlay,meta=_sam3_image5(main['image'],extract_category_terms(dest_desc),
            image_path=main['path'],artifact_dir=os.path.join(artifact,'sam3'),vision_tool_url=vision_tool_url)
        record['evidence']=dict(comparison_path=comparison_path,sam3=meta,view_id=main['view_id'])
        evidence_views=list(selected_views)+[
            dict(main,role='comparison',view_id=main['view_id']+'_comparison',image=comparison,
                 description='Proposal comparison: purple = projected Search proposal; orange = initial Confirmation. Auxiliary evidence only; do not localize on this image.'),
            dict(main,role='sam3',view_id=main['view_id']+'_sam3',image=overlay,
                 description='SAM3 evidence on the confirmation main view. Tool marks are fallible auxiliary evidence; do not localize on this image.')]
        review=review_candidate_with_evidence(dest_desc,initial,signals,evidence_views,api_key=api_key,url=url,model=model)
        record['review']=review;record['status']='completed';record['applied']=review['action']
        result=deepcopy(initial)
        if review['action']=='revise':
            result.update(bbox_2d=review['bbox_2d'],confidence=review['confidence'],reason=review['reason'])
        elif review['action']=='reject':
            result.update(dest_present=False,selected_candidate_id=None,view_id=None,bbox_2d=None,
                          confidence=review['confidence'],reason=review['reason'])
        audit.event('confirmation_review_decision', candidate_id=candidate['candidate_id'], review=review)
        return result,record
    except Exception as exc:
        record.update(status='operational_error_keep_initial',error_type=type(exc).__name__,error=str(exc))
        audit.event('confirmation_review_error', error_type=type(exc).__name__, error=str(exc), fallback='keep_initial')
        return initial,record


def normalize_reply(parsed, raw, confirmation_views=None, allowed_source_ids=None):
    obj = parsed if isinstance(parsed, dict) else {}
    result = dict(grid_5x5=obj.get('grid_5x5') or [], raw=raw,
                  reason=str(obj.get('reason') or ''), validation_errors=[])
    errors = result['validation_errors']
    if confirmation_views is None:
        candidates = obj.get('candidates')
        result['candidates'] = []
        if not isinstance(candidates, list):
            errors.append('candidates_must_be_list')
            return result
        seen = set()
        if len(candidates) > 2:
            errors.append('candidate_budget_exceeded')
        for index, proposal in enumerate(candidates[:2]):
            if not isinstance(proposal, dict):
                errors.append('candidate_%d_not_object' % index)
                continue
            box = valid_box(proposal.get('bbox_2d'))
            source_id=proposal.get('source_view_id')
            if source_id not in (allowed_source_ids or ['search_main']) or box is None:
                errors.append('candidate_%d_invalid_source_or_bbox' % index)
                continue
            key=(source_id,tuple(box))
            if key in seen:
                errors.append('candidate_%d_duplicate' % index)
                continue
            seen.add(key)
            result['candidates'].append(dict(
                source_view_id=source_id, bbox_2d=box,
                confidence=_confidence(proposal.get('confidence')),
                supporting_evidence=proposal.get('supporting_evidence', ''),
                unresolved_attributes=proposal.get('unresolved_attributes', [])))
        return result

    result.update(dest_present=False, selected_candidate_id=None, view_id=None,
                  bbox_2d=None, confidence=_confidence(obj.get('confidence')))
    if obj.get('dest_present') is False:
        if obj.get('selected_candidate_id') is not None or obj.get('view_id') is not None:
            errors.append('absent_requires_null_ids')
        return result
    if obj.get('dest_present') is not True:
        errors.append('dest_present_must_be_boolean')
        return result
    pair = (obj.get('selected_candidate_id'), obj.get('view_id'))
    allowed = [(v['candidate_id'], v['view_id']) for v in confirmation_views if v['role'] == 'main']
    box = valid_box(obj.get('bbox_2d'))
    if pair not in allowed or box is None:
        errors.append('invalid_candidate_view_pair_or_bbox')
        return result
    # Match the original Confirmation edge clamp, then recheck nonzero extent.
    box = [max(1.0, min(v, FIXED_CROP_SIDE - 2.0)) for v in box]
    if box[2] <= box[0] or box[3] <= box[1]:
        errors.append('degenerate_confirmation_bbox_after_clamp')
        return result
    result.update(dest_present=True, selected_candidate_id=pair[0], view_id=pair[1], bbox_2d=box)
    return result


def build_request(focus_text, baseline_system, dest_desc, search_images,
                  prev_grid_5x5=None, confirmation_views=None):
    """Reuse baseline analysis instructions; replace only roles and output protocol."""
    import json
    confirmation = confirmation_views is not None
    if confirmation:
        # The baseline analysis remains, applied independently within each group.
        focus = focus_text[focus_text.index('Step 0 - Mode Selection'):]
        focus = focus.replace('and consult Image 4 (the minimap) for global orientation / trajectory. ', '')
        focus = focus.replace('and use Image 4 only for high-level position / heading / trajectory context. ', '')
        focus = focus.replace('Image 2, Image 3, or Image 4', 'detail or context views')
        focus = focus.replace('Image 1', 'the candidate main view').replace('IMAGE 1', 'THE CANDIDATE MAIN VIEW')
        focus = focus.replace('Image 2', 'the same candidate detail view').replace('Image 3', 'the same candidate context view')
        rules = (
            'Apply the following baseline localization procedure independently within each candidate group. '
            'Compare the candidate-centered views against the complete destination description. '
            'Select at most one existing candidate and independently localize it on that candidate\'s current confirmation main view. '
            'Only that selected main view supplies the output bbox and grid_5x5. '
            'Return selected_candidate_id, view_id, bbox_2d, confidence and reason; '
            'return dest_present=false with null IDs and null bbox_2d when none is sufficiently supported.\n'
        )
        schema = ('{"dest_present":true|false,"selected_candidate_id":"existing ID or null",'
                  '"view_id":"selected confirmation main view ID or null","bbox_2d":[x1,y1,x2,y2]|null,'
                  '"confidence":0.0,"reason":"up to 500 characters","grid_5x5":[]}')
        views = confirmation_views
        user = f'Destination description: {dest_desc or "(None)"}\nAll coordinates must use the selected candidate main view pixel grid.'
    else:
        focus = focus_text[focus_text.index('1) Destination parsing:'):]
        focus = focus.replace(
            'Choose EXACTLY ONE final region. If you cannot isolate one unambiguous region in Image 1, you must report dest_present=false.',
            'Propose up to 2 distinct plausible candidates on the current main search view. Do not invent candidates to fill the budget.')
        focus = focus.replace('the final chosen region', 'each proposed region').replace('the chosen region', 'each proposed region')
        focus = focus.replace('Return dest_present (true/false), bbox_2d, confidence in [0.0,1.0], and a brief reason (<500 chars).',
                              'Return each candidate\'s source_view_id, bbox_2d, confidence, supporting_evidence, and unresolved_attributes.')
        focus = focus.replace('If you are not sufficiently confident in ONE clear match in Image 1, output dest_present=false.',
                              'If no plausible candidate is supported, return an empty candidates list.')
        prior = baseline_system[baseline_system.index('If a previous-step grid_5x5 is provided:'):baseline_system.index('IMPORTANT:')]
        rules = prior + 'All candidate boxes use source_view_id="search_main". Candidate IDs are assigned by the program.\n'
        schema = ('{"candidates":[{"source_view_id":"search_main","bbox_2d":[x1,y1,x2,y2],'
                  '"confidence":0.0,"supporting_evidence":"visual evidence",'
                  '"unresolved_attributes":[]}],"reason":"up to 500 characters","grid_5x5":[]}')
        views = search_images
        user = f'Destination description: {dest_desc or "(None)"}\nAll candidate coordinates must use search_main (Image 1). Return up to 2 candidates; the list may be empty.'
        if any(v['role'] == 'sam3' for v in views):
            # Preserve baseline tool-evidence caveats, changing only singular output.
            from .model import IMAGE5_PROMPT
            rules += IMAGE5_PROMPT.replace(
                'Return exactly one final bbox in Image 1 coordinates, never a bbox for Image 5\nor a tool candidate ID. All original output rules and JSON fields still apply.',
                'SAM3 marks remain auxiliary evidence. Return candidate boxes only in search_main coordinates, never boxes on Image 5 or tool candidate IDs.')
    # Retain the baseline's explicit nested grid example (not a flat list).
    schema = schema.replace('"grid_5x5":[]', '"grid_5x5":[[c11,c12,c13,c14,c15],[c21,c22,c23,c24,c25],[c31,c32,c33,c34,c35],[c41,c42,c43,c44,c45],[c51,c52,c53,c54,c55]]')
    system = ('You are a visual localization assistant. You must return ONLY strict JSON, with no extra text.\n'
              'Each image has its own role and view_id in the input. Main views have top = current heading. '
              'Detail/context images have the same center and heading as their main view, with narrower/wider fields of view. '
              'Localize only on the permitted main view; all auxiliary images provide context only. '
              f'Coordinates use a {FIXED_CROP_SIDE}x{FIXED_CROP_SIDE} pixel grid.\n'
              + rules + '\n' + focus + '\nOutput JSON (grid_5x5 is a 5x5 array of strings):\n' + schema)
    content = [{'type': 'text', 'text': user}]
    for view in views:
        if view['image'] is None:
            continue
        content.append({'type': 'text', 'text': view['description']})
        content.append({'type': 'image_url', 'image_url': {'url': data_url_from_image_array(view['image'])}})
        if not confirmation and view['role'] == 'main' and prev_grid_5x5:
            content.append({'type': 'text', 'text': 'Previous step grid_5x5 (Image 1, soft prior; ignore if inconsistent): ' + json.dumps(prev_grid_5x5)})
    return system, content
