import json
import requests
from . import telemetry as _audit
from ..config import DEFAULT_API_KEY, QWEN_URL, QWEN_MODEL, FIXED_CROP_SIDE
from util import data_url_from_image_array, _norm_bool, _parse_json_loose, _extract_content

requests = _audit.RequestsProxy(requests)

IMAGE5_PROMPT = """
Image 5 - optional visual segmentation tool evidence:
Image 5 shows candidate bounding boxes produced by SAM3 from target-category
keywords on Image 1. It has exactly the same view and pixel coordinates as
Image 1. These marks are reference evidence only, not ground truth: the tool
may miss the actual target, produce false positives, or mark other objects of
the same category that do not satisfy the full destination description.
Independently inspect Image 1 using the complete destination description and
Images 2-4 for context. Do not rely exclusively on Image 5 or copy its boxes
without checking color, shape, structure, texture and spatial relations.
You may choose a matching target in Image 1 that is NOT marked in Image 5.
An image with no tool marks does NOT imply that the destination is absent.
Return exactly one final bbox in Image 1 coordinates, never a bbox for Image 5
or a tool candidate ID. All original output rules and JSON fields still apply.
"""

REVIEW_PROMPT = """Review the selected candidate using the COMPLETE destination description, the original confirmation views, the proposal-comparison image, and the new SAM3 evidence. The reported geometric changes are reasons to inspect, not proof that either box is wrong. Search and initial Confirmation boxes are fallible proposals, not ground truth. Decide whether added or removed regions belong to the described object or group, using visible attributes and boundaries. SAM3 may miss the target or mark unrelated objects; do not copy its boxes without checking the full description. Keep the selected candidate ID fixed. Return action=keep to preserve the initial confirmation, action=revise with a newly localized bbox on the original confirmation main view, or action=reject when the target is not sufficiently supported. All output coordinates must use that main view; auxiliary comparison and SAM3 images are evidence only. Explain the decisive new evidence briefly; do not change the box merely because it grew, shifted, or differs from SAM3."""


@_audit.model_call(kind='review')
def review_candidate_with_evidence(dest_desc, initial, signals, confirmation_views,
                                  api_key=DEFAULT_API_KEY, url=QWEN_URL, model=QWEN_MODEL):
    from .candidates import normalize_review
    system = REVIEW_PROMPT + ('\nReturn only JSON with action (keep/revise/reject), selected_candidate_id, '
        'view_id, bbox_2d, confidence (0..1), and reason. For keep/reject, bbox_2d must be null. '
        'For revise, bbox_2d is [x1,y1,x2,y2] in the original main view 768x768 pixel grid. '
        'Keep selected_candidate_id and view_id equal to the supplied initial confirmation IDs for every action.')
    content=[dict(type='text',text=json.dumps(dict(destination=dest_desc,
        initial_confirmation={k:initial[k] for k in ('selected_candidate_id','view_id','bbox_2d')},
        geometric_observations=signals),ensure_ascii=False))]
    for view in confirmation_views:
        content.extend([dict(type='text',text=view['description']),
            dict(type='image_url',image_url=dict(url=data_url_from_image_array(view['image'])))])
    response=requests.post(url,headers={'Authorization':'Bearer '+api_key,'Content-Type':'application/json'},
        json=dict(model=model,temperature=0,stream=False,messages=[dict(role='system',content=system),
             dict(role='user',content=content)]),timeout=600)
    if response.status_code != 200:raise RuntimeError('Review HTTP '+str(response.status_code))
    body=response.json()
    if body.get('error'):raise RuntimeError('Review API error')
    raw=(_extract_content(body) or '').strip();parsed=_parse_json_loose(raw)
    _audit.parsed(parsed)
    result=normalize_review(parsed,initial['selected_candidate_id'],initial['view_id'])
    result['raw']=raw
    return result

@_audit.model_call
def qwen_locate_bbox_in_view(
    dest_desc,
    patch_main_bgr,
    patch_s4_bgr=None,
    patch_s8_bgr=None,
    minimap_bgr=None,
    api_key=DEFAULT_API_KEY,
    url=QWEN_URL,
    model=QWEN_MODEL,
    timeout=120,
    restrict_top_half=True,
    prev_grid_5x5=None,
    segmentation_image=None,
    candidate_count=0,
    confirmation_views=None,
    search_context=None,
):
    if restrict_top_half is True:
        focus_text = (
            f"[Task Decomposition / Thought Process]\n"
            f"\n"
            f"Views:\n"
            f"- Image 1: main view ({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}), top = current heading. THIS IS THE ONLY IMAGE YOU MAY OUTPUT A BOX FOR.\n"
            f"- Image 2: same center and heading as Image 1 but with a narrower, more zoomed-in view of the same moment (local detail context).\n"
            f"- Image 3: same center and heading as Image 1 but with a wider, more zoomed-out view of the same moment (broader surrounding context).\n"
            f"- Image 4: north-up global map; red dot = current position, red arrow = current heading, "
            f"yellow trail = recent trajectory.\n"
            f"Note: Image 2 and Image 3 show exactly the same location and heading as Image 1, only with different field-of-view sizes. "
            f"Use them to confirm ambiguous regions in Image 1. For example, if the target in Image 1 is blurry, unclear, "
            f"or lies near the border of Image 1, check Image 2 / Image 3 to verify what that region actually is before deciding.\n"
            f"\n"
            f"1) Destination parsing:\n"
            f"   - Parse the destination description and extract key visual/physical attributes "
            f"(shape, texture, material, color), geometric structure, and any directional / positional cues.\n"
            f"\n"
            f"2) Scene understanding:\n"
            f"   - Summarize the scene and spatial layout in Image 1 (main view).\n"
            f"   - Summarize local spatial / contextual structure seen in Image 2 and Image 3 "
            f"(they provide tighter or wider context of the exact same place and heading).\n"
            f"   - Separately summarize global position / heading / trajectory cues from Image 4 (the minimap).\n"
            f"\n"
            f"3)Generate semantic grid map on Image 1:\n"
            f"   - Partition Image 1 into 5 rows × 5 columns (top→bottom, left→right).\n"
            f"   - For EACH cell, output 1–2 concise lowercase English words describing the dominant content "
            f"(e.g., road, building, container yard, parking, runway, water, field, forest, roof, shadow, rail, vehicle); "
            f"use 'unknown' if unclear.\n"
            f"   - Return this grid in JSON under key \"grid_5x5\" as a 5×5 array of strings.\n"
            f"\n"
            f"4) Target localization ON IMAGE 1 ONLY:\n"
            f"   - First, restrict your search to the TOP HALF of Image 1 (forward / current heading direction).\n"
            f"   - Propose candidate regions in Image 1 that satisfy the destination description, using appearance, "
            f"geometry, texture, shadows, arrangement, and nearby structures.\n"
            f"   - If nothing in the top half matches, you may extend to the full Image 1, "
            f"but the final chosen region must still come from Image 1.\n"
            f"   - Choose EXACTLY ONE final region. If you cannot isolate one unambiguous region in Image 1, "
            f"you must report dest_present=false.\n"
            f"\n"
            f"5) Bounding box output:\n"
            f"   - Produce the minimal square bounding box that fully encloses the chosen region ON IMAGE 1 ONLY.\n"
            f"   - All bounding box coordinates MUST be given in Image 1's {FIXED_CROP_SIDE}x{FIXED_CROP_SIDE} pixel grid.\n"
            f"   - Return dest_present (true/false), bbox_2d, confidence in [0.0,1.0], "
            f"and a brief reason (<500 chars).\n"
            f"\n"
            f"Rules:\n"
            f"- NEVER output or reference a bbox on Image 2, Image 3, or Image 4; they are context only.\n"
            f"- If you are not sufficiently confident in ONE clear match in Image 1, output dest_present=false.\n"
            f"- These are overhead / satellite views; color contrast may be weak. "
            f"Rely on shape, texture, layout, shadows, and stable spatial relations, not just raw color."
        )
    else:
        focus_text = (
            f"[Decision Mode and Localization Procedure]\n"
            f"\n"
            f"Views:\n"
            f"- Image 1: main view ({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}), top = current heading. THIS IS THE ONLY IMAGE YOU MAY OUTPUT A BOX FOR.\n"
            f"- Image 2: same center and heading as Image 1 but with a narrower, more zoomed-in view of the same moment (local detail context).\n"
            f"- Image 3: same center and heading as Image 1 but with a wider, more zoomed-out view of the same moment (broader surrounding context).\n"
            f"- Image 4: north-up global map; red dot = current position, red arrow = current heading, "
            f"yellow trail = recent trajectory.\n"
            f"Note: Image 2 and Image 3 show exactly the same location and heading as Image 1, only with different field-of-view sizes. "
            f"Use them to disambiguate candidates in Image 1, especially if a region is small, noisy, or sits at the border of Image 1.\n"
            f"\n"
            f"Step 0 - Mode Selection (Simple Direct Localization vs. Reasoning Chain):\n"
            f"- If the destination description is just a single category / attribute phrase "
            f"(only color / material / shape / category, e.g. 'a white house', 'a circular water tower'), "
            f"and it does NOT include relational / compositional constraints "
            f"(left/right/center/adjacent/along/between/leftmost/rightmost/near/connected to/parallel to), "
            f"then use [Simple Direct Localization].\n"
            f"- Otherwise, use [Reasoning Chain].\n"
            f"\n"
            f"[Simple Direct Localization]:\n"
            f"- Work directly on Image 1. Start from the TOP HALF of Image 1 (the forward / heading direction), "
            f"expanding to the full Image 1 only if necessary.\n"
            f"- Find the best-matching region in Image 1 using category- and attribute-level cues "
            f"(roof shape, texture pattern, shadow geometry, surroundings, etc.).\n"
            f"- If multiple candidates in Image 1 look similar, consult Image 2 and Image 3 "
            f"to clarify local spatial context (e.g. cluster layout, adjacency to a runway / road), "
            f"and consult Image 4 (the minimap) for global orientation / trajectory. "
            f"Use this context to decide which ONE candidate in Image 1 is most consistent.\n"
            f"\n"
            f"[Reasoning Chain]:\n"
            f"Step 1 - Generate an explicit reasoning chain. Break the destination description into ordered, "
            f"checkable sub-steps. Example: 'the vehicle on the far left of the central parking line' → "
            f"'identify the central parking line' → 'scan along that line for the leftmost vehicle'.\n"
            f"Step 2 - Follow those sub-steps on Image 1. "
            f"For each sub-step, progressively narrow the candidate area in Image 1, "
            f"starting with the top half first whenever it applies. "
            f"Use Image 2 / Image 3 only to clarify local geometry and layout "
            f"(e.g. which elongated strip is a runway, which dense blob is a parking lot), "
            f"and use Image 4 only for high-level position / heading / trajectory context. "
            f"Record at least one concrete visual evidence item for each critical sub-step.\n"
            f"\n"
            f"[Generate semantic grid map on Image 1]:\n"
            f"   - Partition Image 1 into 5 rows × 5 columns (top→bottom, left→right).\n"
            f"   - For EACH cell, output 1–2 concise lowercase English words describing the dominant content "
            f"(e.g., road, building, container yard, parking, runway, water, field, forest, roof, shadow, rail, vehicle); "
            f"use 'unknown' if unclear.\n"
            f"   - Return this grid in JSON under key \"grid_5x5\" as a 5×5 array of strings.\n"
            f"\n"
            f"[Bounding box output]:\n"
            f"- Draw one tight square bounding box that encloses the final chosen region ON IMAGE 1 ONLY.\n"
            f"- All bbox coordinates MUST be reported in Image 1's {FIXED_CROP_SIDE}x{FIXED_CROP_SIDE} pixel grid.\n"
            f"- Return dest_present (true/false), bbox_2d, confidence in [0.0,1.0], and 1–3 short justification sentences.\n"
            f"  · In Simple Direct Localization mode: name the decisive visual attributes.\n"
            f"  · In Reasoning Chain mode: cite the key reasoning steps and supporting evidence.\n"
            f"\n"
            f"Critical rules:\n"
            f"- DO NOT output bounding boxes for Image 2, Image 3, or Image 4; they are context only.\n"
            f"- If you cannot isolate exactly one high-confidence match within Image 1, "
            f"you MUST output dest_present=false.\n"
            f"- These are overhead / satellite views; color may be low-contrast. "
            f"Use structure, texture, shape, arrangement, shadows, and spatial relations, not just raw color."
        )

    system_prompt = (
        "You are a visual localization assistant. You must return ONLY strict JSON, with no extra text.\n"
        "\n"
        "You will receive up to 4 images:\n"
        f"- Image 1: main view ({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}), top = current heading. "
        "This is the ONLY image for which you are allowed to output a bounding box.\n"
        "- Image 2: same center and same heading as Image 1 but with a narrower, more zoomed-in view of the exact same moment. "
        "Use this only as local-detail context.\n"
        "- Image 3: same center and same heading as Image 1 but with a wider, more zoomed-out view of the exact same moment. "
        "Use this only as surrounding-context.\n"
        "- Image 4: a north-up global map. A red solid dot marks the current position, "
        "a red arrow marks the current heading, and a yellow trail shows the recent trajectory. "
        "Use this only as high-level positional / directional prior.\n"
        "\n"
        "If a previous-step grid_5x5 is provided:\n"
        "- Treat it as a SOFT PRIOR for Image 1 ONLY (top→bottom rows, left→right cols).\n"
        "- Use it just to prioritize scanning cells and keep temporal consistency.\n"
        "- If it conflicts with current evidence in Image 1/2/3, IGNORE the prior.\n"
        "- Always regenerate the current step's grid_5x5 from the present images; do NOT copy the prior.\n"
        "\n"
        "IMPORTANT:\n"
        "- Image 2, Image 3, and Image 4 are context only. NEVER output a bbox for them.\n"
        "- All bbox coordinates MUST be reported in Image 1's pixel grid "
        f"({FIXED_CROP_SIDE}x{FIXED_CROP_SIDE}).\n"
        "- If you cannot confidently identify exactly one valid region in Image 1, "
        "you must set dest_present=false.\n"
        "\n"
        "Your task is described below. Follow it exactly.\n"
        f"{focus_text}\n"
        "\n"
        "You must output ONLY the following JSON object:\n"
        "{"
        "\"dest_present\": true|false, "
        "\"bbox_2d\": [x1,y1,x2,y2], "
        "\"confidence\": 0.0-1.0, "
        "\"reason\": \"a justification of up to 500 characters\", "
        "\"grid_5x5\": [[c11,c12,c13,c14,c15],[c21,c22,c23,c24,c25],[c31,c32,c33,c34,c35],[c41,c42,c43,c44,c45],[c51,c52,c53,c54,c55]]"
        "}"
    )

    if segmentation_image is not None:
        system_prompt = system_prompt.replace("up to 4 images", "up to 5 images") + IMAGE5_PROMPT

    user_msg = (
        f"Destination description: {dest_desc or '(None)'}\n"
        f"All coordinates MUST be in Image 1's {FIXED_CROP_SIDE}x{FIXED_CROP_SIDE} pixel grid.\n"
        f"If not sure, return dest_present=false."
    )

    content = [
        {"type": "text", "text": user_msg},

        {"type": "image_url", "image_url": {
            "url": data_url_from_image_array(patch_main_bgr)
        }},
    ]

    if prev_grid_5x5:
        content.append({
            "type": "text",
            "text": "Previous step grid_5x5 (Image 1, soft prior; ignore if inconsistent): " + json.dumps(prev_grid_5x5)
        })

    if patch_s4_bgr is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url_from_image_array(patch_s4_bgr)}
        })
    if patch_s8_bgr is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url_from_image_array(patch_s8_bgr)}
        })

    if minimap_bgr is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url_from_image_array(minimap_bgr)}
        })

    if segmentation_image is not None:
        content.append({
            "type": "image_url",
            "image_url": {"url": data_url_from_image_array(segmentation_image)}
        })

    if candidate_count:
        from .candidates import build_request
        search_images = [
            dict(role=role, image=image, description=description)
            for role, image, description in (
                ('main', patch_main_bgr, 'Image 1 · search_main：当前搜索主图，可定位。'),
                ('detail', patch_s4_bgr, 'Image 2 · search_detail：同位置、同朝向的细节视图。'),
                ('context', patch_s8_bgr, 'Image 3 · search_context：同位置、同朝向的广域视图。'),
                ('minimap', minimap_bgr, 'Image 4 · minimap：北向地图，红点为当前位置、红箭头为朝向、黄线为历史轨迹。'),
                ('sam3', segmentation_image, 'Image 5 · sam3：搜索主图上的 SAM3 辅助标注。'),
            )
        ]
        system_prompt, content = build_request(
            focus_text, system_prompt, dest_desc, search_images,
            prev_grid_5x5=prev_grid_5x5, confirmation_views=confirmation_views)
        if search_context is not None:
            if confirmation_views is not None:
                raise ValueError('Spatial context is Search-only')
            from .spatial_map import build_spatial_request
            system_prompt, content = build_spatial_request(system_prompt, dest_desc, search_context, prev_grid_5x5)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": model,
        "temperature": 0,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": content}
        ]
    }

    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/json" not in ct:
            return {
                "dest_present": False,
                "bbox_2d": [0,0,0,0],
                "confidence": 0.0,
                "reason": "fallback: non-json response",
                "grid_5x5": [],
                "raw": r.text[:200]
            }
        resp = r.json()
    except Exception as e:
        return {
            "dest_present": False,
            "bbox_2d": [0,0,0,0],
            "confidence": 0.0,
            "reason": f"fallback: request/json error: {e}",
            "grid_5x5": [],
            "raw": None
        }

    if isinstance(resp, dict) and "error" in resp:
        return {
            "dest_present": False,
            "bbox_2d": [0,0,0,0],
            "confidence": 0.0,
            "reason": f"fallback: api error {resp.get('error')}",
            "grid_5x5": [],
            "raw": resp
        }

    raw = (_extract_content(resp) or "").strip()
    parsed = _parse_json_loose(raw) or {}
    _audit.parsed(parsed)

    if candidate_count:
        from .candidates import normalize_reply
        return normalize_reply(parsed, raw, confirmation_views,
                               search_context['main_ids'] if search_context is not None else None)

    def _box_ok(b):
        try:
            x1, y1, x2, y2 = [float(v) for v in b]
            return x2 > x1 and y2 > y1
        except Exception:
            return False

    dest_present = _norm_bool(parsed.get("dest_present", False))
    bbox = parsed.get("bbox_2d") or [0,0,0,0]
    try:
        confidence = float(parsed.get("confidence", 0.0))
        confidence = min(max(confidence, 0.0), 1.0)
    except Exception:
        confidence = 0.0
    reason = (parsed.get("reason") or "").strip() or "N/A"

    grid_5x5 = parsed.get("grid_5x5") or []

    if not dest_present or not _box_ok(bbox):
        return {
            "dest_present": False,
            "bbox_2d": [0,0,0,0],
            "confidence": confidence,
            "reason": reason,
            "grid_5x5": grid_5x5,
            "raw": raw
        }

    side = float(FIXED_CROP_SIDE)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = max(0.0, min(x1, side-1))
    y1 = max(0.0, min(y1, side-1))
    x2 = max(0.0, min(x2, side-1))
    y2 = max(0.0, min(y2, side-1))

    return {
        "dest_present": True,
        "bbox_2d": [x1,y1,x2,y2],
        "confidence": confidence,
        "reason": reason,
        "grid_5x5": grid_5x5,
        "raw": raw
    }

