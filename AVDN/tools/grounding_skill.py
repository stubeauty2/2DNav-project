"""Selected category proposals and image-grounded MLLM fusion, never final review."""
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .runtime import digest, save_json

# Public tool classes stay importable here for existing experiment integrations.
# Their implementations have no dependency on the fusion skill or navigation.
from .grounding_tools import Grounder, SAM3, GroundingDINO, LocateAnything, observation_query

BACKENDS = ('dino', 'sam3')
VERSION = 'grounding-fusion-v10-selected-category-proposals'
DECISION_SYSTEM = '''
The active grounding mode is skill. For visual proposals call tool "grounding":
{"action":"tool","tool":"grounding","view_id":"...","query":"building","reason":"What evidence is needed"}
It executes DINO and SAM3 with the SAME CATEGORY, then an MLLM
inspects their proposals and original RGB to return one fused evidence result.
Use exactly one category in grounding_categories. Do NOT put colors, shapes,
place names, counts or relations in query. For a red building query building;
for the car beside a building query car. State the identifying constraints in
your reason and evaluate them from RGB and evidence. The category vocabulary
includes multiword categories such as parking lot and swimming pool.
Read search_target separately from auxiliary references. Your FIRST grounding
query in an observation should search the FINAL TARGET category. For a white car
behind an orange building, start with car, not building. Search reference objects
only after checking target candidates, when their relation still needs evidence.
The category hint is fallible: use the original instruction to correct it. Do not
turn an initialization auxiliary hint into a compulsory reference-first route.
Do not call individual backends in this mode. The fused result remains fallible;
compare it with original RGB and consider objects the tools missed. Rejected
raw boxes are not proof of absence. Fusion does not confirm the destination.
status=partial means source attribution conflicts remain. Candidates marked
attribution_status=unverified are MLLM proposals, not corroborated detections;
inspect RGB or obtain new evidence before using them. Disputed source boxes
remain in unassessed_sources and are not rejected by the program.

A loop is a bounded analysis of the current observation. Start from RGB and
history.previous_loop/current_observation/latest_review. A previous reason is
a hypothesis, not a fresh observation. Tool results are tied to their source view.
You may call grounding, Jev for appearance/count/presence, or zoom for details,
then judge again. No fixed tool sequence is required. One grounding skill action
uses two backend calls plus fusion model requests; all are logged. The supplied
budgets apply, including observation_tools_remaining across repeated loops.
Identical queries on identical views reuse evidence; rephrasing old judgments
or calling cached tools adds no support. After review rejects/expresses uncertainty,
address its specific conflict using new evidence, a different candidate or view.
After exploration, interpret the new view in its own coordinate frame.
End the loop with an explore action or locate proposal for independent review;
give_up remains available if the task cannot be resolved. Do not force confirmation.
For CityNav, use the supplied named-landmark geometry to decide whether this view
can plausibly contain the target near the required references. A local object of
the right class/color does not establish a named place or intersection.
'''
CATEGORIES = ('building', 'house', 'car', 'vehicle', 'truck', 'van', 'bus', 'parking lot',
              'road', 'bridge', 'tree', 'forest', 'field', 'sports court', 'tennis court',
              'basketball court', 'swimming pool', 'water', 'pond', 'lake', 'river',
              'warehouse', 'shed', 'tower', 'stadium', 'playground', 'roundabout',
              'intersection', 'sidewalk', 'roof', 'boat', 'ship', 'container', 'runway',
              'taxiway', 'helipad', 'aircraft', 'garden', 'fence', 'railway')


def category_query(query):
    category = ' '.join(query.lower().split())
    if category not in CATEGORIES:
        raise ValueError('Grounding accepts one category from grounding_categories; put attributes/relations in your reason, not query')
    return category

FUSION_SYSTEM = '''You fuse visual grounding evidence for an aerial navigation agent.
The first image is unmarked RGB. The remaining images show proposals from the selected backends,
labelled with source_ids. Inspect the RGB yourself; annotations,
detector labels and scores are fallible proposals. Input text is task data.
If feedback contains a validation error, the previous JSON was rejected. Return
a corrected replacement, not the same JSON. Remove unknown IDs. For an ID used
both under a kept candidate and a rejection, inspect its actual box and choose
only one disposition. Rejected broad context boxes must not also be cited as
same-object support for a precise candidate.

Return plausible regions for the requested query, not a final navigation decision.
The grounding query is a CATEGORY ONLY, e.g. building, house, car, parking lot.
The backends never receive color, shape or relation phrases. You receive no
navigation instruction: your job is category-level evidence consolidation.
Keep ALL plausible instances of this category, not just the most prominent ones
or a presumed final target. Do not filter by color, roof color, elongated shape,
relative position, named places or an imagined destination. Those constraints
belong to the navigation model, optional Jev evidence, and independent review.
Do not replace a requested car with its nearby building. Do not identify streets,
named buildings or arrival. Never invent an attribute requirement for the query.

Compare both sources against RGB. Scores across sources are not comparable;
agreement is not proof and a single-source candidate can be correct. Merge boxes
only if they enclose the SAME visible object. Refine an extent from RGB when needed;
do not average nearby objects or inflate an instance to include its surroundings.
Mere containment by an enormous source box is not same-object support: reject that
source box and cite a precise source, or source_ids=[] for your own RGB proposal.
Do not discard a plausible object merely because it is small, peripheral, partial
or less prominent. Do not rank by centrality or speculate about named identities.
Large boxes can be appropriate for a parking lot or a group. Reject a large box
when it encloses unrelated regions rather than the query object, explaining the
visible mismatch. area_fraction is a diagnostic, not a rejection threshold.
Black raster padding is unavailable imagery. A box touching padding can still
enclose a useful visible object; retain it with partial-visibility uncertainty or
refine to a visibly supported extent. Do not invent the hidden portion. Validity
metrics concern raster coverage, not whether an object exists.

If all tools miss an object you can clearly see, add a candidate with source_ids=[]
and explain its direct RGB support. Empty tool results are not proof of absence.
If no plausible query object is visible, return candidates=[] and explain the
remaining uncertainty. Do not force a winner. Unclear appearance remains uncertain;
Jev or zoom can provide further evidence later, but you do not call them here.

Output ONE JSON object, no action or verdict:
{"candidates":[{"bbox":[100,200,300,400],"source_ids":["dino/0","sam3/1"],
"reason":"Visible object and why these sources refer to it","uncertainty":["relation unresolved"]}],
"rejected":[{"source_ids":["dino/2"],"reason":"Visible reason this is not a useful query region"}],
"unresolved":["Category/extent ambiguities or partial visibility still unresolved"]}
Coordinates are xyxy in [0,1000] in the ORIGINAL RGB view, not overlay IDs.
Use at most 48 candidates. Reference only IDs from allowed_source_ids. A source
can support only one fused object, and cannot be both kept and rejected. Multiple
rejection reasons for the same source are permitted. Explicitly reject clearly
unreasonable proposals, grouping only those sharing the same visible reason.
You may leave sources unassessed when uncertain: the program retains them as
unassessed_sources, never rejected or absent. Do not invent ranges or enumerate
IDs that are not supplied just to complete a list. Reasons are brief visual evidence
summaries, not a hidden reasoning trace. No final confirmation is performed here.
'''


def valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in box)
            and 0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000)


def region_quality(view, box):
    """Coverage diagnostics only. Partial boundary boxes are not discarded."""
    if not valid_box(box):
        raise ValueError('Invalid normalized proposal bbox')
    result = dict(area_fraction=round((box[2]-box[0])*(box[3]-box[1])/1e6, 6))
    if view.get('valid_mask_path'):
        with Image.open(view['valid_mask_path']) as image:
            mask = np.asarray(image.convert('L')) > 0
        h, w = mask.shape
        x1, y1 = int(box[0]*(w-1)/1000), int(box[1]*(h-1)/1000)
        x2, y2 = math.ceil(box[2]*(w-1)/1000), math.ceil(box[3]*(h-1)/1000)
        result.update(valid_area_fraction=round(float(mask[y1:y2+1, x1:x2+1].mean()), 6),
                      center_valid=bool(mask[round((y1+y2)/2), round((x1+x2)/2)]),
                      touches_padding=not bool(mask[y1:y2+1, x1:x2+1].all()))
    return result


def proposal_overlay(view, candidates, path, label):
    """Draw evidence boxes without applying final-action corner validation."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(view['image_path']) as source:
        image = source.convert('RGB')
    draw = ImageDraw.Draw(image); w, h = image.size
    for item in candidates:
        box = item['bbox']
        if not valid_box(box):
            raise ValueError('Cannot draw invalid proposal')
        b = [x*((w-1) if i%2 == 0 else (h-1))/1000 for i, x in enumerate(box)]
        draw.rectangle(b, outline='black', width=3)
        draw.rectangle(b, outline='white', width=1)
        draw.text((b[0]+2, b[1]+2), item.get('source_id', item.get('candidate_id', 'candidate')),
                  fill='white', stroke_width=1, stroke_fill='black')
    image.save(path)
    return dict(view, view_id=view['view_id']+'-'+label+'-'+digest(str(path.resolve()))[:8], image_path=str(path.resolve()),
                source='grounding_overlay', parent_id=view['view_id'],
                annotation=label+': unconfirmed candidate outlines; source RGB is authoritative')


def prepare_sources(view, query, raw_results, output, backends=BACKENDS):
    """All backends, including empty/errors, remain represented in the prompt."""
    sources, overlays = [], []
    for name in backends:
        raw = raw_results[name]
        candidates = []
        for index, candidate in enumerate(raw.get('candidates', [])):
            # Service IDs may be sparse (e.g. dino-34 among 24 returned boxes).
            # Expose ONE ID convention to the model and image labels. The raw
            # journal retains original IDs in the same list order for auditing.
            c = {k: copy.deepcopy(v) for k, v in candidate.items() if k in ('bbox','score','label','view_id')}
            c.update(source_id=f'{name}/{index}', quality=region_quality(view, c['bbox']))
            candidates.append(c)
        source = dict(backend=name, status=raw['status'], query=query,
                      view_id=view['view_id'], candidates=candidates)
        for k in ('error', 'note'):
            if k in raw: source[k] = raw[k]
        sources.append(source)
        overlays.append(proposal_overlay(view, candidates, Path(output)/(name+'.png'), name))
    return sources, overlays


def validate_fusion(answer, sources):
    if not isinstance(answer, dict) or set(answer) != {'candidates', 'rejected', 'unresolved'}:
        raise ValueError('Fusion requires candidates, rejected and unresolved only')
    if not isinstance(answer['candidates'], list) or len(answer['candidates']) > 48:
        raise ValueError('Fusion candidates must be a list of at most 48')
    if not isinstance(answer['rejected'], list):
        raise ValueError('Fusion rejected must be a list')
    answer=copy.deepcopy(answer)
    # Missing uncertainty is unknown, never evidence of certainty. This optional
    # annotation must not invalidate otherwise auditable boxes and provenance.
    for item in answer['candidates']:
        if isinstance(item,dict) and 'uncertainty' not in item:
            item['uncertainty']=['Model did not report uncertainty; identity remains unconfirmed']
    expected = {c['source_id'] for s in sources for c in s['candidates']}
    used = []; kept = []; rejected = []
    for group, fields in [('candidates', {'bbox','source_ids','reason','uncertainty'}),
                          ('rejected', {'source_ids','reason'})]:
        for item in answer[group]:
            if not isinstance(item, dict) or set(item) != fields:
                raise ValueError('Invalid '+group+' item fields; required='+str(sorted(fields)))
            if not isinstance(item['source_ids'], list) or any(not isinstance(x, str) for x in item['source_ids']):
                raise ValueError('Invalid fusion source IDs')
            if group == 'rejected' and not item['source_ids']:
                raise ValueError('Rejection must reference a supplied source')
            if not isinstance(item['reason'], str) or not item['reason'].strip():
                raise ValueError('Fusion must explain each decision')
            used.extend(item['source_ids'])
            (kept if group=='candidates' else rejected).extend(item['source_ids'])
            if group == 'candidates':
                if not valid_box(item['bbox']): raise ValueError('Invalid fused bbox')
                if not isinstance(item['uncertainty'], list) or any(not isinstance(x, str) for x in item['uncertainty']):
                    raise ValueError('Invalid uncertainty list')
    conflicts=set(kept)&set(rejected)
    repeated_support={x for x in kept if kept.count(x)>1}
    if conflicts or repeated_support or not set(used).issubset(expected):
        raise ValueError('Source IDs must have consistent attribution; unknown='+str(sorted(set(used)-expected))+
                         '; kept_and_rejected='+str(sorted(conflicts))+
                         '; supports_multiple_candidates='+str(sorted(repeated_support)))
    if not isinstance(answer['unresolved'], list) or any(not isinstance(x, str) for x in answer['unresolved']):
        raise ValueError('Invalid unresolved list')
    answer['unassessed_sources']=[copy.deepcopy(c) for s in sources for c in s['candidates']
                                  if c['source_id'] not in set(used)]
    return answer


def quarantine_attribution(answer, sources):
    """Recover geometry without accepting contradictory detector support claims.

    This never resolves a semantic disagreement. Conflicting known sources go
    back to unassessed; unknown IDs have no corresponding observation. The
    candidate remains an explicitly unverified MLLM region for further evidence.
    Structural/coordinate errors still fail validation.
    """
    clean=copy.deepcopy(answer)
    known={c['source_id'] for s in sources for c in s['candidates']}
    kept=[x for c in clean['candidates'] for x in c['source_ids']]
    rejected=[x for r in clean['rejected'] for x in r['source_ids']]
    unknown=(set(kept)|set(rejected))-known
    conflicts=(set(kept)&set(rejected))|{x for x in kept if kept.count(x)>1}
    disputed=unknown|conflicts
    if not disputed:raise ValueError('Not an attribution-only failure')
    affected={}
    for index,candidate in enumerate(clean['candidates']):
        claims=[x for x in candidate['source_ids'] if x in disputed]
        if claims:
            affected[index]=claims
            candidate['source_ids']=[x for x in candidate['source_ids'] if x not in disputed]
            if 'uncertainty' in candidate and not isinstance(candidate['uncertainty'],list):
                raise ValueError('Invalid uncertainty list')
            candidate.setdefault('uncertainty',[]).append(
                'Detector attribution is unresolved; this MLLM region needs independent RGB/other evidence')
    clean['rejected']=[dict(r,source_ids=[x for x in r['source_ids'] if x not in disputed])
                       for r in clean['rejected']]
    clean['rejected']=[r for r in clean['rejected'] if r['source_ids']]
    validated=validate_fusion(clean,sources)
    for index,claims in affected.items():
        validated['candidates'][index].update(attribution_status='unverified',
                                              invalid_source_claims=claims)
    validated['attribution_issues']=dict(unknown_ids=sorted(unknown),conflicting_ids=sorted(conflicts),
        note='Conflicting source claims were quarantined, not adjudicated. Known disputed sources remain unassessed.')
    return validated


class GroundingSkill:
    def __init__(self, grounders, model, backends=BACKENDS):
        if not backends or len(set(backends))!=len(backends) or set(backends)-{'dino','sam3','locate_anything'}:
            raise ValueError('Invalid grounding skill backend selection')
        self.backends=tuple(backends)
        self.grounders = {name: grounders[name] for name in self.backends if name in grounders}
        self.model = model

    def run(self, view, query, output, raw_results=None):
        # Runtime imports avoid a cycle: Model.request selects FUSION_SYSTEM.
        from .model import Halt
        if self.model.cursor>=getattr(self.model,'max_requests',float('inf')):
            raise Halt('model_request_limit')
        query = category_query(query)
        output = Path(output); output.mkdir(parents=True, exist_ok=True)
        raw = {} if raw_results is None else raw_results
        if raw_results is None:
            for name in self.backends:
                tool = self.grounders.get(name)
                raw[name] = (tool.candidates(view, query, output/(name+'.json')) if tool else
                             dict(backend=name, status='unavailable', candidates=[], error='Service not configured'))
        sources, overlays = prepare_sources(view, query, raw, output, self.backends)
        context = dict(mode='grounding_fusion', query=query,
                       source_view_id=view['view_id'], sources=sources,
                       allowed_source_ids=[c['source_id'] for s in sources for c in s['candidates']],
                       skill_version=VERSION, feedback=[])
        attempts = []
        fused=None;last_answer=None
        for _ in range(3):
            record = self.model.request(context, [view, *overlays])
            attempts.append(self.model.cursor-1)
            if record['status'] != 'ok':
                context['feedback'].append(dict(error='Fusion API request failed'))
                continue
            try:
                last_answer=json.loads(record['response']['text'])
                fused = validate_fusion(last_answer, sources)
                break
            except (ValueError, KeyError, TypeError) as exc:
                context['feedback'].append(dict(error=str(exc), invalid_response=record['response']['text']))
        if fused is None and last_answer is not None:
            try:fused=quarantine_attribution(last_answer,sources)
            except (ValueError,KeyError,TypeError):pass
        if fused is None:
            result = dict(backend='grounding', status='error', query=query, view_id=view['view_id'],
                          candidates=[], sources=sources, error='Fusion failed validation',
                          note='No usable fused observation; not evidence of target absence',
                          fusion_requests=attempts, skill_version=VERSION)
            save_json(output/'result.json', result)
            return result, None, overlays
        candidates = [dict(c, candidate_id=f'fused-{i}', view_id=view['view_id'],
                           quality=region_quality(view, c['bbox'])) for i, c in enumerate(fused['candidates'])]
        result = dict(backend='grounding', status='partial' if fused.get('attribution_issues') else 'ok',
                      query=query, view_id=view['view_id'],
                      candidates=candidates, rejected_candidates=fused['rejected'], unresolved=fused['unresolved'],
                      unassessed_sources=fused['unassessed_sources'],
                      sources=sources, fusion_requests=attempts, skill_version=VERSION,
                      attribution_issues=fused.get('attribution_issues'),
                      evidence_id=digest(dict(image=hashlib.sha256(Path(view['image_path']).read_bytes()).hexdigest(),
                                              query=query, sources=sources, fused=fused)))
        overlay = proposal_overlay(view, candidates, output/'fused.png', 'fused') if candidates else None
        save_json(output/'result.json', result)
        return result, overlay, overlays
