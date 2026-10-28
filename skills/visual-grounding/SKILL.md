---
name: visual-grounding
description: Run and audit independent aerial grounding tools and the configurable category-grounding skill, leaving navigation and review to the decision agent.
---

Use for the AVDN and CityNav runtimes in C:/Levir/2DNav/2DNav-project.

## Current runtime contract

The default `--grounding-mode adaptive` exposes tools and skills independently.
The navigation model first sees RGB and chooses one action:

```json
{"action":"tool","tool":"dino","view_id":"r00-main","query":"building","reason":"Get category candidates"}
```

```json
{"action":"skill","skill":"grounding","view_id":"r00-main","query":"building","backends":["dino","sam3"],"reason":"Consolidate category proposals"}
```

Raw tools are SAM3, DINO and LocateAnything when their services are configured and
healthy; zoom is geometric, Jev and optional visual_question provide visual evidence.
`available_grounding_tools`, `available_evidence_tools` and `available_skills` are
separate registries. Never call an unavailable service. A skill can select one or
more available backends; omitted backends use `--skill-backends` or available DINO
and SAM3 defaults. Raw tools remain independently callable regardless of that subset.

Raw DINO/SAM3 and the skill accept one category from grounding_categories. The skill
passes exactly the SAME category to every selected backend, including LocateAnything.
Do not add colors, shapes, place names, routes or relational sentences. Raw
LocateAnything separately supports a short visible referring phrase. Destination
constraints remain in the navigation context; fusion receives no navigation instruction.

Client implementations live in tools/grounding_tools.py; category fusion lives in
[grounding_skill.py](C:/Levir/2DNav/2DNav-project/AVDN/tools/grounding_skill.py).
The skill combines unmarked RGB and one labeled proposal image per selected source.
It returns candidates, provenance, visible reasons, explicit rejection and unresolved
or unassessed sources. It never chooses the destination. CityNav carries equivalent
self-contained modules for freezing. Do not duplicate prompts in this document.

Unknown/conflicting source references trigger bounded repairs. Attribution-only
failures can yield partial results: disputed references lose corroboration, known
disputed boxes remain unassessed and model proposals are marked unverified. Invalid
coordinates or structural output errors fail. Unassessed is not rejected; errors or
empty proposals do not establish absence. RGB proposals with source_ids=[] are allowed.

## Evidence loop

The main decision model defaults to qwen3.5-flash; fusion and independent review use
that same model. `--visual-model qwen3.8-flash` or an explicitly available alternative
adds a separate visual_question tool. Only that role uses the optional model. All
model requests share one global request budget and replay journal.

Identify the actual target noun/group from the full instruction before interpreting
reference objects. Inspect plausible visible candidates before moving away. Obtain
an initial map with a raw tool or skill when needed, then choose evidence for the
specific ambiguity: Jev for object-bound presence/color/count/shape, zoom for detail,
visual_question for competing regions, relations or ordinals. No fixed tool sequence
or confidence-based confirmation is imposed. Preserve public landmark geometry in
CityNav; a broad landmark rectangle is not proof of semantic identity.

Jev questions use neutral objects and tight ROIs; uncertain presence limits color
interpretation. Its full uncalibrated distribution remains visible. visual_question
returns observations/contradictions/unobservable details, never a locate verdict.
Only the separate MLLM review confirms a final destination. Its claims remain fallible.

Optional visual answers may also contain up to eight unverified RGB proposals.
Their boxes stay in the source view; the agent decides which evidence to use.
The adaptive context removes repeated presentations, retaining all candidates and
Jev probability distributions, and keeps the immediately preceding main view after
movement. Distinguish the requested destination noun/group from its reference objects
in both decision and review; a matching reference is not the destination.

After an unresolved review contradiction or two consecutive cached calls, the
adaptive runtime requests a bounded evidence-recovery decision. The same 3.5 model
chooses a new non-Jev source/question, explores, or stops. It cannot locate or repeat
Jev during that recovery step; normal decisions resume after new source evidence.
This is a source-acquisition constraint, not a target selection rule. It applies only
while evidence and model budgets permit recovery; no helper is called automatically.
Callable evidence actions reserve enough global requests for a later decision and
review, including bounded fusion repairs. Independent adaptive review sees current
observations but no old review verdict/prose. The main agent retains old reviews
to address their conflicts; only the independent reviewer omits those opinions.

Memory, view-bound coordinates, evidence ledger and cache live in tools/memory.py.
Identical raw calls can be reused by a later skill, and skill backend results can be
reused as raw tools. Cached calls still consume logical actions/model decisions and
do not add fresh evidence. Errors cannot unlock a duplicate review in adaptive mode.
Keep history and view coordinate frames distinct; do not recreate removed context.py,
grounding.py, tool_protocol.py or agent_memory.py compatibility shells.

## Legacy comparison and validation

`--grounding-mode skill` preserves the older tool:grounding interface and two-source
DINO/SAM3 policy: it does not construct or probe LocateAnything. `direct` preserves
historical independent-tool operation. Frozen experiments use their own runtimes.

Use `tests/adaptive_agent/test_adaptive.py` for current independent dispatch, subset
selection, multimodal model routing, budgets and replay. Existing tests under
grounding_skill, jev_evidence and agent_refactor protect historical behavior.

`tests/adaptive_agent/experiment.py prepare` freezes fresh seeded paired batches,
public inputs, source hashes and isolated evaluator-only GT. `visible` is explicitly
GT-conditioned checkpoint localization, not full navigation SR or semantic visibility;
`navigation` starts at the original pose. Exclude previous selected IDs, do not replace
failed samples, and do not feed GT to workers, tools, fusion, questions or review.
Compare correct proposals, review retention and final metric success separately.

Use real CUDA DINO/SAM3 services for new local smoke tests. Keep credentials only in
process memory. Compare current changes to out/16_adaptive_agent/20261027/before,
which preserves prior authorized workspace edits; do not use Git HEAD as the baseline.
The dated stage-10/12/14/15 experiment outputs remain historical, write-once evidence.
