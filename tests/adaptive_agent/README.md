# Adaptive evidence loop

The online packages default to `adaptive` mode and `qwen3.5-flash`. Raw tools and
the grounding skill are independently callable. Adaptive entry points now enable
`qwen3.8-flash` visual questions by default; `--no-visual-model` (or Python
`visual_model=None`) disables them. Only evidence questions use 3.8; initialization,
decision, fusion and review keep their existing models. Existing `skill` and `direct`
modes remain available for historical comparisons.

```powershell
python -m AVDN.run --inputs public-episodes.jsonl --root out/new-adaptive-run `
  --model qwen3.5-flash --grounding-mode adaptive --tool-protocol structured_v2 `
  --dino-url http://127.0.0.1:8766 --sam3-url http://127.0.0.1:8765 `
  --jev-url http://127.0.0.1:8770 --jev-model imajev-4b `
  --visual-model qwen3.8-flash
```

Replace `AVDN` with `CityNav` for the other dataset. Configure
`--locate-anything-url` to expose that raw tool. `--skill-backends dino` selects a
single default fusion source without disabling other raw tools; the agent can
also select a source subset on each skill call. The optional visual model can be
any supported multimodal model ID on the configured endpoint. Its availability
must be checked through a real request; it does not replace decision/fusion/review.

The main model sees the raw image first, then requests evidence for the actual
ambiguity before deciding locate or explore. Jev supports neutral object-bound
presence, color, count and shape questions. `visual_question` supports comparisons,
relations and ordinal ambiguity with up to four explicit regions. Only independent
review confirms the destination. Errors, unknown answers and detector agreement
do not confirm identity. Every model role shares the request budget and journal.

The adaptive prompt is a single contract. Repeated presentations of the same
evidence are compacted without dropping candidate alternatives or Jev probability
distributions. The immediate preceding main image stays available after movement,
with its own coordinate frame. Optional visual answers can include up to eight
unverified RGB proposals; these are evidence, not confirmed destinations.

After a non-confirming review with no fresh non-Jev evidence, or two consecutive
cached calls, a bounded recovery decision exposes new evidence sources, exploration
and stopping. The same 3.5 model chooses the source, question and ROI; the runtime
does not choose a target or automatically call the larger model. New successful
non-Jev evidence restores the usual actions. Recovery is disabled when evidence or
model budgets cannot support it. Main and review prompts explicitly distinguish
the requested destination noun/group from reference objects in its modifiers.
Callable tools reserve a subsequent main decision and review within the global
request budget (plus up to three requests for skill fusion/repair). The execution
layer enforces the same reserve. Independent adaptive review receives current
observations but no previous review verdict or prose; the main agent retains that
history for recovery. This prevents an old rejection from being presented as the
starting opinion about a different candidate.

Implementation:

- `AVDN/tools/grounding_tools.py`: independent SAM3, DINO, LocateAnything clients.
- `AVDN/tools/grounding_skill.py`: selected-backend fusion and source attribution.
- `AVDN/tools/adaptive_policy.py`: instance-dependent evidence selection policy.
- `AVDN/tools/visual_evidence.py`: scoped visual question/answer validation.
- `AVDN/tools/memory.py`: separate registries, cache sharing, observation evidence.
- `AVDN/loop.py`: validated tool/skill/locate/explore/review dispatch.

CityNav carries the equivalent modules plus its public landmark geometry policy.

## Regression checks

```powershell
python -B tests/adaptive_agent/test_adaptive.py
python -B tests/adaptive_agent/test_experiment.py
python -B tests/grounding_skill/test_skill.py
python -B tests/jev_evidence/test_evidence.py
python -B tests/tool_iteration/test_protocol.py
python -B tests/agent_refactor/test_entrypoint.py
python -B -m unittest discover -s tests/andn_core -p test_continuous_agent.py
```

These verify execution, provenance, role routing, budgets and replay. They do not
measure semantic quality or localization gains.

## Fresh bounded paired experiments

`experiment.py prepare` freezes the current runtime and the pre-change workspace
snapshot under `out/16_adaptive_agent/20261027/before`. The prior analysis audit
supplies the sampling population. Recent ten-case tuning examples and all IDs
selected by earlier batches are excluded. Sampling does not filter on success.

```powershell
python -B tests/adaptive_agent/experiment.py prepare --batch visible_02 --per-dataset 3
python -B out/16_adaptive_agent/20261027/visible_02/harness.py run --batch visible_02
```

The run command reads `DASHSCOPE_API_KEY` from the process or requests a hidden
terminal input. It never writes the key to artifacts. CUDA DINO/SAM3 and the Jev
service must already be healthy. It verifies 3.5 and the visual model before any
trajectory, alternates arm execution order, and refuses already attempted batches.
Failed samples are preserved, not replaced. Use a new batch for a changed runtime.
`prepare --thinking-budget 1024` requests the same optional reasoning allowance
in both arms without changing their model IDs; the default is zero. Report this
condition separately. Different fresh samples cannot isolate the causal effect of
enabling reasoning, even when the within-batch before/after comparison is matched.

The `visible` scope starts both arms at the first historical main view geometrically
containing the GT center. The original instruction, initialization and initial pose
remain intact; only current camera position changes. This is an explicitly
GT-conditioned checkpoint localization experiment, with no historical decision
memory. It cannot estimate natural coverage or dataset navigation success rate.
Use `--scope navigation` in a new batch for a full original-pose comparison.
The frozen visible batches use four rounds, six logical tools per round, ten new
evidence executions per observation, and 32 shared model requests. These are
experiment settings in `plan.json`, not a change to all CLI runtime defaults.

GT is read only by preparation/evaluation. The navigation worker receives public
episode data and the selected observation pose, with no target coordinates, GT
boxes, expected identity or success label. The report separates correct proposals,
correct proposals rejected by review and final metric success. AVDN uses the
existing square-footprint IoU > 0.4; CityNav uses center distance <= 20 m. Neither
metric alone proves exact semantic identity. Manually inspect successful cases too.

The stage-01 six-case local smoke exercises real CUDA proposals and Jev without
paid navigation calls. It is evidence of service operation, not agent improvement.

`audit_report.py --batches visible_02 visible_03` builds an offline HTML comparison
and PNGs under the stage root's `audit_report/`. Green marks evaluator targets;
magenta marks final predictions. These overlays are never read by online workers.
Do not pool successive batches as a fixed-policy estimate when their frozen
after runtimes differ. Preserve unsuccessful and interrupted attempts in reporting.

`key_session.py` accepts a key through a hidden terminal prompt and starts an
independent hidden broker. Queue files contain only a frozen batch name; the broker
passes credentials through child process environments. No key is written to disk.
The session defaults to six cases per batch with the 3.5 decision model. An explicitly
expanded run can use `--session api_session_05 --max-cases 20 --exit-after-batch`.
The hidden broker exits after a successful batch; a failed preflight retains the
credential session for an explicitly queued retry. Alternatively, write `stop.json`
in its session directory to stop it after the current batch. Verify services are
ready before dispatch, and never retry an existing trajectory attempt silently.

For a larger fixed-version validation, `paired_stats.py --primary visible_08_expanded
--combine visible_06` reports new cases separately from a secondary pooled summary.
It refuses different runtime hashes, budgets, models, reasoning settings or repeated
episodes. It reports paired wins/losses, Wilson intervals, paired bootstrap intervals
and an exact two-sided McNemar test. These quantify metric outcomes; semantic image
audits remain separate. `audit_report.py --batches visible_08_expanded --output
expanded_audit_report` preserves the earlier report in its original directory.

## Observed results, 2026-10-27

76 offline regression checks pass. Six disjoint online batches contain 36 new
cases and 72 paired trajectories. Each dataset contributes three cases per batch.
The table reports final metric successes, before → after:

| Batch | After version / condition | AVDN | CityNav |
| --- | --- | --- | --- |
| visible_02 | Independent tools and skill | 1 → 1 | 0 → 0 |
| visible_03 | Compact context and RGB proposals | 0 → 0 | 0 → 0 |
| visible_04 | Evidence recovery decision | 1 → 1 | 1 → 0 |
| visible_05 | Destination/reference binding | 1 → 0 | 1 → 2 |
| visible_06 | Current default: reserved review budget and independent review history | 0 → 1 | 0 → 1 |
| visible_07_thinking | Same current runtime; both arms request 1024 reasoning tokens | 0 → 0 | 0 → 1* |

The current default batch improves from 0/6 to 2/6 while total model requests change
from 70 to 65. Image inspection supports the newly localized warehouse and gray
vehicle. This is small-sample checkpoint evidence, not established population-wide
or end-to-end navigation improvement; earlier prototypes also regress. Different
versions and reasoning settings must not be pooled as one policy estimate.

*The reasoning-batch CityNav metric success is in a neighboring courtyard, 18.20 m
from GT. It passes the 20 m threshold but is not an exact semantic localization
gain. Default reasoning remains disabled; these different samples do not establish
the causal benefit or harm of reasoning.

See `out/16_adaptive_agent/20261027/audit_report/index.html` for every paired image,
`manual_audit.json` for inspected semantic limitations, and each batch's immutable
`plan.json`, requests, trace and `evaluation.json` for source and evaluation evidence.
LocateAnything was covered by interface/dispatch regression checks; these online
comparisons used the real CUDA DINO/SAM3 and Jev services, without LocateAnything.

## Expanded fixed-version validation, 2026-10-27

`visible_08_expanded` adds 20 previously unused cases (10 per dataset), with all
40 paired trajectories completed. The runtime hashes match `visible_06`; decision,
fusion and review remain Qwen 3.5, optional visual evidence remains Qwen 3.8, and
reasoning is disabled. Sampling and the analysis plan were frozen before execution.
No failed case was replaced and no runtime tuning occurred during this batch.

| Dataset | Before | After | Paired wins / losses | Exact McNemar p | Model requests before / after |
| --- | --- | --- | --- | --- | --- |
| AVDN | 2/10 | 2/10 | 0 / 0 | 1.00 | 174 / 132 |
| CityNav | 1/10 | 3/10 | 2 / 0 | 0.50 | 97 / 215 |

The new sample does not establish a reliable localization improvement. Both new
CityNav metric successes pass the 20 m threshold while selecting a neighboring
vehicle or roof region; neither provides verified exact-identity improvement.
AVDN's confirmed-but-metric-failed finals increase from 1 to 7 despite unchanged
success count. CityNav's request cost more than doubles. More confirmations and
more evidence calls therefore cannot be treated as accuracy improvements.

Wilson 95% intervals are 5.7–51.0% for both AVDN rates, and 1.8–40.4% before versus
10.8–60.3% after for CityNav. CityNav's paired bootstrap difference interval is
0–50 percentage points. AVDN's observed pair differences are all zero, producing
a degenerate bootstrap interval; that is not evidence of equivalence or certainty.

Pooling only the identical default configuration in `visible_06` gives 13 cases
per dataset: AVDN 2/13 → 3/13 and CityNav 1/13 → 4/13. This is supplementary
descriptive evidence because those earlier six cases were already inspected.
Other runtime versions and the reasoning batch remain separate. Across all seven
batches there are 56 distinct cases and 112 trajectories, not one fixed-policy
estimate. The current eligible historical pool has only one unused CityNav
visible-checkpoint case left; a larger fresh CityNav evaluation needs more source
episodes or a separately defined original-pose experiment.

The expanded batch records 618 model requests, zero model API errors, and one
invalid visual-evidence response (retained as a tool error). Grounding `partial`
statuses are reported separately from API failures. An initial service-preflight
failure occurred before any trajectory attempt; its logs are preserved. The later
successful run completed once, and its credential broker and owned GPU services
have exited. Five dedicated paired-statistics tests pass; runtime code is unchanged
from the earlier 76-check validation.

See `visible_08_expanded/paired_statistics.html` and
`expanded_audit_report/index.html` under the stage root for statistics and all 40
paired images. `visible_08_expanded/expanded_summary.json` records integrity checks,
tool outcomes, limitations and cleanup. These remain GT-conditioned checkpoint
results, not full navigation success rates.
