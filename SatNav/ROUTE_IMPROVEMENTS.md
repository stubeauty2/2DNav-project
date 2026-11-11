# Language route guidance experiments

## Current outcome

Independent validation completed at **54/100**, seen **26/50 (52%)**, unseen
**28/50 (56%)**, meeting the approximately 50% project-adapter target on this
sample. The selected configuration is baseline prompts, route guidance, route
stop gate, partial route and search prior, with 18 rounds / 72 requests, plus
decision, max review, flash initialization/planning and SAM3. See `RUNNING.md`
for exact commands. All hundred selected cases are included; the sample excludes
250 previously tested trajectories. Overall Wilson 95% interval is 44.3%–63.4%.
This is not native SatSim protocol parity. The preceding independent 39/100 used
a different cohort, so the 15-point difference is not a paired causal estimate.

The audit verified 100 exact public initialization inputs, 100 text-only planning
requests, 100 navigation outcomes, four source snapshots, 974 navigation requests
and 1,483 observed-image asset occurrences, with no errors. Current runtime matches
the evaluated freeze. All four workers exited zero. One semantic-plan parse failure
remained in the denominator and used fallback. No GT coordinates were used for
initialization, planning or navigation; search priors use training language statistics.

By category: numeric 25/32, partial ray 14/27, other 15/41. Remaining failures:
33 incorrect confirmed stops, 12 step limits and one give-up. The method still has
target-confusion and search failures; reaching this sample's goal does not remove
those limitations. Authoritative artifacts are `validation_summary.json`,
`inference_integrity_audit.json`, `completion_audit.json` and `failure_categories.json`
under `out/satnav/validation100_search_prior_20261111`. No validation workers remain
active, and future independent sampling must exclude the now-observed 350 trajectories.

## Historical development record

The initially selected configuration was `--prompt-variant baseline --route-guidance`.
`--route-guard` is an additional experimental motion constraint and is not part
of the selected configuration. Existing AVDN callers retain their defaults.

Complete distances and turns extracted from the public instruction provide a
predicted stopping area. The agent receives its current relative bearing and
remaining distance every round. After visual target confirmation, a predicted
stop inside the current view can determine the physical stopping pose. Missing
distances or angles disable this geometry; they are not filled from annotations.
This is an additional language-derived navigation prior, not merely a change to
initial heading. It preserves the exploration, grounding, and review loop.

## Development results

| Sample | Baseline | Selected guidance | Gained / lost |
| --- | --- | --- | --- |
| Original 10 | 2/10 | 4/10 | 2 / 0 |
| Previously used extension 20 | 6/20 | 9/20 | 3 / 0 |
| Combined development 30 | 8/30 (26.7%) | 13/30 (43.3%) | 5 / 0 |

Complete numeric language accounts for 13 of these 30 episodes: baseline 4/13,
guidance 8/13. On the other 17 episodes the result is 4/17 versus 5/17, despite
unchanged handling; this difference must not be attributed to numeric guidance.
API outputs can vary between runs. All 30 have previously informed development,
so these numbers are not an independent final evaluation.

## Frozen independent expansion

Before starting navigation, 60 distinct new trajectories were selected uniformly
within each split (30 seen, 30 unseen), excluding all 30 development trajectories.
One instruction was selected per trajectory. Both arms use the same saved
initialization outputs; all 60 initialized successfully. Selection and runtime
hashes, configuration, API journals, and source snapshots are retained.

- Improved: `out/satnav/scale60_20261109`
- Baseline: `out/satnav/scale60_baseline_20261109`
- Selection decision: improved directory `frozen_selection.json`

For throughput, indices 11–60 are assigned to five disjoint ten-episode shards
per arm under `out/satnav/scale60_shards_20261109`. The original sequential
processes supply indices 1–10 only. This partition was fixed before any of the
sharded episodes began navigation; `partition.json` records the exact IDs.
Any sequential work beyond index 10 is excluded by this predetermined partition,
regardless of its outcome. The selected algorithm and initialization are unchanged.

The initial expansion attempt was invalidated in full after widespread proxy/TLS
failures, including failures in the original sequential processes. Its files are
retained. A threaded startup also crashed in Windows OpenBLAS before producing
trajectories. The replacement under `out/satnav/scale60_direct_20261109` uses six
independent workers, single-thread BLAS, and process-local `NO_PROXY` for the
same model endpoint. It includes all 60 episodes in both arms, with identical
initialization and unchanged navigation code; no successful original outcomes
are selectively carried over. See its `partition.json` for provenance.

The `scale60_direct_20261109` attempt also finished with pervasive transport
failures: baseline 57/60 and guidance 56/60 ended with `interface_error`.
Its raw evaluator output is not a valid method-performance comparison. The
previous GET/unauthenticated checks were insufficient to establish stable
authenticated multimodal inference over a complete batch.

On 2026-11-10, actual authenticated image requests succeeded for both models.
`scale60_verified_20261110` repeats the complete fixed comparison with four
independent workers. A new batch circuit breaker stops after an API-related
interface failure and preserves all completed results and journals. Normal
navigation failures do not activate this breaker. Eight runtime/geometry/transport
tests passed. This attempt was stopped after the same failure recurred.

### Upload timeout diagnosis (2026-11-10)

An actual failed two-image request reproduced `Connection aborted: The write
operation timed out` twice in three probes with the original 20-second connect
timeout. The requests/urllib3 connection timeout also affects this upload phase.
With a 90-second connect/upload timeout and 180-second response deadline, all
three identical-image probes succeeded (27.80, 28.84, 11.25 seconds total).
Evidence: `out/satnav/upload_timeout_probe_20261110.json`. This addresses a
reproduced failure mode, not a guarantee against all network failures.

SatNav now explicitly opts into these timeouts, recorded in its run manifest.
AVDN defaults remain 20/90 seconds. Nine tests cover the opt-in transport setting,
batch breaker, geometry and navigation control flow. The fixed 60-episode paired
comparison completed under `out/satnav/scale60_upload90_20261110`, with four
workers, the same frozen policy and inputs, and the breaker enabled for both arms.
All prior attempts and their costs are retained separately. Do not combine their
successful episodes selectively with this replacement attempt.

## Completed independent expansion (2026-11-10)

All 120 episode runs completed; all 12 workers exited successfully. No episode
terminated for an interface error. Inputs/initialization and runtime hashes match
across the paired runs. The 60 trajectories are distinct and exclude the prior 30.

| Split | Baseline | Guidance |
| --- | --- | --- |
| Seen | 6/30 (20.0%) | 6/30 (20.0%) |
| Unseen | 7/30 (23.3%) | 8/30 (26.7%) |
| Total | 13/60 (21.7%) | 14/60 (23.3%) |

There are six gains and five losses (exact paired McNemar p=1.0). The net gain of
one episode does not establish a reliable improvement. Complete numeric routes
account for 19 episodes (8 to 11 successes); the other 41 regress from 5 to 3.
Of the latter 41, 26 confirmed STOPs never visited the 30m region and 12 reached
the step limit without visiting it. Final-target-only search is insufficient for
the region-conditioned intermediate-landmark instructions.

Evidence: expansion `paired_summary.json`, `sample_integrity.json`, and
`seen_failure_audit.json`. Visual review distinguishes a visible but incorrectly
positioned solar-panel roof in London from a different pool location in TheBayArea.

The user has now set an approximately 50% SR target. This has NOT been achieved.
The 60-episode set is henceforth diagnostic/development data for new variants;
subsequent reuse must not be called a fresh independent test.

## Stop-gate follow-up

`--route-guidance --route-stop-gate` keeps an early visual match provisional when
the complete language stopping area is outside the current view. It clears the
final result, moves at most 70% of a view footprint toward that area, and returns
to the ordinary grounding/review loop. A fresh visual confirmation is required;
no STOP is issued merely from the language coordinate. The motion guard remains off.
The feature is opt-in. Thirty-two tests pass, including rejection after movement
and repeated bounded approaches. Original-ten online development run:
`out/satnav/stop_gate_dev_20261110`: completed 6/10 (seen 4/5, unseen 2/5),
versus baseline 2/10 (four gains, no losses), and numeric guidance 4/10
(three gains, one loss). The fixed additional twenty completed 9/20 in three
independent processes at `out/satnav/stop_gate_extension_20261110`.
This fix alone cannot address the
dominant nonnumeric route-following failures and is not evidence of reaching 50%.

## Semantic-stage follow-up toward 50%

`SatNav.semantic_plan` extracts ordered forward legs, intermediate/final landmark
descriptions, explicit distances, and before/after-turn image regions using only
the public instruction. Unknown distances/angles remain null. The original ten
development instructions all produced validated plans; their request journals
are in `out/satnav/semantic_dev_20261110/semantic_plan_requests`.

`--semantic-route --route-guidance --route-stop-gate` uses semantic stages for
episodes without complete numeric geometry and preserves numeric guidance for
the others. Current-stage landmarks drive the automatic grounding query; stage
changes invalidate the observation-tool cache. Forward exploration uses overlapping
views along the current leg heading. Confirmed intermediate landmarks satisfy
image-region/explicit-distance conditions before the next turn. Final alignment
requires a new visual confirmation after movement. Ground-truth annotations are
never inputs to this controller. Region centers are explicit policy approximations,
not recovered exact annotations. The round/request budgets remain 12/48.

The shared AVDN loop has overridable target/cache/confirmation hooks whose defaults
preserve its previous behavior. Thirty-eight tests pass, including stage-specific
query switching, auxiliary positioning rather than center arrival, turn geometry,
and final reobservation. The initial semantic run at `out/satnav/semantic_dev_20261110` was interrupted
because the generic system prompt contradicted the ordered stage controller. Its
partial outcomes are retained but are not used for performance selection. All ten
are rerun from scratch at `out/satnav/semantic_dev_v2_20261110`, with explicit
stage decision/review prompts and matching context. The completed corrected ten scored 7/10.
The twenty-case extension has instruction-only plans prepared (19 valid, one
parse failure retained for ordinary-policy fallback); navigation subsequently completed 8/20. No failed parser case is removed from the denominator.

Success here means confirmed STOP within 30 m of the evaluator-only goal. The
continuous-action project adapter is not native SatSim benchmark parity. A saved
final image is not an additional visual review, and endpoint success alone does
not prove correct target identity. Ground truth is used only for offline scoring.

## Completed stop-gate expansion and frozen validation

The fixed twenty-case expansion completed 9/20 (seen 5/10, unseen 4/10),
versus baseline 6/20 (three gains, no losses) and earlier route guidance 9/20
(one gain, one loss). All workers exited successfully; no terminal interface
failures. Combining the original ten gives 15/30 = 50% (seen 9/15, unseen 6/15),
versus original baseline 8/30. Numeric routes: 10/13; others: 5/17. These are
development results, not an independent estimate.

The stop-gate variant is selected and frozen before navigation on a new sixty
(30 seen / 30 unseen), excluding every previous ninety trajectory. The completed independent run is at `out/satnav/validation60_frozen_20261110`; its
`validation_plan.json` commits the variant and full-denominator metric before
any navigation outcome. This is independent SR validation, not a new paired
baseline comparison. No validation-driven tuning will occur during this run.

The corrected semantic-stage ten finished 7/10 (seen 4/5, unseen 3/5),
versus stop-gate 6/10: one gain (Orlando 997), no losses. The original twenty
semantic extension completed 8/20 as a separate development comparison, with
one failed plan retained for ordinary-policy fallback. The independent sixty
remains the frozen stop-gate variant; semantic results will not switch it midway.

## Independent sixty completed

The frozen stop-gate method scored **24/60 (40%)**, seen **15/30 (50%)**,
unseen **9/30 (30%)**. Overall Wilson 95% interval: 28.6%-52.6%. The development
50% did not persist as the independent point estimate; do not claim stable 50%
performance or a matched improvement over the older, different sixty.

Complete numeric routes: **22/23 (95.7%)**. Other routes: **2/37 (5.4%)**.
The one numeric failure exhausted steps after visiting within 30m. Other-route
failures: 22 wrong confirmed stops, 12 step limits, one request limit. All 274
grounding calls returned ok, which does not certify target identity. There were
no terminal interface failures. All sixty remain in the denominator.

`validation_summary.json`, `failure_audit.json`, and `sample_integrity.json`
under `out/satnav/validation60_frozen_20261110` hold the complete evidence.
Execution-only handoffs repartitioned an unfinished shard after its active
episode completed; assignments were fixed before delegated outcomes. The
original worker nonzero exit is documented in `execution_handoff.json`, while
all authoritative episode results are complete. Code/prompts/budgets stayed
frozen throughout. Ground truth was used only for offline scoring and audits.

A development-only semantic failure audit is saved at
`semantic_dev_v2_20261110/public_geometry_failure_audit.json`. In LosAngeles 709,
the first verified landmark center is 9.45m LEFT of the original forward axis,
although the first cue requests lower-right after forward flight. Arbitrary
lateral camera alignment can manufacture a requested image region instead of
checking consistency with the forward course. Subsequent region-center yaw is
also approximate. This is a diagnostic hypothesis, not an implemented or tested
fix; no frozen validation code was changed.

## Final semantic development comparison

The corrected semantic ten scored 7/10; the fixed extension scored 8/20
(seen 4/10, unseen 4/10), below stop-gate 9/20. Combined: 15/30 for both
variants, with three gains and three losses relative to stop-gate.
On the sixteen episodes where semantic policy actually activates, it scored
4/16 versus stop-gate 5/16. On thirteen unchanged numeric episodes, outcomes
were 11/13 versus 10/13; this is run variation, not semantic-policy benefit.
The one parse-failure fallback remained a failure under both. Therefore this
variant is **not promoted**. No additional validation run is justified by its
development evidence. See `out/satnav/semantic_development30_summary.json`.

The practical retained option is `--route-guidance --route-stop-gate`, with
`--semantic-route` and `--route-guard` off. The independent score remains 40%,
so stable 50% has not been demonstrated. All inference used public instructions,
starting pose and observed images; evaluator ground truth stayed offline.

## Geometry-constrained semantic follow-up (completed, not promoted)

The 40% independent result leaves the requested approximately 50% goal open.
The next working-tree semantic variant rejects landmarks on the wrong side of
the public phase forward axis, aligns arrival positions along that axis rather
than moving laterally to manufacture a quadrant, and bounds local search after
an explicit distance. A visible-only final cue no longer implies flying to an
object center. Geometry tests plus existing checks total 41 passing tests; the
local-search regression suite was rerun after the final sweep adjustment.

`out/satnav/geometry_dev16_20261110` freezes every policy-active episode from
the development thirty: eight seen and eight unseen. Selection uses only plan
validity and numeric-route availability, never outcomes or GT. Historical
controls are 5/16 for stop-gate and 4/16 for semantic v2; their original
results/provenance are retained. Three independent processes test the modified
policy with the same 12/48 budgets. The prespecified development gate is at
least 8/16 before considering another independent validation. This is not yet
evidence of improved online success, and all prior results remain unchanged.

Completed result: 3/16 (seen 2/8, unseen 1/8), versus stop-gate 5/16.
There was one gain and three losses. The 8/16 gate was not met, so this
experimental semantic variant is not selected for expanded validation.

## Search-bearing and exploration-stride development follow-ups

`bearing_prior_dev10_20261110` changes only unresolved initial search bearings
using frozen statistics from training instruction text. It scored 1/10 against
the historical 2/10 control (zero gains, one loss). Better bearing alignment
did not establish better navigation; this option is not promoted. Imputed
distances and turns never become exact stopping coordinates.

`prior_bounded_dev17_20261110` tests the same search prior with exploration
hops capped at 50m on all seventeen nonnumeric development episodes. Its
prespecified gate is 9/17 versus the historical 5/17. Results must be complete
before selection; pending episodes are not failures. The retained independent
validation remains 24/60, and neither follow-up changes that frozen result.

The bounded seventeen completed at 1/17 against 5/17 (zero gains, four losses).
It is not promoted. Direction alignment alone was insufficient, and indiscriminate
shorter strides did not solve false landmark matches.

## Partial numeric final-ray experiments, 2026-11-11

`--partial-route` is opt-in and requires route guidance; it excludes semantic
stages. An explicit numeric prefix followed by a known final heading but unknown
final distance defines a search ray. The adapter approaches its anchor with
overlapping observations, then scans forward. Identity still requires the shared
visual tool/review loop; no unknown distance or turn is filled statistically.
Complete numeric routes and unsupported cases retain their previous behavior.
The 30m landmark-to-pose constraint comes from the paper's public Appendix C.2,
not episode goal coordinates. Observed bounding-box centers are approximate,
so the constraint does not itself guarantee correct object identity or success.

All six eligible episodes from the original development thirty were frozen in
`partial_ray_dev6_20261111`. Version one scored 3/6 versus 3/6, gaining two and
losing two, and did not pass the 4/6 gate. It incorrectly left a multi-candidate
view after rejecting one object and accepted an opposite-side roof at the ray
anchor. Version two checks the final instruction side and gives same-view
candidate reselection opportunities. Thirteen geometry/control-flow tests
passed at that stage. The seven eligible episodes from the older development
sixty were prepared before selection; extension requires the six-episode gate
first. No eligible-only score is an overall SatNav SR.

Version two also completed at 3/6 (two gains, two losses). Version three adds
geometric consistency to tool proposal selection, avoids repeatedly reviewing
premature candidates before the known prefix, and explicitly names the FINAL
target during final review. It scored **6/6 versus 3/6, three gains and no losses**.
This is development evidence, not an independent overall success rate.
The runtime hash is
`199054127370ed8653eda448bf74518a97c3d57dffdb0497a17f7cecdd894d00`.
The prepared seven-episode extension was launched with exactly that source
hash and unchanged budgets/models; its gate remains at least 5/7. Fourteen
targeted geometry/control-flow tests passed across the final checks.

A fresh all-category cohort is frozen at
`out/satnav/validation100_partial_route_20261111`: fifty seen and fifty unseen,
excluding all previous 150 unique trajectories, with no numeric-route filtering.
It remains unstarted until the extension gate passes. All one hundred must be
reported, including initialization/interface failures, without mid-run tuning.

The frozen seven-episode extension completed at **1/7**, identical to its
historical control (zero gains, zero losses), so the 5/7 gate failed and the
one-hundred-episode validation was NOT launched. The final-ray option remains
experimental and is not promoted over the retained 40% overall method.
Development-only diagnosis shows the extracted rays are geometrically accurate,
but some trajectories spend remaining rounds repeatedly reviewing rejected
candidates instead of advancing, while some visually similar objects produce
premature stops. Object-center projection also discards the final vertical
image-region condition. These are hypotheses for subsequent development work,
not fixes already shown to generalize. All thirteen ray-subset results must
remain distinguished from an overall, all-category success estimate.

## Review recovery and independent expansion

The recovery variant advances after two unsuccessful reviews without movement.
It also increases the explicitly recorded budget to 18 rounds / 72 requests.
`partial_ray_recovery13_20261111` completed at **9/13 versus 7/13**, two gains
and no losses: the original six remained 6/6; the seven improved from 1/7 to
3/7. This is a combined policy/budget result, not an isolated prompt ablation.
Its frozen runtime SHA is
`3e358e1335f046f1e5230b36dce6bdc0e3d6748d0a770e5497ae8b079b130fd4`.

The earlier self-imposed subset gates (10/13 overall and 5/7 extension) were
NOT met. Before any validation prediction, `selection_amendment.json` recorded
the decision to independently test this observed improvement rather than keep
tuning the same thirteen cases. This is a transparent protocol amendment, not
a claim that the gates passed. The frozen all-category validation hundred was
then launched from the recovery source snapshot with 18/72 budgets. Its samples,
exclusions, all-episode denominator, and no-mid-run-tuning rule are unchanged.
No overall success claim is available until that run completes.

A separate public-image-only stopping-pose replay on all ten visually confirmed
v3 cases regressed from 7/10 to 2/10 (zero gains, five losses), so asking the
model again to choose the stop is rejected. A fixed 3x3-region-center diagnostic
scored 8/10 versus 7/10 but lost one previous success; it is not online navigation
evidence. `--ray-region-stop` exists only as an unvalidated opt-in experiment and
is excluded from validation100, which executes the earlier source snapshot.
Sixteen targeted geometry/control tests passed for the current working tree.

## Completed independent hundred: 39%, goal remains open

`validation100_partial_route_20261111` completed every selected trajectory:
**39/100**, seen **20/50**, unseen **19/50**. Overall Wilson 95% interval is
30.0%–48.8%. All four workers exited normally; all hundred initializations
succeeded and no terminal interface failures occurred. The result does not
demonstrate an overall improvement over the previous independent 24/60 result;
the cohorts differ, so this is not a paired comparison.

By public-instruction geometry: complete numeric 28/33, final-ray 9/19, other
2/48. The other category contributes 46 failures (34 incorrect confirmed stops,
12 step limits); only one of those failures ever visited within the success
radius. This makes missing-route search and target identity the main remaining
bottleneck. Partial-ray failures include six incorrect confirmed stops, three
step limits and one request limit. All five complete-numeric failures hit the
step limit; two had visited within the success radius.

The completed audit verifies all 100 exact public initialization inputs, four
source manifests, 1,093 navigation request contexts, and 1,541 image-asset
occurrences, with no audit errors. These are occurrences, not unique images.
All results remain in the denominator. This run is not promoted as achieving
the approximately 50% objective. Any future independent cohort must exclude
the now-observed 250 trajectories.

Post-run diagnosis found one explicit sign error: Orlando 1059 said RIGHT
120 degrees while initialization extracted -120. The current parser now repairs
only an unambiguous single-turn sign contradiction with matching magnitude,
using public instruction text alone. On a replay of all hundred initialization
responses, exactly one changed; its offline language-endpoint error fell from
277.47m to 0.39m. This is a parser regression check, NOT a new online success.
Unknown angles, ambiguous multi-turn text and magnitude conflicts are not filled.

Of the 48 other cases, 45 had no resolved initial final-goal bearing; the three
resolved bearings were within 90 degrees in the offline audit. Simply adding a
bearing prior already failed on development (1/10), so the next search hypothesis
must address where/how to explore, not repeat that experiment. A possible next
development experiment is an explicitly uncertain search region derived from
the frozen training-text distance/turn priors, followed by visual grounding.
Such a prior must never enter `numeric_route`, masquerade as GT, or directly
authorize STOP at an imputed coordinate.

## Search-region development expansion

The uncertain search-region policy is now implemented as `--search-prior`.
It preserves explicit instruction geometry and imputes missing values from
frozen training-instruction statistics only. It approaches the estimated region,
then searches overlapping views. A confirmed target stops at its observed
grounded center, never at the statistical search center. Complete numeric routes
and supported final rays retain their existing policies. Unsupported semantic
plans fall back and remain in the denominator.

The paired ten-case experiment `search_region_dev10_20261111` completed at
**4/10 versus 2/10**, with two gains and no losses, at equal 18-round / 72-request
budgets. It passed the preregistered expansion gate (at least four successes and
two net paired gains). The frozen runtime hash is
`ada23a24ceee23b1b9649ceddfcb4b3b3252a8de16559267cb812c85b3494abd`.

`search_region_development48_20261111` expands to all 48 incomplete-route cases
from the completed hundred, selected by route category rather than outcome.
Their historical paired baseline is 2/48 with identical public inputs and
initialization labels. This is development evaluation, not independent overall
SR. The recorded gate for further fresh all-category testing is at least 13/48.
No source changes are permitted during this frozen run. Public-text semantic
planning adds one planning call per episode; failed plans are not excluded.

Completed expansion: **11/48 (22.9%) versus 2/48 (4.2%)**, ten gains and one
loss. Seen: 8/22 versus 1/22; unseen: 3/26 versus 1/26. All four workers exited
zero. The 13/48 gate was not met, so this result does not trigger another fresh
all-category evaluation. The development improvement does not establish 50%
overall SR. Of 37 failures, 31 were incorrect confirmed stops and six reached
the step limit; five failures had visited within the success radius.

Semantic plans were valid in 41 cases and had parse errors in seven; all seven
remain included through fallback. The audit verified unchanged historical
initializations for all 48, four frozen source manifests, 48 public-text planning
requests, 433 navigation requests and 579 observed-image asset occurrences,
with no audit errors. See `comparison.json`, `failure_analysis.json` and
`inference_integrity_audit.json` in the experiment directory. Further development
should diagnose the seven plan errors and incorrect target confirmations before
repeating a fresh representative evaluation.

## Semantic-parser recovery: three additional paired successes

The seven parse failures were three `lower_middle` region aliases, two bare leg
arrays, and two legitimate qualitative turns without numeric angle/alignment.
`parse_plan` normalizes the first two syntactic variants. The opt-in
`allow_qualitative` path preserves unknown geometry for uncertain search only;
the exact semantic-navigation validator remains strict. No geometric value is
filled by parsing. Replay recovered all seven while preserving all 41 previously
valid plans exactly.

`search_parse_recovery7_20261111` retested every parse-failure case, with identical
public inputs, initialization, training priors and 18/72 budgets. It completed
**3/7 versus 0/7**, three gains and no losses; both workers exited zero. This
passes its recorded gate of at least two net gains and zero losses for broader
confirmation. It is not an independent overall evaluation, and combining these
results with the earlier 48 would only be a mixed-version diagnostic, not a new
48-case run. All 42 SatNav tests passed after the parser change.

Visual inspection of Auckland 1412's observed `r05-main` also identified a real
review failure: a green hipped residential roof was confirmed as a lined tennis
court. Review improvements should be evaluated on both existing correct and
incorrect confirmations to measure false-confirmation reduction and success
retention; rejecting more candidates alone is not evidence of better SR.

## Full parser confirmation: 12/48, expansion gate not met

`search_parse_confirmation48_20261111` reran all 48 under one frozen runtime
(`7e0c3e4360cd53a0ef07c513168810d71cf255d5d1406bae7aef60dd8fc88fc9`),
identical public inputs/initializations, cached semantic responses reparsed with
the fix, and 18/72 budgets. All four workers completed normally. Result:
**12/48 (25.0%) versus 11/48**, three gains and two losses. Seen stayed 8/22;
unseen rose from 3/26 to 4/26. The seven recovered-plan cases scored 2/7 in
this rerun versus 3/7 in the preceding small experiment. Thus gains cannot be
added across runs; inference/navigation outcomes vary even with unchanged inputs.
There were 31 incorrect confirmed stops and five step limits. The 13/48 gate
was not met; no independent overall success improvement is claimed.

A fresh all-category cohort at `validation100_search_prior_20261111` has been
preselected (50 each split, excluding all 250 previously observed trajectories)
but has NOT been initialized or navigated. Its launcher enforces the failed
development gate and must not be bypassed. It remains an unobserved prepared
cohort, not 100 more completed trajectories. A review replay cohort contains all
42 historical final confirmations (11 successes, 31 failures) from the original
search-region development run, with a prospective screening criterion of retaining
at least 10 successes and rejecting at least six failed candidates. No review
replay model calls have yet been made, and its screening is not navigation SR.

## Review diagnostics and model-only online experiment

The 42-case paired replay is complete: the original flash reviewer retained all
11 historical successes, rejected two historical failures and marked one uncertain.
Adding a neutral physical-structure comparison prompt retained 11, rejected zero
and marked one uncertain. All 84 requests used identical contexts and image
assets to their historical source. The added prompt is rejected and has NOT
been added to runtime navigation.

A model-only replay with `qwen3.8-max` and the original prompt retained all 11,
rejected four failed candidates and marked three uncertain, with no API/parse
errors. All 42 contexts/assets/prompts matched their sources. These are acceptance
diagnostics on historical observations, not new navigation SR or proof of refined
box correctness. The original gate of six explicit rejections was NOT met.

Before online inference, `search_max_review48_20261111/selection_amendment.json`
recorded the decision to test the actual mechanism: both reject and uncertain
prevent immediate STOP, so seven withheld false confirmations with all eleven
successful candidates retained merit a full development test. This is an explicit
screening amendment, not a retroactive passed gate. That 48-case run uses the
same frozen runtime and inputs as parser confirmation, with only the reviewer
changed to max. Decision remains `qwen3.7-plus`; budgets remain 18/72. It compares
against 12/48 and still requires at least 13/48 before independent validation.

Completed max-review development: **14/48 versus 12/48**, four gains and two
losses; seen 9/22 versus 8/22, unseen 5/26 versus 4/26. All workers exited zero.
Five successful trajectories continued after a rejected/uncertain review. There
were still 29 incorrect confirmed stops and five step limits. The online 13/48
gate passed; this does not itself demonstrate overall 50% SR.

The preselected independent `validation100_search_prior_20261111` was then launched
under the same frozen source, max reviewer, plus decision model and 18/72 budgets.
`selection_update.json` records the selected development configuration before any
new-cohort inference; the original public sample hash and 250 exclusions remain
unchanged. Initialization/semantic planning use flash. All-category validation also
tests the reviewer change on numeric/final-ray routes, which were absent from the
48-case development subset. No tuning occurred during this run. All 100 outcomes
and the completion audit now pass; see the current outcome at the top of this file.
