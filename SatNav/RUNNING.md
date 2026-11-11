# Running the SatNav adapter

Use the current `navAgent-SatNav` branch from the repository root. The downloaded
dataset resides in `datasets/SatNav`. The adapter imports the existing AVDN
navigation loop, geometry, grounding clients, API transport and contracts.

## Current validated configuration: 54/100

`out/satnav/validation100_search_prior_20261111` is COMPLETE: overall **54/100**,
seen **26/50**, unseen **28/50**. All workers exited normally. The hundred were
selected before inference, excluded 250 previously observed trajectories, and
included every language/route category. The completion and inference audits pass.
This meets the approximately 50% project-adapter target on this independent
sample; it is not native SatSim parity or a guarantee of the full-dataset SR.
Overall Wilson 95% interval: 44.3%–63.4%. Historical experiment sections below
describe earlier configurations; use this section for the selected configuration.

For a fresh prepared directory containing `inputs.jsonl` (and evaluator-only truth
for later scoring), use the existing environment and SAM3 service described below:

```powershell
$runRoot = 'out/satnav/my_fresh_run'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
Copy-Item out/satnav/training_text_prior_20261110.json "$runRoot/frozen_training_text_prior.json"
python -m SatNav.cli initialize --root $runRoot --model qwen3.8-flash
python -m SatNav.semantic_plan --root $runRoot --allow-qualitative
python -m SatNav.cli run --root $runRoot --model qwen3.8-max --prompt-variant baseline --route-guidance --route-stop-gate --partial-route --search-prior --max-rounds 18 --max-requests 72
python -m SatNav.cli evaluate --root $runRoot
```

The run's `--model` selects the reviewer; decision requests remain `qwen3.7-plus`.
Initialization and semantic planning use `qwen3.8-flash`. The frozen prior uses
training instruction text statistics, never evaluation goal/path coordinates.
Do not enable the rejected experimental route-guard, semantic-route or
ray-region-stop options. Preserve the completed experiment; use a fresh directory
for new work. Future independent selection must exclude all 350 now-observed
trajectories. The validated Python runtime hash is
`7e0c3e4360cd53a0ef07c513168810d71cf255d5d1406bae7aef60dd8fc88fc9`.

## Environment

The navigation environment needs numpy, Pillow, rasterio, requests and jsonschema.
The existing local `2DNav` conda environment provides these dependencies.
Set `DASHSCOPE_API_KEY` (or `API_KEY`) in the launching process. No credential is
stored in the adapter or request journals. Set `SAM3_URL=http://127.0.0.1:8765`.

Start the existing SAM3 service in its GPU conda environment:

```powershell
C:/Users/admin/anaconda3/envs/sam3/python.exe -m AVDN.tools.sam3_service --checkpoint C:/Levir/2DNav/models/sam3/sam3.pt --port 8765 --output out/satnav/sam3_service_cache
```

## Commands

```powershell
python -m SatNav.cli prepare --root out/satnav/new_run --count 5
python -m SatNav.cli initialize --root out/satnav/new_run
python -m SatNav.cli run --root out/satnav/new_run --grounder sam3
python -m SatNav.cli evaluate --root out/satnav/new_run
python -m SatNav.audit_views --root out/satnav/new_run
```

Preparation freezes five distinct trajectories per evaluation split, then chooses
one instruction per trajectory. Existing samples cannot be overwritten. Initialize
and run keep API journals; changed prompts/code require a new experiment directory.
An interrupted pending API request is not automatically resent. All initialization
outcomes remain in `initialization_results.jsonl`; valid outputs go to `episodes.jsonl`.

## Inference and evaluation separation

The policy receives instruction, start pose and local observations. Named landmark
coordinates are empty. The private `evaluator_only.jsonl` contains endpoint truth;
only evaluation and offline visual audit read it. No reference path or waypoint is
used for initialization, exploration or stopping. The evaluator separately reports
direction error, physical stop distance, and success within 30 m after STOP.

The current adapter uses the project's continuous movement actions and 768px views
with a 100m ground footprint. It is not an implementation of native SatSim's
448px observations, 10m forward increments and 15-degree turns. Comparisons must
identify this protocol difference rather than claim official action parity.

Initialization extracts relative turns; code integrates headings and known lengths.
If lengths are unknown but every heading lies inside a cone narrower than 180deg,
the midpoint supplies a direction prior. Its geometric bound assumes the extracted
headings and straight positive-length segments are correct. Unknown visual turns
remain unresolved. The base loop then uses the original initial heading as an
exploration fallback, explicitly distinguished from a final-goal direction.

Target confirmation does not directly count as success. A final model call chooses
a stopping pose from the visible target and language; the adapter moves and stops.
The stopping image is saved, but this version does not perform another model review
after that move. A matching object can be visible while the stop is outside 30m.

Offline contact sheets include GT stop-area crops and must NEVER become policy
inputs. These crops do not constitute official target bounding-box annotations.
Visual identity and first target visibility require explicit review of the frames.

Online models: initialization/review `qwen3.8-flash`; decisions (including stop pose)
`qwen3.7-plus`, following the existing model client. Thinking is disabled. Pricing
records reuse the existing project's rate table; provider billing is authoritative.

## Verification

## Prompt adaptation experiments

`run --prompt-variant baseline` preserves the previous prompts and is the default.
`--prompt-variant satnav` selects the full dataset-specific exploration/review/stop
guidance; `--prompt-variant exploration` isolates exploration guidance with the
original review prompt. Both alternatives are experimental. Full adaptation
regressed on the paired 20-episode extension (2/20 vs baseline 6/20); do not treat
its dataset-specific wording as evidence of superiority. Prompt choice is stored
in the manifest and the exact text in every API journal. Base AVDN callers keep
their original default prompts. The navigation loops and action schemas are unchanged.

Use `python -m SatNav.compare --baseline BASE_DIR --adapted VARIANT_DIR --output REPORT.json`
for paired evaluation; inputs and initialization must match exactly. Pending
episodes are not scored as failures. The original ten episodes are development
data. The additional twenty exclude those trajectories; exploration-only followup
reuses this extension after partial full-prompt results were observed and must be
reported as secondary validation rather than a fresh blind test.

`python -m unittest SatNav.test_runtime -v` checks the real raster adapter and
movement/localization/stop control flow using a scripted model. It is not evidence
of online model performance. Real pilot outputs are under
`out/satnav/pilot_20261109` and `out/satnav/pilot_v2_20261109`.

## Expanded validation, 2026-11-10

Retained experimental configuration (on a fresh prepared and initialized root):

```powershell
python -m SatNav.cli run --root <prepared-root> --route-guidance --route-stop-gate
python -m SatNav.cli evaluate --root <prepared-root>
```

Keep `--route-guard` and `--semantic-route` off for this configuration. Models:
decision/physical stop `qwen3.7-plus`; initialization/review `qwen3.8-flash`;
local SAM3. Configure credentials via the environment, never in source. The
run fixes 12 rounds / 48 requests and 180s response / 90s connect-upload
timeouts. Do not change an existing run manifest to resume under new settings.

Development: 6/10 then 9/20, combined 15/30. Independent sixty excluding the
previous ninety: 24/60, seen 15/30, unseen 9/30. These are continuous-project
actions and a confirmed STOP within 30m, not native SatSim protocol parity.
The semantic-stage option scored 7/10 then 8/20 and is not promoted. Detailed
results and limitations are in `ROUTE_IMPROVEMENTS.md`.

## Completed independent 100-trajectory experiment

`out/satnav/validation100_partial_route_20261111` uses the frozen source under
`out/satnav/partial_ray_recovery13_20261111/shard_0/source`, with route guidance,
stop gate, partial route, 18 rounds and 72 requests. Run/resume it only from that
source directory with its existing manifest, never from a changed working tree.
`out/satnav/run_validation100.py collect` collects progress without starting new
episodes. The parent launcher is single-use. The root's selection amendment
documents that prior subset screening gates were not met; the independent
expansion is exploratory, and all hundred selected trajectories remain in scope.
The current working-tree `--ray-region-stop` option is not part of this run.

Completed result: 39/100, seen 20/50 and unseen 19/50. The run did not establish
the requested approximately 50% SR. No workers remain active for it. Retain the
source snapshot and all outcomes; do not rerun its launcher. The current parser
contains a subsequent public-text turn-sign correction, which is not retroactively
included in that result. Future independent sampling must exclude the previous
250 trajectories.

## Completed search-prior development expansion

The experimental `--search-prior` option requires `--partial-route` and prepared
public-text semantic plans plus `frozen_training_text_prior.json`. It estimates
an uncertain search region; only observed grounded targets authorize STOP.
It is incompatible with `--semantic-route`. Keep `--ray-region-stop` disabled.

The paired ten improved 2/10 to 4/10. The subsequent frozen 48-case development
run `out/satnav/search_region_development48_20261111` completed 11/48 versus 2/48,
with ten gains and one loss, using 18 rounds / 72 requests. Seen was 8/22 and
unseen 3/26. All workers have completed; do not rerun its single-use launcher.
The preregistered 13/48 expansion gate was not met. These are incomplete-route
development cases, not a new independent overall success rate.

For search-prior preparation, use `python -m SatNav.semantic_plan --root <root>
--allow-qualitative` (one shell command). This permits incomplete turns only in
the uncertain search input. Exact semantic navigation keeps strict validation.
The completed parser-recovery development seven improved 0/7 to 3/7, with no
losses; its output is `out/satnav/search_parse_recovery7_20261111`. Do not rerun
the single-use launcher or overwrite historical semantic-plan files. Broader
confirmation must use a fresh directory and frozen updated source.

Full confirmation is now complete at `search_parse_confirmation48_20261111`:
12/48 versus 11/48, three gains and two losses; seen 8/22, unseen 4/26. All
workers completed. The 13/48 gate failed. `validation100_search_prior_20261111`
contains a preselected fresh cohort only, with no inference yet; do not describe
it as a running/completed validation or bypass the gate in its launcher.

Subsequent max-review development completed **14/48 versus 12/48**, passing the
online expansion gate. The explicit `selection_update.json` records this later
selection before inference; `validation100_search_prior_20261111` subsequently
completed at 54/100 with the unchanged preselected hundred. Use
`out/satnav/run_search_validation100.py collect` only to recollect artifacts.
Never rerun its single-use launcher. Runtime
hash remains `7e0c3e4360cd53a0ef07c513168810d71cf255d5d1406bae7aef60dd8fc88fc9`;
review is max, decision plus, initialization/planning flash. Every selected outcome
has been collected and audited; preserve the evaluated source and completed artifacts.
