# 2DNav-project

## Tool-agent visual grounding

Target-event Search uses a bounded Qwen3-VL tool loop. The
navigation environment remains on Python 3.9/Torch 1.11; SAM3/SAM2/DINOv3 run
in a loopback-only service in the separate `sam3` environment.

Start the default SAM3 + DINOv3 service from the repository root:

```powershell
$env:PYTHONPATH = "$PWD/PSC-AVDN/script"
conda run --no-capture-output -n sam3 python -m visual_grounding.service `
  --backend sam3 `
  --sam3-checkpoint E:/Levir/Graduation_Project/sam3/sam3.pt
```

`GET http://127.0.0.1:8765/health` reports the resident backend, exact
weights, thresholds, CUDA memory and prewarm state. Select the A/B backend by
starting a separate run with `--backend grounded-sam2`; only one proposal
backend is resident in a service process.

Run navigation in its normal environment:

```powershell
conda run --no-capture-output -n 2DNav python `
  PSC-AVDN/script/ANDH-Full/Search_Confirmation.py `
  --grounding-backend sam3 `
  --vision-tool-url http://127.0.0.1:8765 `
  --max-tool-calls 4 `
  --qwen-api-timeout 600
```

Service failures or backend mismatches fail during route preflight. A target
event succeeds only when the Qwen tool agent selects a legal SAM candidate in
the scale-5 main view and supplies a validated relation-aware pixel point.
Search JSON records the Qwen point, selected SAM candidate, candidate bbox,
forward-sector pixel evidence, tool trace, and visual artifact paths. For the
final goal only, a separate clean-context Qwen call returns the tight target
bbox; pixel-to-geographic conversion remains local.

The evaluation-only annotation and metric tools are available as modules:

```powershell
$env:PYTHONPATH = "$PWD/PSC-AVDN/script"
python -m visual_grounding.annotation_tool annotate manifest.jsonl --output labels.jsonl
python -m visual_grounding.annotation_tool validate labels.jsonl
python -m visual_grounding.evaluate labels.jsonl predictions.jsonl --split test
```

Validation enforces 60 labeled items, including 20 calibration / 40 held-out
test items and 15 absent-target examples. These labels are not connected to
any training or fine-tuning path.

## Sequential event execution

The ANDH-Full runner executes the flat Parse result in event-list order:

- `TURN` updates heading without changing position.
- `MOVE` advances 60 meters by default without visual search.
- `REACH` and `AVOID` localize only the active event target through Qwen +
  SAM3, then apply the event-specific end pose. Passing, flying over, crossing,
  going through, and travelling along a target are `REACH` events whose
  `relation` field preserves that motion.
- Each visual search renders and submits only the heading-up scale-5 main view.
  SAM3 candidates expose only identity plus the code-computed ±15-degree
  forward-sector intersection and nearest pixel distance; metric and
  cross-scale geometry are not sent to Qwen.
- A failed or timed-out visual search does not move the agent or trigger a
  second scale attempt.
- A failed target event stops the route before any later event can run.

```powershell
python PSC-AVDN/script/ANDH-Full/Search_Confirmation.py `
  --max-routes 1
```

Results are written to
`out/preds/andh_full_sample3/sequential_search_output` by default. Each route
contains ordered `search_steps`, `event_progress`, `completed_event_ids`, and
the executed `physical_trajectory`. A successful `goal=true` REACH also writes
`final_target_grounding`; intermediate events retain the SAM candidate bbox as
evidence while navigation uses Qwen's relation-aware pixel point. The
old confirmation/window execution path and its CLI switches have been removed.
