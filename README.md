# 2DNav-project
# 2DNav-project

## PSC-AVDN baseline with optional SAM3 grounding

The baseline follows the PSC-AVDN Parse/Search/Confirmation inference layout
described in the authors' repository and paper. The original Qwen bbox path is
kept as the default. SAM3 is an optional loopback service so the navigation
environment (Python 3.9/Torch 1.11) does not import the newer visual-model
stack.

Start the service from the repository root in a dedicated visual environment:

```powershell
$env:PYTHONPATH = "$PWD/PSC-AVDN/script"
conda run --no-capture-output -n sam3 python -m visual_grounding.service `
  --sam3-checkpoint E:/Levir/Graduation_Project/sam3/sam3.pt
```

Run the unchanged ANDH-Full pipeline with SAM3 enabled:

```powershell
python PSC-AVDN/script/ANDH-Full/Search_Confirmation.py `
  --grounding-backend sam3 `
  --vision-tool-url http://127.0.0.1:8765 `
  --qwen-url "$env:QWEN_URL" `
  --qwen-model qwen3-vl-plus
```

The service accepts only loopback connections and exposes `GET /health` and
`POST /v1/candidates`. It generates candidates from the scale-5 main view;
Qwen selects one registered target candidate and the legacy PSC code performs
the existing confirmation and pixel-to-geographic conversion. If the service
is unavailable, SAM3 mode records the reason and falls back to the legacy Qwen
path. Use `--grounding-backend legacy` to force the baseline path.

## Offline grounding pairs from recorded predictions

Run `python scripts/export_grounding.py` to export the current full-test Parse
run to `out/grounding/`. Open `out/grounding/index.html` to inspect selected
frames, target descriptions, GT polygons/bboxes, and the original agent results.
`inputs.jsonl` contains only images and text for independent grounding tools;
`selected.jsonl` and `coco.json` contain the corresponding offline labels.

The exporter projects the annotation's `destination` polygon through the exact
recorded view geometry, including the renderer's map-edge clipping. It prefers
fully visible missed targets and records coverage for every GT route. Final
visual-stage text alignment is explicitly weak, not human verified; intermediate
targets without their own GT are separated into `review_candidates.jsonl`.
See the generated README for coordinate conventions and filtering details.

Check projection regressions with
`python -m unittest discover -s scripts -p test_export_grounding.py`.
