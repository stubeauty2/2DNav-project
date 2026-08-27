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
