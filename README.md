# 2DNav-project

## 当前在线 Agent（2026-10-23）

当前工作目录对应 [第 06 阶段：harness 重构版](out/06_harness_refactor/README.md)，已从 Jev 复核实验回退至加入 Jev 之前；AVDN/CityNav 的 32 个 Python 文件与 `tests/refactor_online_compare/runtime_after` 一致。第 07 阶段已归档，第 08 阶段是尚未实现的 Jev 搜索证据工具设计。

[01–08 实验输出总索引](out/README.md) · [当前代码身份与哈希](out/current_code_version.json) · [下一版设计](out/08_jev_evidence_design/jev_evidence_design_20261023/plan.md)

本次代码快照包含当前在线包、`tests/nav_offline` 离线工具及配套回归测试。`out/`、数据集、权重和服务环境仅保存在本地；上面的实验报告链接需要本地归档。入口测试中的两项历史一致性检查还依赖 `out/agent_refactor_20261021/before_refactor.zip`。

旧 Jev 复核代码保存在本地分支 `codex/jev-review-archive-20261023`（提交 `b100fa589b77a1298a7077b94f26f3e343552154`），完整源码压缩包为 `out/07_jev_review/jev_review_archive_20261023/jev_review_code.zip`。恢复旧版时使用独立工作目录，避免覆盖当前版本。

AVDN 和 CityNav 各自使用 `run.py → analysis.py → loop.py`；全部系统提示位于各自 `prompts.py`，执行能力位于 `tools/`。

- [AVDN 入口与输入说明](AVDN/README.md)
- [CityNav 入口与输入说明](CityNav/README.md)
- [离线准备与评测](tests/nav_offline/README.md)
- [重构与源码说明](out/agent_refactor_20261021/index.html)

历史实验保留在 `out/`，近期冻结运行时保留在各自 `tests/` 实验目录。下面是旧基线的历史说明，不作为当前 AVDN/CityNav 入口。

## Version status: preliminary harness exploration (2026-10-10)

This version records an initial exploration of a navigation harness. It does
not implement autonomous action or tool selection: navigation still follows
predefined search, candidate confirmation, memory, and movement policies.

The current exploration has lower navigation metrics than the SAM3 version
in the experiments conducted so far. It is an experimental checkpoint, not
an improved baseline; this checkpoint does not establish autonomous navigation
or a performance gain. No new online evaluation was run for this publication.

`andh_core` contains the current multi-candidate exploration and does not call
SAM3 during navigation. `andn_core` preserves the modular SAM3 baseline and
optional experiments for comparison. The historical SAM3 baseline remains
available at commit `57b9ecb`.

Offline regression tests and fixed-response replay validate implementation
consistency, not online navigation quality or autonomous behavior.

## Current Search/Confirmation entry point

Run the modular navigation pipeline in the existing navigation environment:

```powershell
python PSC-AVDN/script/andh_core/runner.py --qwen-model qwen3-vl-plus
```

`runner.py` owns `main()`, data loading, stage orchestration, and evaluation.
It reads existing Parse CSV outputs and does not load files from `ANDH-Full/`.
The `andh_core` pipeline is multi-candidate Search followed by candidate
Confirmation. It does not require a SAM3 service or a mode-selection flag.
The default output is `out/andh_core/current`, separate from historical baselines.
The old Search/Confirmation entry point remains a redirect to this current pipeline.
There is no `andh_core/__main__.py`. See
[andh_core/README.md](PSC-AVDN/script/andh_core/README.md) for current options,
dependencies, and offline regression checks. The baseline notes below describe
earlier experiment interfaces.

## Original SAM3 baseline in a modular layout

`andn_core` is a separate, mechanically reorganized copy of the original SAM3
baseline from commit `57b9ecb`. Existing `andh_core` files are unchanged. It retains
the baseline Parse, Search/Confirmation prompts, SAM3 Image 5, position updates,
and fallback policy. The default output is isolated under `out/andn_core/`.

```powershell
conda run --no-capture-output -n 2DNav python PSC-AVDN/script/andn_core/runner.py `
  --anno-dir datasets/sample2 --dataset-dir datasets/sample2 `
  --results-csv out/01_parse_only/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv `
  --out-dir out/andn_core/sample2_sam3
```

Start its SAM3 service separately with `python -m andn_core.tools.grounding.service`
in the SAM3 environment and `PSC-AVDN/script` on `PYTHONPATH`. See
[andn_core/README.md](PSC-AVDN/script/andn_core/README.md) for setup and offline
equivalence verification. This baseline does not introduce autonomous tool calls.

## PSC-AVDN baseline with optional SAM3 grounding

The historical baseline follows the PSC-AVDN Parse/Search/Confirmation inference layout
described in the authors' repository and paper. These notes describe that archived
experiment, not the current `andh_core` runner. SAM3 is a loopback service so the navigation
environment (Python 3.9/Torch 1.11) does not import the newer visual-model
stack.

SAM3 implementation lives in `andh_core/tools/grounding/`. The old
`visual_grounding` imports and service command remain compatibility redirects.
Start the service from the repository root in a dedicated visual environment:

```powershell
$env:PYTHONPATH = "$PWD/PSC-AVDN/script"
conda run --no-capture-output -n sam3 python -m andh_core.tools.grounding.service `
  --sam3-checkpoint E:/Levir/Graduation_Project/sam3/sam3.pt
```

To reproduce the historical SAM3 policy, use `andn_core`, the source snapshot
stored with that run, or its original Git revision. The current `ANDH-Full/Search_Confirmation.py`
redirect no longer implements the archived SAM3 pipeline or accepts its old flags.

The service accepts only loopback connections and exposes `GET /health` and
`POST /v1/candidates`. It generates candidates from the scale-5 main view;
Qwen selects one registered target candidate and the legacy PSC code performs
the existing confirmation and pixel-to-geographic conversion. If the service
is unavailable, SAM3 mode records the reason and falls back to the legacy Qwen
path. These service options belong to the historical pipeline.

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
