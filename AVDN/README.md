# AVDN 在线导航 Agent

当前唯一入口是 `run.py`。原 `agent/`、`initialization/`、`navigation.py` 和旧路径重定向已经移除。

```text
run.py
analysis.py
loop.py
prompts.py
config.py
requirements.txt
tools/
  model.py
  runtime.py
  geometry.py
  views.py
  context.py
  grounding.py
  sam3_service.py
  dino_service.py
  locate_anything_service.py
```

## 执行流程

`run → analysis（缺少初始化标签时）→ loop → tools`

- `analysis.py` 输出 `direction`、`final_target`、`auxiliary`。AVDN 从阶段路线合成方向；CityNav 可选择公开地标并由代码计算方位。
- `loop.py` 保留原来的探索视图、搜索候选、双图 review 过程。
- 辅助目标确认会移动；最终目标确认只产生定位，并不自动将实际位置移动到目标。
- 三种服务分别需要各自模型依赖，保留独立服务文件；轻量客户端已合并。

## 运行

在项目根目录执行（路径为示例；需要先准备对应公开输入）：

```powershell
python -m AVDN.run --inputs out/my_inputs/episodes.jsonl --root out/my_run --model qwen3.8-flash
python -m AVDN.run --inputs out/my_inputs/episodes.jsonl --root out/my_analysis --analysis-only
python -m AVDN.run --help
```

也支持 `python AVDN/run.py ...`。key 从进程环境 `DASHSCOPE_API_KEY` / `API_KEY` / `QWEN_API_KEY` 读取。
默认接口为官方百炼原生多模态接口，可用 `--url` 或 `QWEN_URL` 覆盖。
`--analysis-model` 可单独指定初始化模型；未指定时使用导航模型。
`--thinking-budget 0` 关闭 thinking；这是 token 参数，不是费用预算。
`--direction-mode direct|staged` 仅影响 AVDN 路线解析。

## 输入与输出

输入每行一个公开 episode：

```text
episode_id       唯一的字母、数字、下划线或连字符 ID
agent_input      dataset / setting / instruction_raw / turns /
                 initial_pose / coordinate_frame / named_landmarks / views
environment      image_path / image_size / world_to_raster /
                 main_width_m / main_height_m
initialization   可选：direction / final_target / auxiliary
```

如果 initialization 存在则验证后复用；缺省或 null 时调用 analysis。
如要重新分析已有样本，请在新输入中移除 initialization，使用新的输出根目录。
初始化优先使用输入 views；为空时按公开环境标定裁出初始观察，不读 GT。

输出到 `--root/avdn/`：`run_manifest.json`、`source/`、公开输入副本，以及
`episodes/<id>/analysis/`、`requests/`、`views/`、`tools/`、`trace.json`、`result.json`。
输入文件已位于该根目录时原文件不重写；使用 `--limit` 的已选输入摘要仍记录于 manifest。
改变源码、模型、端点或输入需使用新实验根目录；pending 请求不自动重发。

## 离线工具与历史实验

标注适配、采样准备、方向评测和最终评测均位于 [tests/nav_offline](../tests/nav_offline/README.md)，在线包不导入。
运行评测需要另外提供匹配的 evaluator_only.jsonl；在线程序不会复制或读取隐藏真值。

旧实验保留各自 source/runtime。当前文件从 36 个 Python 文件精简为 16 个。
重构前备份在 `out/agent_refactor_20261021/before_refactor.zip`，用于审计和恢复，不是在线依赖。
完整新结构与源码见 [重构说明](../out/agent_refactor_20261021/index.html)。
