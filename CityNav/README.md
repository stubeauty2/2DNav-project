# CityNav 在线导航 Agent

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
  jev.py
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
python -m CityNav.run --inputs out/my_inputs/episodes.jsonl --root out/my_run --model qwen3.5-flash
python -m CityNav.run --inputs out/my_inputs/episodes.jsonl --root out/my_analysis --analysis-only
python -m CityNav.run --help
```

也支持 `python CityNav/run.py ...`。key 从进程环境 `DASHSCOPE_API_KEY` / `API_KEY` / `QWEN_API_KEY` 读取。
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

输出到 `--root/citynav/`：`run_manifest.json`、`source/`、公开输入副本，以及
`episodes/<id>/analysis/`、`requests/`、`views/`、`tools/`、`trace.json`、`result.json`。
输入文件已位于该根目录时原文件不重写；使用 `--limit` 的已选输入摘要仍记录于 manifest。
改变源码、模型、端点或输入需使用新实验根目录；pending 请求不自动重发。

## 离线工具与历史实验

标注适配、采样准备、方向评测和最终评测均位于 [tests/nav_offline](../tests/nav_offline/README.md)，在线包不导入。
运行评测需要另外提供匹配的 evaluator_only.jsonl；在线程序不会复制或读取隐藏真值。

旧实验保留各自 source/runtime。当前文件从 39 个 Python 文件精简为 16 个。
重构前备份在 `out/agent_refactor_20261021/before_refactor.zip`，用于审计和恢复，不是在线依赖。
完整新结构与源码见 [重构说明](../out/agent_refactor_20261021/index.html)。

## OneJev 强制复核实验

默认 `--review-backend mllm` 保持原双图 MLLM review。可选
`--review-backend mllm_fixed` 使用独立固定候选提示、相同双图，明确只判原候选，
不得通过换对象或修框再确认；确认时固定使用原提议框，单列 `review_box_preserved` 事件。
固定提示与 Jev 共享逐字相同的视觉判别准则；输出格式适配各自接口，默认 MLLM 提示不变。

`--review-backend jev --jev-url http://127.0.0.1:8770 --jev-model OneJev1.6-0.8B
--jev-min-probability 0.7 --jev-service-manifest <冻结服务JSON路径>`
接入官方 OneJev `GET /health` 和 `POST /v1/systemone`。服务另行启动；在线客户端
只依赖 requests，不在导航 Python 环境导入模型。服务 manifest 应包含 `served_model`、
权重 SHA、官方源码 revision、加载参数及依赖版本；CLI 将其内容和哈希冻结进运行清单。

Jev 接收与 MLLM 相同的主图和带黄色 UNCONFIRMED 标记的局部图、公开候选与公开上下文，
回答 confirm/reject/uncertain 三选一。共享问题明确不可见墙体颜色、门窗、楼梯或人不构成
否定；真实可见的形状、明确屋顶颜色或关系矛盾才支持 reject。类别和分数本身不能确认。
预声明的接受门槛是 **最大选项概率 ≥0.7**，不足则 uncertain；不是官方按均匀分布缩放的
`confidence`。两者均记录，概率与 confidence 通过有限性、归一性、选项和公式验证。
该门槛未经 AVDN/CityNav 校准，不允许根据测试结果调参后仍声称是原实验。

确认只保留原提议 bbox 和原 view_id，Jev 不定位、不改框。服务/协议错误记录为 uncertain，
不作为反对候选的视觉证据，也不静默回退 MLLM。health 模型不符或冻结身份变化会明确停止；
相同输入缓存可离线重放，pending 不会自动重发。每次请求的公开 payload、原图路径及 SHA、
原始输出、模型身份、服务 manifest、阈值和延时保存到 `tools/rXX-jev-review.json`；
base64 仅在线传输，不重复写入日志。Jev 请求独立统计，不计入 MLLM 请求数。

`result.json` 的 `jev_calls` 与 `jev_error_count` 将服务/协议故障单独计数，
不能把这些运行的 uncertain 当作模型的正常语义不确定。集成回归：
`python -B -m unittest discover -s tests/jev -p test_integration.py -v`。
