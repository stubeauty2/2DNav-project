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
  memory.py
  grounding_tools.py
  grounding_skill.py
  visual_evidence.py
  adaptive_policy.py
  jev_evidence.py
  sam3_service.py
  dino_service.py
  locate_anything_service.py
```

## 当前 adaptive 模式（2026-10-27）

默认主决策模型为 `qwen3.5-flash`。`action:tool` 独立调用 SAM3、DINO、LocateAnything、
zoom、Jev；`action:skill, skill:grounding` 选择后端并融合类别候选。后端可按实例选择，
也可由 `--skill-backends` 配置默认组合。未配置或不健康的服务不会列为可用工具。

adaptive 模式默认启用 `qwen3.8-flash` 作为独立的 `visual_question` 证据工具；
用 `--visual-model` 指定其他可用模型，或 `--no-visual-model` 关闭。只有问答角色使用该模型，主决策、融合和 review 仍用 `--model`。
工具回答包含观察、矛盾和不可见信息，不确认最终目标。所有模型调用共享请求预算。
问答接收原始指令、原始姿态及该视图的区域几何，避免把图像上下误当成指令方向；
黑边覆盖与不可见部分单独标注。可恢复的部分回答保留有效证据并标记缺项，
错误视图或越界框仍拒绝。融合结果保留原始检测框，供比较部件与整体范围。
初始化仍使用本分支原有解析和提示词，没有引入 RSI 或统一初始化策略。

先看原图，按需要调用原始 grounding 或 skill 获得候选，再针对实例中的外观、数量、形状、
关系或序数歧义选择 Jev、zoom 或视觉问答，最后 locate 或 explore。Jev 色彩问题绑定对象
ROI，存在性未知时不能据颜色确认身份；不强制每次使用全部工具。

相同视图/参数的原始检测在 tool 与 skill 间共享缓存。缓存调用不构成新证据；失败工具
不能解除同一候选的重复复核限制。`skill` 和 `direct` 保留旧接口；历史记录用冻结源码复现。

代码回归及新抽样对照入口见 [adaptive 实验说明](../tests/adaptive_agent/README.md)。
真实服务 smoke 验证可执行性，成功率改善必须由配对在线试验和逐例语义检查证明。

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

## 可选 Jev 视觉证据工具（实验性）

通过 `--jev-url http://127.0.0.1:8770` 启用，默认关闭。主控可以在搜索时调用
`tool=jev`，询问已观察区域的 `color`、`count`、`presence`，并接收原始选项概率。
每次最多 3 个问题，每条轨迹默认最多 24 个问题（`--jev-max-questions`）。
问题使用该视图的 0–1000 `bbox` 或 grounding 返回的 `object_ref`；裁剪来自未标记 RGB。
证据只进入 decision，双图 MLLM review 保持原实现。工具不会自动确认、移动或定位。
概率未校准；错误、unknown、低区分度答案不应被当成明确的视觉事实。

```powershell
python -m CityNav.run --inputs out/my_inputs/episodes.jsonl --root out/my_jev_run --model qwen3.5-flash --jev-url http://127.0.0.1:8770
```

验证脚本在 `tests/jev_evidence/`，本地结果在 `out/08_jev_evidence_design/`。
默认校验服务模型为 `OneJev1.6-4B`。使用实验 imajev 网关时显式指定 `--jev-model imajev-4b`；网关保留其原生 unknown 质量，返回包含 unknown 的联合概率。两种模型都只提供证据。

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

## 四工具返回格式实验

当前默认 `--tool-protocol structured_v2`；`legacy` 和 `structured_v1` 保留用于历史对照。
v1 为 SAM3、DINO、LocateAnything 候选补充查询、源视图和区域信息，为 Jev
补充实际 ROI 像素尺寸和同区域存在性绑定。v2 增加已执行工具索引、当前预算、
执行错误、重复访问及复核拒绝信息；决策上下文不重复播放历史动作理由。
所有有效候选保留，主模型自行选取，原 MLLM 复核不变。Jev 仍需显式启用。

实验脚本在 `tests/tool_iteration/`；冻结源码、双模型原始请求和分阶段指标见
[第 09 阶段报告](../out/09_tool_iteration/20261023/index.html)。未证明成功率提升前，
旧阶段未把实验格式改成默认。旧实验模型固定为 `qwen3.8-flash`、`qwen3.5-flash`。

## DINO + SAM3 grounding skill（第 12 阶段）

第 12 阶段使用 --grounding-mode skill；当前默认 adaptive，skill/direct 均保留供历史比较。以下描述为旧 skill 模式。
模型先看 RGB，再调用 grounding；该调用对 DINO、SAM3 使用同一个类别词，
然后由独立的融合请求生成候选证据。两个后端只接收 building、house、car、parking lot 等类别；
颜色、形状、位置和地标关系由导航模型结合原图、Jev 和 review 判断。融合不接收导航指令，
不负责选择最终目标。可用类别列表通过 grounding_categories 提供。

默认每 loop 最多 4 次逻辑工具动作，每个相同观察最多 8 次新工具执行。
一次 grounding 包含 DINO、SAM3 两次后端尝试及最多三次融合模型请求；融合计入全局模型请求额度。
融合输入为原始 RGB 与两张后端标注图，来源引用仅接受 dino/* 和 sam3/*。
skill 模式不实例化、探测或调用 LocateAnything，也不会向模型暴露它；即使配置了对应 URL 或旧能力缓存也会忽略。
LocateAnything 在 adaptive/direct 中可独立调用，在旧 skill 模式中仍禁用。

当前 GPU 实验由 `python tests/grounding_gpu/experiment.py start` 启动正式的
`AVDN.tools.dino_service` 和 `AVDN.tools.sam3_service`，两者均显式传入 `--device cuda`；
`health` 检查 CUDA 就绪，`smoke` 使用三张固定历史原图进行双后端测试，`cleanup` 清理本轮服务。
本轮输出在 `out/12_grounding_gpu/20261026/`；启动器拒绝覆盖已有 manifest。
旧 `tests/grounding_skill/experiment.py` 和 CPU 包装启动器只用于历史归档复现，不作为新实验默认。
Jev 需配置 --jev-url，仍只是视觉证据；zoom 仍是几何工具。原 MLLM review 保留确认权。
skill 模式不允许同时设置 --observation-grounder，以保证模型先看图。

history 明确携带上一轮判断、实际执行、工具证据和 review 冲突；旧理由标为假设。
相同视图/工具/参数复用结果。原图、融合图及 zoom 图可同时交给模型。
同一观察中无新增证据时，世界框 IoU >= 0.95 的已拒绝/不确定候选不会重复送 review；
该阈值只用于去除重复执行，不判断目标正确性。

融合状态 partial 表示来源冲突被隔离，相关候选仅为未验证的模型提议。
未评估原始框仍保留，空框或错误都不证明目标不存在。触碰黑边的候选不会在证据阶段
因为某个角落无效而直接删除；最终 locate 的几何执行检查仍保留。

技能说明见 ../skills/visual-grounding/SKILL.md。
第 10 阶段三后端的固定视图实验、真实模型请求和 HTML 见 ../out/10_grounding_skill/20261026/；
第 12 阶段双后端修改前快照、测试和 GPU 实验记录见 ../out/12_grounding_gpu/20261026/。
这些是诊断与有界联调，不是全量成功率提升结论。旧实验必须使用原冻结 runtime；
新源码需要新输出目录。历史两次工具预算需显式 --max-tools 2。


## tools 职责与目录清理（第 11 阶段）

- memory.py：持久记忆、每轮证据账本、当前请求 context 的构建、工具结果展示和重复执行检查。
- geometry.py：坐标转换、候选方向/距离、CityNav 地标覆盖与距离计算。这里只计算几何，不判断目标。
- grounding_tools.py：独立的 SAM3、DINO、LocateAnything HTTP 客户端。
- grounding_skill.py：可选后端的类别词候选整合、标注图与 MLLM 融合。
- visual_evidence.py / adaptive_policy.py：可选大模型视觉问答与自适应证据选择策略。
- jev_evidence.py：Jev 视觉证据请求及公开元数据。
- prompts.py：导航、review 与可选工具协议提示词；融合提示词仍与 grounding skill 定义一起维护。

context 是 memory 对当前请求的视图，不再单独放一个 context.py；HTTP 客户端和融合也不再拆成 grounding.py 与 grounding_skill.py。
本轮只调整职责归属和导入，导航/复核/融合提示词与迁移函数保持等价，历史 frozen runtime 不改写。

## 离线工具与历史实验

标注适配、采样准备、方向评测和最终评测均位于 [tests/nav_offline](../tests/nav_offline/README.md)，在线包不导入。
运行评测需要另外提供匹配的 evaluator_only.jsonl；在线程序不会复制或读取隐藏真值。

旧实验保留各自 source/runtime。重构时从 39 个 Python 文件精简为 16 个，此后新增视觉证据客户端与工具上下文格式模块，每包现有 18 个 Python 文件。
此前清理移除了 context.py、grounding.py、tool_protocol.py、agent_memory.py；当前新增独立客户端和视觉问答模块，见上面的文件树。

重构前备份在 `out/agent_refactor_20261021/before_refactor.zip`，用于审计和恢复，不是在线依赖。
完整新结构与源码见 [重构说明](../out/agent_refactor_20261021/index.html)。

## CityNav 地标区域优先提示

`prompts.py` 中的 `LANDMARK_FIRST_POLICY` 要求先匹配原指令中的命名地标，相关搜索区域未进入当前视图时优先使用 `explore(mode="landmark")` 或有效的区域接近动作，暂不对当前位置的相似物体进行目标取证或最终定位。到达后再搜索最终目标类别；已经在区域内则直接搜索。两个地标之间的目标应观察二者之间区域，不能用更近的同名词街道替代原指令中的建筑。

`LANDMARK_REVIEW_POLICY` 要求最终复核核对候选与命名地标的空间关系，明显矛盾应拒绝，尚不能确定应返回 uncertain。辅助候选检查其自身参考角色。此修改仅约束 decision/review 提示，运行时动作校验器没有新增硬拦截，融合模型仍只处理类别与视觉候选；是否改善真实轨迹需要单独实验验证。
