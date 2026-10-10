# andh_core 多候选搜索与确认

本目录只保留当前多候选导航流程，入口为 `runner.py` 中的 `main()`。
不新增 `__main__.py`，不再使用 `--multiview-mode candidates` 启用模式。
当前代码不依赖 `ANDH-Full/` 或 SAM3 服务；仍依赖项目的 `script/util.py`、
`src/env.py`、导航 Python 环境、地图数据和模型 API，不能只复制本目录独立运行。

## 运行

在项目根目录、已配置 `API_KEY` 的导航环境运行：

```powershell
python PSC-AVDN/script/andh_core/runner.py `
  --anno-dir datasets/sample2 --dataset-dir datasets/sample2 `
  --results-csv out/ANDH-Full/sample2_20260924_171805/parse/parsing_results_full.csv `
  --out-dir out/andh_core/sample2_candidates `
  --candidate-count 3 --preview-offset-ratio 0.5 `
  --history-size 1 --relative-pose --no-strict-confirmation --trace
```

也可在 `PSC-AVDN/script` 目录执行 `python -m andh_core.runner`。
默认输出目录为项目根目录下的 `out/andh_core/current`，不覆盖历史 SAM3 基线目录。
本入口读取已有完整轨迹 Parse CSV，不会自动解析指令；CSV 的存放路径不是代码依赖。

```powershell
python tests/run_andh_core_sample2.py --results-csv <已有Parse文件路径>
```

sample2 脚本默认执行当前候选流程，保留 `MC-no-strict` 作为运行目录名称，
使用一张历史图、相对位姿和 trace，关闭严格确认；无需启动 SAM3。
脚本从 `API_KEY` / `QWEN_API_KEY` 或隐藏输入获取凭证，保存独立时间戳目录、
代码快照、配置、逐轨迹增量结果及汇总。历史 B0/H1/H1-R/MV 模式已移除。

## 唯一导航流程

```text
获取当前位置与左右/正前方预览、拼接图
  → Search 提出多个候选 → 地图投影、合并、写入 memory
  → 按各候选中心裁剪确认主图及 detail/context（此时不移动 agent）
  → Confirmation 比较候选，在选中的确认主图输出新 bbox
      confirmed → 投影新框 → 移动到最终目标
      证据不足 → 前进指定距离 → 下一轮 Search
      达到轮数上限 → 未找到，交给 runner 执行既有失败回退
```

预览图与当前主图保持同朝向。令 d 为主图实际覆盖短边乘 `preview_offset_ratio`，
正前方偏移 d；左右分别偏45°前移 √2d，形成适合拼接的布局。
越界时按 1、1/2、1/4、1/8 缩短偏移。预览和候选确认裁剪均不计入真实移动轨迹。

Search 每轮最多提出 `candidate_count` 个候选，不足填 null；常规原始图可定位，
拼接图等辅助图只提供上下文。Confirmation 查看未排除的累计候选，
每个候选都有按原 Search 朝向和尺度居中的确认主图，以及 scale=3/7 的细节/周边图。
Confirmation 仅在确认主图上输出框；默认主图 scale=5。

Prompt 保存在 `prompts/candidates.md`，分别包含 Search 和 Confirmation 两段。
每张图附带 ID、尺寸、能否定位和简短说明，不在通用 prompt 中固定图像数量。
本轮目录合并不改动 prompt 文本。

memory 内部保存每次来源框、地图位置、候选状态和确认记录。模型摘要包含候选的
语义、支持证据、冲突及确认结论，确认时附带本轮候选与视图对应关系，
不传旧 bbox 坐标或历史证据图 ID。未选中不等于排除，不确定候选可以跨轮确认。
单个非法候选不会丢弃其他合法候选；整批失败后仍可使用剩余搜索轮次。

## 参数

CLI 优先于环境变量。几何一致性检查默认关闭，图片历史和 trace 默认关闭。
候选 memory、预览和最终确认位置更新始终启用。

| CLI | 环境变量 | 默认值 / 作用 |
| --- | --- | --- |
| `--candidate-count` | `ANDH_CANDIDATE_COUNT` | 3；每轮候选槽位数，范围1～6 |
| `--preview-offset-ratio` | `ANDH_PREVIEW_OFFSET_RATIO` | 0.5；预览偏移相对视图短边的比例 |
| `--max-steps` | `ANDH_MAX_STEPS` | 3；每阶段最多搜索轮数 |
| `--step-meters` | `ANDH_STEP_METERS` | 120；两轮搜索之间的前进距离 |
| `--history-size` | `ANDH_HISTORY_SIZE` | 0；额外提供0～2张阶段内历史原图 |
| `--relative-pose` | `ANDH_RELATIVE_POSE` | 关闭；为通用请求增加相对位姿，Search已有前方偏好所需位姿 |
| `--strict-confirmation` | `ANDH_STRICT_CONFIRMATION` | 关闭；开启新旧框地图重叠一致性检查 |
| `--model-retries` | `ANDH_MODEL_RETRIES` | 1；每次请求额外重试次数，范围0～3 |
| `--trace` | `ANDH_TRACE` | 关闭；额外输出JSONL调用及运动记录 |

布尔参数支持对应 `--no-...` 形式，环境变量使用0/1或false/true。
数据路径、模型地址等参数见 `python PSC-AVDN/script/andh_core/runner.py --help`。
为保持已有评估口径，runner 的失败回退框仍固定使用第一阶段起点前方120m的scale=2视图，
不随 `--step-meters` 改变。

已删除旧 `off/select/joint` 模式和 `--multiview-mode`、`--branch-preview`、`--memory`、
`--fix-confirmation-position`、`--max-confirmations`、`--vision-tool-url` 开关。
旧实验结果和HTML报告仍保留；重现旧算法应使用对应代码快照，不应用当前入口标记成旧基线。
旧 `run_andh_full_sample1.py`、`run_andh_full_sample2.py`、`run_andh_agent_sample1.py`
已在入口明确停止并提示使用对应快照，避免历史脚本误调用当前算法或在模型调用后才报旧参数错误。

## 文件职责

根目录保留 runner、navigation、memory、config 等上层编排。工具层移动到 `tools/`；
2026-10-03 本次整理仅更新文件位置、导入和 prompt 路径，导航与提示词内容不变。
SAM3 实现统一位于 `tools/grounding/`；相邻 `visual_grounding/` 只保留兼容入口。
当前导航尚未自主调用它，服务仍在独立视觉环境运行。


| 文件 | 职责 |
| --- | --- |
| `runner.py` | main入口、加载Parse与地图、阶段编排、失败回退、final_IoU与结果保存 |
| `navigation.py` | Search → 多候选Confirmation → 实际移动的唯一导航循环 |
| `tools/model.py` | 逐图消息、模型请求、候选/确认输出校验、重试与调用归档 |
| `tools/views.py` | Observation、裁剪、预览布局、确认视图渲染 |
| `tools/geometry.py` | 像素/地图投影、相对位姿、地图框IoU和最终方框计算 |
| `tools/mosaic.py` | 搜索视图拼接与标记 |
| `memory.py` | 候选合并、状态、确认记录及模型摘要 |
| `config.py` | 参数与环境变量 |
| `tools/telemetry.py` | 可选JSONL记录 |
| `tools/grounding/client.py` | 轻量 HTTP 客户端，调用独立 SAM3 服务 |
| `tools/grounding/service.py` | 本机 HTTP 服务与 CLI |
| `tools/grounding/backends.py` | 延迟加载 SAM3 权重，文字提示得到 mask/box/score |
| `tools/grounding/contracts.py` | 校验视图和文本概念请求 |
| `tools/grounding/geometry.py` | mask 连通域、外接框、mask IoU |
| `tools/grounding/postprocess.py` | 候选过滤、mask NMS、结果文件与元数据 |
| `__init__.py` | 包标识 |

每阶段 `*_search.json` 的 `model_calls` 保存实际system prompt、文本、
图片ID/说明/路径以及每次模型响应和校验结果；trace额外记录运动和请求事件。
不归档API Key或图片base64。实际运动记录不包含虚拟预览位置。

## 离线验证

```powershell
conda run --no-capture-output -n 2DNav python -B -m unittest discover -s tests/andh_core -p "test_*.py" -v
conda run --no-capture-output -n 2DNav python -B PSC-AVDN/script/andh_core/runner.py --help
python -B tests/run_andh_core_sample2.py --help
```

离线固定响应回放验证流程、裁剪、模型输入、坐标转换和结果保持一致，不调用模型API，
也不能代替在线sample2性能评测。运行临时文件位于 `tests/tmp/`。

## SAM3 工具服务

实现统一放在 `tools/grounding/`，但模型仍运行在独立的视觉环境中。
导航环境只需导入客户端，不会因此加载 SAM3 或 Torch 权重：

```python
from andh_core.tools.grounding.client import VisionToolClient
client = VisionToolClient("http://127.0.0.1:8765")
# client.health() / client.candidates(payload)
```

在仓库根目录启动（需现有 SAM3 环境和权重）：

```powershell
$env:PYTHONPATH = "$PWD/PSC-AVDN/script"
conda run --no-capture-output -n sam3 python -m andh_core.tools.grounding.service `
  --sam3-checkpoint E:/Levir/Graduation_Project/sam3/sam3.pt
```

旧 `python -m visual_grounding.service` 以及旧 Python 导入仍可用。
本次只迁移服务代码，不改提示词、过滤阈值、HTTP 协议或导航逻辑，
也没有把 SAM3 自动启用到 Search/Confirmation 中。
