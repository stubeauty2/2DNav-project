# andn_core：原始 SAM3 基线的目录整理版

来源为 Git `57b9ecb31c2b64967d6dea9aaebdbce92c8f02a8` 的
`ANDH-Full/Parse.py`、`Search_Confirmation.py`、`category_terms.py` 和
`visual_grounding/`。这是第一步结构整理实验，现有 `andh_core/` 保持不动。

默认 `--candidate-count 0` 走原 SAM3 基线分支，原导航决策、prompt、SAM3 阈值及后处理保持原样。
STEP 0 的被动记录继续保留；STEP 1 的 memory 代码和参数已回退，历史实验归档保留。
STEP 2 C1 用 `--candidate-count 2` 显式启用本轮多候选与独立确认。没有左右预览、跨轮候选累计或反思。
运行入口位于 `runner.py` 的 `main()`，没有 `__main__.py`。

## 文件组织

```text
andn_core/
  runner.py                 数据加载、阶段推进、评估、命令行入口
  parser.py                 原始完整对话解析与 CSV 后处理编排
  navigation.py             原始 Search → Confirmation 导航策略
  config.py                 原始参数与数据路径；输出目录隔离
  tools/
    views.py                原始裁剪、画框；共享视图辅助函数
    geometry.py             原始像素到地图坐标转换、距离计算
    model.py                原始 Qwen 请求；可选 C1 prompt 增补入口
    candidates.py           C1 协议校验、基线分析文本增补、候选裁剪、方框转换
    sam3.py                 原始 SAM3 Image 5 生成、跨类别 NMS
    category_terms.py       原始目标类别提取
    parsing.py              原始 Parse 模型请求、prompt、方向和 CSV 辅助函数
    telemetry.py            STEP 0：请求/图像/框/动作归档，不参与导航决策
    grounding/              原始 SAM3 客户端、服务、后端和依赖
```

不从 `ANDH-Full/`、`andh_core/` 或 `visual_grounding/` 加载实现。
仍复用项目 `PSC-AVDN/script/util.py`、`src/env.py`、数据集及 Python 环境；
因此不是复制本目录即可在任意项目中独立运行。

## 保留的基线行为

- Search 使用主图、同位置 detail/context、minimap 和可用时的 SAM3 Image 5。
  SAM3 返回的候选框作为辅助证据，Qwen 在主图输出一个目标框。
- 没有找到时，沿原方向前进 120m，默认最多 3 轮。
- 找到后围绕 Search 框中心裁剪 Confirmation 图；确认输入保留原主图和两种尺度图。
  Confirmation 不调用 SAM3。
- 最终显示/评估框仍来自原确认与方框转换过程；向下一阶段返回的位置仍是原来的
  Search 中心。没有在本次整理中将其改为 Confirmation 中心。
- 确认无有效框时回用 Search 框、首轮下半区远目标特殊规则、未找到时的回退、
  `destination` 特殊阶段等全部原样保留。
- SAM3 不可用时保留原有告警和无 Image 5 降级行为。做 SAM3 对照实验时须核对服务
  与运行日志，避免把降级运行当作有效的 SAM3 实验。
- 保留所有后处理 SAM3 候选以及原跨类别 mask NMS；不是服务默认 top-12。
- Parse 的模型、prompt 和方向转换也原样保留。复现实验时优先复用历史 Parse CSV，
  避免把解析变化混入目录整理对照。

初始目录整理只调整了导入/目录路径、独立输出目录和运行入口的数据路径参数：
导航默认输出 `out/andn_core/current`；重新解析默认输出 `out/andn_core/parse`。
导航默认读取原历史 Parse CSV，不自动重解析。

## 运行 sample2

先在独立 SAM3 环境启动工具服务（仓库根目录）：

```powershell
$env:PYTHONPATH = "$PWD/PSC-AVDN/script;D:/Graduation_Project/sam3"
conda run --no-capture-output -n sam3 python -m andn_core.tools.grounding.service `
  --sam3-checkpoint D:/Graduation_Project/sam3/sam3.pt
```

在导航环境中配置环境变量 `API_KEY`，然后运行：

```powershell
conda run --no-capture-output -n 2DNav python -B PSC-AVDN/script/andn_core/runner.py `
  --anno-dir datasets/sample2 `
  --dataset-dir datasets/sample2 `
  --results-csv out/01_parse_only/parse/preds_out_Parse_test_unseen_full/parsing_results_full.csv `
  --out-dir out/andn_core/sample2_sam3 `
  --vision-tool-url http://127.0.0.1:8765
```

新实验请选择新的输出目录。也可在 `PSC-AVDN/script` 下执行 `python -m andn_core.runner`。
`python -m andn_core.parser` 对原默认数据集执行解析；解析使用原来的 `API_TOKEN` 配置。
若要在导航中使用新解析结果，显式指定 `--results-csv`。

## STEP 0：冻结基线与记录（2026-10-07）

正常运行默认启用记录，显式 `--trace` 也可；用 `--no-trace` 关闭。
Python 调用 `run_turn_and_crop_full_trajectory(..., trace=False)` 同样可关闭。
这里只给完整轨迹入口启用记录，旧单阶段入口及重新 Parse 不在本次范围内。

```text
<out_dir>/trace/<UTC时间戳-进程ID>/
  manifest.json             运行参数、模型/数据路径、源码及输入哈希、运行环境
  snapshot/                 源码副本、解析 CSV；正常无密钥源码按字节保留
  events.jsonl              逐阶段、逐轮动作、框、工具调用与异常
  calls/call00001.request.json  发出请求前即保存的输入
  calls/call00001.json      完整输入、HTTP响应、模型原文、解析结果、耗时
  images/<sha256>.jpg       实际 data-URL 解码后的图像字节，去重保存
  summary.json             调用/图像计数、耗时、记录错误数与结束状态
```

每个模型输入保存原 system/user 文本和图片顺序。图片以 `asset:images/...`
代替大段 base64；结合 `images` 元数据中的 `data_url_prefix`，重新 base64 编码
对应文件即可重建原请求。`request_payload_sha256` 用于检查重建后的完整 JSON。
图像角色、尺寸及所属视图的地图四角分别记录，说明只写入归档，不加入模型 prompt。

框分为：模型原文中的原始框、客户端规范化的 Search/Confirmation 框、确认实际使用框、
最终评估方形区域。像素框绑定视图四角，并记录地图投影；评估裁剪事件标记
`actual_move=false`，不会被当作移动。`position_assignment` 是原代码中的局部位置赋值，
`stage_position_commit` 是向下一阶段采用 Search 中心的提交，二者分开记录。

Search/Confirmation 和共享 Land 调用均可记录实际请求。Land 使用原函数代码和默认参数，
只为该调用单独包装 transport，不修改共享 `util.py` 或全局 requests。
SAM3 记录实际服务请求、原始响应、Image 5 元数据和失败事件。
本步不额外查询 SAM3 服务，因此服务端 checkpoint/设备仍由独立服务的启动配置确定，
不能仅靠客户端 manifest 确认；客户端参数和代码快照会保留。

API 错误、非 JSON、空/无法解析的模型回复与正常“未找到目标”分别标记。
不增加自动重试、不修复 Land 接口、不改变原回退。
不保存 Authorization 请求头；已知密钥及密钥格式会脱敏。
记录失败只告警，不改变导航返回值；检查 `summary.json` 的 `recording_errors`，
若进程被强制结束而缺少 summary，则应将这次记录视为不完整。

STEP 0 的源码检查允许指定的独立记录语句，并验证移除它们后原有决策 AST 完全一致。
异常分支、归档重建和开关一致性另由测试验证；不会用“加了记录所以应该一致”代替回放。

## 离线验证

```powershell
conda run --no-capture-output -n 2DNav python -B -m unittest discover -s tests/andn_core -p "test_*.py" -v
```

`tests/andn_core/replay_sample2.py` 分别回放原文件和新目录，重用历史模型回答、
SAM3 Image 5 和 Land 响应，校验请求、轨迹与最终 IoU。它不调用 API、不加载 GPU 权重，
不能当作新一轮在线成功率实验。验证记录位于 `out/sam3_core_restructure_20261006/`。

2026-10-06 验证结果：5 项源码一致性测试通过；sample2 的 20 条轨迹回放中，
75 次模型请求的文本、图像及参数完全一致，各阶段结果和最终预测完全一致。
平均 final_IoU 为 `0.3109567567009118`，成功 `10/20`（阈值 `> 0.3`），
与对应历史结果的最大 final_IoU 误差为 `0`。
现有 `andh_core` 的 21 个文件 SHA-256 均未变化。
详见 `out/sam3_core_restructure_20261006/verification.json`。

2026-10-07 STEP 0：15 项测试通过。开启记录、关闭记录与原始 SAM3 的 sample2
固定回答回放完全一致：20 条轨迹、75 次模型请求、各阶段输出及最终结果一致；
323 次图片输入均可从归档重建，记录错误为 0，源码快照与输入哈希检查通过。
现有 `andh_core` 保持不动。验证结果为离线回放的 10/20、平均 IoU 0.310957，
不能视为重新在线运行的成功率改善。

验证汇总：`out/step0_observability_20261007/verification.json`。
示例归档：`out/step0_observability_20261007/verified/replay_after/trace/`。
完整回放的 SAM3 图片和 Land 回复来自历史记录，不调用 GPU/API；真实请求包装的
一致性与异常记录由 `tests/andn_core/test_telemetry.py` 的受控响应测试独立验证。

## STEP 1：已回退，保留历史实验

按用户要求移除了 `memory.py`、memory 命令行开关、模型摘要输入、专用测试和回放脚本。
实验代码快照在 `out/step2_candidates_20261007/before/`，在线记录仍在原输出目录。
以下 M1 表格是回退前的实验，不代表当前代码功能。

### M1 在线结果（2026-10-07）

两组已各完成 sample2 的 20 条，输出根目录 `out/andn_core/sample2_step1_20261007/`。

| 配置 | 平均 final_IoU | 成功（>0.3） | 相对历史提升 / 退化 / 不变 |
| --- | --- | --- | --- |
| 历史 SAM3 | 0.310957 | 10/20 | — |
| M1-S | 0.185771 | 5/20 | 6 / 7 / 7 |
| M1-SC | 0.209092 | 5/20 | 5 / 7 / 8 |

两组各有 5 条历史成功→失败，没有历史失败→成功。M1-SC 的均值比 M1-S 高
0.023321，但成功率相同，均低于历史基线。当前已回退此模块。
完整 20 条结果见该目录的 `comparison.csv`、`comparison.json`。

M1-S 的 47 次 Search 均有 SAM3，其中 20 次加入摘要；18 次 Confirmation 不加摘要。
M1-SC 的 43 次 Search 均有 SAM3，其中 16 次加入摘要；21 次 Confirmation 全部加入摘要。
两组定位 API 无回退，记录错误为 0；各有 1 次原有 Land HTTP 401 回退。
源码、Parse CSV、模型和 SAM3 配置一致，检查通过，详见 `verification.json`。

每组只跑一次，B0 为历史结果而非同期重跑。已观察到 21 对完整输入相同的请求中，
9 对存在性判断或框坐标仍有差异；不能将所有组间变化解释为 memory 配置的因果效果。
每步模型输入输出、图片、memory 和最终框均已保存在各组 `trace/` 中。

## STEP 2 C1：本轮多候选与独立确认

启动参数 `--candidate-count 2`；默认 `0` 完全沿用原基线分支。仅完整轨迹入口开放 C1。

1. Search 在原搜索主图提出最多 2 个候选，允许少于两个或为空。主图、detail/context、minimap、SAM3 Image 5 和原有前方优先保持；只允许 `search_main` 坐标。
2. 程序赋予 `rN_cM` ID，把各 Search 框中心映射到地图；每个中心裁剪确认 main/detail/context。预览不提交位置，不进入实际轨迹。主图红点表示仍在原 anchor 的 agent。
3. Confirmation 一次比较候选，保留基线简单定位/关系分析与目标属性检查。只输入完整目标和附有候选 ID / view_id / 角色说明的图像，不传 Search 旧框、语义位置、置信度、reason 或网格。
4. 确认只能返回一个已有候选及其确认主图上的新框；无把握返回 false / null。非法 ID、错配视图、辅助图上的框或无效框视为未确认，不回用 Search 框、不默认选首个候选。
5. 选中后下一阶段位置沿用该候选的 Search 中心；最终评估框来自 Confirmation 主图上的新框，使用基线方框转换。原下半区且远于 120m 的特殊规则保留，但在选中候选后应用，日志单列该特殊结束原因。
6. 未确认时从原搜索 anchor 前进 120m，最多 3 轮；最终未找到仍采用基线返回第一步位置的评估回退。本轮候选不传入下一轮。Search 轮间的原 `prev_grid_5x5` 保留，不新增 STEP 1 文本摘要。

`<stage_id>_search.json` 存候选、来源框/地图四角、候选地图中心、原始证据、预览图及 ID、确认选择和动作。
未选中候选保留 pending；一次确认不通过记 uncertain；主图裁剪失败记 unavailable；不把未选中等同 rejected。
这些是归档记录，不是再次传给模型的 memory。`trace/` 保存实际请求、全部图像、原始回答和校验错误；每张确认图单独绑定候选和地图四角。

运行 C1 并实时统计与历史 SAM3 对比：

```powershell
conda run --no-capture-output -n 2DNav python -B tests/andn_core/run_sample2.py `
  --candidate-count 2 --out-dir out/andn_core/sample2_step2_c1_20261007
```

需先配置 `API_KEY` 并启动 SAM3 服务。统计文件为 `live_vs_sam3.json/.csv` 和最终 `final_vs_sam3.json`。
这轮仅验证 C1；C2 跨轮累计/更多候选留待单独实验。

### C1 在线结果（2026-10-07）

20/20 完成：平均 final_IoU **0.196055**（历史 **0.310957**），成功 **6/20**（历史 **10/20**）。
4 改善、9 退化、7 不变；成功→失败：2078__10、2384__11、1913__8、432__2；无失败→成功。
39 次 Search 都有 SAM3，17 次提出两个候选；33 次 Confirmation 中 25 次确认，2 次选择第二个候选。
Search/Confirmation 无接口或协议错误，归档错误为 0；原 Land 路径出现 1 次凭证失败回退。
本次未同期重跑在线 B0，结果尚未维持历史基线水平，不作为默认替换。

结果目录 `out/andn_core/sample2_step2_c1_20261007/`：20 条对比 CSV、final_vs_sam3.json、runtime_audit.json、geometry_diagnostics.json、逐调用 trace。
初始运行因归档标签缺失而停止，另存 `out/andn_core/sample2_step2_c1_initial_trace_issue_20261007/`，不计入正式统计。
23 项测试通过；C1 关闭后的完整回放与冻结基线一致，见 `out/step2_candidates_20261007/verification.json`。

### STEP 3：有依据的 Confirmation 复核

`--confirmation-review off|observe|once`，默认 `off`，启用时需要 `--candidate-count 2`。
`observe` 只记录同一地图坐标系下的框面积、覆盖、偏移与边界信息；`once` 在疑点触发后，
每个导航阶段最多调用一次确认图 SAM3 和一次模型复核。复核可 `keep`、`revise` 或 `reject`。
修订框只使用选定的原始确认主图坐标，不能换候选 ID；对照图和 SAM3 图只能提供证据。
拒绝后沿用 C1 从原搜索位置继续前进的流程；工具或协议错误保留初次确认。

初始触发配置位于 `tools/geometry.py` 的 `REVIEW_THRESHOLDS`，不读取 GT，也不自动拒绝框。
每轮 `verification` 保存初次确认、触发信号、工具证据、复核原文和最终采用结果；trace 单独记录 `review` 调用。
首次 STEP 3 实验使用已有的 Search S1 增补，正常初次 Confirmation 保持 C1；Search S1 由实验入口加载，
并非上述 CLI 开关自动启用的内容。实验入口及结果在 `out/andn_core/sample2_step3_r1_20261007/`。

新增复核检查后 29 项离线测试通过；默认关闭时 sample2 的 20 条轨迹、75 次请求与冻结基线完整回放一致。
R0 历史诊断、默认关闭验证和报告脚本在 `out/step3_verification_20261007/`。

STEP 3 在线结果（2026-10-07）：20/20，平均 final_IoU **0.236573**；按 `final_IoU > 0.3` 成功
**8/20（40%）**，历史 SAM3 **10/20（50%）**，此前 C1 **6/20（30%）**。
原 Natural Path Eval 的 SR 使用 IoU ≥ 0.4 加双方中心互含条件，本次 **35%**（SAM3 50%、C1 25%）。
相对 SAM3：3 改善、10 退化、7 不变；成功→失败为 2078__10、432__2。
相对 C1：2384__11、1913__8 恢复成功，没有成功→失败。
7 次复核（6 revise、1 reject），最终目标框直接改善 IoU 3 次，但未直接跨越 0.3 成功阈值。
本轮同时启用 Search S1，不能将全部收益归因于复核；功能继续默认关闭。
源码运行期间未变、归档和复核错误为 0；原 Land 路径有 1 次回退，仍纳入统计。
详细报告：`docs/andn_core_step3_sample2_20261007.html`；指标：实验目录的 `final_comparison.json`。

### G0–G4：二维空间证据地图（2026-10-08）

`--spatial-mode off|g0|g1|g2|g3|g4`，默认关闭。G0 可以用于原 SAM3 路径；
G1–G4 使用 `--candidate-count 2`，Confirmation 仍独立比较候选居中视图。

| 模式 | 功能 |
| --- | --- |
| g0 | 记录实际裁剪变换、有效覆盖、位置与搜索事件；模型请求不变 |
| g1 | 原有主图/细节/广域图，加已观察地图和方位关系 |
| g2 | 当前、左前、前方、右前主图共同搜索；合计最多两个候选 |
| g3 | G2 相同采样，增加地理对齐地图与结构化空间关系 |
| g4 | G3 加同阶段最多两张相关历史主图及搜索事件引用 |

核心实现为 `tools/spatial_map.py`。地图只重投影已获得的图像，不从原地图填充图间空隙。
图像坐标转换与原裁剪的边界裁剪行为一致；地图使用局部东/北米制坐标，并标注实际图像上方的地理方位。
阶段起点和解析方向固定为阶段参照；实际位置与预览位置独立保存。
固定预览前移约 0.4H、横移 ±0.35W，裁剪朝向保持一致；每阶段仍最多 3 轮。
SAM3 只处理实际位置主图，历史/地图/SAM3 图不允许输出定位框。
图像均附独立 view_id 和说明；当前主图语义网格沿用原接口，历史网格明确属于上一主图。

每条轨迹的地图跨阶段保留；搜索历史检索限制在相同阶段，旧 bbox 和模型自由文本不进入历史摘要。
`spatial/<trajectory>/` 保存观察图片、地图及输入/搜索/阶段状态快照；trace 保存实际模型请求与返回。

```powershell
conda run --no-capture-output -n 2DNav python -B PSC-AVDN/script/andn_core/runner.py `
  --anno-dir datasets/sample2 --dataset-dir datasets/sample2 `
  --candidate-count 2 --spatial-mode g4 --confirmation-review off `
  --out-dir out/andn_core/sample2_spatial_g4
```

完整对照入口为 `tests/andn_core/run_spatial_sample2.py`，一次读取凭证后运行指定配置。
本轮运行目录为 `out/map_g0_g4_20261008/online/`，包括 G1–G4 与同期 C1 off 对照。
Search S1 与 STEP 3 复核均关闭；使用相同原 Parse CSV 和模型，所有结果均保留。
实时统计脚本为 `out/map_g0_g4_20261008/status.py`；HTML 报告为 `docs/andn_core_spatial_g0_g4_20261008.html`。

初始验证：41 项离线测试通过；G0 与 off 的 20 条轨迹、75 次请求均与冻结 SAM3 回放完全一致。
在线结果以 `comparison_matrix.json` 为准；地图功能不因回放达到 50% 就视为在线达到基线。

首轮主消融已完成（每组相同 20 条，复核关闭）：

| 版本 | final_IoU > 0.3 | 平均 final_IoU | Natural SR |
| --- | --- | --- | --- |
| 历史 SAM3 | 10/20（50%） | 0.310957 | 50% |
| 同期 C1 off | 6/20（30%） | 0.193914 | 20% |
| G1 | 6/20（30%） | 0.216079 | 30% |
| G2 | 6/20（30%） | 0.201062 | 30% |
| G3 | 6/20（30%） | 0.203528 | 25% |
| G4 | 8/20（40%） | 0.254126 | 40% |

G4 对历史 SAM3 的成功→失败为 `2078__10`、`432__2`，没有失败→成功。
对同期 C1 恢复 `2384__11`、`1468__8`、`2031__6`，但失去 C1 新成功的 `2567__7`。
G4 未达到历史基线；功能保留为可选实验开关。各组源码冻结，SAM3/归档错误均为 0；
G2 一次候选超出预算，按原协议保留前两个。各组都有原 Land 路径的一次回退，未从统计中剔除。
G1–G4 按候选实际地理位置判断前后，因此 G1 对 off 不只是增加文本；在线轨迹也可能分化。

额外 G4 + STEP 03 复核为独立完整实验，目录 `out/map_g0_g4_20261008/online_r1/`，
使用入口参数 `--modes g4 --confirmation-review once`；不启用 Search S1。
初次 Confirmation 仍不接收 Search 框；仅几何疑点触发后的复核读取本轮 proposal 对照与新 SAM3 证据。
组合结果同样汇总到 HTML 和 `online/comparison_matrix.json`，不合并各版本的逐条最优结果。

G4 + R1 已完成：8/20（40%），平均 IoU 0.245971，Natural SR 35%。两次复核均 revise；
最终目标阶段的 `432__2` 从初次确认 0.0821 降至 0.0515，另一次是 `1462__4` 的中间阶段，
不对中间目标套用最终 GT。组合没有恢复 50%，也没有优于纯 G4。
两次 G4 运行中，30 次完整请求及图像哈希一致，16 次候选/框决策不同（包含微小坐标变化），
见 `online/reproducibility.json`。单次在线差异不能全部归因于复核或 memory。

所有在线运行结束后修复了历史记录中的错误分类：传输或协议失败不作为“未找到候选”供模型使用。
42 项测试通过；重新检查全部 273 次已归档 Search，修复没有改变其中任何一次历史评估分类。
没有修改 prompt，也未为该错误分支重新在线运行；冻结实验源码保存在各 trace 的 `snapshot/`。
验证记录为 `out/map_g0_g4_20261008/post_run_validation.json`。


### G5：原 SAM3 主流程 + 失败后地图补救

`--spatial-mode g5 --candidate-count 0 --confirmation-review off` 保留原主流程，
仅在阶段 `found=false` 后追加最多一次 G4 Search 和一次 C1 Confirmation。
不增加探索移动；补救未确认则保留原结果；不改写原 `found=true` 的 Search 框回退语义。
地图复用已批准 prompt，没有新增 prompt。每条轨迹保存原流程配对结果。

完整 sample2：7/20（35%），平均 final_IoU 0.225883，与同次原 SAM3 主流程完全一致。
4次补救均未形成可确认候选，没有补救收益或退化。历史成功→失败为
`2078__10`、`2031__6`、`432__2`，没有历史失败→成功；未维持历史50%。
49项离线测试通过，冻结回放75次请求及20条原预测一致；在线源码冻结、SAM3/归档错误0。
报告：`docs/andn_core_spatial_g5_20261008.html`；完整统计及审计：`out/map_g5_20261008/`。
G0–G5 均为可选实验，默认 off。
