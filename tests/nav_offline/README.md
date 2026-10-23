# 离线数据准备与评测

这些代码从原 AVDN/、CityNav/ 移出。在线包不导入本目录。
输出应指定到 `out/实验名称/`，不要写入源码目录。

每个数据集目录包含：

- `adapter.py`：读取原始标注，分离公开指令／地标与 evaluator 真值。
- `geometry.py`、`views.py`：数据集坐标适配及初始化输入裁图。
- `initialization.py`：准备初始化输入、离线计算方向误差。
- `prepare.py`：读取已有开发初始化缓存，生成导航 episode。保留的缓存路径用于历史开发集转换，不能当作任意测试集抽样器。
- `evaluate.py`：读取 episode、trace、result 和 evaluator_only，计算定位／导航指标。
- `citynav/compact.py`：按指令名称缩减公开地标目录。

统一入口：

```powershell
python tests/nav_offline/cli.py --dataset AVDN --help
python tests/nav_offline/cli.py --dataset AVDN evaluate --folder out/my_run/avdn
python tests/nav_offline/cli.py --dataset CityNav evaluate --folder out/my_run/citynav
```

评测需要匹配的 `episodes.jsonl`、`evaluator_only.jsonl` 和 `episodes/<id>/` 结果。
`prepare-init` 生成的是初始化输入，不含完整导航 environment；导航入口接受的
公开 episode 结构见 `AVDN/README.md` / `CityNav/README.md`。

本次重构不改指标：通用 AVDN `success` 仍使用 IoU > 0.3，并额外给出
`successes_iou_gt04`；近期模型实验使用 IoU > 0.4。CityNav 使用预测中心距目标 ≤20m。
历史实验自身的冻结源码和指标保持原样，不应换成当前代码后续跑同一输出目录。
