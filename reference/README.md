# 根数预测的 CPU/GPU 实测结果

仅保留三种方法：**最新 TLE＋SGP4、根数线性外推＋SGP4、TimesFM 根数预测＋SGP4**。

- [elements/cpu](elements/cpu/) 与 [elements/gpu](elements/gpu/)：各有四个固定窗口，以及整体 `by_window.csv`、`summary.csv`、`comparison.png`。
- 每窗保存根数输入、根数预测、三种方法的 XYZ、指标、运行说明和图。
- `history.csv` 是仅用于显示的过去 SGP4 轨迹；真正模型输入见 `element_input.csv`。
- 全部结果来自真实运行，未挑选窗口。CPU/GPU 指 TimesFM 推理设备，SGP4 均在 CPU 计算。

## 实测环境（2026-11-16）

| 项目 | CPU | GPU |
|---|---|---|
| 系统 / Python | Windows / 3.12.7 | Windows / 3.12.7 |
| PyTorch | 2.8.0+cpu | 2.8.0+cu128 |
| GPU | 不使用 | NVIDIA GeForce RTX 5070 Ti |
| NumPy / Matplotlib / SGP4 | 1.26.4 / 3.8.4 / 2.27 | 1.26.4 / 3.8.4 / 2.27 |
| TimesFM 包 | 3.0.2 | 3.0.2 |

- 模型：`google/timesfm-3.0-pytorch`，未微调。
- 源码提交：`e51928e27119cb17bebc005be2696b75e0a9e688`。
- 权重 revision：`43046b85ec22d584a13f8098c2ed39c889e129c2`。
- 权重 SHA-256：`a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`。
- `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`、`per_core_batch_size=1`。
- `make_positive=False`、`use_symmetric_averaging=False`、`return_quantiles=False`。

各设备目录的 `environment.json` 记录实际环境及运行代码指纹。跨设备不要求浮点结果逐位一致。

## 复现

```bash
python compare_elements.py --device cpu
python compare_elements.py --device gpu
```

新结果写入 `outputs/elements-cpu/` 或 `outputs/elements-gpu/`，不会覆盖归档。已有模型目录可加 `--checkpoint 路径`，请确认权重版本一致。

基础测试会从归档 XYZ 重算指标、从预测根数重建轨道，并验证历史输入与时间边界。CI 检查 Windows/Linux 的基线流程和结果核算，不在云端重跑大模型。

[根数原理与逐窗结果](../ELEMENTS_EXPERIMENT.md) · [数据和评价约定](../DATA_SOURCES.md)
