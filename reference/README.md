# 已实测的 CPU/GPU 结果

这些文件来自真实运行，没有手工修改预测值或挑选窗口。

- `cpu/`、`gpu/`：分别包含每个固定窗口的 history.csv、prediction.csv、metrics.csv、run.json、result.png，以及整体 by_window.csv、summary.csv、comparison.png。
- 根目录 README 展示 GPU 图；CPU 结果保留在这里，数值可能略有不同。
- 动力学均为 CPU 计算；CPU/GPU 标签指 TimesFM 的推理设备。
- 不包含模型权重或原始大型数据文件。

## 实测环境（2026-11-16）

| 项目 | CPU | GPU |
|---|---|---|
| 系统 | Windows | Windows |
| Python | 3.12.7 | 3.12.7 |
| PyTorch | 2.8.0+cpu | 2.8.0+cu128 |
| GPU | 不使用 | NVIDIA GeForce RTX 5070 Ti |
| NumPy / Matplotlib | 1.26.4 / 3.8.4 | 1.26.4 / 3.8.4 |
| TimesFM 包 | 3.0.2 | 3.0.2 |

CPU 与 GPU 都已使用官方稳定版 PyTorch 2.8.0 实测全部窗口。GPU 在独立环境中重新安装验证；TimesFM 使用上述固定提交的源码。完整包版本记录在各设备目录的 environment.json，跨设备不要求浮点结果逐位一致。

- 上游源码提交：`e51928e27119cb17bebc005be2696b75e0a9e688`。
- 模型：`google/timesfm-3.0-pytorch`，未微调。
- 模型 revision：`43046b85ec22d584a13f8098c2ed39c889e129c2`。
- 模型权重 SHA-256：`a7592b0a8432baee54483254e5647856911ce69e09d09a9bb65904b2d98f17da`。
- per_core_batch_size=1；make_positive=False；use_symmetric_averaging=False；return_quantiles=False。
- 数据及其哈希见上一级 `data/orbit.manifest.json`。

复现命令：

```bash
python compare_windows.py --device cpu
python compare_windows.py --device gpu
```

新结果写入 `outputs/`，不会覆盖本目录。已有官方模型目录时可加 `--checkpoint 路径`，避免重复下载；请自行确认其版本与上述 revision 一致。

基础测试会重新从归档 XYZ 计算三维 RMS，并检查结果与参考数据的时间对齐。CI 运行基础物理模型和结果核算，不重新运行大模型。不要将 CI 通过理解为在云端重新验证了 TimesFM 推理。

## 误差补偿版更新

本目录全部窗口已重新运行，并替换为包含 TimesFM 误差补偿的结果。`history.csv` 记录实际历史输入与残差；`run.json` 记录补偿基线及残差构造方式。直接预测与误差预测共用同一原始模型，没有微调。

本次 CPU/GPU 运行均设置 `OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`；设备与线程配置引起的微小浮点差异正常。完整环境以本目录更新后的 environment.json 为准。

本碎片实验使用六份多历元 TLE，以及显式的“历元后 24 h 可用”离线假设；已知最新 TLE 的 SGP4 也作为基线。留后参考不是独立实测真值。此前单 TLE 结果与本次结果不属于同一任务，不能直接比较数值。
