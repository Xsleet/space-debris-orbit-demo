# 空间碎片轨道预测 Demo

用 **COSMOS 2251 DEB（NORAD 33757）的 6 份不同历元 TLE**，演示动力学与时间序列模型的结合。

**过去 6 小时 → 起报虚线 → 未来 60 分钟**，新增 **动力学＋TimesFM 误差预报**，保留 CPU/GPU。

也可试验 **TimesFM 预测轨道根数 → SGP4**：[实测对比与原理](ELEMENTS_EXPERIMENT.md)。

## 1. 先理解这三步

本例用三份按历元排列的 TLE：**A（旧）、B（最新已知）、C（留后评分）**。

1. 起报时用 B 回算过去 6 小时轨迹，计算 `历史误差 = SGP4(B) - SGP4(A)`。
2. 原始 TimesFM 预测未来 60 分钟误差。
3. `补偿预测 = SGP4(A) 的未来轨道 + 预测误差`；用 C 传播的轨道评分。

另外比较 **最新已知 TLE 的 SGP4、二体、J2、TimesFM 直接预测 XYZ**。保留最新 SGP4 很重要：补偿旧 TLE 后变好，可能只是恢复了新 TLE 已包含的信息。

**过去轨迹是起报时由 B 回算的参考序列，未来参考由 C 传播，都不是实测真值。** 原始归档没有发布时间，因此明确假设每份 TLE 在“历元后 24 小时”可用；仅在这个假设下作因果回放，不声称复现真实在线发布流程。

先读约 130 行的 [main.py](main.py)，核心是 `predict()`；画图单独放在 [plotting.py](plotting.py)，动力学在 [orbit.py](orbit.py)。**全部 TimesFM 分支使用同一个官方原始模型，没有微调。**

## 2. 下载与运行

安装 Python **3.11 或 3.12** 和 Git：

```bash
git clone https://github.com/Xsleet/space-debris-orbit-demo.git
cd space-debris-orbit-demo
python -m venv .venv
```

Windows 命令提示符：`.venv\Scripts\activate.bat`；Linux/macOS：`source .venv/bin/activate`。PowerShell 可将下文 `python` 换为 `.\.venv\Scripts\python.exe`。

**CPU：**

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-timesfm.txt
python main.py --device cpu
```

**GPU：NVIDIA 显卡，驱动支持 CUDA 12.8。**

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-timesfm.txt
python main.py --device gpu
```

CPU/GPU 只改变 TimesFM 推理设备，动力学仍在 CPU。首次自动下载约 **1.32 GB** 模型，不需要训练；正常运行使用附带数据，无需轨道数据网站登录或服务器。macOS 将 CPU 安装首条命令换为 `python -m pip install torch==2.8.0`。[PyTorch 官方安装说明](https://pytorch.org/get-started/previous-versions/#v280)。

只跑动力学、不下载模型：`python -m pip install -r requirements.txt`，然后 `python main.py --model physics`。

## 3. 看懂结果

![过去轨迹、历史误差与未来预测](data/example.png)

- 上图：Z 坐标的过去轨迹与未来预测。
- 中图：**去除动力学轨道后的 Z 误差 → 起报虚线 → 未来误差预报**；零线代表不补偿。黑色点线仅用于事后评分。
- 下图：全部 XYZ 的三维误差，使用对数纵轴；不能只凭 Z 曲线是否重合判断精度。

碎片图在起报前后切换 B/C 两份参考 TLE，黑色留后曲线不必接上历史曲线。该差异保留，不做拼接平滑。

**全部 4 个固定窗口合并的三维 RMS（m）**，相对留后较晚 TLE，不是实测精度：

| 方法 | CPU | GPU |
|---|---:|---:|
| 二体 | 33,195.013 | 33,195.013 |
| J2 | 391.027 | 391.027 |
| 旧 TLE 的 SGP4 | 634.808 | 634.808 |
| 最新已知 TLE 的 SGP4 | 362.454 | 362.454 |
| TimesFM 直接预测 XYZ | 42,887.874 | 42,888.340 |
| **旧 SGP4＋TimesFM 误差预报** | 360.902 | 360.902 |

补偿相对旧 SGP4 改善明显，但与**最新已知 TLE 的 SGP4**非常接近，合并 RMS 仅降低约 **0.43%**，其中一个窗口反而变差。这个小样本不能证明对真实轨道预报有稳定提升。

![全部窗口比较](data/comparison.png)

`outputs/timesfm-cpu/window-01/` 中：`result.png` 是图，`history.csv` 保存历史参考、动力学与误差，`prediction.csv` 保存未来参考及各方法预测，`metrics.csv` 保存 RMS/60 分钟误差，`run.json` 保存方法和窗口说明。GPU 对应 `timesfm-gpu`。

```text
三维误差 = sqrt(dx² + dy² + dz²)
RMS(3D) = sqrt(全部未来点的三维误差平方的平均值)
```

## 4. 换窗口与自主分析

六份 TLE 的历元为 **2013-10-07 至 21 日**。按顺序取相邻三份组成 4 个窗口，起报时刻为中间 TLE 的假设可用时间向上取整到分钟：

| 窗口 | 起报时间 UTC |
|---|---|
| 1 | 2013-10-11 02:09 |
| 2 | 2013-10-15 19:50 |
| 3 | 2013-10-18 03:27 |
| 4 | 2013-10-20 03:59 |

相邻窗口共享部分 TLE，不是四个独立目标。每个窗口未来参考 TLE 的历元都晚于预测结束。

```bash
python main.py --device cpu --window 3
python compare_windows.py --device cpu
python compare_windows.py --device gpu
```

每个窗口取 360 点（-359 至 0 分钟），预测 +1 至 +60 分钟；间隔 60 s。批处理只加载一次模型，保存所有窗口及 `by_window.csv`、`summary.csv`、`comparison.png`。也支持 `--model physics`。

给学生的三个问题：补偿在哪些窗口有效？为何直接预测 XYZ 比预测误差困难？补偿旧 TLE 的收益，有多少来自最新 TLE 本身？

数据来源、坐标、时间假设和限制见 [DATA_SOURCES.md](DATA_SOURCES.md)；完整 CPU/GPU 实测结果见 [reference](reference/README.md)。`python -m unittest discover -v` 检查时间边界、误差加回、TLE 选择和归档指标。CI 检查 Windows/Linux、Python 3.11/3.12 的基础流程，不在线下载大模型。

## 5. 预测轨道根数会更好吗？

本次小样本中，**会**。相同 4 个起报窗口、相同未来 60 分钟、相同留后 TLE 评分，实测三维合并 RMS（m）：

| 方法 | CPU | GPU |
|---|---:|---:|
| 最新已知 TLE → SGP4 | 362.454 | 362.454 |
| 旧 SGP4＋TimesFM 误差预报 | 360.902 | 360.902 |
| 根数线性外推 → SGP4 | 354.956 | 354.956 |
| **TimesFM 预测根数 → SGP4** | **206.150** | **206.176** |

相对最新已知 TLE 的 SGP4，整段 RMS 降低约 **43.1%**，4 个窗口均改善。但只看第 60 分钟的末端误差，有 2 个窗口变差；不能理解为每个预报时刻都更好。

```bash
python compare_elements.py --device cpu
python compare_elements.py --device gpu
```

安装方式同第 2 节。核心在约 90 行的 [elements.py](elements.py)。输出到 `outputs/elements-cpu/` 或 `outputs/elements-gpu/`，已实测结果在 [reference/elements](reference/elements)。

根数分支使用起报前全部 **2～5 份真实 TLE**，小时插值用于适配 TimesFM，不能算作新增观测；误差分支仍使用 A/B 构造的过去 6 小时序列。**这比较的是两套预报方案，不是只替换输入坐标的受控实验。** 数据只涉及一个碎片，评分参考仍为后续 TLE。[详细协议和每窗结果](ELEMENTS_EXPERIMENT.md)。

自编代码为 MIT；数据和模型遵循各自许可。TimesFM 3.0 权重用于非商业、非生产教学研究，仓库不分发权重，见 [模型许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。

同系列：[精密钟差预报](https://github.com/Xsleet/gracefo-clock-prediction) · [低轨卫星精密轨道预测](https://github.com/Xsleet/leo-precise-orbit-demo)
