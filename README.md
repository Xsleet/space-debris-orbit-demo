# 空间碎片轨道根数预测 Demo

**历史 TLE → TimesFM 预测六个轨道根数 → SGP4 预报未来 60 分钟。**

使用 COSMOS 2251 DEB（NORAD 33757）的 6 份真实 TLE，比较三种方法：

- 最新已知 TLE＋SGP4：基线。
- 根数线性外推＋SGP4：简单的趋势预测。
- **TimesFM 根数预测＋SGP4**：原始模型，无微调，支持 CPU/GPU。

## 0. 先认识几个概念

| 名称 | 简单理解 | 在本 Demo 中做什么 |
|---|---|---|
| **TLE** | Two-Line Element，两行文本记录的一份轨道参数 | 提供不同历元的历史轨道信息 |
| **历元** | 一份轨道参数对应的参考时刻 | 区分每份 TLE 描述的是哪个时刻；不等同于发布时间 |
| **轨道根数** | 描述轨道大小、形状、朝向和物体相位的参数 | 本例预测 n、e、i、Ω、ω、M 六个平均根数 |
| **SGP4** | 根据轨道参数和指定时间，计算位置、速度的传播模型 | 把 TLE 或预测根数转换成未来 XYZ 轨道 |
| **TimesFM** | 已预训练的时间序列预测模型 | 根据历史根数序列，推算起报时的六根数 |

例如：**TLE 给出某时刻的轨道参数，SGP4 用它计算一小时后的位置；TimesFM 则尝试先更新这些参数，再交给 SGP4。**

TLE 的标准格式见 [CelesTrak](https://celestrak.org/NORAD/documentation/tle-fmt.php)；SGP4 的输入输出见 [作者文档](https://pypi.org/project/sgp4/2.27/)。

第一次接触这些名词，先读 [3 分钟基础知识：真实 TLE 示例与预报流程](BASICS.md)，再运行下面的命令。

## 1. 下载与运行

安装 Python **3.11 或 3.12** 和 Git：

```bash
git clone https://github.com/Xsleet/space-debris-orbit-demo.git
cd space-debris-orbit-demo
python -m venv .venv
```

激活环境：Windows 命令提示符用 `.venv\Scripts\activate.bat`；Linux/macOS 用 `source .venv/bin/activate`。PowerShell 可将下文 `python` 换为 `.\.venv\Scripts\python.exe`。

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

首次下载约 **1.32 GB** 模型，不需要训练。CPU/GPU 只改变 TimesFM 推理设备，SGP4 在 CPU 计算。macOS 的 CPU 安装首条命令改为 `python -m pip install torch==2.8.0`。

换窗口或运行全部四个窗口：

```bash
python main.py --device cpu --window 3
python compare_elements.py --device cpu
python compare_elements.py --device gpu
```

只运行两种基线、不下载模型：`python -m pip install -r requirements.txt`，然后 `python main.py --model linear`。批处理同样支持 `--model linear`；旧命令 `compare_windows.py` 也是根数预测入口。

## 2. 实测结果

全部 **4 个固定窗口、240 个未来位置点**的合并三维 RMS（m），相对留后较晚 TLE：

| 方法 | CPU | GPU |
|---|---:|---:|
| 最新已知 TLE＋SGP4 | 362.454 | 362.454 |
| 根数线性外推＋SGP4 | 354.956 | 354.956 |
| **TimesFM 根数预测＋SGP4** | **206.150** | **206.176** |

相对最新 TLE＋SGP4，整段 RMS 降低约 **43.1%**，四个窗口均改善。但第 60 分钟的末端误差有两个窗口变大，不能理解为每个预报时刻都更好。

![四个窗口的根数预测比较](data/comparison.png)

![过去轨迹与根数预测得到的未来轨道](data/example.png)

上图展示过去 6 小时的 SGP4 轨迹、起报虚线与未来轨道；下图比较三种方法的三维偏差。**过去轨迹仅用于显示，模型真正输入的是起报前 2～5 份 TLE 的根数。**

## 3. 本科生从哪里读起？

先读约 90 行的 [elements.py](elements.py)，理解“选历史 → 处理角度 → 预测根数 → SGP4”。[main.py](main.py) 运行单个窗口，[compare_elements.py](compare_elements.py) 批量比较，[plotting.py](plotting.py) 保存与画图。

`outputs/elements-cpu/window-01/` 中：

- `element_tle_knots.csv`、`element_input.csv`：真实 TLE 节点及小时插值输入。
- `element_forecast.csv`、`elements.json`：根数预测和起报时的六根数。
- `prediction.csv`、`metrics.csv`：三种方法的 XYZ、整段 RMS 和末端误差。
- `history.csv`、`result.png`、`run.json`：用于显示的过去轨迹、图和实验说明。

GPU 对应 `elements-gpu`；基线模式对应 `linear-cpu`。批处理另存 `by_window.csv`、`summary.csv`、`comparison.png`。已实测结果见 [reference](reference/README.md)。

## 4. 如何理解这个实验？

仅有一个碎片、6 份 TLE、4 个相关窗口。每窗使用全部已知 TLE，插值到小时网格以适配 TimesFM；**插值没有增加观测信息**。归档没有发布时间，因此假设每份 TLE 在历元后 24 h 可用。后续 TLE 仅用于评分，属于参考轨道，不是独立精密真值。

[根数含义、时间处理和逐窗结果](ELEMENTS_EXPERIMENT.md) · [数据来源和评价约定](DATA_SOURCES.md)

`python -m unittest discover -v` 验证时间边界、角度相位、SGP4 重建和归档指标。GitHub CI 检查 Windows/Linux、Python 3.11/3.12 的基线流程，不下载 TimesFM。

自编代码为 MIT；数据与模型遵循各自许可。TimesFM 3.0 权重用于非商业、非生产教学研究，仓库不分发权重，见 [模型许可](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)。

同系列：[精密钟差预报](https://github.com/Xsleet/gracefo-clock-prediction) · [低轨卫星精密轨道预测](https://github.com/Xsleet/leo-precise-orbit-demo)
