# 数据、坐标与实验约定

## 真实的是 TLE，参考轨迹是传播得到的

对象：**COSMOS 2251 DEB，NORAD 33757**。2026-11-16 从 [CelesTrak 单目标 TLE 接口](https://celestrak.org/NORAD/elements/gp.php?CATNR=33757&FORMAT=TLE) 获取快照。目标身份可在 [COSMOS 2251 碎片目录](https://celestrak.org/NORAD/elements/table.php?FORMAT=tle&GROUP=cosmos-2251-debris) 核对。

[data/debris.tle](data/debris.tle) 保存固定快照，名称去掉末尾填充空格、统一为 LF 换行；TLE 两行数值保持获取时的内容。历元为 **2026-11-15 17:52:35.014079 UTC**。来源、历元、坐标系和 SHA-256 见 [data/orbit.manifest.json](data/orbit.manifest.json)。在线接口会更新，不能用今天的在线响应代替归档快照核验。

使用 `sgp4==2.27`、WGS72 常数，从 TLE 历元传播至 +25 h，每 60 s 生成一行，共 1,501 行。**过去和未来都来自同一份 TLE 的 SGP4 传播**；没有独立测站观测、独立精密轨道或真实未来真值。

因此，这是固定对象的传播模型对比：TimesFM 尝试延续 SGP4 生成的轨迹，二体/J2 从 SGP4 在起报时刻的瞬时状态传播。SGP4 本身是参考生成器，不能再把“SGP4 对自身误差为 0”当作性能优势。

## 复现

安装基础依赖后运行：

```bash
python advanced/prepare_data.py
```

脚本验证 TLE 的固定列格式、校验和与 SGP4 错误码，将 km、km/s 转为 m、m/s。采用拆分的儒略日和小数日计算，避免将大儒略日与小时间间隔直接相加损失精度。正常实验读取已附带的 CSV，无需网络。

原始 TLE 是平均根数，适用于配套的 SGP4 模型；不能直接把它当作二体瞬时根数。若要更新对象或 TLE，应同时修改脚本中的对象来源等元数据并重新评估，不能继续引用本仓库的固定快照结果。

参考：[CelesTrak GP 格式说明](https://celestrak.org/NORAD/documentation/gp-data-formats.php)、[SGP4 作者文档](https://pypi.org/project/sgp4/2.27/)、[Vallado 等：Revisiting Spacetrack Report #3](https://celestrak.org/publications/AIAA/2006-6753/)。

## 时间和坐标

- CSV 的 `seconds_since_tle_epoch` 是相对固定 TLE 历元的秒数；图中起报时间显示为 UTC。
- 位置和速度为 **TEME**，不是 ECEF，也不是 J2000。没有把 TEME 数值直接标成地理经纬度。
- 窗口按历元后 +6、+12、+18、+24 h 固定；每段历史为 -359 至 0 min 的 360 点，未来为 +1 至 +60 min 的 60 点。
- 数据覆盖多个窗口，但都是同一对象、同一 TLE，不能代表跨目标泛化。

## 简化动力学

[orbit.py](orbit.py) 用 RK4（5 s 内部步长）积分二体/J2。常数采用 WGS72：GM=3.986008e14 m³/s²，赤道半径=6378135 m，J2=1.082616e-3，与 SGP4 参考的常数体系一致。

在一小时短弧内将 TEME 近似作为固定惯性坐标系，不加 ECEF 的科里奥利或离心项；没有建模框架随时间的细微变化及全部摄动。J2 不是 SGP4 的另一种实现，二者动力学细节不同。

这里没有观测定轨、轨道协方差或独立轨道误差验证。要评价真实预测精度，需要独立实测或高质量轨道基准；只比较同一 TLE 的传播结果不能做到这一点。

## TimesFM 与评分

使用官方原始 [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch)，没有微调。源码固定到 `e51928e27119cb17bebc005be2696b75e0a9e688`；权重固定到 `43046b85ec22d584a13f8098c2ed39c889e129c2`。

XYZ 分别预测。输入在 float64 下减末值、由 m 转 km 后转 float32；输出转回 m、加回末值。官方内部归一化及去趋势保持默认。物理基线还使用起报点速度，而 TimesFM 只用历史位置，输入特征并不完全相同。

预测函数只接收历史；留后 60 点只用于评分。每点先计算 XYZ 欧氏距离，再计算 60 点 RMS；汇总时合并全部固定窗口。该量是**与 SGP4 的偏差**，不是实测精度。

MIT 只适用于本项目自编代码；原始数据、依赖和模型遵循来源方许可。仓库不分发 TimesFM 权重，模型由用户从官方源下载。
