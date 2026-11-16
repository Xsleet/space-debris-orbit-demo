# 多历元 TLE、时间切分与实验约定

## 固定数据来源

对象为 **COSMOS 2251 DEB，NORAD 33757**。使用 Aleksander Lidtke 公开研究仓库归档的 **6 份真实 TLE**，历元为 2013-10-07 至 21 日，保留全部记录。

- [原始数据固定提交](https://github.com/AleksanderLidtke/ConjunctionDetection/blob/c21596737757d87ea5bd2353c4b128c0795632d0/Release/Covariance%203LEs/33757.txt)
- [作者仓库及论文说明](https://github.com/AleksanderLidtke/ConjunctionDetection)
- [本地 TLE](data/debris.tle) · [上游许可及署名](data/TLE_SOURCE_LICENSE.txt)
- [SHA-256、历元及窗口信息](data/orbit.manifest.json)

这是公开研究归档，没有测站观测、独立精密轨道或原始发布时间。

## 起报前可以使用哪些数据？

假设每份 TLE 在**历元后 24 h 可用**。依次取第 2～5 份 TLE 的假设可用时刻，向上取整到分钟，作为四个起报时刻：

| 窗口 | 起报时间 UTC | 起报前已知 TLE 数 |
|---|---|---:|
| 1 | 2013-10-11 02:09 | 2 |
| 2 | 2013-10-15 19:50 | 3 |
| 3 | 2013-10-18 03:27 | 4 |
| 4 | 2013-10-20 03:59 | 5 |

每窗使用全部已知 TLE 的六个平均根数。最新已知 TLE 直接传播形成 SGP4 基线；线性外推和 TimesFM 则预测起报历元的根数，再经 SGP4 生成未来 +1 至 +60 分钟的位置。

最新已知 TLE 的历元距起报约 24 h，所以根数要先跨越这段时间。各根数按历史标准差缩放；角度展开，平近点角先去除快速相位，预测后恢复。详见 [实验说明](ELEMENTS_EXPERIMENT.md)。

**后续第 3～6 份 TLE 分别作为四窗评分参考，历元均晚于预测结束。** 它们不进入模型输入。相邻窗口共享历史记录，不能当作四个独立目标。

图中的过去 6 小时轨迹由最新已知 TLE 在起报时回算，仅作显示，不是逐分钟实测数据，也不是 TimesFM 输入。历史与未来参考来自不同 TLE，边界可能有跳变，不作拼接平滑。

## 数据文件和坐标

正常实验直接读附带数据。重建显示轨迹及评分参考：

```bash
python advanced/prepare_data.py
```

脚本检查 TLE 固定列、校验和、对象编号、历元及 SGP4 错误码。生成四个 420 点窗口，共 1,680 行；段内间隔 60 s，段间有时间空隙。

- 时标为 UTC；CSV 时间列为自 2000-01-01 00:00 UTC 起的秒数，本例不跨闰秒。
- SGP4 使用 `sgp4==2.27`、WGS72，输出 km、km/s 转为 m、m/s。
- 坐标系为 **TEME**，不是 ECEF 或 J2000。
- 模型使用 SGP4 配套的 TLE 平均根数；SGP4 初始化时保留最新已知 B*。

参考：[CelesTrak GP 说明](https://celestrak.org/NORAD/documentation/gp-data-formats.php) · [SGP4 作者文档](https://pypi.org/project/sgp4/2.27/)

## 模型与指标

原始 [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch)，无微调。源码固定 `e51928e27119cb17bebc005be2696b75e0a9e688`；权重 revision 固定 `43046b85ec22d584a13f8098c2ed39c889e129c2`。六个根数独立预测，保留官方内部归一化、去趋势默认设置。

每个未来点的三维偏差为 XYZ 欧氏距离；整段 RMS 是这些偏差平方均值的平方根。合并 RMS 使用四窗全部 240 点，不能直接平均四个窗口的 RMS。

评价对象是与后续 TLE 的接近程度，不能直接当作实测精度。24 h 可用时间是离线假设；小时插值不增加独立观测。CPU/GPU 均保留全部四个窗口，不按未来误差选样本或调整参数。

自编代码为 MIT；上游数据和模型遵循各自许可，仓库不分发模型权重。
