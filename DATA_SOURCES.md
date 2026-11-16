# 多历元 TLE、时间切分与实验约定

## 固定数据来源

对象为 **COSMOS 2251 DEB，NORAD 33757**。使用 Aleksander Lidtke 公开研究代码仓库中归档的 **6 份不同历元真实 TLE**，日期为 2013-10-07 至 21 日。本次替换了旧版只有一份 2026 年 TLE 的传播示例。

- [原始数据固定提交](https://github.com/AleksanderLidtke/ConjunctionDetection/blob/c21596737757d87ea5bd2353c4b128c0795632d0/Release/Covariance%203LEs/33757.txt)
- [作者仓库及论文说明](https://github.com/AleksanderLidtke/ConjunctionDetection)
- 本地副本：[data/debris.tle](data/debris.tle)，保留原始字节和全部六个历元，未根据实验表现筛选。
- 上游许可及作者署名：[data/TLE_SOURCE_LICENSE.txt](data/TLE_SOURCE_LICENSE.txt)。保留上游 MIT 许可不意味着将第三方轨道数据重新授权。

SHA-256、六个 UTC 历元、窗口与参考选择记录在 [data/orbit.manifest.json](data/orbit.manifest.json)。来源是公开研究归档，**不是本次从 Space-Track 下载的历史库**，也没有原始测站观测和独立精密轨道。

## 三份 TLE 分别做什么

按时间排序，将相邻三份记为 A（旧）、B（更新）、C（留后）。六份恰好组成四组三元组，不按误差挑选。

1. 归档未给出实际发布时间，**假设历元后 24 h 可用**。起报时刻取 B 的假设可用时刻向上取整到分钟。这只是离线回放假设，并非对真实发布延迟的测量或保证。
2. 起报时用 B 回算过去 360 个位置、速度，得到历史参考；历史不是逐分钟实测轨迹，也不是每个历史时刻在线可用的轨迹。
3. 用 A 传播相同历史时刻，构造 `e_past = r_B - r_A`；TimesFM 预测未来误差，`r_hybrid = r_A_future + e_forecast`。
4. 保留 **A 的 SGP4、B 的 SGP4、二体、J2、TimesFM 直接预测**。二体/J2 从 B 在起报点的瞬时状态初始化；B 的 SGP4 是必须比较的最新信息基线。
5. C 的历元晚于预测结束，只在准备脚本中生成留后评分参考，不传给模型。预测端 `available_tles()` 严格按假设可用时间筛选 A/B。

历史与未来参考由不同 TLE 生成，起报边界可有跳变，图上不连接、不平滑、不把 C 回填到历史输入。C 仍是另一份估计轨道，**相对 C 的偏差不是相对实测真值的精度**。C 与起报间隔较长时，其回算误差也影响评分。相邻窗口共享 TLE，不能当作四个独立目标。

误差补偿接近 B 的 SGP4 可能主要说明模型恢复了 A/B 的轨道差，不能据此宣称发现了真实 SGP4 误差规律。若评价实际轨道精度，需要独立观测或精密轨道；若评价在线能力，需要可靠的历史发布时间记录。

## 复现和坐标

```bash
python advanced/prepare_data.py
```

脚本只读附带 TLE，检查固定列、校验和、对象编号、不同历元及 SGP4 错误码；不联网、不修改根数。生成四个 420 点窗口，共 1,680 行。正常实验直接读取附带 CSV。

- 时标为 UTC，CSV 时间列是自 **2000-01-01 00:00 UTC** 起的秒数；本例时段不跨闰秒。
- 每段历史为 -359 至 0 min，未来 +1 至 +60 min；段内间隔 60 s，段间有时间空隙。
- SGP4 使用 `sgp4==2.27`、WGS72，拆分儒略日计算；输出 km、km/s 转为 m、m/s。
- 位置速度为 **TEME**，不是 ECEF 或 J2000。不能将 TLE 平均根数当作瞬时二体根数。
- `orbit.py` 用 RK4（5 s 内部步长）积分二体/J2；在一小时短弧内近似将 TEME 视为固定惯性系，未包括框架细微变化及全部摄动。SGP4 使用配套平均根数，J2 不是 SGP4 的另一种实现。

参考：[CelesTrak GP 说明](https://celestrak.org/NORAD/documentation/gp-data-formats.php)、[SGP4 作者文档](https://pypi.org/project/sgp4/2.27/)、[Vallado 等传播模型说明](https://celestrak.org/publications/AIAA/2006-6753/)。

## 模型与指标

使用原始 [TimesFM 3.0](https://huggingface.co/google/timesfm-3.0-pytorch)，无微调。源码固定 `e51928e27119cb17bebc005be2696b75e0a9e688`；权重固定 `43046b85ec22d584a13f8098c2ed39c889e129c2`。

三个坐标独立预测；float64 下减末值后转 float32。直接位置分支用 km，误差分支用 m，输出再还原单位和末值。保持官方内部归一化、去趋势默认设置；不利用留后参考选择模型或参数。

保存历史参考、动力学、残差及未来各模型 XYZ，方便学生重算。先计算每个未来点的 XYZ 欧氏距离，再计算 RMS；所有窗口等长，合并 RMS 是各窗口 RMS 平方均值的平方根。CPU/GPU 均归档全部四个窗口，旧版同一 TLE 的结果可通过 Git 历史查看。

自编代码为 MIT；上游数据和模型遵循各自许可。仓库不分发 TimesFM 权重。
