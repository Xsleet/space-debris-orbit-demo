"""空间碎片轨道预测：多历元 TLE、SGP4 和 TimesFM 误差补偿。"""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np
from plotting import save_result as write_result

from orbit import propagate
from tle import EPOCH, available_tles, positions

ROOT = Path(__file__).resolve().parent
HISTORY, HORIZON, STEP = 360, 60, 60
WINDOWS = 4
ROTATING = False
FRAME = "TEME"
INFO = json.loads((ROOT / "data/orbit.manifest.json").read_text(encoding="utf-8"))
OBJECT = INFO["object"]
REFERENCE = "Held-out later TLE"
ERROR_LABEL = "3-D deviation from later TLE (m)"
BASE, HYBRID = "sgp4_old", "sgp4_timesfm"
RESIDUAL_CONSTRUCTION = "Latest known TLE minus older TLE on past arc; both available by assumed epoch + 24 h"
MODEL_ID = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def origin_seconds(number):
    return INFO["windows"][number - 1]["origin_seconds"]


def time_label(seconds):
    return (EPOCH + timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M:%S UTC")


def select_window(data, number):
    if number not in range(1, WINDOWS + 1) or data.ndim != 2 or data.shape[1] != 7:
        raise ValueError("窗口编号或七列位置速度数据格式不正确")
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("时间必须严格递增")
    expected = origin_seconds(number) + np.arange(1 - HISTORY, HORIZON + 1) * STEP
    start = np.searchsorted(data[:, 0], expected[0])
    sample = data[start:start + len(expected)]
    if not np.array_equal(sample[:, 0], expected) or not np.isfinite(sample).all():
        raise ValueError("窗口存在缺测或无效值，不插值")
    return sample


def forecast_xyz(values, evaluator, scale):
    """三个轴独立预测；位置以 km 输入，误差以 m 输入，均先减末值。"""
    last = values[-1]
    inputs = ((values - last) / scale).T.astype(np.float32)
    outputs = list(evaluator.predict_batch(list(inputs), horizon=HORIZON,
        return_quantiles=False, make_positive=False, use_symmetric_averaging=False))
    return np.stack([out.forecast for out in outputs], axis=1).astype(np.float64) * scale + last


def predict(history, evaluator=None):
    """只接收过去 360 点；误差预测加回同一条动力学轨道，没有微调。"""
    if history.shape != (HISTORY, 7) or not np.isfinite(history).all():
        raise ValueError("预测只接受完整的 360 点历史，不接收未来参考")
    result = {"two_body": propagate(history[-1, 1:7], HORIZON, STEP, False, ROTATING)[:, :3],
              "j2": propagate(history[-1, 1:7], HORIZON, STEP, True, ROTATING)[:, :3]}
    old, latest = available_tles(history[-1, 0])
    future = history[-1, 0] + np.arange(1, HORIZON + 1) * STEP
    past_base = positions(old, history[:, 0])[:, :3]
    result["sgp4_old"] = positions(old, future)[:, :3]
    result["sgp4_latest"] = positions(latest, future)[:, :3]  # 必须保留最新已知 TLE 基线。
    past_error = history[:, 1:4] - past_base
    if evaluator is not None:
        result["timesfm"] = forecast_xyz(history[:, 1:4], evaluator, scale=1000)
        result[HYBRID] = result[BASE] + forecast_xyz(past_error, evaluator, scale=1)
    if any(values.shape != (HORIZON, 3) or not np.isfinite(values).all() for values in result.values()):
        raise ValueError("模型未返回完整、有效的 XYZ 预测")
    return result, past_error


def save_result(sample, predictions, past_error, number, output, device):
    metadata = {"window": number, "object": OBJECT, "origin": time_label(sample[HISTORY - 1, 0]),
        "frame": FRAME, "reference": REFERENCE, "error_label": ERROR_LABEL, "device": device,
        "history_points": HISTORY, "horizon_points": HORIZON, "step_seconds": STEP,
        "timesfm_revision": REVISION, "residual_baseline": BASE, "hybrid_model": HYBRID,
        "timesfm_input": "Independent XYZ (km) and residual XYZ (m), last-value centered; no fine-tuning",
        "residual_construction": RESIDUAL_CONSTRUCTION}
    metadata["tle_window"] = INFO["windows"][number - 1]
    metadata["availability_assumption_seconds"] = INFO["availability_assumption_seconds"]
    write_result(sample, predictions, past_error, metadata, output)


def load_evaluator(device, checkpoint):
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    if device == "gpu" and not torch.cuda.is_available():
        raise ValueError("未发现 NVIDIA GPU，请使用 --device cpu")
    print("加载原始 TimesFM；首次下载约 1.32 GB 权重……", flush=True)
    return TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, revision=REVISION,
        device="cuda" if device == "gpu" else "cpu", per_core_batch_size=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", choices=["timesfm", "physics"], default="timesfm")
    parser.add_argument("--window", type=int, choices=range(1, WINDOWS + 1), default=1)
    parser.add_argument("--checkpoint", default=MODEL_ID)
    args = parser.parse_args()
    if args.model == "physics" and args.device == "gpu":
        parser.error("动力学模型使用 CPU；GPU 选项用于 TimesFM")
    print("多历元 TLE 离线回放；评分参考为留后的较晚 TLE，不是实测真值。", flush=True)
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    try:
        sample = select_window(data, args.window)
    except ValueError as error:
        parser.error(str(error))
    evaluator = None
    if args.model == "timesfm":
        evaluator = load_evaluator(args.device, args.checkpoint)
    predictions, past_error = predict(sample[:HISTORY], evaluator)
    output = ROOT / "outputs" / f"{args.model}-{args.device}" / f"window-{args.window:02d}"
    save_result(sample, predictions, past_error, args.window, output, args.device)
    print(f"结果：{output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
