"""空间碎片轨道根数预测：历史 TLE → 六根数 → SGP4 预报一小时。"""
import argparse
from datetime import timedelta
import json
from pathlib import Path
import sys

import numpy as np
from elements import predict_elements
from plotting import save_result as write_result
from tle import EPOCH, positions

ROOT = Path(__file__).resolve().parent
HISTORY, HORIZON, STEP = 360, 60, 60
WINDOWS = 4
INFO = json.loads((ROOT / "data/orbit.manifest.json").read_text(encoding="utf-8"))
OBJECT = INFO["object"]
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


def predict(origin, evaluator=None):
    """只接收起报时刻；实际输入为已知 TLE 根数，不接收未来参考位置。"""
    future = origin + np.arange(1, HORIZON + 1) * STEP
    predictions, audit = predict_elements(origin, future, evaluator)
    latest = audit["known"][-1]
    return {"sgp4_latest": positions(latest, future)[:, :3], **predictions}, audit


def save_result(sample, predictions, audit, number, output, device):
    metadata = {"window": number, "object": OBJECT, "origin": time_label(sample[HISTORY - 1, 0]),
        "frame": "TEME", "reference": "Held-out later TLE", "device": device,
        "models": list(predictions), "display_history_points": HISTORY,
        "display_history_kind": "Latest known TLE/SGP4 past arc; display only, not model input",
        "horizon_points": HORIZON, "step_seconds": STEP,
        "timesfm_revision": REVISION if "elements_timesfm" in predictions else None,
        "model_input": "Six TLE mean elements on hourly grid; see element_input.csv; no fine-tuning",
        "known_tle_count": len(audit["known"]),
        "tle_window": INFO["windows"][number - 1],
        "availability_assumption_seconds": INFO["availability_assumption_seconds"]}
    write_result(sample, predictions, audit, metadata, output)


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
    parser.add_argument("--model", choices=["timesfm", "linear"], default="timesfm")
    parser.add_argument("--window", type=int, choices=range(1, WINDOWS + 1), default=1)
    parser.add_argument("--checkpoint", default=MODEL_ID)
    args = parser.parse_args()
    if args.model == "linear" and args.device == "gpu":
        parser.error("线性根数与 SGP4 使用 CPU；GPU 选项用于 TimesFM")
    print("历史 TLE 根数预报；评分参考为留后的较晚 TLE，不是实测真值。", flush=True)
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    sample = select_window(data, args.window)
    evaluator = load_evaluator(args.device, args.checkpoint) if args.model == "timesfm" else None
    predictions, audit = predict(sample[HISTORY - 1, 0], evaluator)
    mode = "elements" if evaluator is not None else "linear"
    output = ROOT / "outputs" / f"{mode}-{args.device}" / f"window-{args.window:02d}"
    save_result(sample, predictions, audit, args.window, output, args.device)
    print(f"结果：{output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
