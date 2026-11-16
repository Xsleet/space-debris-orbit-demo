"""空间碎片轨道预测Demo：用固定 TLE/SGP4 参考轨迹对比 TimesFM、二体和 J2。"""
import argparse
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from orbit import propagate

ROOT = Path(__file__).resolve().parent
HISTORY, HORIZON, STEP = 360, 60, 60
WINDOWS = 4
ROTATING = False
FRAME = "TEME"
INFO = json.loads((ROOT / "data/orbit.manifest.json").read_text(encoding="utf-8"))
OBJECT = INFO["object"]
REFERENCE = "SGP4 reference (not observations)"
ERROR_LABEL = "3-D deviation from\nSGP4 (m)"
MODEL_ID = "google/timesfm-3.0-pytorch"
REVISION = "43046b85ec22d584a13f8098c2ed39c889e129c2"


def origin_seconds(number):
    return number * 6 * 3600


def time_label(seconds):
    return (datetime.fromisoformat(INFO["tle_epoch_utc"]) + timedelta(seconds=float(seconds))).strftime("%Y-%m-%d %H:%M:%S UTC")


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


def predict(history, evaluator=None):
    """这里没有未来数据参数。物理模型用末点位置速度；TimesFM 用历史 XYZ 位置。"""
    result = {"two_body": propagate(history[-1, 1:7], HORIZON, STEP, False, ROTATING)[:, :3],
              "j2": propagate(history[-1, 1:7], HORIZON, STEP, True, ROTATING)[:, :3]}
    if evaluator is not None:
        # float64 下先减末值、转为 km，再转 float32。三个轴分别预测，不加载微调权重。
        last = history[-1, 1:4]
        inputs = ((history[:, 1:4] - last) / 1000).T.astype(np.float32)
        outputs = list(evaluator.predict_batch(list(inputs), horizon=HORIZON,
            return_quantiles=False, make_positive=False, use_symmetric_averaging=False))
        result["timesfm"] = np.stack([out.forecast for out in outputs], axis=1).astype(np.float64) * 1000 + last
    if any(values.shape != (HORIZON, 3) or not np.isfinite(values).all() for values in result.values()):
        raise ValueError("模型未返回完整、有效的 XYZ 预测")
    return result


def save_result(sample, predictions, number, output, device):
    reference = sample[HISTORY:, 1:4]  # 所有模型预测完后，才读取留后参考值评分。
    errors = {name: np.linalg.norm(values - reference, axis=1) for name, values in predictions.items()}
    metrics = [{"model": name, "rms_3d_m": float(np.sqrt(np.mean(error**2))),
                "endpoint_3d_m": float(error[-1])} for name, error in errors.items()]
    output.mkdir(parents=True, exist_ok=True)
    columns = [sample[HISTORY:, 0], *reference.T]
    header = ["time_seconds", "reference_x_m", "reference_y_m", "reference_z_m"]
    for name, values in predictions.items():
        columns.extend([*values.T, errors[name]])
        header.extend([f"{name}_{axis}_m" for axis in "xyz"] + [f"{name}_deviation_m"])
    np.savetxt(output / "prediction.csv", np.column_stack(columns), delimiter=",",
               header=",".join(header), comments="", fmt="%.17g")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    metadata = {"window": number, "origin": time_label(sample[HISTORY - 1, 0]),
        "frame": FRAME, "reference": REFERENCE, "device": device, "history_points": HISTORY,
        "horizon_points": HORIZON, "step_seconds": STEP, "timesfm_revision": REVISION,
        "timesfm_input": "Independent XYZ positions, last-value centered in km; no fine-tuning"}
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    # 上图选 Z 分量便于看周期；下图用全部 XYZ 的三维误差，不能只看重合的轨迹线。
    past = np.arange(1 - HISTORY, 1) * STEP / 60
    future = np.arange(1, HORIZON + 1) * STEP / 60
    names = {"two_body": "Two-body", "j2": "J2", "timesfm": "TimesFM (original)"}
    colors = {"two_body": "#d76a13", "j2": "#168568", "timesfm": "#2463a6"}
    fig, (ax, err_ax) = plt.subplots(2, 1, figsize=(10, 6.3), gridspec_kw={"height_ratios": [1.5, 1]})
    ax.plot(past, sample[:HISTORY, 3] / 1000, color="#687580", label="Past reference")
    for name, values in predictions.items():
        ax.plot(future, values[:, 2] / 1000, color=colors[name], label=names[name])
        err_ax.plot(future, np.maximum(errors[name], 1e-9), color=colors[name], label=names[name])
    ax.plot(future, reference[:, 2] / 1000, ":", color="black", label=REFERENCE)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvspan(0, future[-1], color="#f1b75c", alpha=.16)
    ax.set(xlim=(-360, 60), ylabel=f"Z in {FRAME} (km)", xlabel="Minutes relative to forecast start")
    ax.set_title(f"{OBJECT} | {time_label(sample[HISTORY - 1, 0])} | window {number} | {device.upper()}", fontsize=11)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.13), ncol=3, fontsize=8)
    err_ax.set(xlim=(1, 60), yscale="log", ylabel=ERROR_LABEL, xlabel="Forecast lead (minutes)")
    err_ax.legend(fontsize=8)
    for panel in (ax, err_ax):
        panel.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "result.png", dpi=150)
    plt.close(fig)
    for row in metrics:
        print(f"{row['model']}: RMS(3D) = {row['rms_3d_m']:.3f} m；60 min = {row['endpoint_3d_m']:.3f} m")


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
        parser.error("二体/J2 在 CPU 上积分；GPU 选项用于 TimesFM")
    print("参考轨迹由同一 TLE 经 SGP4 生成；以下数值是相对传播模型的偏差，不是实测轨道精度。", flush=True)
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    try:
        sample = select_window(data, args.window)
    except ValueError as error:
        parser.error(str(error))
    evaluator = None
    if args.model == "timesfm":
        evaluator = load_evaluator(args.device, args.checkpoint)
    predictions = predict(sample[:HISTORY], evaluator)
    output = ROOT / "outputs" / f"{args.model}-{args.device}" / f"window-{args.window:02d}"
    save_result(sample, predictions, args.window, output, args.device)
    print(f"结果：{output}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
