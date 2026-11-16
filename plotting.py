"""画图和保存结果；所有留后参考值只在预测完成后用于评分。"""
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"two_body": "Two-body", "j2": "J2", "timesfm": "TimesFM: direct XYZ",
         "j2_timesfm": "J2 + TimesFM", "sgp4_old": "SGP4: older TLE",
         "sgp4_latest": "SGP4: latest known TLE", "sgp4_timesfm": "Older SGP4 + TimesFM"}
COLORS = {"two_body": "#c27415", "j2": "#168568", "timesfm": "#3879bf",
          "j2_timesfm": "#b33178", "sgp4_old": "#806b9b", "sgp4_latest": "#345447",
          "sgp4_timesfm": "#b33178"}


def save_result(sample, predictions, past_error, metadata, output):
    history, horizon = len(past_error), len(next(iter(predictions.values())))
    reference = sample[history:, 1:4]
    errors = {name: np.linalg.norm(values - reference, axis=1) for name, values in predictions.items()}
    metrics = [{"model": name, "rms_3d_m": float(np.sqrt(np.mean(error**2))),
                "endpoint_3d_m": float(error[-1])} for name, error in errors.items()]
    output.mkdir(parents=True, exist_ok=True)
    columns = [sample[history:, 0], *reference.T]
    header = ["time_seconds", "reference_x_m", "reference_y_m", "reference_z_m"]
    for name, values in predictions.items():
        columns.extend([*values.T, errors[name]])
        header.extend([f"{name}_{axis}_m" for axis in "xyz"] + [f"{name}_deviation_m"])
    np.savetxt(output / "prediction.csv", np.column_stack(columns), delimiter=",",
               header=",".join(header), comments="", fmt="%.17g")
    # 保存实际输入，学生可以逐点核算“参考 - 动力学 = 历史误差”。
    past_reference = sample[:history, 1:4]
    columns = np.column_stack([sample[:history, 0], past_reference, past_reference - past_error, past_error])
    header = ["time_seconds"] + [f"{kind}_{axis}_m" for kind in ("reference", "baseline", "residual") for axis in "xyz"]
    np.savetxt(output / "history.csv", columns, delimiter=",", header=",".join(header), comments="", fmt="%.17g")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    past, future = np.arange(1 - history, 1), np.arange(1, horizon + 1)
    base, hybrid = metadata["residual_baseline"], metadata["hybrid_model"]
    fig, (ax, residual_ax, err_ax) = plt.subplots(3, 1, figsize=(10, 9),
        gridspec_kw={"height_ratios": [1.1, 1, 1]})
    ax.plot(past, past_reference[:, 2] / 1000, color="#687580", label="Past reference")
    residual_ax.plot(past, past_error[:, 2], color="#687580", label="Past residual: reference - baseline")
    residual_ax.axhline(0, color=COLORS[base], linewidth=1, label="Dynamics: zero correction")
    residual_ax.plot(future, (reference - predictions[base])[:, 2], ":", color="black", label="Held-out residual")
    for name, values in predictions.items():
        ax.plot(future, values[:, 2] / 1000, color=COLORS[name], label=NAMES[name])
        err_ax.plot(future, np.maximum(errors[name], 1e-9), color=COLORS[name], label=NAMES[name])
    if hybrid in predictions:
        correction = predictions[hybrid] - predictions[base]
        residual_ax.plot(future, correction[:, 2], color=COLORS[hybrid], label="TimesFM residual forecast")
    ax.plot(future, reference[:, 2] / 1000, ":", color="black", label=metadata["reference"])
    for panel in (ax, residual_ax):
        panel.axvline(0, color="black", linestyle="--", linewidth=1)
        panel.axvspan(0, horizon, color="#f1b75c", alpha=.16)
        panel.set(xlim=(-360, 60), xlabel="Minutes relative to forecast start")
    ax.set(ylabel=f"Z in {metadata['frame']} (km)")
    fig.suptitle(f"{metadata['object']} | {metadata['origin']} | window {metadata['window']} | {metadata['device'].upper()}", fontsize=11)
    ax.legend(loc="upper left", ncol=3, fontsize=7)
    residual_ax.set(ylabel="Z residual (m)")
    residual_ax.legend(loc="best", fontsize=8)
    err_ax.set(xlim=(1, 60), yscale="log", ylabel=metadata["error_label"], xlabel="Forecast lead (minutes)")
    err_ax.legend(fontsize=8, ncol=2)
    for panel in (ax, residual_ax, err_ax):
        panel.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "result.png", dpi=150)
    plt.close(fig)
    for row in metrics:
        print(f"{row['model']}: RMS(3D) = {row['rms_3d_m']:.3f} m; 60 min = {row['endpoint_3d_m']:.3f} m")
