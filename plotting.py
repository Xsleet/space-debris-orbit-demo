"""保存根数输入、三种方法的轨道结果与图；未来参考只用于评分。"""
import csv
from datetime import timedelta
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from elements import INPUT_NAMES, NAMES as ELEMENT_NAMES
from tle import EPOCH

NAMES = {"sgp4_latest": "Latest TLE + SGP4",
         "elements_linear": "Linear elements + SGP4",
         "elements_timesfm": "TimesFM elements + SGP4"}
COLORS = {"sgp4_latest": "#345447", "elements_linear": "#cb7c26", "elements_timesfm": "#136cb0"}


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_inputs(output, metadata, audit):
    def array_file(name, times, values, names):
        np.savetxt(output / name, np.column_stack([times, values]), delimiter=",",
                   header=",".join(["hours_from_latest_tle_epoch", *names]),
                   comments="", fmt="%.17g")
    def utc(seconds):
        return (EPOCH + timedelta(seconds=float(seconds))).isoformat()
    array_file("element_tle_knots.csv", audit["hours"], audit["raw"], ELEMENT_NAMES)
    array_file("element_input.csv", audit["grid"], audit["inputs"], INPUT_NAMES)
    if audit["forecast"] is not None:
        array_file("element_forecast.csv", np.arange(len(audit["forecast"])),
                   audit["forecast"], INPUT_NAMES)
    info = {"origin": metadata["origin"], "known_tle_count": len(audit["known"]),
        "known_tle_epochs": [utc(r["epoch"]) for r in audit["known"]],
        "known_tle_availability": [utc(r["available"]) for r in audit["known"]],
        "input_grid_hours": 1, "input_grid_points": len(audit["grid"]),
        "element_extrapolation_hours": audit["target_hour"],
        "predicted_element_epoch": metadata["origin"], "bstar_source": "latest known TLE",
        "phase_handling": "Remove latest mean-motion phase before unwrap; restore after forecast",
        "linear_baseline": "Least-squares slope over actual known TLEs, anchored at latest TLE",
        "parameters_at_origin": {name: dict(zip(ELEMENT_NAMES, p.tolist()))
                                 for name, p in audit["parameters"].items()},
        "caveat": "Only 2-5 actual TLEs per window; hourly interpolation is not new observations"}
    (output / "elements.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def save_result(sample, predictions, audit, metadata, output):
    history, horizon = metadata["display_history_points"], metadata["horizon_points"]
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
    # 这段 SGP4 历史仅用于显示；真正模型输入在 element_input.csv。
    np.savetxt(output / "history.csv", sample[:history, :4], delimiter=",",
               header="time_seconds,sgp4_x_m,sgp4_y_m,sgp4_z_m", comments="", fmt="%.17g")
    write_csv(output / "metrics.csv", metrics)
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    save_inputs(output, metadata, audit)

    past, future = np.arange(1 - history, 1), np.arange(1, horizon + 1)
    fig, (ax, err_ax) = plt.subplots(2, 1, figsize=(10, 7))
    ax.plot(past, sample[:history, 3] / 1000, color="#687580", label="Past SGP4 (display only)")
    for name, values in predictions.items():
        ax.plot(future, values[:, 2] / 1000, color=COLORS[name], label=NAMES[name])
        err_ax.plot(future, errors[name], color=COLORS[name], label=NAMES[name])
    ax.plot(future, reference[:, 2] / 1000, ":", color="black", label=metadata["reference"])
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axvspan(0, horizon, color="#f1b75c", alpha=.16)
    ax.set(xlim=(-360, 60), ylim=(-8000, 10000), xlabel="Minutes relative to forecast start",
           ylabel=f"Z in {metadata['frame']} (km)")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    err_ax.set(xlim=(1, 60), ylim=(0, max(e.max() for e in errors.values()) * 1.35),
               ylabel="3-D deviation from later TLE (m)", xlabel="Forecast lead (minutes)")
    err_ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Orbital-element prediction | {metadata['object']} | window {metadata['window']} | {metadata['device'].upper()}\n{metadata['origin']}", fontsize=11)
    for panel in (ax, err_ax):
        panel.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output / "result.png", dpi=150)
    plt.close(fig)
    for row in metrics:
        print(f"{row['model']}: RMS(3D) = {row['rms_3d_m']:.3f} m; 60 min = {row['endpoint_3d_m']:.3f} m")


def save_summary(folder, rows, device):
    models = list(dict.fromkeys(r["model"] for r in rows))
    windows = sorted({int(r["window"]) for r in rows})
    summaries = []
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model in models:
        rms = np.array([float(r["rms_3d_m"]) for r in rows if r["model"] == model])
        summaries.append({"model": model, "windows": len(windows),
            "pooled_rms_3d_m": float(np.sqrt(np.mean(rms**2))),
            "median_window_rms_3d_m": float(np.median(rms)),
            "worst_window_rms_3d_m": float(rms.max())})
        ax.plot(windows, rms, "o-", label=NAMES[model], color=COLORS[model])
    write_csv(folder / "by_window.csv", rows)
    write_csv(folder / "summary.csv", summaries)
    ax.set(xticks=windows, ylim=(0, max(float(r["rms_3d_m"]) for r in rows) * 1.4),
           xlabel="Forecast window (2, 3, 4, 5 known TLEs)",
           ylabel="3-D RMS vs. held-out TLE (m)", title=f"Orbital-element forecasting | {device.upper()}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    return summaries
