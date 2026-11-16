"""小样本实验：直接预测轨道根数，是否优于 XYZ 和误差预报？"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys

import numpy as np
from elements import INPUT_NAMES, NAMES as ELEMENT_NAMES, predict_elements
from main import (ROOT, WINDOWS, HISTORY, MODEL_ID, REVISION, load_evaluator,
                  predict, save_result, select_window, time_label)
from plotting import COLORS, NAMES, plt


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_inputs(output, origin, audit):
    def array_file(name, times, values, names):
        np.savetxt(output / name, np.column_stack([times, values]), delimiter=",",
                   header=",".join(["hours_from_latest_tle_epoch", *names]),
                   comments="", fmt="%.17g")
    array_file("element_tle_knots.csv", audit["hours"], audit["raw"], ELEMENT_NAMES)
    array_file("element_input.csv", audit["grid"], audit["inputs"], INPUT_NAMES)
    array_file("element_forecast.csv", np.arange(len(audit["forecast"])),
               audit["forecast"], INPUT_NAMES)
    metadata = {"origin": time_label(origin), "known_tle_count": len(audit["known"]),
        "known_tle_epochs": [time_label(r["epoch"]) for r in audit["known"]],
        "known_tle_availability": [time_label(r["available"]) for r in audit["known"]],
        "input_grid_hours": 1, "input_grid_points": len(audit["grid"]),
        "element_extrapolation_hours": audit["target_hour"],
        "predicted_element_epoch": time_label(origin), "bstar_source": "latest known TLE",
        "phase_handling": "Remove latest mean-motion phase before unwrap; restore after forecast",
        "linear_baseline": "Least-squares slope over actual known TLEs, anchored at latest TLE",
        "parameters_at_origin": {name: dict(zip(ELEMENT_NAMES, p.tolist()))
                                 for name, p in audit["parameters"].items()},
        "caveat": "Only 2-5 actual TLEs per window; hourly interpolation is not new observations"}
    (output / "elements.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--checkpoint", default=MODEL_ID)
    args = parser.parse_args()
    evaluator = load_evaluator(args.device, args.checkpoint)
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    folder = ROOT / "outputs" / f"elements-{args.device}"
    rows = []
    for number in range(1, WINDOWS + 1):
        print(f"Window {number}/{WINDOWS}", flush=True)
        sample = select_window(data, number)
        history, future = sample[:HISTORY], sample[HISTORY:, 0]
        origin = history[-1, 0]
        # 预测函数只收到历史或未来时间，不收到留后轨道。
        predictions, past_error = predict(history, evaluator)
        element_predictions, audit = predict_elements(origin, future, evaluator)
        predictions.update(element_predictions)
        output = folder / f"window-{number:02d}"
        save_result(sample, predictions, past_error, number, output, args.device)
        save_inputs(output, origin, audit)
        with (output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows.extend({"window": number, "known_tle_count": len(audit["known"]), **row}
                        for row in csv.DictReader(stream))
    summaries = []
    for model in predictions:
        selected = [r for r in rows if r["model"] == model]
        rms = np.array([float(r["rms_3d_m"]) for r in selected])
        summaries.append({"model": model, "windows": WINDOWS,
            "pooled_rms_3d_m": float(np.sqrt(np.mean(rms**2))),
            "median_window_rms_3d_m": float(np.median(rms)),
            "worst_window_rms_3d_m": float(rms.max())})
    write_csv(folder / "by_window.csv", rows)
    write_csv(folder / "summary.csv", summaries)
    # 主图只保留与研究问题直接相关的四条曲线，便于学生阅读。
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model in ["sgp4_latest", "sgp4_timesfm", "elements_linear", "elements_timesfm"]:
        y = [float(r["rms_3d_m"]) for r in rows if r["model"] == model]
        ax.plot(range(1, WINDOWS + 1), y, "o-", label=NAMES[model], color=COLORS[model])
    ax.set(xticks=range(1, WINDOWS + 1), ylim=(0, 600), xlabel="Forecast window (2, 3, 4, 5 known TLEs)",
           ylabel="3-D RMS vs. held-out TLE (m)", title=f"Orbital elements vs. residual forecasting | {args.device.upper()}")
    ax.grid(alpha=.2)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=160)
    plt.close(fig)
    import torch
    environment = {"python": platform.python_version(), "platform": platform.platform(),
        "device": args.device, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if args.device == "gpu" else None,
        "threads": torch.get_num_threads(), "model": MODEL_ID, "revision": REVISION,
        "packages": {p: importlib.metadata.version(p) for p in ["numpy", "matplotlib", "sgp4", "timesfm"]},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in ["elements.py", "compare_elements.py", "main.py", "tle.py"]}}
    (folder / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"全部结果：{folder}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
