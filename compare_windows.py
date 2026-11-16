"""按预先固定的时间顺序检查全部窗口，不根据未来误差筛选。模型只加载一次。"""
import argparse
import csv
import sys

import numpy as np
from main import (ROOT, WINDOWS, HISTORY, MODEL_ID, OBJECT, REFERENCE, ERROR_LABEL,
                  select_window, predict, save_result, load_evaluator)
from plotting import plt, NAMES, COLORS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", choices=["timesfm", "physics"], default="timesfm")
    parser.add_argument("--checkpoint", default=MODEL_ID)
    args = parser.parse_args()
    if args.model == "physics" and args.device == "gpu":
        parser.error("GPU 选项用于 TimesFM，动力学模型使用 CPU")
    print(f"Reference: {REFERENCE}", flush=True)
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    samples = [select_window(data, n) for n in range(1, WINDOWS + 1)]
    evaluator = load_evaluator(args.device, args.checkpoint) if args.model == "timesfm" else None
    folder = ROOT / "outputs" / f"{args.model}-{args.device}"
    rows = []
    for number, sample in enumerate(samples, 1):
        print(f"Window {number}/{WINDOWS}", flush=True)
        predictions, past_error = predict(sample[:HISTORY], evaluator)
        output = folder / f"window-{number:02d}"
        save_result(sample, predictions, past_error, number, output, args.device)
        with (output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows.extend({"window": number, **row} for row in csv.DictReader(stream))
    summaries = []
    fig, ax = plt.subplots(figsize=(8, 4))
    for model in predictions:
        selected = [row for row in rows if row["model"] == model]
        rms = np.array([float(row["rms_3d_m"]) for row in selected])
        endpoint = [float(row["endpoint_3d_m"]) for row in selected]
        # 每个窗口长度相同，因此合并 RMS 是各窗口 RMS 平方均值的平方根。
        summaries.append({"model": model, "windows": WINDOWS,
            "pooled_rms_3d_m": float(np.sqrt(np.mean(rms**2))),
            "median_window_rms_3d_m": float(np.median(rms)),
            "worst_window_rms_3d_m": float(rms.max()),
            "mean_endpoint_3d_m": float(np.mean(endpoint))})
        ax.plot(range(1, WINDOWS + 1), rms, "o-", label=NAMES[model], color=COLORS[model])
    for filename, records in [("by_window.csv", rows), ("summary.csv", summaries)]:
        with (folder / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    ax.set(xticks=range(1, WINDOWS + 1), xlabel="Fixed forecast window", yscale="log",
           ylabel="RMS: " + ERROR_LABEL, title=f"{OBJECT} | {args.device.upper()} | {WINDOWS} windows")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(folder / "comparison.png", dpi=150)
    plt.close(fig)
    print(f"全部窗口及汇总：{folder}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    main()
