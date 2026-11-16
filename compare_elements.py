"""按固定顺序运行全部四个根数预测窗口；模型只加载一次。"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys

import numpy as np
from main import (ROOT, WINDOWS, HISTORY, MODEL_ID, REVISION, load_evaluator,
                  predict, save_result, select_window)
from plotting import save_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--model", choices=["timesfm", "linear"], default="timesfm")
    parser.add_argument("--checkpoint", default=MODEL_ID)
    args = parser.parse_args()
    if args.model == "linear" and args.device == "gpu":
        parser.error("线性根数与 SGP4 使用 CPU；GPU 选项用于 TimesFM")
    evaluator = load_evaluator(args.device, args.checkpoint) if args.model == "timesfm" else None
    data = np.loadtxt(ROOT / "data/orbit.csv", delimiter=",", skiprows=1)
    mode = "elements" if evaluator is not None else "linear"
    folder = ROOT / "outputs" / f"{mode}-{args.device}"
    rows = []
    for number in range(1, WINDOWS + 1):
        print(f"Window {number}/{WINDOWS}", flush=True)
        sample = select_window(data, number)
        predictions, audit = predict(sample[HISTORY - 1, 0], evaluator)
        output = folder / f"window-{number:02d}"
        save_result(sample, predictions, audit, number, output, args.device)
        with (output / "metrics.csv").open(encoding="utf-8", newline="") as stream:
            rows.extend({"window": number, "known_tle_count": len(audit["known"]), **row}
                        for row in csv.DictReader(stream))
    summaries = save_summary(folder, rows, args.device)
    packages = ["numpy", "matplotlib", "sgp4"]
    environment = {"python": platform.python_version(), "platform": platform.platform(),
        "device": args.device, "models": list(predictions), "model": None, "revision": None,
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                          for p in ["elements.py", "compare_elements.py", "main.py", "plotting.py", "tle.py"]}}
    if evaluator is not None:
        import torch
        packages.append("timesfm")
        environment.update({"model": MODEL_ID, "revision": REVISION, "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name() if args.device == "gpu" else None,
            "threads": torch.get_num_threads()})
    environment["packages"] = {p: importlib.metadata.version(p) for p in packages}
    (folder / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"全部结果：{folder}", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
