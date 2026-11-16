"""六份固定 TLE -> 四组历史/留后参考窗口；正常运行无需执行本脚本。"""
import hashlib
import json
from pathlib import Path
import sys
from datetime import timedelta

import numpy as np
import sgp4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tle import EPOCH, DELAY, load_tles, positions

SOURCE = "https://raw.githubusercontent.com/AleksanderLidtke/ConjunctionDetection/c21596737757d87ea5bd2353c4b128c0795632d0/Release/Covariance%203LEs/33757.txt"


def main():
    records = load_tles()
    windows, samples = [], []
    for i in range(1, len(records) - 1):
        recent, held_out = records[i:i + 2]
        origin = np.ceil(recent["available"] / 60) * 60
        if held_out["epoch"] <= origin + 3600:
            raise ValueError("留后 TLE 历元必须晚于本窗口预测结束")
        times = origin + np.arange(-359, 61) * 60
        # 在起报时回算最新已知 TLE 的历史；留后 TLE 只生成评分参考。
        states = np.vstack([positions(recent, times[:360]), positions(held_out, times[360:])])
        samples.append(np.column_stack([times, states]))
        windows.append({"number": i, "origin_seconds": float(origin),
            "origin_utc": (EPOCH + timedelta(seconds=float(origin))).isoformat(),
            "known_tle_indices": list(range(i + 1)), "recent_tle_index": i, "reference_tle_index": i + 1})
    output = ROOT / "data/orbit.csv"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        np.savetxt(stream, np.vstack(samples), fmt="%.17g", delimiter=",", comments="",
            header="seconds_since_2000_utc,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s")
    manifest = {"object": "COSMOS 2251 DEB", "norad_id": 33757, "source_url": SOURCE,
        "tle_sha256": hashlib.sha256((ROOT / "data/debris.tle").read_bytes()).hexdigest(),
        "tle_count": len(records), "tle_epochs_utc": [(EPOCH + timedelta(seconds=r["epoch"])).isoformat() for r in records],
        "availability_assumption_seconds": DELAY, "availability_is_observed": False,
        "frame": "TEME", "gravity_model": "WGS72", "sgp4_version": sgp4.__version__,
        "rows": sum(len(s) for s in samples), "sample_seconds": 60, "windows": windows,
        "reference_kind": "Later-epoch TLE/SGP4; not independently measured ground truth",
        "history_kind": "Display only: latest known TLE propagated over past six hours; model input is historical TLE elements",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} TLE epochs, {len(windows)} windows")


if __name__ == "__main__":
    main()
