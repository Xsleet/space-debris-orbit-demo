"""从附带 TLE 生成固定 SGP4 参考轨迹；没有下载、拟合或制造实测真值。"""
import hashlib
import json
from pathlib import Path

import numpy as np
import sgp4
from sgp4.api import Satrec, WGS72
from sgp4.conveniences import sat_epoch_datetime
from sgp4.io import verify_checksum, twoline2rv
from sgp4.earth_gravity import wgs72

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "data/debris.tle"
    name, line1, line2 = source.read_text(encoding="ascii").splitlines()
    verify_checksum(line1, line2)
    twoline2rv(line1, line2, wgs72)  # 严格检查固定列格式，不能把平均根数直接当作二体瞬时根数。
    satellite = Satrec.twoline2rv(line1, line2, WGS72)
    seconds = np.arange(0, 25 * 3600 + 1, 60, dtype=np.float64)
    jd = np.full(len(seconds), satellite.jdsatepoch)
    fraction = satellite.jdsatepochF + seconds / 86400
    errors, r_km, v_km_s = satellite.sgp4_array(jd, fraction)
    if np.any(errors != 0):
        raise ValueError(f"SGP4 传播失败，错误码 {np.unique(errors)}")
    data = np.column_stack([seconds, r_km * 1000, v_km_s * 1000])
    if not np.isfinite(data).all():
        raise ValueError("SGP4 返回无效数值")
    output = ROOT / "data/orbit.csv"
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        np.savetxt(stream, data, fmt="%.17g", delimiter=",", comments="",
            header="seconds_since_tle_epoch,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s")
    manifest = {"object": name, "norad_id": satellite.satnum,
        "source_url": "https://celestrak.org/NORAD/elements/gp.php?CATNR=33757&FORMAT=TLE",
        "snapshot_retrieved_utc_date": "2026-11-16", "tle_epoch_utc": sat_epoch_datetime(satellite).isoformat(),
        "tle_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "frame": "TEME", "gravity_model": "WGS72", "sgp4_version": sgp4.__version__,
        "rows": len(data), "sample_seconds": 60, "propagation_hours_after_epoch": [0, 25],
        "reference_kind": "SGP4 propagated reference, not observed or independently measured ground truth",
        "history_kind": "Same-TLE SGP4 propagated positions; not historical tracking observations",
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(data)} SGP4 reference states for NORAD {satellite.satnum}")


if __name__ == "__main__":
    main()
