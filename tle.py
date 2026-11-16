"""多历元 TLE：只按时间选数据，SGP4 输出 TEME 米、米/秒。"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sgp4.api import Satrec, WGS72
from sgp4.conveniences import sat_epoch_datetime
from sgp4.earth_gravity import wgs72
from sgp4.io import verify_checksum, twoline2rv

EPOCH = datetime(2000, 1, 1, tzinfo=timezone.utc)
DELAY = 24 * 3600  # 归档没有发布时间：假设历元后 24 h 可用，仅用于离线回放。


def load_tles(path=None):
    path = path or Path(__file__).parent / "data/debris.tle"
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) % 3:
        raise ValueError("每份 TLE 必须有名称和两行根数")
    records = []
    for i in range(0, len(lines), 3):
        name, line1, line2 = lines[i:i + 3]
        verify_checksum(line1, line2)
        twoline2rv(line1, line2, wgs72)
        satellite = Satrec.twoline2rv(line1, line2, WGS72)
        if satellite.satnum != 33757:
            raise ValueError("本教学样本只接受 NORAD 33757")
        epoch = (sat_epoch_datetime(satellite) - EPOCH).total_seconds()
        records.append({"satellite": satellite, "epoch": epoch, "available": epoch + DELAY})
    if len(records) < 3 or not np.all(np.diff([r["epoch"] for r in records]) > 0):
        raise ValueError("至少需要三个严格递增、不同历元的 TLE")
    return records


def available_tles(origin, records=None):
    """预测端只使用已到假设可用时间的记录；不能把历元直接当发布时间。"""
    records = load_tles() if records is None else records
    past = [record for record in records if record["available"] <= origin]
    if len(past) < 2:
        raise ValueError("起报前至少需要两份可用 TLE")
    return past[-2:]


def positions(record, seconds):
    seconds = np.asarray(seconds, dtype=float)
    days = np.floor(seconds / 86400)
    codes, r, v = record["satellite"].sgp4_array(2451544.5 + days, (seconds - days * 86400) / 86400)
    if np.any(codes) or not np.isfinite(r).all() or not np.isfinite(v).all():
        raise ValueError("SGP4 传播失败")
    return np.column_stack([r, v]) * 1000
