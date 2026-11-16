"""预测六个 TLE 平均根数，再交给 SGP4；不把瞬时二体根数混入 TLE。"""
import numpy as np
from sgp4.api import Satrec, WGS72

from tle import load_tles, positions

TAU = 2 * np.pi
NAMES = ["mean_motion_rev_day", "eccentricity", "inclination_rad", "raan_rad",
         "arg_perigee_rad", "mean_anomaly_rad"]
INPUT_NAMES = NAMES[:-1] + ["mean_anomaly_residual_rad"]


def parameters(record):
    sat = record["satellite"]
    return np.array([sat.no_kozai * 1440 / TAU, sat.ecco, sat.inclo,
                     sat.nodeo, sat.argpo, sat.mo])


def prepare_history(origin, records=None):
    """仅用起报前可用的真实 TLE；小时插值只适配模型，不增加观测信息。"""
    records = load_tles() if records is None else records
    known = [r for r in records if r["available"] <= origin]
    if len(known) < 2:
        raise ValueError("至少需要两份已可用 TLE")
    times = np.array([r["epoch"] for r in known])
    raw = np.array([parameters(r) for r in known])
    hours = (times - times[-1]) / 3600
    values = raw.copy()
    values[:, 3:5] = np.unwrap(values[:, 3:5], axis=0)
    # M 每天绕行十多圈，不能对稀疏 M 直接 unwrap。先去除最新 n 的快相位。
    phase_rate = raw[-1, 0] * TAU / 24  # rad/hour
    residual = raw[:, 5] - raw[-1, 5] - phase_rate * hours
    values[:, 5] = np.unwrap((residual + np.pi) % TAU - np.pi)
    values[:, 5] -= values[-1, 5]
    grid = np.arange(np.ceil(hours[0]), 1., 1.)
    inputs = np.column_stack([np.interp(grid, hours, y) for y in values.T])
    return known, hours, raw, values, grid, inputs


def restore_parameters(values, hours, latest):
    """恢复快相位和角度范围；其余根数越界时明确报错，不按评分结果修剪。"""
    out = np.array(values, dtype=float, copy=True)
    p = parameters(latest)
    out[5] += p[5] + p[0] * TAU / 24 * hours
    out[3:6] %= TAU
    if (not np.isfinite(out).all() or out[0] <= 0 or
            not 0 <= out[1] < 1 or not 0 <= out[2] <= np.pi):
        raise ValueError("预测根数无效；请检查根数预测输出")
    return out


def make_record(values, epoch, latest):
    """构造新历元的 SGP4 平均根数；B* 保留起报前最新已知值。"""
    n, e, inc, node, arg, anomaly = values
    base = latest["satellite"]
    sat = Satrec()
    sat.sgp4init(WGS72, "i", base.satnum, 18263. + epoch / 86400,
                 base.bstar, base.ndot, base.nddot, e, arg, inc, anomaly,
                 n * TAU / 1440, node)
    if sat.error:
        raise ValueError(f"SGP4 根数初始化失败：{sat.error}")
    return {"satellite": sat, "epoch": epoch}


def predict_elements(origin, future, evaluator, records=None):
    """固定协议：全部已知 TLE → 小时网格 → 起报时六根数 → 未来一小时 SGP4。"""
    known, hours, raw, values, grid, inputs = prepare_history(origin, records)
    latest = known[-1]
    target_hour = (origin - latest["epoch"]) / 3600
    steps = int(np.ceil(target_hour))
    # 线性基线锚定最新记录，用真实 TLE 节点拟合，不使用插值点伪造样本数。
    slope = hours @ (values - values[-1]) / (hours @ hours)
    linear = values[-1] + target_hour * slope
    # 各根数独立预测；在 float64 中去末值并缩放，保护很小的根数变化。
    scale = np.maximum(np.std(inputs, axis=0), 1e-12)
    centered = ((inputs - inputs[-1]) / scale).T.astype(np.float32)
    outputs = list(evaluator.predict_batch(list(centered), horizon=steps,
        return_quantiles=False, make_positive=False, use_symmetric_averaging=False))
    forecast = np.stack([o.forecast for o in outputs], axis=1).astype(float)
    if forecast.shape != (steps, 6) or not np.isfinite(forecast).all():
        raise ValueError("TimesFM 未返回有效的六根数序列")
    forecast = np.vstack([inputs[-1], forecast * scale + inputs[-1]])
    predicted = np.array([np.interp(target_hour, np.arange(steps + 1), y)
                          for y in forecast.T])
    params = {"elements_linear": restore_parameters(linear, target_hour, latest),
              "elements_timesfm": restore_parameters(predicted, target_hour, latest)}
    result = {name: positions(make_record(p, origin, latest), future)[:, :3]
              for name, p in params.items()}
    audit = {"known": known, "hours": hours, "raw": raw, "values": values,
             "grid": grid, "inputs": inputs, "forecast": forecast,
             "target_hour": target_hour, "parameters": params}
    return result, audit
