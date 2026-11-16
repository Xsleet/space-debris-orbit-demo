"""简化动力学：二体/J2 引力 + 可选地球自转，四阶 Runge–Kutta 积分。单位为 m、s。"""
import numpy as np

# WGS-72 常数，与本例 SGP4 初始化一致；来源见 DATA_SOURCES.md。
MU = 3.986008e14
R_EARTH = 6378135.0
J2 = 1.082616e-3
OMEGA = np.array([0., 0., 7.292115e-5])


def gravity(position, use_j2=True):
    r = np.linalg.norm(position)
    acceleration = -MU * position / r**3
    if use_j2:
        x, y, z = position
        q = 5 * (z / r)**2
        acceleration += 1.5 * J2 * MU * R_EARTH**2 / r**5 * np.array(
            [x * (q - 1), y * (q - 1), z * (q - 3)])
    return acceleration


def derivative(state, use_j2, rotating):
    position, velocity = state[:3], state[3:]
    acceleration = gravity(position, use_j2)
    if rotating:
        # ECEF 不是惯性系：必须加入科里奥利和离心项。
        acceleration -= 2 * np.cross(OMEGA, velocity) + np.cross(OMEGA, np.cross(OMEGA, position))
    return np.r_[velocity, acceleration]


def propagate(initial_state, points=60, interval=60., use_j2=True, rotating=False, max_step=5.):
    """仅从起报时刻的位置/速度向后传播；第一行对应 +interval，不是 t=0。"""
    state = np.asarray(initial_state, dtype=np.float64).copy()
    if state.shape != (6,) or not np.isfinite(state).all() or np.linalg.norm(state[:3]) < R_EARTH:
        raise ValueError("需要地表以上、有限的六维位置速度状态")
    if points < 1 or interval <= 0 or max_step <= 0:
        raise ValueError("点数、采样间隔和积分步长必须为正")
    steps = int(np.ceil(interval / max_step))
    h = interval / steps
    result = []
    for _ in range(points):
        for _ in range(steps):
            k1 = derivative(state, use_j2, rotating)
            k2 = derivative(state + h * k1 / 2, use_j2, rotating)
            k3 = derivative(state + h * k2 / 2, use_j2, rotating)
            k4 = derivative(state + h * k3, use_j2, rotating)
            state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        result.append(state.copy())
    return np.array(result)
