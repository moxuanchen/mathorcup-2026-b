"""一维准恒定断面和经验能量线诊断；不解守恒输运。"""
from __future__ import annotations

import numpy as np

G = 9.80665


def _slope_profile(s: np.ndarray, z: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """河段尺度比降。

    D8 沿程采样会给出"台阶—平台"式纵剖面，平台段瞬时坡度为 0 并不代表
    河段坡度为 0。故对纵剖面先做窗口平滑，再取比降，并设下限 slope_min
    以避免下游平原段除零。
    """
    ds = float(np.median(np.diff(s)))
    if win > 1 and len(z) > win:
        k = np.ones(win) / win
        zp = np.pad(z, (win // 2, win - 1 - win // 2), mode="edge")
        zs = np.convolve(zp, k, mode="valid")[: len(z)]
    else:
        zs = z
    slope = np.clip(-np.diff(zs, prepend=zs[0]) / ds, 1e-4, 1.5)
    sin_t = slope / np.sqrt(1.0 + slope ** 2)
    cos_t = 1.0 / np.sqrt(1.0 + slope ** 2)
    return slope, sin_t, cos_t


def solve_flow(s_m: np.ndarray, z_m: np.ndarray, p: dict) -> dict:
    """准恒定 Manning 流 + Fahrböschung 停止判据。

    Parameters
    ----------
    s_m, z_m : 中心线沿程水平里程与高程（m）
    p : 参数字典
        V0_m3      初始释放体积 (m³)
        B0_m       沟道初始宽度 (m)
        n_man      泥石流曼宁糙率 (s·m^{-1/3})
        mu         等效摩擦系数 (= tan 等效摩擦角)
        T_pass_s   过流历时 (s)，用于由体积折算特征流量
        slope_window  平滑窗口（节点数）
    """
    s = np.asarray(s_m, dtype="float64")
    z = np.asarray(z_m, dtype="float64")
    n = len(s)
    ds = float(np.median(np.diff(s)))

    V0 = float(p["V0_m3"]); B0 = float(p["B0_m"])
    nman = float(p["n_man"]); mu = float(p["mu"])
    T_pass = float(p.get("T_pass_s", 900.0))
    slope, sin_t, cos_t = _slope_profile(s, z, int(p.get("slope_window", 7)))

    # Common injection section is the zero of both propagation clocks.
    i0 = int(p.get("start_index", 0))
    if not 0 <= i0 < n - 1:
        raise ValueError("invalid common injection index")
    E0 = float(p.get("E0_m2s2", 0.0))

    # 沟宽沿程放大（下游汇流增大）
    B = B0 * (1.0 + float(p.get("beta_B", 0.8)) * s / max(s[-1], 1.0))
    # 特征流量：V0 在 T_pass 内通过任一断面
    Q_pk = V0 / max(T_pass, 1.0)

    # 断面流深与流速（Manning）。陡坎处 √S 极大、水深会被压到亚米级，
    # 平原段 √S→0 又会使水深发散；这两种情形都不代表真实断面，
    # 故对坡度取工程下限 slope_floor（默认 0.01，约 0.6°）并对水深设上限。
    s_eff = np.maximum(slope, float(p.get("slope_floor", 0.01)))
    h = (nman * Q_pk / (B * np.sqrt(s_eff))) ** 0.6
    h = np.minimum(h, float(p.get("h_cap", 15.0)))
    v = Q_pk / (B * h)
    v = np.minimum(v, float(p.get("v_cap", 30.0)))
    continuity = (B * h * v - Q_pk) / max(Q_pk, 1e-12)
    F = v / np.sqrt(G * h)

    # 停止位置（Fahrböschung / Heim 原理）：
    #   E(x) = E0 + 2g[ (z_i0 − z(x)) − μ·(s(x) − s_i0) ]
    # 该式几何上表示自释放点按倾角 arctan(μ) 作直线，与纵剖面的交点即
    # 堆积前锋的理论停止位置（Heim 1932；Rickenmann 2005）。E0 为冰崩
    # 在起始点注入的比能，使平坦的粒雪盆段不至于被误判为终点。
    # 停止判据需连续 sustain 个节点跌破 0，以容忍局部小凸起（等高线台阶）。
    sustain = int(p.get("sustain_nodes", 5))
    E = E0 + 2.0 * G * ((z[i0] - z) - mu * (s - s[i0]))
    E[:i0] = -np.inf
    E[i0] = max(E0, 0.0) + 1e-6
    below = E <= 0.0
    stop_idx = n - 1
    for i in range(i0 + 1, n):
        if i + sustain <= n and np.all(below[i:i + sustain]):
            stop_idx = i
            break
    if stop_idx <= i0 and i0 < n - 1:
        stop_idx = min(i0 + 2, n - 1)

    wet = (np.arange(n) >= i0) & (np.arange(n) <= stop_idx)
    h = np.where(wet, h, 0.0)
    v = np.where(wet, v, 0.0)

    # 到达时刻（梯形积分，起点之前不计时）
    t = np.full(n, np.nan)
    t[i0] = 0.0
    for i in range(i0 + 1, stop_idx + 1):
        t[i] = t[i - 1] + (s[i] - s[i-1]) / max(0.5 * (v[i - 1] + v[i]), 1e-9)

    # 几何强度 h·v²（m²·s⁻²）。强度指标 I = ρ_m h v² 在 step04_simulation
    # 中由 C_v 换算，这样密度参数有唯一的进入点，不会两处各算一套。
    inten = h * v ** 2
    # 体积诊断（修正版）。Q_pk = V0/T_pass 是定义式，用它反算 V0 必然恒等
    # （相对误差 0），不构成检验。真正有信息量的量是：
    #   沟槽储水体积 S(x) = ∫_{i0}^{x} B·h ds
    # 即按 Manning 流深把整条过流段同时充满所需的物质体积。
    #
    # 需要注意其正确解读：Heim 停止判据 E=V0 无关，只含 μ 与地形，故
    # **停止位置恒由 μ 控制**，与 V0 无关。V0 决定的是"同时有多少沟道在
    # 过流"，即波列长度 L_surge = V0 / mean(B·h)。S/V0 因此是"波列占运移
    # 路程的比例的倒数"，用于判断这是脉冲型（surge）还是满槽型流动，而
    # 不是判定物质是否充足的判据。
    dseg = np.zeros(n)
    dseg[i0:] = ds
    S_cum = np.cumsum(B * h * dseg)
    S_stop = float(S_cum[stop_idx]) if stop_idx >= i0 else 0.0
    bh = np.where(wet, B * h, 0.0)
    mean_bh = float(np.mean(bh[wet])) if np.any(wet) else 0.0
    L_surge_m = V0 / mean_bh if mean_bh > 0 else 0.0
    run_m = float(s[stop_idx] - s[i0]) if stop_idx >= i0 else 0.0
    return {
        "s_km": s / 1000.0, "z_m": z, "slope": slope, "B_m": B,
        "h_m": h, "v_m_s": v, "Q_m3s": np.where(wet, Q_pk, 0.0),
        "Fr": F, "E": E, "t_s": t, "intensity": inten,
        # 波列长度与占运移路程的比例
        "波列长度_m": float(L_surge_m),
        "波列占路程比": float(min(L_surge_m / run_m, 1.0)) if run_m > 0 else np.nan,
        "液型": ("脉冲型（surge）" if L_surge_m < 0.6 * run_m else "近满槽型"),
        # 满槽储水体积（诊断量，非守恒校验）
        "沟槽储水体积_m3": S_stop,
        "物质充裕度": float(S_stop / V0) if V0 > 0 else np.nan,
        "运移控制因素": "停止位置由 μ 控制；V0 决定波列长度",
        "定义恒等式_非守恒检验": float(abs(V0 - Q_pk * T_pass) / V0),
        "stop_idx": stop_idx,
        "runout_km": float(s[stop_idx] / 1000.0),
        "total_km": float(s[-1] / 1000.0),
        "reached_outlet": bool(stop_idx >= n - 1),
        "arrival_outlet_s": (float(t[-1]) if stop_idx == n - 1
                             and np.isfinite(t[-1]) else None),
        "stop_time_s": float(t[stop_idx]) if np.isfinite(t[stop_idx]) else None,
        "termination_reason": ("right_censored_at_port" if stop_idx == n - 1
                               else "natural_energy_stop"),
        "runout_right_censored": bool(stop_idx == n - 1),
        "start_index": i0,
        "common_clock": "common_injection_section",
        "continuity_relative_residual_max": float(np.max(np.abs(continuity[wet]))),
        "slope_floor_nodes": int(np.sum((slope < p.get("slope_floor", 0.01)) & wet)),
        "params": dict(p),
        "v_max_m_s": float(np.max(v)), "v_median_m_s": float(np.median(v[wet])),
        "h_max_m": float(np.max(h)), "h_median_m": float(np.median(h[wet])),
        "I_max": float(np.max(inten)),
        "Fr_median": float(np.median(F[wet])),
        "drop_m": float(z[0] - z[-1]),
        "stopped_drop_m": float(z[0] - z[stop_idx]),
        "mu_crit": float((z[0] - z[-1]) / s[-1]),
        "mu": mu, "n_man": nman, "V0": V0, "Q_pk": Q_pk,
        "等效摩擦角_deg": float(np.degrees(np.arctan(mu))),
    }


def calibrate_mu(s_m, z_m, target_km: float, params: dict,
                 lo: float = 0.02, hi: float = 0.5) -> dict:
    """二分反演使运动距离等于目标值的 μ（L_max ≈ Δz/μ 单调，收敛稳定）。"""
    def runout(m):
        q = dict(params); q["mu"] = float(m)
        return solve_flow(s_m, z_m, q)["runout_km"]

    a, b = lo, hi
    for _ in range(70):
        mid = 0.5 * (a + b)
        if runout(mid) > target_km:
            a = mid
        else:
            b = mid
        if abs(b - a) < 1e-6:
            break
    mu = 0.5 * (a + b)
    q = dict(params); q["mu"] = mu
    r = solve_flow(s_m, z_m, q)
    return {"mu": float(mu), "runout_km": r["runout_km"], "目标_km": float(target_km),
            "残差_km": float(r["runout_km"] - target_km),
            "等效摩擦角_deg": r["等效摩擦角_deg"]}


def runout_curve(s_m, z_m, mus, params: dict) -> dict:
    rows = []
    for m in mus:
        q = dict(params); q["mu"] = float(m)
        r = solve_flow(s_m, z_m, q)
        rows.append({"mu": float(m), "等效摩擦角_deg": r["等效摩擦角_deg"],
                     "runout_km": r["runout_km"], "reached_outlet": r["reached_outlet"],
                     "arrival_outlet_s": r["arrival_outlet_s"],
                     "v_max_m_s": r["v_max_m_s"], "I_max": r["I_max"]})
    return {"曲线": rows}


def lateral_spread(cl: dict, reach_km: float, half_angle_deg: float, I0: float,
                   dem: np.ndarray, grid: dict) -> dict:
    """出山口以下的横向扩散影响区（扇形 + 沿程指数衰减）。"""
    h, w = dem.shape
    cell_x = abs(grid["dlon"]) * 111320.0 * np.cos(np.radians(28.4))
    cell_y = abs(grid["dlat"]) * 110574.0
    apex = cl["end_rc"]
    k = max(2, len(cl["s_m"]) // 20)
    ang = float(np.arctan2(cl["y_m"][-1] - cl["y_m"][-k], cl["x_m"][-1] - cl["x_m"][-k]))
    rmax = reach_km * 1000.0
    phi = np.radians(half_angle_deg)

    rr, cc = np.mgrid[0:h, 0:w]
    px = (cc - apex[1]) * cell_x
    py = (rr - apex[0]) * cell_y
    r = np.hypot(px, py)
    theta = np.arctan2(py, px)
    dtheta = np.abs((theta - ang + np.pi) % (2 * np.pi) - np.pi)
    inside = (r <= rmax) & (dtheta <= phi) & (r > 0)
    inten = np.zeros((h, w), dtype="float32")
    inten[inside] = (I0 * np.exp(-r[inside] / max(rmax, 1.0)) *
                     np.cos(dtheta[inside] / max(phi, 1e-6) * np.pi / 2)).astype("float32")
    cell_area = cell_x * cell_y
    return {"intensity": inten, "affected_cells": int(inside.sum()),
            "affected_area_km2": float(inside.sum() * cell_area / 1e6),
            "apex_rc": [int(apex[0]), int(apex[1])],
            "main_direction_deg": float(np.degrees(ang)),
            "half_angle_deg": float(half_angle_deg), "reach_km": float(reach_km),
            "I0": float(I0), "max_intensity": float(np.max(inten))}


def intensity_classes(inten, mask, cell_area: float, edges=None) -> dict:
    """按单位宽度冲击力 I = ρ_m·h·v² 分级（默认阈值 10 / 100 / 1000 kN·m⁻¹）。

    泥石流的破坏力由流深、流速与密度共同决定。指标取单位宽度冲击力：
        I = ρ_m·h·v²,   ρ_m = C_v·ρ_s + (1−C_v)·ρ_w
    量纲 kg·m⁻³ × m × m²·s⁻² = N/m（不是压强 Pa，二者相差一个流深因子）。
    乘上 ρ_m 是关键：若写成 h·v²（m²·s⁻²），含沙量 15% 与 80% 的流动会给出
    相同破坏力，与题面把固相含量作为定义性特征的设定冲突。

    默认分级界 10 / 100 / 1000 kN·m⁻¹（ρ_m≈1742 kg·m⁻³）：
      10 kN/m   ≈ h=0.5 m、v=3.4 m/s —— 可推动沟床松散物质；
      100 kN/m  ≈ h=1.5 m、v=6.2 m/s —— 可破坏浆砌石护岸；
      1000 kN/m ≈ h=3.0 m、v=13.8 m/s —— 可摧毁钢筋混凝土结构。
    edges 若给出则按 edges 分级（N/m），供调用方与参数表保持同一口径。
    """
    edges = tuple(edges) if edges is not None else (0.0, 1.0e4, 1.0e5, 1.0e6, np.inf)
    names = ("低", "中", "高", "极高")
    out = {}
    for i, name in enumerate(names):
        lo, hi = edges[i], edges[i + 1]
        m = (inten >= lo) & (inten < hi) & mask
        out[name] = {"像元数": int(m.sum()), "面积_km2": float(m.sum() * cell_area / 1e6)}
    out["公式"] = "I = ρ_m·h·v²（单位宽度冲击力，N·m⁻¹）"
    out["分级阈值_N_m"] = [float(v) for v in edges[:-1]] + ["inf"]
    out["最大强度_N_m"] = float(np.nanmax(inten[mask])) if mask.any() else 0.0
    return out
