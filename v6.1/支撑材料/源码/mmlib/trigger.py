"""泥石流触发判据：三条件耦合的水源—地形—物质链。

统一框架
--------
1. 水源供给 Q_w(t)：SCS-CN 产流（降雨）+ 能量平衡融水（冰/雪），均折算为沟口流量。
2. 地形起动门槛 Q_c(t)：Manning 流深 + 库仑床面抗力的渠道松散体起动判据，
   逐沟由矢量属性（面积、主沟长、高差、比降、扇宽/扇长）计算。
3. 触发指数 T(t) = Q_w / Q_c；仅在 water_controlled 且 Q_c>0 时定义。

标定
-----
旧版把 Q_c 中的集总系数 k 由 2026-08-26 事件窗口反标定，使窗口内 T 峰值为 1。
该 k 只作为 v4.3 旧方法审计对照，不得进入正式触发状态、问题三阈值或预测区间。

集水区比降口径
--------------
本模块提供三种口径，用途严格区分（详见 建模成果/问题2.md §2.4.2）：

1. `geometric`：√像元数×30 m×1.6 的几何近似，**只作面积—形状粗估，不进入 q_c**；
2. `traced`：自出口按最大汇流上游追迹的像元链，用于空间可靠性检验；
3. `centerline`：问题 4 中心线（源区冰厚高值像元 → 口岸），**q_c 的唯一计算口径**，
   并使问题 2 与问题 4 共用同一沟长/比降。

早期实现把口径 1 的比降再乘 3 后代入 q_c，该系数既无来源也未在论文披露，
会使 q_c 偏低约 4.2 倍；现已删除，改由口径 3 直接计算。

沟宽与 κ
--------
A_charge = q_c·B/κ 中，B 优先取扇宽/4，缺测时取 B = 9√A (km²)（百米量级经验式）；
κ = 6 为"有效过流宽度占全沟宽的比例"的倒数式集总系数。两者均为**显式假设**；
v5.0 不用事件窗 k 吸收其不确定度，而是分别报告参数情景与状态分支。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io_data as io

G = 9.80665
RHO_W = 1000.0
LF = 334.0            # 融化潜热 MJ/m³（水当量），仅用于 MJ·m⁻² 口径的换算
LF_J = 334000.0       # 融化潜热 J/kg（用于 W·m⁻² 通量口径的换算）

# ---- 沟道起动判据中的显式假设常数（论文参数表须逐项列出，不得隐去）----
N_MAN_GULLY = 0.06    # 沟道曼宁糙率（砾石质沟道）
TAN_PHI = 0.55        # 松散体有效内摩擦角正切（约 28.8°）
ZB = 1.0              # 可蚀层厚度 m
DRHO = 1050.0         # 饱和松散体与水的密度差 kg/m³
KAPPA_Q = 6.0         # Q_c = q_c·B/κ 的集总系数（有效过流宽度比例的倒数式）
KAPPA_WIDTH = 9.0     # 缺测沟宽经验式 B = 9·√A(km²)（百米量级）
EPS_S = 1e-6          # 摩擦平衡状态的无量纲比降容差
CN0 = 85.0            # 独立声明的SCS-CN工程情景，不由土地覆盖类码推导
API_TRAIN_YEARS = (2021, 2022, 2023, 2024, 2025)
API_WARM_MONTHS = (6, 7, 8, 9)


def _density_difference(bed_solid_fraction: float) -> float:
    """Return rho_sat-rho_w for rho_sat=1000+1650*Cv."""
    return max(float(bed_solid_fraction) * 1650.0, 0.0)

# ---- 集水区比降口径 ----
SLOPE_DEFS = ("geometric", "traced", "centerline")
# ------------------------------------------------------------------ 沟道参数
def gully_table() -> pd.DataFrame:
    """84 条沟道的几何与地形参数（含派生的沟口流量门槛）。"""
    haz = io.load_hazard_points()
    df = haz.copy()
    df["面积_km2"] = pd.to_numeric(df["area_km2"], errors="coerce")
    df["主沟长_m"] = pd.to_numeric(df["length_m"], errors="coerce")
    df["高差_m"] = pd.to_numeric(df["relief_m"], errors="coerce")
    df["比降"] = pd.to_numeric(df["grad"], errors="coerce")
    df["沟口高程_m"] = pd.to_numeric(df["mouth_m"], errors="coerce")
    df["流域最高高程_m"] = pd.to_numeric(df["zmax_m"], errors="coerce")
    fan_w = pd.to_numeric(df["fan_w"], errors="coerce") if "fan_w" in df else pd.Series(np.nan, index=df.index)
    fan_l = pd.to_numeric(df["fan_l"], errors="coerce") if "fan_l" in df else pd.Series(np.nan, index=df.index)
    df["扇宽_m"] = fan_w
    df["扇长_m"] = fan_l
    # 沟道宽度：优先用扇宽/4（扇体远宽于沟槽），否则由面积经验式 B = 9·√A(km²) 估计。
    # 该式为显式经验假设，非推导结果，论文须按"经验式取定"表述（见模块 docstring）。
    width = (fan_w / 4.0).where(fan_w.between(20, 4000))
    width = width.fillna(KAPPA_WIDTH * np.sqrt(df["面积_km2"].clip(lower=0.01)))
    df["沟宽_m"] = width.clip(3.0, 300.0)
    mechanics = [_mechanics(s) for s in df["比降"]]
    df["mechanical_regime"] = [m["regime"] for m in mechanics]
    df["mechanical_reason"] = [m["reason"] for m in mechanics]
    df["h_crit_m"] = [m["h_crit_m"] for m in mechanics]
    df["tau_w_c_Pa"] = [m["tau_w_c_Pa"] for m in mechanics]
    df["tau_g_parallel_Pa"] = [m["tau_g_parallel_Pa"] for m in mechanics]
    df["tau_r_Pa"] = [m["tau_r_Pa"] for m in mechanics]
    df["q_crit_m2s"] = [m["q_crit_m2s"] for m in mechanics]
    df["Q_crit_m3s"] = df["q_crit_m2s"] * df["沟宽_m"] / KAPPA_Q
    # 旧式无床体沿坡自重项，只用于关闭/解释历史结果。
    df["q_crit_old_m2s"] = [_q_crit_old(s) for s in df["比降"]]
    df["Q_crit_old_m3s"] = df["q_crit_old_m2s"] * df["沟宽_m"] / KAPPA_Q
    return df


def _mechanics(slope: float, bed_solid_fraction: float = DRHO / 1650.0) -> dict:
    """完整无黏聚、无超孔压床层的力学分支与临界量。

    几何约定：z_b 为竖直床厚，应力和临界量均按单位坡面面积报告；
    h 为垂直床面的水深。令 S=tanθ，则
    τ_g,∥=Δρ g z_b sinθ cosθ，
    τ_r=Δρ g z_b cos²θ tanφ，
    τ_w,c=Δρ g z_b cos²θ(tanφ−tanθ)。
    当 tanθ 超过 tanφ 时，声明模型下不需要外部水流即可自失稳；这不是现实安全性结论。
    """
    invalid = {
        "regime": "not_evaluable",
        "reason": "非有限或非正比降",
        "h_crit_m": None,
        "tau_w_c_Pa": None,
        "tau_g_parallel_Pa": None,
        "tau_r_Pa": None,
        "delta_rho_kg_m3": None,
        "q_crit_m2s": None,
    }
    if (
        not np.isfinite(slope)
        or slope <= 1e-4
        or not np.isfinite(bed_solid_fraction)
        or bed_solid_fraction <= 0.0
        or bed_solid_fraction > 1.0
    ):
        return invalid
    s = float(slope)
    cos2 = 1.0 / (1.0 + s * s)
    sin_theta = s / np.sqrt(1.0 + s * s)
    cos_theta = 1.0 / np.sqrt(1.0 + s * s)
    delta_rho = _density_difference(bed_solid_fraction)
    tau_g = delta_rho * G * ZB * sin_theta * cos_theta
    tau_r = delta_rho * G * ZB * cos2 * TAN_PHI
    net = TAN_PHI - s
    if net > EPS_S:
        regime = "water_controlled"
        reason = "tan(phi)-tan(theta)>epsilon_S"
        tau_wc = delta_rho * G * ZB * cos2 * net
        h = tau_wc / (RHO_W * G * sin_theta)
        q = h ** (5.0 / 3.0) * np.sqrt(s) / N_MAN_GULLY
    elif abs(net) <= EPS_S:
        regime = "friction_equilibrium"
        reason = "abs(tan(phi)-tan(theta))<=epsilon_S"
        tau_wc, h, q = 0.0, 0.0, 0.0
    else:
        regime = "gravity_unstable"
        reason = "tan(theta)-tan(phi)>epsilon_S"
        tau_wc, h, q = 0.0, 0.0, 0.0
    return {
        "regime": regime,
        "reason": reason,
        "h_crit_m": float(h),
        "tau_w_c_Pa": float(tau_wc),
        "tau_g_parallel_Pa": float(tau_g),
        "tau_r_Pa": float(tau_r),
        "delta_rho_kg_m3": float(delta_rho),
        "q_crit_m2s": float(q),
    }


def _q_crit_unit(slope: float, area_km2: float = 0.05,
                 bed_solid_fraction: float = DRHO / 1650.0) -> float:
    """单宽临界流量；零门槛分支返回 0，不能用于定义 T=Q/Q_c。"""
    del area_km2
    return float(_mechanics(slope, bed_solid_fraction)["q_crit_m2s"])


def _q_crit_old(slope: float, area_km2: float = 0.05,
                bed_solid_fraction: float = DRHO / 1650.0) -> float:
    """v4.3 旧式无床体沿坡自重项的单宽临界流量，仅供审计。"""
    del area_km2
    if not np.isfinite(slope) or slope <= 1e-4:
        return np.nan
    if not np.isfinite(bed_solid_fraction) or not (0.0 < bed_solid_fraction <= 1.0):
        return np.nan
    s = float(slope)
    delta_rho = _density_difference(bed_solid_fraction)
    tau_r = delta_rho * G * ZB * TAN_PHI / (1.0 + s * s)
    h = tau_r / (RHO_W * G * s)
    return float(h ** (5.0 / 3.0) * np.sqrt(s) / N_MAN_GULLY)


# ------------------------------------------------------------------ 产流
def catchment_glacier_fraction(return_mask: bool = False):
    """口岸所在集水区的冰川覆盖率与高程特征（用于融水面积权重）。

    使用附件3的RGI6编目边界；高程带仅作审计对照，不进入融水面积。
    RGI6不是2026年实测冰川或积雪边界，编目年代差异另行披露。

    return_mask=True 时额外返回集水区掩膜（供静态特征统计复用，避免重算）。
    """
    from . import terrain as tr, geodesy as gd

    dem, grid = io.load_dem()
    z = dem.astype("float64")
    d = np.load(io.CACHE / "dir.npy")
    acc = np.load(io.CACHE / "acc.npy")
    r, c = io.lonlat_to_rowcol(grid, io.EVENT["port_lon"], io.EVENT["port_lat"])
    ri, ci = int(round(r)), int(round(c))
    # 取距口岸最近的主河道像元（汇流累积 ≥ 5×10⁵）作为集水区出口，
    # 避免沿程向下游搜索时误取更远处的干流像元
    cand = np.argwhere(acc >= 5e5)
    if len(cand):
        dist = np.hypot(cand[:, 0] - ri, cand[:, 1] - ci)
        best = tuple(int(v) for v in cand[int(np.argmin(dist))])
        best_acc = float(acc[best])
    else:
        best, best_acc = (ri, ci), float(acc[ri, ci])
    mask = tr.upstream_mask(d, best)
    from .terrain_prep import cell_meters
    lat_c = 28.4
    dy_cell, dx_cell = cell_meters(grid)
    cell_area = dy_cell * dx_cell
    area = float(mask.sum()) * cell_area / 1e6
    zc = z[mask]

    ice = z >= 4800.0
    # 附件（4）源区冰川像元
    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    lon_g = float(grid["lon0"]) + xx * float(grid["dlon"])
    lat_g = float(grid["lat0"]) - yy * abs(float(grid["dlat"]))
    src = np.zeros_like(mask)
    try:
        g = io.load_glacier_nc()
        gt = g["geotransform"]
        valid = np.isfinite(g["thickness"]) & (g["thickness"] > 1.0)
        gy, gx = np.nonzero(valid)
        for row, col in zip(gy, gx):
            e = gt[0] + (col + 0.5) * gt[1]
            n = gt[3] + (row + 0.5) * gt[5]
            lo, la = gd.utm_to_lonlat(e, n, 45)
            rr = int(round((float(grid["lat0"]) - la) / abs(float(grid["dlat"]))))
            cc = int(round((lo - float(grid["lon0"])) / float(grid["dlon"])))
            if 0 <= rr < z.shape[0] and 0 <= cc < z.shape[1]:
                src[rr, cc] = 1
    except (KeyError, ValueError, OSError) as exc:
        raise RuntimeError("glacier coverage mapping failed; cannot silently omit attachment 4") from exc
    from .glacier_masks import rgi6_mask
    boundary_mask, boundary_meta = rgi6_mask(grid, z.shape)
    cover = boundary_mask & mask
    f_gl = float(cover.sum()) / max(int(mask.sum()), 1)

    # 三种沟长—比降口径并列（用途严格区分，见模块 docstring）。
    # 几何近似：√像元数×30 m×1.6，仅作面积—形状粗估。
    L_geo = float(np.sqrt(mask.sum()) * 30.0 * 1.6)
    z_range = float(np.nanmax(zc) - np.nanmin(zc))
    # 追迹口径：自出口按最大汇流上游追迹的像元链
    from . import terrain as _tr
    stem = _tr.trace_upstream_mainstem(d, acc, best, min_acc=2000.0)
    stem_z = np.array([z[p] for p in stem], dtype="float64")
    steps = np.diff(np.asarray(stem), axis=0)
    dy = abs(float(grid["dlat"])) * 110574.0
    dx = abs(float(grid["dlon"])) * 111320.0 * np.cos(np.radians(lat_c))
    L_trace = float(np.hypot(steps[:, 0] * dy, steps[:, 1] * dx).sum())
    S_trace = float((stem_z[-1] - stem_z[0]) / max(L_trace, 1.0)) if len(stem) > 5 else np.nan
    out = {
        "出口像元": [int(best[0]), int(best[1])],
        "出口汇流累积": int(best_acc),
        "集水区像元数": int(mask.sum()),
        "集水区面积_km2": round(area, 3),
        "像元面积_m2": round(cell_area, 1),
        # 高程带代理与冰厚场源区两个分量必须分别报告：二者量级差近两个数量级，
        # 并集值几乎完全由高程带决定，不能称为"冰川覆盖率"（论文失效点 9）。
        "冰川覆盖率": round(f_gl, 5),
        "覆盖率分解": {
            "高程带代理_ge4800m": round(float((ice & mask).sum() / mask.sum()), 5),
            "冰厚场源区": round(float(((src > 0) & mask).sum() / mask.sum()), 5),
            "旧高程代理并集_仅审计": round(float(((ice | (src > 0)) & mask).sum()/mask.sum()),5),
            "RGI6编目边界": round(f_gl, 5),
        },
        "冰川面积来源": boundary_meta,
        "冰川覆盖语义": "RGI6编目边界栅格化；非2026年消融区、积雪或实时冰川范围",
        "冰川区平均高程_m": round(float(np.nanmean(zc[cover[mask]])) if f_gl > 0 else float(np.nanmax(zc)), 1),
        "集水区平均高程_m": round(float(np.nanmean(zc)), 1),
        "集水区最高高程_m": round(float(np.nanmax(zc)), 1),
        "集水区最低高程_m": round(float(np.nanmin(zc)), 1),
        "集水区高差_m": round(z_range, 1),
        # 几何近似口径（仅粗估，不进入 q_c）
        "集水区主沟长_m": round(L_geo, 1),
        "集水区比降": round(z_range / max(L_geo, 1.0), 5),
        # 追迹口径（空间可靠性检验用）
        "追迹主沟长_m": round(L_trace, 1),
        "追迹比降": round(S_trace, 5) if np.isfinite(S_trace) else None,
    }
    return (out, mask) if return_mask else out


# 类码字典未核实，本版停用土地覆盖物理权重，仅保留原码统计。
def catchment_static_features(base: dict | None = None) -> dict:
    """集水区静态特征：坡度统计 + 土地覆盖构成 + 松散物质可蚀性指数。

    B1 成果直接携带 02_静态特征.json 而未随附生成代码，从原始数据冷启动
    即 KeyError；本函数补全其生产链。坡度用 Horn 法（与二维引擎同式，
    在填洼 DEM 上计算），土地覆盖（附件5, 2023）按 DEM 像元中心最近邻
    重采样，类码未核实，物理可蚀性指数返回 None。
    """
    from . import terrain as tr

    if base is None:
        base, mask = catchment_glacier_fraction(return_mask=True)
    else:
        _, mask = catchment_glacier_fraction(return_mask=True)
    dem, grid = io.load_dem()
    zf = np.load(io.CACHE / "filled.npy")
    # 与 terrain_prep.cell_meters 同式（纬度参考 28.4°）
    dy = abs(float(grid["dlat"])) * 110574.0
    dx = abs(float(grid["dlon"])) * 111320.0 * np.cos(np.radians(28.4))
    slp = np.degrees(np.arctan(tr.slope_horn(zf.astype("float64"), (dy, dx))))
    s = slp[mask & np.isfinite(slp)]

    # 土地覆盖 → DEM 网格（最近邻，按像元中心经纬度）
    lc, lg = io.load_landcover()
    yy, xx = np.nonzero(mask)
    lon_c = float(grid["lon0"]) + xx * float(grid["dlon"])
    lat_c = float(grid["lat0"]) + yy * float(grid["dlat"])
    rr_f = (lat_c - float(lg["lat0"])) / float(lg["dlat"])
    cc_f = (lon_c - float(lg["lon0"])) / float(lg["dlon"])
    rr = np.round(rr_f).astype(int)
    cc = np.round(cc_f).astype(int)
    ok = (rr >= 0) & (rr < lc.shape[0]) & (cc >= 0) & (cc < lc.shape[1])
    lcv = np.asarray(lc[rr[ok], cc[ok]]).astype(int)
    vals, cnts = np.unique(lcv, return_counts=True)
    comp = {str(int(v)): int(n) for v, n in zip(vals, cnts)}
    # No physical weights without the product's class dictionary.
    erod = None

    L_km = float(base["集水区主沟长_m"]) / 1000.0
    return {
        "集水区面积_km2": base["集水区面积_km2"],
        "平均坡度_deg": round(float(np.mean(s)), 4),
        "坡度p90_deg": round(float(np.percentile(s, 90)), 4),
        "陡坡面积比_gt30deg": round(float((s > 30.0).mean()), 5),
        "冰川面积_km2": round(float(base["集水区面积_km2"]) * float(base["冰川覆盖率"]), 3),
        "冰川覆盖率": base["冰川覆盖率"],
        "松散物质可蚀性指数": erod,
        "可蚀性状态": "unavailable: supplied class dictionary not verified; physical weighting disabled: class dictionary unverified",
        "可蚀性权重_先验": {},
        "土地覆盖构成": comp,
        "土地覆盖有效比例": float(ok.mean()),
        "土地覆盖名称": {str(k): v for k, v in io.LC_NAMES.items()},
        "土地覆盖重采样": "附件5(2023) 按 DEM 像元中心最近邻；越界像元不冒充0类；原码语义未确认",
        "平均高程_m": base["集水区平均高程_m"],
        "高差_m": base["集水区高差_m"],
        "比降": base["集水区比降"],
        "主沟长_m": base["集水区主沟长_m"],
        "流域形状系数": round(float(base["集水区面积_km2"]) / max(L_km ** 2, 1e-6), 5),
    }


def api_climate_thresholds(feat: pd.DataFrame) -> dict:
    """Return the frozen warm-season API terciles used for AMC classification."""
    api = pd.to_numeric(feat["API"], errors="coerce")
    preset = feat.attrs.get("api_quantiles")
    mask = np.isin(feat.index.year, API_TRAIN_YEARS) & np.isin(
        feat.index.month, API_WARM_MONTHS
    )
    training_api = api.loc[mask]
    baseline = training_api.where(np.isfinite(training_api) & training_api.ge(0)).dropna()
    if preset is not None:
        q1, q3 = (float(value) for value in preset)
        source = "frozen feat.attrs['api_quantiles']"
    else:
        if len(baseline) < 100:
            raise ValueError(
                "API warm-season training sample is too small; refusing an all-season fallback"
            )
        q1, q3 = (float(value) for value in baseline.quantile([1 / 3, 2 / 3]))
        source = "2021-2025 June-September observed training sample"
    if not (np.isfinite(q1) and np.isfinite(q3) and 0 <= q1 <= q3):
        raise ValueError(f"invalid API terciles: {(q1, q3)!r}")
    return {
        "q1": q1,
        "q3": q3,
        "source": source,
        "training_years": list(API_TRAIN_YEARS),
        "training_months": list(API_WARM_MONTHS),
        "training_valid_hours": int(len(baseline)),
        "index_start": str(feat.index.min()),
        "index_end": str(feat.index.max()),
    }


def event_rain_runoff(
    feat: pd.DataFrame, cn: float = CN0, dry_hours: int = 6,
) -> pd.DataFrame:
    """Event-cumulative SCS-CN with causal, observed-zero event boundaries.

    The index must be sorted, unique and on the hour. Gaps and invalid rain
    invalidate the event history; only ``dry_hours`` consecutive observed
    P=0 hours recover a known boundary. Positive rain before recovery has no
    event_id. An event freezes CN from API exactly one hour before its start;
    unavailable/invalid antecedent API makes all its positive rain unevaluable.
    Zero-rain runoff is always zero, including during unknown history.

    States: unknown_boundary, invalid_rain, dry_boundary, event, event_api_missing.
    ``Q`` is cumulative direct runoff in mm, not discharge. Unknown history has
    NaN cumulative_P/Q; a known boundary has zero cumulative_P/Q and no event_id.
    On the confirming dry hour the previous event is cleared; boundary_confirmed
    marks that hour only. Event IDs are local to this complete chronological call.
    No calendar reset or future event-closure requirement is imposed.
    """
    index = feat.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("rainfall requires a DatetimeIndex")
    if index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("rainfall index must be monotonic, unique and non-missing")
    if ((index.minute != 0) | (index.second != 0)
            | (index.microsecond != 0) | (index.nanosecond != 0)).any():
        raise ValueError("rainfall index must contain whole-hour timestamps")
    if (isinstance(dry_hours, (bool, np.bool_))
            or not isinstance(dry_hours, (int, np.integer)) or dry_hours < 1):
        raise ValueError("dry_hours must be a positive integer")
    cn = float(cn)
    if not np.isfinite(cn) or not 0 < cn <= 100:
        raise ValueError("base CN must be finite and in (0, 100]")
    p = pd.to_numeric(feat["P"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    api = pd.to_numeric(feat["API"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    meta = api_climate_thresholds(feat)
    q1, q3 = meta["q1"], meta["q3"]
    known_boundary = False
    dry_run = event_count = 0
    event_id = None
    event_cn = event_api = cumulative_p = cumulative_q = np.nan
    rows = []
    for i, value in enumerate(p):
        gap = i > 0 and index[i] - index[i - 1] != pd.Timedelta(hours=1)
        if gap:
            known_boundary, dry_run, event_id = False, 0, None
            event_cn = event_api = cumulative_p = cumulative_q = np.nan
        started = confirmed = False
        runoff = np.nan
        if not np.isfinite(value) or value < 0:
            known_boundary, dry_run, event_id = False, 0, None
            event_cn = event_api = cumulative_p = cumulative_q = np.nan
            state = "invalid_rain"
        elif value == 0:
            runoff = 0.0
            dry_run += 1
            if dry_run == dry_hours:
                known_boundary, confirmed, event_id = True, True, None
                event_cn = event_api = np.nan
                cumulative_p = cumulative_q = 0.0
            state = (
                "event" if np.isfinite(event_cn) else "event_api_missing"
            ) if event_id is not None else (
                "dry_boundary" if known_boundary else "unknown_boundary"
            )
        else:
            dry_run = 0
            if not known_boundary:
                state = "unknown_boundary"
            else:
                if event_id is None:
                    event_count += 1
                    event_id, started = event_count, True
                    event_api = api[i - 1]
                    cumulative_p = 0.0
                    if np.isfinite(event_api) and event_api >= 0:
                        factor = 0.75 if event_api < q1 else 1.15 if event_api > q3 else 1.0
                        event_cn = float(np.clip(cn * factor, 30, 98))
                        cumulative_q = 0.0
                    else:
                        event_cn = cumulative_q = np.nan
                cumulative_p += value
                if not np.isfinite(cumulative_p):
                    raise ValueError("event cumulative rainfall overflow")
                if np.isfinite(event_cn):
                    s_max = 25400.0 / event_cn - 254.0
                    excess = cumulative_p - 0.2 * s_max
                    new_q = excess / (1.0 + s_max / excess) if excess > 0 else 0.0
                    runoff = new_q - cumulative_q
                    cumulative_q = new_q
                    state = "event"
                else:
                    state = "event_api_missing"
        rows.append((runoff, state, event_id, event_cn, event_api,
                     cumulative_p, cumulative_q, dry_run, known_boundary,
                     started, confirmed, gap))
    out = pd.DataFrame(rows, index=index, columns=[
        "runoff_mm", "state", "event_id", "event_CN", "event_API",
        "cumulative_P", "Q", "dry_hours", "boundary_known",
        "event_start", "boundary_confirmed", "time_gap",
    ])
    out["event_id"] = pd.array(out["event_id"], dtype="Int64")
    out.attrs.update({
        "api_quantiles": (q1, q3), "api_quantiles_record": meta,
        "rainfall_event_protocol": {
            "dry_hours": int(dry_hours), "dry_rule": "finite P == 0",
            "CN_rule": "freeze at event start from API exactly one hour earlier",
            "cumulative_units": "mm", "calendar_reset": False,
        },
    })
    return out


def rain_runoff(
    feat: pd.DataFrame, cn: float = CN0, dry_hours: int = 6,
) -> pd.Series:
    """Hourly runoff increments (mm); event protocol/audit: event_rain_runoff."""
    return event_rain_runoff(feat, cn, dry_hours)["runoff_mm"].rename(None)


def melt_water(feat: pd.DataFrame, temp_offset: float = -7.5) -> pd.Series:
    """能量平衡融水（mm w.e./h）：有效融化能量 → 水当量，按源区气温调整有效时段。

    量纲：melt_energy 为辐射通量余量（W·m⁻²），故 1 h 的能量为 E×3600 J·m⁻²。
    化水当量深度 = E×3600 ⁄ (ρ_w·L_f)，ρ_w = 1000 kg·m⁻³、L_f = 334000 J·kg⁻¹，
    结果以 m 表示，乘 1000 得 mm。合并常数即 3600×1000 ⁄ (1000×334000) = 0.010778 mm 每 (W·m⁻²)。
    """
    energy = feat["melt_energy"]                   # W·m⁻²; missing is not zero
    t_src = feat["Ta_2m"] + temp_offset
    # 源区气温低于 0 ℃ 时融化受抑制（指数过渡，避免硬截断）
    factor = 1.0 / (1.0 + np.exp(-(t_src - 0.0) / 1.5))
    mm_per_wm2 = 3600.0 / (RHO_W * LF_J) * 1000.0      # mm 每 (W·m⁻²)
    return energy * mm_per_wm2 * factor


# ------------------------------------------------------------------ 触发指数
class TriggerModel:
    """正式水源—地形状态模型；旧事件 k 只能显式启用为审计模式。"""

    def __init__(self, area_km2: float, glacier_frac: float, q_crit_m2s: float,
                 width_m: float, cn: float = CN0, temp_offset: float = -7.5,
                 k: float = 1.0, mechanical_regime: str = "water_controlled",
                 use_legacy_k: bool = False):
        self.area = float(area_km2)
        self.f_gl = float(glacier_frac)
        self.q_crit = float(q_crit_m2s)
        self.width = float(width_m)
        self.cn = float(cn)
        self.temp_offset = float(temp_offset)
        self.k = float(k)
        self.mechanical_regime = str(mechanical_regime)
        self.use_legacy_k = bool(use_legacy_k)

    def discharge(self, feat: pd.DataFrame) -> pd.DataFrame:
        r = rain_runoff(feat, self.cn)
        m = melt_water(feat, self.temp_offset) * self.f_gl
        # Positional routing weights must not jump across missing clock hours.
        grid = pd.date_range(feat.index.min(), feat.index.max(), freq="h") if len(feat) else feat.index
        r, m = r.reindex(grid), m.reindex(grid)
        # 汇流：降雨按 1 h 滞后三角形权重，融水按 3 h 平滑
        w_rain = pd.Series([0.6, 0.3, 0.1], index=[0, 1, 2])
        q_rain = sum(r.shift(i) * w for i, w in w_rain.items())
        q_melt = m.rolling(3, min_periods=3).mean()
        water = (q_rain + q_melt) * self.area * 1000.0 / 3600.0            # m³/s
        out = pd.DataFrame({"Q_rain": q_rain * self.area * 1000 / 3600,
                            "Q_melt": q_melt * self.area * 1000 / 3600,
                            "Q_water": water}).reindex(feat.index)
        base_q = self.base_q_crit()
        out["Q_crit"] = base_q * (self.k if self.use_legacy_k else 1.0)
        if self.mechanical_regime == "water_controlled" and base_q > 0:
            out["T_index"] = out["Q_water"] / out["Q_crit"]
            hydraulic = pd.Series("not_evaluable", index=out.index, dtype=object)
            valid = (
                out["Q_water"].notna()
                & out["Q_crit"].notna()
                & out["Q_crit"].gt(0)
                & out["T_index"].notna()
            )
            hydraulic.loc[valid] = np.where(
                out.loc[valid, "T_index"] >= 1.0,
                "hydraulic_candidate",
                "below_threshold",
            )
            out["hydraulic_state"] = hydraulic
        else:
            out["T_index"] = np.nan
            hydraulic = pd.Series("inapplicable", index=out.index, dtype=object)
            hydraulic.loc[out["Q_water"].isna()] = "not_evaluable"
            out["hydraulic_state"] = hydraulic
        out["mechanical_regime"] = self.mechanical_regime
        out["legacy_k_used"] = self.use_legacy_k
        return out

    def base_q_crit(self) -> float:
        """未标定的起动流量 q_c·B/κ；零门槛分支返回 0。"""
        if self.mechanical_regime == "not_evaluable":
            return np.nan
        return float(self.q_crit * self.width / KAPPA_Q)

    def calibrate(self, feat: pd.DataFrame, window: tuple[str, str]) -> float:
        """旧方法审计：令事件窗口内触发指数峰值为 1，解出集总系数 k。

        此函数只在显式设置 ``use_legacy_k=True`` 后进入输出，不应被问题三调用。
        k = 窗口内 Q_w 峰值 / (q_c·B/κ)。零门槛或无效几何返回 NaN，不伪造有限标定。
        """
        d = self.discharge(feat)
        peak = float(d.loc[window[0]:window[1], "Q_water"].max())
        if not np.isfinite(peak) or peak <= 0:
            return 1.0
        base_q = self.base_q_crit()
        if not np.isfinite(base_q) or base_q <= 0:
            self.k = np.nan
            return self.k
        self.k = peak / base_q
        return self.k


def centerline_slope() -> dict:
    """问题 4 中心线的沟长与比降——q_c 的唯一计算口径。

    读 求解结果/数据/04_中心线校验.json（由 step04_simulation.py 先行产出）。
    该文件缺失时抛出明确错误，而不是静默回退到别的口径：口径混用正是
    "同一物理量出现多个互斥值"的根源，必须让依赖顺序的违约显式失败。
    """
    path = io.DATA_OUT / "04_中心线校验.json"
    if not path.exists():
        raise FileNotFoundError(
            "缺少 04_中心线校验.json：q_c 依赖问题 4 的中心线口径，"
            "请先运行 step04_simulation.py（run_all.py 已按此顺序排列）。"
        )
    import json
    d = json.loads(path.read_text(encoding="utf-8"))
    z = d["高程范围_m"]                       # [沟口, 源头]
    L = float(d["中心线长度_km"]) * 1000.0
    drop = float(z[1]) - float(z[0])
    return {"沟长_m": L, "落差_m": drop, "比降": drop / max(L, 1.0),
            "来源": "问题4 中心线（04_中心线校验.json）"}


def default_model(feat: pd.DataFrame | None = None, calibrate: bool = False,
                  slope_def: str = "centerline") -> TriggerModel:
    """以口岸集水区为对象的默认触发模型。

    slope_def 只接受 "centerline"（默认，q_c 的正式口径）；其余两种口径
    由 `slope_variants()` 单独计算并用于敏感性/可靠性讨论，不进入正式模型。
    """
    if slope_def != "centerline":
        raise ValueError(
            f"q_c 只允许使用 centerline 口径（收到 {slope_def!r}）；"
            "三种口径并列见 slope_variants()，不得混用。"
        )
    info = catchment_glacier_fraction()
    sl = centerline_slope()
    # 集水区整体比降（中心线口径）→ 完整力学分支。不做任何未披露的放大。
    mechanics = _mechanics(sl["比降"])
    qc = float(mechanics["q_crit_m2s"]) if mechanics["q_crit_m2s"] is not None else np.nan
    w = KAPPA_WIDTH * np.sqrt(info["集水区面积_km2"])
    geometry = io.load_glacier_nc()
    valid = np.isfinite(geometry["thickness"]) & (geometry["thickness"] > 1.0)
    source_elevation = float(np.nanmean(geometry["dem"][valid]))
    # The source DEM meaning is uncertain; this is the declared unraised-DEM scenario.
    station_elevation = 3980.0
    offset = -(source_elevation - station_elevation) * 0.006
    m = TriggerModel(
        info["集水区面积_km2"], info["冰川覆盖率"], qc, min(w, 400.0),
        cn=CN0, temp_offset=offset, mechanical_regime=mechanics["regime"],
        use_legacy_k=bool(calibrate),
    )
    m.mechanics = mechanics
    m.q_crit_old = _q_crit_old(sl["比降"])
    m.temperature_geometry = {"source_elevation_m": source_elevation,
                              "station_elevation_m": station_elevation,
                              "lapse_K_per_m": 0.006, "offset_C": offset}
    m.cn0 = CN0
    m.slope_info = sl
    if calibrate and feat is not None:
        m.calibrate(feat, ("2026-08-26 08:00", "2026-08-26 13:00"))
    return m


def slope_variants() -> dict:
    """三种沟长—比降口径的并列结果，供论文与可靠性讨论逐项引用。"""
    info = catchment_glacier_fraction()
    cl = centerline_slope()
    out = {
        "geometric": {"沟长_m": info["集水区主沟长_m"], "比降": info["集水区比降"],
                      "用途": "几何近似，仅面积—形状粗估，不进入 q_c"},
        "traced": {"沟长_m": info.get("追迹主沟长_m"), "比降": info.get("追迹比降"),
                   "用途": "D8 主河道追迹，用于分辨率敏感性与沟道迁移性"},
        "centerline": {"沟长_m": cl["沟长_m"], "比降": cl["比降"],
                       "用途": "问题 4 中心线，q_c 的唯一计算口径"},
    }
    for k, v in out.items():
        s = v["比降"]
        mechanics = _mechanics(s)
        v["mechanical_regime"] = mechanics["regime"]
        v["q_c_m2s"] = mechanics["q_crit_m2s"]
        v["q_c_old_m2s"] = _q_crit_old(s) if s and s > 0 else None
    return out
