"""气象与近地层特征工程，能量输出为融化潜势代理而非实测消融。

所有函数以 pandas DataFrame（逐小时索引）为输入/输出，列名遵循：
  Ta_<z>m / RH_<z>m / WS_<z>m / WD_<z>m / soilT_<d>m / VWC_<d>m / G_<d>m
"""
from __future__ import annotations

import numpy as np
import pandas as pd

G = 9.80665
LV = 2.501e6
CP = 1004.67
R_D = 287.058

LEVELS_T = [1, 2, 4, 10, 20]
LEVELS_W = [1, 2, 4, 10, 20]
DEPTHS = ["0.1", "0.2", "0.4", "0.8", "1.6"]
RI_SHEAR_EPS = 1e-6


def hourly_regular(df: pd.DataFrame) -> pd.DataFrame:
    """显式补齐小时索引；只补缺测位置，不插值或把缺测当无雨。"""
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_unique:
        raise ValueError("观测必须有唯一的 DatetimeIndex")
    if not (df.index == df.index.floor("h")).all():
        raise ValueError("特征工程只接受整点小时记录")
    out = df.sort_index().reindex(pd.date_range(df.index.min(), df.index.max(), freq="h"))
    out.attrs = df.attrs.copy()
    return out


# ---------------------------------------------------------------- 热力学量
def es_hpa(t_celsius: pd.Series | np.ndarray) -> np.ndarray:
    """饱和水汽压（Bolton 1980），hPa。"""
    t = np.asarray(t_celsius, dtype="float64")
    return 6.112 * np.exp(17.67 * t / (t + 243.5))


def dewpoint(e_hpa: pd.Series | np.ndarray) -> np.ndarray:
    """由水汽压反算露点温度（℃）。"""
    e = np.clip(np.asarray(e_hpa, dtype="float64"), 1e-3, None)
    ln = np.log(e / 6.112)
    return 243.5 * ln / (17.67 - ln)


def specific_humidity(e_hpa: pd.Series | np.ndarray, p_hpa: pd.Series | np.ndarray) -> np.ndarray:
    """比湿（kg/kg）。"""
    e = np.asarray(e_hpa, dtype="float64")
    p = np.asarray(p_hpa, dtype="float64")
    return 0.622 * e / (p - 0.378 * e)


def potential_temperature(t_celsius: pd.Series | np.ndarray, p_hpa: pd.Series | np.ndarray) -> np.ndarray:
    """位温（K）。"""
    t = np.asarray(t_celsius, dtype="float64") + 273.15
    p = np.asarray(p_hpa, dtype="float64")
    return t * (1000.0 / p) ** 0.2857


def virtual_potential_temperature(t_celsius, q, p_hpa) -> np.ndarray:
    """虚位温（K）。"""
    return potential_temperature(t_celsius, p_hpa) * (1.0 + 0.61 * np.asarray(q, dtype="float64"))


def air_density(p_hpa, t_celsius) -> np.ndarray:
    """空气密度（kg/m³）。"""
    p = np.asarray(p_hpa, dtype="float64") * 100.0
    t = np.asarray(t_celsius, dtype="float64") + 273.15
    return p / (R_D * t)


def hydrostatic_profiles(df: pd.DataFrame, z_ref: int = 2,
                         corrected: bool = True) -> pd.DataFrame:
    """Five-layer common-sample thermodynamics; reference height is assumed, not observed."""
    if z_ref not in LEVELS_T:
        raise ValueError("reference height must be one of the observed levels")
    needed = [f"{family}_{z}m" for family in ("Ta", "RH") for z in LEVELS_T] + ["Pressure"]
    t = df.reindex(columns=[f"Ta_{z}m" for z in LEVELS_T]).to_numpy(float) + 273.15
    rh = df.reindex(columns=[f"RH_{z}m" for z in LEVELS_T]).to_numpy(float)
    pref = df.reindex(columns=["Pressure"]).to_numpy(float)
    valid = (np.isfinite(df.reindex(columns=needed).to_numpy(float)).all(axis=1)
             & (t > 0).all(axis=1) & ((rh >= 0) & (rh <= 100)).all(axis=1)
             & (pref[:, 0] > 0))
    e = es_hpa(t - 273.15) * rh / 100
    p = np.repeat(pref, len(LEVELS_T), axis=1)
    valid &= (p - .378 * e > 0).all(axis=1)
    p[~valid] = np.nan
    residual = np.zeros(len(df), dtype=float)
    ref_index = LEVELS_T.index(z_ref)
    if corrected:
        for _ in range(3):
            q = specific_humidity(e, p)
            tv = t * (1 + .61 * q)
            segments = np.diff(LEVELS_T)[None, :] * .5 * (1 / tv[:, :-1] + 1 / tv[:, 1:])
            integral = np.column_stack((np.zeros(len(df)), np.cumsum(segments, axis=1)))
            new_p = pref * np.exp(-G / R_D * (integral - integral[:, ref_index, None]))
            valid &= (np.isfinite(new_p).all(axis=1) & (tv > 0).all(axis=1)
                      & (new_p - .378 * e > 0).all(axis=1))
            new_p[~valid] = np.nan
            delta = np.abs(new_p - p)
            residual = np.max(np.where(np.isfinite(delta), delta, 0), axis=1)
            p = new_p
    residual[~valid] = np.nan
    q = specific_humidity(e, p)
    theta = t * (1000 / p) ** .2857
    theta_v = theta * (1 + .61 * q)
    out = pd.DataFrame(index=df.index)
    for k, z in enumerate(LEVELS_T):
        for name, values in (("p", p), ("q", q), ("theta", theta), ("thetav", theta_v)):
            out[f"{name}_{z}m"] = values[:, k]
    out["thermo_valid"] = valid
    out["pressure_iteration_residual_hPa"] = residual
    finite_residual = residual[np.isfinite(residual)]
    out.attrs.update({"reference_height_m_assumed": z_ref, "hydrostatic_corrected": corrected,
                      "iterations": 3 if corrected else 0, "five_layer_common_hours": int(valid.sum()),
                      "total_hours": len(df), "missing_fraction": float(1 - valid.mean()),
                      "iteration_3_minus_2_max_hPa": float(finite_residual.max()) if finite_residual.size else None,
                      "iteration_3_minus_2_median_hPa": float(np.median(finite_residual)) if finite_residual.size else None,
                      "pressure_source": "MET Pressure; reference height assumed; FLUX PA not substituted"})
    return out


def vector_shear_squared(df: pd.DataFrame, lo: int = 2, hi: int = 20) -> pd.Series:
    needed = [f"{family}_{z}m" for family in ("WS", "WD") for z in (lo, hi)]
    if not set(needed).issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    a, b = np.radians(df[f"WD_{lo}m"]), np.radians(df[f"WD_{hi}m"])
    du = -df[f"WS_{hi}m"] * np.sin(b) + df[f"WS_{lo}m"] * np.sin(a)
    dv = -df[f"WS_{hi}m"] * np.cos(b) + df[f"WS_{lo}m"] * np.cos(a)
    return (du ** 2 + dv ** 2).where(df[needed].notna().all(axis=1))


# ---------------------------------------------------------------- 边界层结构
def temperature_gradient(df: pd.DataFrame) -> pd.DataFrame:
    """各相邻层的温度梯度（K/100m，按层高差归一）。"""
    out = {}
    for lo, hi in zip(LEVELS_T[:-1], LEVELS_T[1:]):
        col_lo, col_hi = f"Ta_{lo}m", f"Ta_{hi}m"
        if col_lo in df and col_hi in df:
            out[f"dT_{lo}_{hi}m"] = (df[col_hi] - df[col_lo]) / (hi - lo) * 100.0
    return pd.DataFrame(out, index=df.index)


def inversion_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """逆温强度（最大正温度梯度）与逆温层结高度。"""
    grad = temperature_gradient(df)
    if grad.empty:
        return pd.DataFrame(index=df.index)
    out = pd.DataFrame(index=df.index)
    out["inv_max"] = grad.clip(lower=0).max(axis=1)          # K/100m
    out["inv_count"] = (grad > 0.05).sum(axis=1)             # 逆温层数
    out["dT_total_1_20m"] = df["Ta_20m"] - df["Ta_1m"]
    return out


def wind_shear(df: pd.DataFrame) -> pd.DataFrame:
    """风切变与对数律拟合（粗糙度、摩擦速度）。"""
    out = pd.DataFrame(index=df.index)
    if {"WS_2m", "WS_10m"}.issubset(df.columns):
        out["shear_2_10m"] = (df["WS_10m"] - df["WS_2m"]) / 8.0
    if {"WS_1m", "WS_20m"}.issubset(df.columns):
        out["shear_1_20m"] = (df["WS_20m"] - df["WS_1m"]) / 19.0
    out["vector_shear_2_20m"] = np.sqrt(vector_shear_squared(df)) / 18.0
    out["WS_bulk"] = df[[f"WS_{z}m" for z in LEVELS_W if f"WS_{z}m" in df]].mean(axis=1)
    return out


def bulk_richardson(df: pd.DataFrame, profile: pd.DataFrame | None = None,
                    z_ref: int = 2, corrected: bool = True) -> pd.Series:
    """整体理查森数 Ri_b = g·Δθ_v·Δz / (θ_v·ΔU²)，取 2 m 与 20 m 之间。"""
    need = {"Ta_2m", "Ta_20m", "RH_2m", "RH_20m", "WS_2m", "WS_20m",
            "Pressure", "WD_2m", "WD_20m"}
    if not need.issubset(df.columns):
        return pd.Series(np.nan, index=df.index)
    profile = hydrostatic_profiles(df, z_ref, corrected) if profile is None else profile
    th2, th20 = profile["thetav_2m"], profile["thetav_20m"]
    dtheta = th20 - th2
    du2 = vector_shear_squared(df)
    dz = 18.0
    theta_mean = 0.5 * (th2 + th20)
    with np.errstate(divide="ignore", invalid="ignore"):
        ri = G / theta_mean * dtheta * dz / du2.where(du2 >= RI_SHEAR_EPS)
    ri = ri.replace([np.inf, -np.inf], np.nan)
    eligible = df[list(need)].notna().all(axis=1) & profile["thermo_valid"]
    ri.attrs["shear_mask_count"] = int((eligible & (du2 < RI_SHEAR_EPS)).sum())
    ri.attrs["eligible_count"] = int(eligible.sum())
    ri.attrs["shear_epsilon_m2_s2"] = RI_SHEAR_EPS
    return ri


def stability_class(ri: pd.Series | np.ndarray) -> np.ndarray:
    """按 Ri_b 分级：1 强不稳定 … 5 强稳定。"""
    ri = np.asarray(ri, dtype="float64")
    cls = np.full(ri.shape, np.nan)
    cls[ri < -0.2] = 1.0
    cls[(ri >= -0.2) & (ri < -0.01)] = 2.0
    cls[(ri >= -0.01) & (ri <= 0.01)] = 3.0
    cls[(ri > 0.01) & (ri <= 0.25)] = 4.0
    cls[ri > 0.25] = 5.0
    return cls


def pbl_diagnostics(df: pd.DataFrame, profile: pd.DataFrame | None = None,
                    z_ref: int = 2, corrected: bool = True) -> pd.DataFrame:
    """Keep observed threshold crossing separate from right censoring."""
    profile = hydrostatic_profiles(df, z_ref, corrected) if profile is None else profile
    out = pd.DataFrame({"pblh": np.nan, "pbl_crossing_height_m": np.nan,
                        "pbl_right_censored": np.nan}, index=df.index)
    pending = pd.Series(True, index=df.index)
    all_valid = pd.Series(True, index=df.index)
    for hi in (4, 10, 20):
        needed = [f"{fam}_{z}m" for fam in ("Ta", "RH", "WS", "WD") for z in (2, hi)] + ["Pressure"]
        if not set(needed).issubset(df.columns):
            return out
        thlo, thhi = profile["thetav_2m"], profile[f"thetav_{hi}m"]
        du2 = vector_shear_squared(df, 2, hi)
        valid = df[needed].notna().all(axis=1) & profile["thermo_valid"] & (du2 >= RI_SHEAR_EPS)
        all_valid &= valid
        ri = G * (thhi - thlo) * (hi - 2) / (0.5 * (thhi + thlo) * du2.where(valid))
        hit = pending & all_valid & (ri >= 0.25)
        out.loc[hit, ["pblh", "pbl_crossing_height_m"]] = float(hi)
        out.loc[hit, "pbl_right_censored"] = 0.
        pending &= ~hit
    out.loc[pending & all_valid, "pblh"] = 20.0
    out.loc[pending & all_valid, "pbl_right_censored"] = 1.
    out.attrs["right_censored_at_20m"] = int((pending & all_valid).sum())
    return out


def pbl_height(df: pd.DataFrame, profile: pd.DataFrame | None = None) -> pd.Series:
    diagnostics = pbl_diagnostics(df, profile)
    out = diagnostics["pblh"].copy()
    out.attrs.update(diagnostics.attrs)
    return out


# ---------------------------------------------------------------- 辐射收支
def clear_sky_rsd(index: pd.DatetimeIndex, lat: float = 28.4, elev: float = 3980.0) -> np.ndarray:
    """晴空水平面太阳辐射（Haurwitz 模型），W/m²。"""
    doy = index.dayofyear.values.astype(float)
    hour = index.hour.values.astype(float) + index.minute.values / 60.0
    decl = np.radians(23.45 * np.sin(2 * np.pi * (284 + doy) / 365.0))
    B = 2 * np.pi * (doy - 81) / 364.0
    eot = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)     # 分钟
    # 时间按UTC+8假设，中央经线120E；站点经度不等于时区中央经线。
    solar_time = hour + (85.3 - 120.0) / 15.0 + eot / 60.0
    ha = np.radians(15.0 * (solar_time - 12.0))
    phi = np.radians(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, 0.0, None)
    r0 = 1.0 - 0.03344 * np.cos(2 * np.pi * doy / 365.0)
    s = 1361.0 * r0 * cosz
    clear = 1098.0 * cosz * np.exp(-0.057 / np.maximum(cosz, 0.02))
    return np.where(cosz > 0.01, np.minimum(s, clear) * r0, 0.0)


def radiation_budget(df: pd.DataFrame) -> pd.DataFrame:
    """辐射收支：净短波、净长波、净辐射、反照率、云量指数、晴空指数。"""
    out = pd.DataFrame(index=df.index)
    rsd = df["Rsd"]; rsu = df["Rsu"]; rld = df["Rld"]; rlu = df["Rlu"]
    out["Rn"] = rsd - rsu + rld - rlu
    out["Rns"] = rsd - rsu
    out["Rnl"] = rld - rlu
    with np.errstate(divide="ignore", invalid="ignore"):
        alb = rsu / rsd
    out["albedo"] = alb.where((rsd > 50) & (alb.between(0, 1.2)))
    cs = clear_sky_rsd(df.index, lat=float(df.attrs.get("lat", 28.4)))
    out["Rsd_clear"] = cs
    with np.errstate(divide="ignore", invalid="ignore"):
        ki = rsd / np.where(cs > 20, cs, np.nan)
    out["clearness"] = ki.clip(0, 1.4)
    out["cloud_idx"] = (1.0 - out["clearness"]).where(rsd > 20)
    return out


def energy_partition(df: pd.DataFrame, rn: pd.Series | None = None) -> pd.DataFrame:
    """地表能量配分：波文比、闭合比、消融有效能量、正积温。

    Parameters
    ----------
    df : 原始观测表（需含 H、LE、G_0.1m、Ta_2m）
    rn : 净辐射序列；缺省时用降级的 Rsd−Rsu+Rld−Rlu 现算，保证函数可独立调用
    """
    out = pd.DataFrame(index=df.index)
    if rn is None and {"Rsd", "Rsu", "Rld", "Rlu"}.issubset(df.columns):
        rn = df["Rsd"] - df["Rsu"] + df["Rld"] - df["Rlu"]
    h = df.get("H"); le = df.get("LE"); g = df.get("G_0.1m")
    if h is not None and le is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            beta = h / le.where(le.abs() > 5)
        out["bowen"] = beta.clip(-10, 10)
        out["available"] = h + le
        if g is not None and rn is not None:
            avail = rn - g
            out["closure"] = (h + le) / avail.where(avail.abs() > 20)
    if g is not None and rn is not None:
        # 可用于融雪/融冰的能量（净辐射扣除土壤热通量与湍流交换后的余量，限正）
        exchange = h + le if h is not None and le is not None else pd.Series(np.nan, index=df.index)
        resid = rn - g - exchange
        out["melt_energy"] = resid.clip(lower=0)
        out["cum_melt_energy"] = out["melt_energy"] * 3600.0 / 1e6           # MJ/m² per hour
    if "Ta_2m" in df:
        out["PDD_hour"] = df["Ta_2m"].clip(lower=0) / 24.0                   # 正积温（℃·d）
    return out


# ---------------------------------------------------------------- 湍流
def turbulence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """湍流：摩擦速度、拖曳系数、湍流强度、湍流动能。"""
    out = pd.DataFrame(index=df.index)
    if "USTAR" in df:
        out["ustar"] = df["USTAR"].where(df["USTAR"] >= 0)
    if {"USTAR", "WS_10m"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            cd = (df["USTAR"] / df["WS_10m"].where(df["WS_10m"] > 0.5)) ** 2
        out["Cd"] = cd.clip(1e-4, 1.0)
    if "TKE" in df:
        out["TKE_obs"] = df["TKE"].where(df["TKE"] >= 0)
    if {"Ux", "Uy", "Uz"}.issubset(df.columns):
        out["sigma_u"] = df["Ux"].rolling(3, min_periods=3).std()
        out["sigma_w"] = df["Uz"].rolling(3, min_periods=3).std()
    if "FETCH_90" in df:
        out["fetch90"] = df["FETCH_90"]
    return out


# ---------------------------------------------------------------- 降水特征
def precipitation_features(df: pd.DataFrame) -> pd.DataFrame:
    """降水：前累积、有效降雨指数、最大雨强、降水历时、干湿转折。"""
    out = pd.DataFrame(index=df.index)
    p = df["Precip"].where(df["Precip"] >= 0)
    out["P"] = p
    for hz in (1, 3, 6, 12, 24, 72, 168):
        out[f"P_{hz}h_coverage"] = p.notna().rolling(hz, min_periods=hz).mean()
        out[f"P_{hz}h"] = p.rolling(hz, min_periods=hz).sum()
    # 缺测打断递推，并屏蔽重启后的7日热身；不假定未知降雨为0。
    api = np.full(len(p), np.nan)
    acc = 0.0
    valid_run = 0
    decay = 0.85 ** (1.0 / 24.0)
    for i, v in enumerate(p.values):
        if not np.isfinite(v):
            acc, valid_run = 0.0, 0
            continue
        acc = acc * decay + v
        valid_run += 1
        if valid_run >= 168:
            api[i] = acc
    out["API"] = api
    out["wet"] = (p > 0.1).astype(float).where(p.notna())
    out["storm_len"] = out["wet"].groupby((out["wet"] != out["wet"].shift()).cumsum()).cumsum() * out["wet"]
    for hz in (1, 3):
        out[f"Pmax_{hz}h"] = p.rolling(hz, min_periods=hz).max()
    out["AP_7d"] = out["P_168h"]
    return out


# ---------------------------------------------------------------- 综合特征表
def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """拼接全部派生特征，返回与原表同索引的特征表。"""
    df = hourly_regular(df)
    rad = radiation_budget(df)
    parts = [
        df[[c for c in df.columns if c.startswith(("Ta_", "RH_", "WS_", "WD_", "soilT_", "VWC_"))
            or c in ("Pressure", "Precip", "Rsd", "Rsu", "Rld", "Rlu", "H", "LE", "G_0.1m",
                     "USTAR", "TKE", "VPD", "PA")]],
        temperature_gradient(df),
        inversion_metrics(df),
        wind_shear(df),
        rad,
        energy_partition(df, rad["Rn"]),
        turbulence_metrics(df),
        precipitation_features(df),
    ]
    feat = pd.concat(parts, axis=1)
    profile = hydrostatic_profiles(df)
    feat = pd.concat((feat, profile), axis=1)
    ri = bulk_richardson(df, profile)
    feat["Ri_b"] = ri
    feat["stab"] = stability_class(feat["Ri_b"])
    feat["dtheta_1_20"] = feat["theta_20m"] - feat["theta_1m"]
    es = es_hpa(df["Ta_2m"])
    e = es * df["RH_2m"] / 100.0
    feat["e_s"] = es
    feat["e_a"] = e
    feat["VPD2"] = es - e
    feat["Td"] = dewpoint(e)
    feat["q"] = profile["q_2m"]
    feat["rho"] = air_density(profile["p_2m"], df["Ta_2m"])
    t_kelvin = df["Ta_2m"] + 273.15
    feat["theta_e"] = profile["theta_2m"] * np.exp(
        LV * feat["q"] / (CP * t_kelvin.where(t_kelvin > 0)))
    feat["PDD_3d"] = df["Ta_2m"].clip(lower=0).rolling(72, min_periods=72).sum()
    feat["PDD_7d"] = df["Ta_2m"].clip(lower=0).rolling(168, min_periods=168).sum()
    feat["rise_24h"] = df["Ta_2m"] - df["Ta_2m"].shift(24)
    feat["freeze_thaw"] = ((df["Ta_2m"] * df["Ta_2m"].shift() < 0).astype(float)
                            .where(df["Ta_2m"].notna() & df["Ta_2m"].shift().notna())
                            .rolling(24, min_periods=24).sum())
    pbl = pbl_diagnostics(df, profile)
    feat = pd.concat((feat, pbl), axis=1)
    feat.attrs["pbl_right_censored_count"] = pbl.attrs.get("right_censored_at_20m", 0)
    feat.attrs["hydrostatic"] = profile.attrs.copy()
    feat.attrs.update(ri.attrs)
    feat.attrs["api_missing_policy"] = "缺测重置；连续168小时有效后才提供API，不填0"
    feat.attrs["feature_version"] = "v2-hydrostatic-five-layer-3iter"
    feat.attrs["energy_units"] = {"melt_energy": "W/m2, unadjusted proxy",
                                  "cum_melt_energy": "MJ/m2 for one hour"}
    return feat


def closure_sensitivity(feat: pd.DataFrame, n_boot: int = 500, seed: int = 20260826) -> tuple[pd.DataFrame, dict]:
    """以训练年块直接自举均值比C；未调整能量保留为基准。"""
    mask = ((feat.index.year >= 2021) & (feat.index.year <= 2025))
    common = feat.loc[mask, ["Rn", "G_0.1m", "H", "LE"]].dropna()
    x = common["Rn"] - common["G_0.1m"]
    common = common.loc[x > 20].copy()
    common["x"] = common["Rn"] - common["G_0.1m"]
    common["y"] = common["H"] + common["LE"]
    if len(common) < 100:
        raise ValueError("训练期共同有效能量样本不足")
    c = float(common["y"].sum() / common["x"].sum())
    if not np.isfinite(c) or c <= 0:
        raise ValueError("训练期闭合比非正，不能据此调整")
    slope, intercept = np.polyfit(common["x"], common["y"], 1)
    blocks = common.groupby(common.index.year)[["x", "y"]].sum()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        b = blocks.iloc[rng.integers(0, len(blocks), len(blocks))].sum()
        ratio = float(b["y"] / b["x"]) if abs(b["x"]) > 1e-9 else np.nan
        draws.append(ratio if np.isfinite(ratio) and ratio > 0 else np.nan)
    draws = np.asarray(draws)
    finite = draws[np.isfinite(draws)]
    if len(finite) < 0.8 * n_boot:
        raise ValueError("闭合比自举无效成员过多")
    lo, hi = np.percentile(finite, [2.5, 97.5])
    out = feat.copy()
    residual = out["Rn"] - out["G_0.1m"]
    turbulent = out["H"] + out["LE"]
    out["melt_energy_adjusted"] = (residual - turbulent / c).clip(lower=0)
    out["cum_melt_energy_adjusted"] = out["melt_energy_adjusted"] * 0.0036
    summaries = {}
    ev = (out.index >= "2026-08-24") & (out.index < "2026-08-27")
    for tag, cc in (("unadjusted", 1.0), ("C_point", c), ("C_lower", lo), ("C_upper", hi)):
        energy = (residual - turbulent / cc).clip(lower=0) * 0.0036
        summaries[tag] = {"C": float(cc), "训练共同样本平均小时能量_MJm2": float(energy.loc[common.index].mean()),
                          "事件三日有效小时": int(energy.loc[ev].notna().sum()),
                          "事件三日已观测能量和_MJm2": float(energy.loc[ev].sum(min_count=1))}
    return out, {"训练期": [2021, 2025], "共同有效样本": int(len(common)),
                 "选择规则": "Rn-G>20 W/m²且Rn/G/H/LE共同有效；不按各情景另筛样本",
                 "闭合比估计量": "sum(H+LE)/sum(Rn-G)", "闭合比C": c,
                 "回归斜率_非C": float(slope), "回归截距": float(intercept),
                 "自举单位": "完整训练年块", "自举重复": n_boot, "随机种子": seed,
                 "闭合比C_95pct": [float(lo), float(hi)],
                 "自举无效比例": float(1 - len(finite) / n_boot),
                 "自举C样本": draws.tolist(), "能量敏感性": summaries,
                 "口径": "melt_energy为未调整基准；adjusted仅为冻结C的能量代理敏感性，非实测融化"}
