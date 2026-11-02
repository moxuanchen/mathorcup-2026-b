"""问题4一维场景诊断：同一注入断面、参数审计、经验停止判据。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmlib import io_data as io, terrain as tr, trigger as tg, hydro  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = io.FIG_OUT
OUT = io.DATA_OUT
CACHE = io.CACHE

RHO_S = 2650.0           # 固相（岩石/冰碛）密度 kg/m³
RHO_W = 1000.0           # 水密度 kg/m³

PARAMS = {
    # V0 由"冰崩体量 × 可动比例"派生，在 main() 中按附件（4）覆写。
    # 这里不放与派生链无关的硬常数：若 V0 是独立常数，其标称来源便推不出该值。
    "V0_m3": None,
    "Cv": 0.45,            # 固体体积分数（题给 0.15–0.80，取偏洪水性值，非算术中点 0.475）
    # B0 与问题 2 的沟宽口径同源（B = 9√A(km²)），但**不做 κ 折算**。
    # 这不是笔误，而是两问的物理对象不同：
    #   问题 2 的 κ 描述"诱发水流只占全沟宽的一部分"（其余为滩地与堆积体），
    #   故起动流量用 q_c·B/κ；
    #   问题 4 描述的是已经起动的**泥石流浆体**，它充满整个沟道断面——
    #   这正是泥石流区别于洪水的定义性特征，故 Manning 公式用全沟宽 B。
    # Full-width transport is an explicit geometry assumption, not a debris-flow definition.
    "B0_m": 40.0,          # 占位，main() 中按 B=9√A 覆写
    "n_man": 0.10,         # 曼宁糙率（泥石流区间 0.08–0.12 的中值）
    "mu": 0.255,           # 等效摩擦系数（附件（6）反演中位数；main() 中按规模分组覆写）
    "T_pass_s": 900.0,     # 过流历时
    "dx_m": 100.0,         # 中心线重采样间距
    "slope_window": 7,     # 纵剖面平滑窗口（节点数）
    "E0_m2s2": 200.0,
    "start_index": 0,
    "beta_B": 0.8,
    "sustain_nodes": 5,    # 停止判据的持续节点数
    "slope_floor": 0.01,   # 曼宁公式的坡度下限
    "h_cap": 15.0,         # 断面流深上限
    "v_cap": 30.0,         # 断面流速上限
    "half_angle_deg": 25.0,  # 扇体扩散半角
    "mobilizable_fraction": 0.15,  # 冰崩体量中参与形成碎屑流的比例（链B）
    # ---- 链A（冰厚-坡度筛选链，基准链）预登记参数（§4.5.4）----
    "kappa_rel": 0.10,      # 冰体释放比例（敏感性 ±50%；κ 符号保留给问题2 沟宽比）
    "gamma_enrich": 3.0,    # 物源放大系数 γ（沟床松散物补给；先验定值并预登记，
                            #   定值先于与历史堆积规模的量级对照——对照仅校核不定标）
    "h_thr_m": 50.0,        # 链A 源区冰厚阈值（m）
    "sl_thr_deg": 25.0,     # 链A 源区坡度阈值（°）
    "rho_ice": 900.0,       # 冰密度（冰-泥密度换算 ρ_ice/ρ_df 用）
}

# 敏感性水平：V0 与 B0 在 main() 中按派生基准值展开（见 build_sens）。
# V0 的情景族 = 链A 基准 ±50%（κ_rel·γ 乘积水平的等效扫描）∪ 链B 上界，
# 不再使用与派生链无关的 {1e6, 2.5e6, 5e6} 硬常数。
SENS = {
    "n_man": [0.06, 0.10, 0.14],
    # μ 水平在 build_sens 中按采用值 μb（大+特大中位数）锚定展开；
    # 此处留空，不使用与基准值无关的孤立数值。
    "mu": [],
    "B0_m": [20.0, 40.0, 80.0],
    "T_pass_s": [300.0, 900.0, 2700.0],

    "Cv": [0.15, 0.45, 0.80],
    "E0_m2s2": [0.0, 200.0, 400.0],
}

# 强度分级阈值（N/m，即单位宽度冲击力）。口径与 I = ρ_m·h·v² 一致，
# 取 10 / 100 / 1000 kN·m⁻¹（ρ_m≈1742 kg·m⁻³）：
#   10 kN/m   ≈ h=0.5 m、v=3.4 m/s —— 可推动沟床松散物质
#   100 kN/m  ≈ h=1.5 m、v=6.2 m/s —— 可破坏浆砌石护岸
#   1000 kN/m ≈ h=3.0 m、v=13.8 m/s —— 可摧毁钢筋混凝土结构
I_CLASS_EDGES = (0.0, 1.0e4, 1.0e5, 1.0e6, np.inf)


def mixture_density(cv: float) -> float:
    """混合物密度 ρ_m = C_v ρ_s + (1−C_v) ρ_w（kg/m³）。"""
    return float(cv) * RHO_S + (1.0 - float(cv)) * RHO_W


def _tg_kappa_width() -> float:
    """问题2 沟宽经验式的系数（同一真源，避免两问各写一个数）。"""
    from mmlib import trigger as _tg
    return float(_tg.KAPPA_WIDTH)


def _tg_kappa_q() -> float:
    """问题2 起动流量中的 κ（全沟宽 → 有效过流宽度）。"""
    from mmlib import trigger as _tg
    return float(_tg.KAPPA_Q)


def build_sens(v0_base: float, b0_base: float, v_upper: float | None = None,
               mu_base: float | None = None) -> dict:
    """按派生基准值展开敏感性水平，并保证每个参数的扫描序列包含基准值。

    体积情景族（§4.5.4）：链A 基准 ±50%（κ_rel 与 γ 只以乘积 κ·γ 进入
    V_df，故体积水平扫描即覆盖二者联合效应）+ 链B 上界情景。
    μ 水平锚定在采用值 μb 上（0.7×/0.85×/1×/1.2×）：覆盖"更流动→运程
    触顶（前锋抵达中心线末端）"与"更阻滞→未到达口岸"两个方向，
    不再混入与基准无关的孤立数值（旧水平 0.147 并不等于采用值 μb，
    "扫描序列包含基准值"的约定即由此破坏）。
    """
    out = {k: list(v) for k, v in SENS.items()}
    if mu_base is not None:
        out["mu"] = sorted({round(0.7 * mu_base, 4), round(0.85 * mu_base, 4),
                            float(mu_base), round(1.2 * mu_base, 4)})
    lv = sorted({0.5 * v0_base, v0_base, 1.5 * v0_base})
    if v_upper is not None:
        lv = sorted(set(lv) | {float(v_upper)})
    out["V0_m3"] = lv
    out["B0_m"] = sorted({0.5 * b0_base, b0_base, 2.0 * b0_base})
    return out


def _ser(x):
    if isinstance(x, dict):
        return {str(k): _ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ser(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return _ser(x.tolist())
    return x


def save_json(obj, name: str):
    out = OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_ser(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


# ================================================================ 中心线
def channel_centerline(dx: float = 100.0) -> dict:
    """源区至口岸的主沟中心线，等距重采样。

    高程处理：D8 追迹路径逐像元取高程时会出现大量伪"上升步"
    （本数据集实测 139 步，随地形缓存复核并写入 04_中心线校验.json 的
    "原始纵剖面上升步数"），成因是 D8 在近等高线上横向摆动而非真正反坡。
    直接使用会使 Heim 判据的累计比能被虚假上升步不断扣减。
    故对纵剖面做**单调递减回归**（isotonic regression），在最小二乘
    意义下得到物理一致的单调纵剖面。
    """
    dem, grid = io.load_dem()
    z = dem.astype("float64")
    # 地形派生量由 terrain_prep 一次性预计算并缓存（fill/d8/acc 与二维引擎共用）
    from mmlib import terrain_prep as _tpp
    tpz = _tpp.ensure_terrain_cache()
    zf = tpz["filled"].astype("float64")
    d8 = tpz["dir"]
    acc = tpz["acc"]
    cell_x = abs(grid["dlon"]) * 111320.0 * np.cos(np.radians(28.4))
    cell_y = abs(grid["dlat"]) * 110574.0

    gn = io.load_glacier_nc()
    from mmlib import geodesy as gd
    gt = gn["geotransform"]
    mask = (np.isfinite(gn["thickness"]) &
            (gn["thickness"] >= PARAMS["h_thr_m"]) &
            np.isfinite(gn["slope"]) & (gn["slope"] >= PARAMS["sl_thr_deg"]))
    gy_, gx_ = np.nonzero(mask)
    pts = []
    for row, col in zip(gy_, gx_):
        lo, la = gd.utm_to_lonlat(gt[0] + (col + .5) * gt[1],
                                 gt[3] + (row + .5) * gt[5], 45)
        # Geographic metadata uses the upper-left pixel center.
        rr = int(np.floor((la - grid["lat0"]) / grid["dlat"] + .5))
        cc = int(np.floor((lo - grid["lon0"]) / grid["dlon"] + .5))
        if 0 <= rr < z.shape[0] and 0 <= cc < z.shape[1] and np.isfinite(z[rr, cc]):
            pts.append((rr, cc))
    if not pts:
        raise ValueError("chain A source does not intersect DEM")
    start = max(set(pts), key=lambda q: acc[q])
    path = tr.trace_downstream(d8, start)
    rp, cp = io.lonlat_to_rowcol(grid, io.EVENT["port_lon"], io.EVENT["port_lat"])
    distance = np.hypot((np.array([q[0] for q in path]) - rp) * cell_y,
                        (np.array([q[1] for q in path]) - cp) * cell_x)
    ip = int(np.argmin(distance))
    path = path[:ip + 1]
    if len(path) < 3:
        raise ValueError("common source-to-port path too short")
    z_path = np.array([z[p] for p in path])
    z_fill_path = np.array([zf[p] for p in path])
    from sklearn.isotonic import IsotonicRegression
    idx = np.arange(len(z_path), dtype="float64")
    z_iso = IsotonicRegression(increasing=False, out_of_bounds="clip").fit_transform(idx, z_path)

    xy = np.array([[p[1] * cell_x, p[0] * cell_y] for p in path], dtype="float64")
    seg = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(s[-1] / dx) + 1, 20)
    s_new = np.linspace(0.0, s[-1], n)
    sx = np.interp(s_new, s, xy[:, 0])
    sy = np.interp(s_new, s, xy[:, 1])
    zz = np.interp(s_new, s, z_iso)
    zz_fill = np.interp(s_new, s, z_fill_path)
    return {
        "x_m": sx, "y_m": sy, "s_m": s_new, "z_m": zz,
        "z_filled_m": zz_fill, "z_raw_m": np.interp(s_new, s, z_path),
        "length_km": float(s[-1] / 1000), "n": n,
        "start_rc": [int(v) for v in path[0]], "end_rc": [int(v) for v in path[-1]],
        "start_lonlat": [float(grid["lon0"]) + path[0][1] * float(grid["dlon"]),
                         float(grid["lat0"]) + path[0][0] * float(grid["dlat"])],
        "end_lonlat": [float(grid["lon0"]) + path[-1][1] * float(grid["dlon"]),
                       float(grid["lat0"]) + path[-1][0] * float(grid["dlat"])],
        "port_offset_m": float(distance[ip]),
        "injection_z_raw_m": float(z_path[0]),
        "injection_z_filled_m": float(z_fill_path[0]),
        "source_mapping": {"native_cells": int(mask.sum()), "inside_instances": len(pts),
                           "unique_cells": len(set(pts))},
        "clock": "common_chain_A_injection_to_port",
        "path_rc": np.array(path, dtype=int),
        "原始纵剖面上升步数": int((np.diff(z_path) > 0).sum()),
        "单调回归调整量_m": {"mean": float(np.mean(z_iso - z_path)),
                        "max": float(np.max(np.abs(z_iso - z_path)))},
        "与填洼高程差_m": {"mean": float(np.mean(zz - zz_fill)),
                       "max": float(np.max(np.abs(zz - zz_fill)))},
        "高程范围_m": [float(zz[-1]), float(zz[0])],
    }


def source_geometry(gn, mask):
    from mmlib import geodesy as gd
    rr, cc = np.nonzero(mask)
    gt = gn["geotransform"]
    coords = [gd.utm_to_lonlat(gt[0] + (c+.5)*gt[1], gt[3] + (r+.5)*gt[5], 45)
              for r, c in zip(rr, cc)]
    z = np.asarray(gn["dem"])[mask]
    hi = np.asarray(gn["thickness"])[mask]
    good = np.isfinite(z) & np.isfinite(hi)
    return {"centroid_lonlat": np.mean(coords, axis=0).tolist(),
            "z_dem_p90_m": float(np.percentile(z[good], 90)),
            "z_dem_plus_ice_p90_m": float(np.percentile((z+hi)[good], 90)),
            "release_ice_m3": float(np.sum(hi) * abs(gt[1]*gt[5]) * PARAMS["kappa_rel"]),
            "height_semantics": "two hypotheses; equal weight only as declared engineering prior"}


def _jonckheere_terpstra(groups: list[np.ndarray]) -> dict:
    """Jonckheere–Terpstra 趋势检验（正态近似，含并列校正）。

    Kruskal–Wallis 只能回答"组间有差异"，不能区分趋势方向；JT 检验
    组间均值随**预设顺序**单调。J = Σ_{i<j}[#(x_i>x_j) + 0.5·#(x_i=x_j)]，
    E[J] = (N²−Σn²)/4，Var = [N²(2N+3) − Σn²(2n+3) − Σt(t²−1)]/72
    （t 为合并样本中并列组的大小）。J 偏大 ⇔ 后面的组系统性地更小。
    """
    from scipy import stats as _st
    n = np.array([len(g) for g in groups], dtype=float)
    x = np.concatenate(groups)
    N = float(x.size)
    J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            a = np.asarray(groups[i])[:, None]
            b = np.asarray(groups[j])[None, :]
            J += float((a > b).sum() + 0.5 * (a == b).sum())
    EJ = (N ** 2 - float(np.sum(n ** 2))) / 4.0
    _, cnt = np.unique(x, return_counts=True)
    t = cnt.astype(float)
    var = (N ** 2 * (2 * N + 3) - float(np.sum(n ** 2 * (2 * n + 3)))
           - float(np.sum(t ** 3 - t))) / 72.0
    se = float(np.sqrt(max(var, 0.0)))
    z = (J - EJ) / se if se > 0 else 0.0
    return {"J": float(J), "E_J": float(EJ), "SE": se, "z": float(z),
            "p_单侧_递减": float(_st.norm.sf(z))}


# ================================================================ μ 反演
def invert_mu() -> dict:
    """由附件（6）反演等效摩擦系数 μ = Δz / (L_channel + L_fan)。"""
    hp = io.load_hazard_points()
    hp = hp.copy()
    hp["mu"] = hp["relief_m"] / (hp["length_m"] + hp["fan_l"])
    hp["mu_ch"] = hp["relief_m"] / hp["length_m"]
    v = hp["mu"].replace([np.inf, -np.inf], np.nan).dropna()
    out = {
        "样本数": int(len(v)),
        "分位数": {f"p{q}": float(np.percentile(v, q)) for q in (5, 10, 25, 50, 75, 90, 95)},
        "均值": float(v.mean()), "标准差": float(v.std()),
        "按规模": {str(k): {"n": int(len(g)), "mu中位数": float(g["mu"].median())}
                for k, g in hp.groupby("规模")},
        "按类型": {str(k): {"n": int(len(g)), "mu中位数": float(g["mu"].median())}
                for k, g in hp.groupby("类型")},
        "统计检验_规模与mu": {},
    }
    # Kruskal–Wallis（组间差异）+ Jonckheere–Terpstra（有序趋势）：
    # 采用条件 μ（大+特大）的前提是"μ 随规模等级单调递减"，KW 检验不了
    # 方向，JT 补上。组顺序显式固定为 小→中→大→特大——不能用 groupby
    # 的字典序（其顺序由中文码点决定，与规模等级无关）。
    from scipy import stats
    scale_order = [k for k in ("小", "中", "大", "特大")
                   if (hp["规模"] == k).sum() >= 2]
    groups = [hp.loc[hp["规模"] == k, "mu"].dropna().values
              for k in scale_order]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
        out["统计检验_规模与mu"] = {"H": float(H), "p": float(p),
                                "结论": "μ 随规模等级存在显著差异" if p < 0.05
                                else "μ 与规模等级无显著差异"}
    if len(groups) >= 3:
        jt = _jonckheere_terpstra(groups)
        jt.update({"顺序": "小→中→大→特大",
                   "方向": "J 偏大表示 μ 随规模等级递减（条件 μ 的采用前提）",
                   "结论": ("μ 随规模等级显著递减（p<0.05），条件 μ（大+特大）"
                        "取低于全样本中位数的值有统计依据"
                        if jt["p_单侧_递减"] < 0.05 else
                        "未检出单调递减趋势，条件 μ 的采用依据减弱（登记）")})
        out["趋势检验_JonckheereTerpstra"] = jt
    shift = {str(k): float(g["mu"].median() - v.median()) for k, g in hp.groupby("规模")}
    out["规模相对总体中位数的偏移"] = shift
    return out


# ================================================================ 主仿真
def run_once(params: dict, cl: dict | None = None, verbose: bool = True) -> dict:
    if cl is None:
        cl = channel_centerline(params["dx_m"])
    if "状态" in cl:
        return cl
    r = hydro.solve_flow(cl["s_m"], cl["z_m"], params)
    dem, grid = io.load_dem()
    cell_area = (abs(grid["dlon"]) * 111320.0 * np.cos(np.radians(28.4))) * \
                (abs(grid["dlat"]) * 110574.0)
    r["中心线"] = cl
    r["input_parameters"] = dict(params)

    # 强度指标：单位宽度冲击力 I = ρ_m·h·v²（N·m⁻¹），ρ_m 由 C_v 给出。
    # hydro.solve_flow 内部只算几何量 h·v²；密度在这里进入，使 C_v 成为
    # 真正进入方程的参数而不是装饰性定义。
    cv = float(params["Cv"])
    rho = mixture_density(cv)
    wet_mask = r["h_m"] > 0.05
    # 单位宽度冲击力 I = ρ_m·h·v²（N/m = J/m²），是泥石流强度分级的常用量。
    # 量纲：kg·m⁻³ × m × m²·s⁻² = kg·s⁻² = N/m。注意它不是压强（Pa）。
    r["intensity"] = rho * r["h_m"] * r["v_m_s"] ** 2
    r["Cv"] = cv
    r["rho_m"] = rho
    r["I_max"] = float(np.max(r["intensity"]))
    r["I_median"] = float(np.median(r["intensity"][wet_mask])) if wet_mask.any() else 0.0

    # 强度场：沿中心线走廊带，每个像元取最近节点的强度。
    # 走廊半宽 = 该断面沟宽的一半（B/2），这样强度分布反映真实的
    # 沟道宽度变化，而不是人为设定的固定带宽。
    h, w = dem.shape
    inten_field = np.zeros((h, w), dtype="float32")
    for i in range(len(cl["s_m"])):
        if r["v_m_s"][i] <= 0:
            continue
        frac = i / max(len(cl["s_m"]) - 1, 1)
        rr = int(np.floor(cl["y_m"][i] / (abs(grid["dlat"]) * 110574.0)+.5))
        cc = int(np.floor(cl["x_m"][i] / (abs(grid["dlon"]) * 111320.0 * np.cos(np.radians(28.4)))+.5))
        # 半宽（像元数）= 沟宽一半 / 像元尺寸
        bw = max(1, int(round(0.5 * r["B_m"][i] / 30.0)))
        r0_, r1_ = max(0, rr - bw), min(h, rr + bw + 1)
        c0_, c1_ = max(0, cc - bw), min(w, cc + bw + 1)
        inten_field[r0_:r1_, c0_:c1_] = np.maximum(
            inten_field[r0_:r1_, c0_:c1_], float(r["intensity"][i]))
    r["强度场"] = inten_field
    r["强度分级_走廊"] = hydro.intensity_classes(
        inten_field, inten_field > 0.05, cell_area, edges=I_CLASS_EDGES)
    r["强度分级_中心线"] = hydro.intensity_classes(
        r["intensity"], wet_mask, cell_area, edges=I_CLASS_EDGES)
    r["走廊面积_km2"] = float((inten_field > 0).sum() * cell_area / 1e6)

    # 截断诊断：触顶节点数决定流深/流速峰值是否由模型输出还是人为上限决定
    r["触顶节点数"] = {"流深": int((r["h_m"] >= float(params.get("h_cap", 15.0))).sum()),
                   "流速": int((r["v_m_s"] >= float(params.get("v_cap", 30.0))).sum())}

    if verbose:
        print(f"  起始 {r['s_km'][np.where(r['v_m_s'] > 0)[0][0]]:.2f} km，"
              f"停止 {r['runout_km']:.2f} km（总长 {r['total_km']:.2f} km）")
        print(f"  流速 中位 {r['v_median_m_s']:.1f} / 最大 {r['v_max_m_s']:.1f} m/s，"
              f"流深 中位 {r['h_median_m']:.2f} m，Fr 中位 {r['Fr_median']:.2f}")
        print(f"  到达口岸 {r['arrival_outlet_s']} s "
              f"到达秒={r['arrival_outlet_s']}，到达={r['reached_outlet']}")
    return r


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)

    print("[1/6] 中心线提取…")
    cl = channel_centerline(PARAMS["dx_m"])
    print(f"  中心线 {cl['length_km']:.2f} km，{cl['n']} 节点，"
          f"高程 {cl['z_m'][0]:.0f}→{cl['z_m'][-1]:.0f} m")
    save_json({"中心线长度_km": cl["length_km"], "节点数": cl["n"],
               "起点": cl["start_lonlat"], "终点": cl["end_lonlat"],
               "原始纵剖面上升步数": cl["原始纵剖面上升步数"],
               "单调回归调整量_m": cl["单调回归调整量_m"],
               "与填洼高程差_m": cl["与填洼高程差_m"],
               "高程范围_m": cl["高程范围_m"]}, "04_中心线校验.json")

    print("[2/6] 等效摩擦系数反演（附件6，84 条历史沟道）…")
    mu_inv = invert_mu()
    print(f"  μ 中位数 {mu_inv['分位数']['p50']:.3f}（p25 {mu_inv['分位数']['p25']:.3f}，"
          f"p75 {mu_inv['分位数']['p75']:.3f}）")
    # 条件 μ：本沟为**大型链式事件**（冰崩—碎屑流—泥石流—山洪），
    # 故取"大 + 特大"规模沟道 μ 的中位数作为基准，而非全样本中位数。
    # 依据：μ 与规模等级显著负相关（事件越大、越持续、流动性越强），
    # 全样本中位数会把本事件按"中型"处理，导致运移距离被系统性低估。
    hp = io.load_hazard_points().copy()
    hp["mu"] = hp["relief_m"] / (hp["length_m"] + hp["fan_l"])
    big = hp[hp["规模"].isin(["大", "特大"])]["mu"].dropna()
    mu_cond = float(np.median(big))
    PARAMS["mu"] = mu_cond
    print(f"  条件 μ（大+特大，n={len(big)}）中位数 {mu_cond:.3f}")
    mu_inv["条件μ_大特大"] = {"n": int(len(big)), "median": mu_cond,
                          "p25": float(np.percentile(big, 25)),
                          "p75": float(np.percentile(big, 75)),
                          "采用值": mu_cond,
                          "理由": "大型链式事件的条件场景选择，非独立率定", "samples": big.to_list()}
    save_json(mu_inv, "04_摩擦系数反演.json")

    print("[3/6] 初始释放体积（附件4 冰厚场，双链情景族 §4.5.4）…")
    gn = io.load_glacier_nc()
    ice = gn["thickness"]
    thick = np.where(np.isfinite(ice), ice, 0.0)
    cell_ice = abs(gn["geotransform"][1] * gn["geotransform"][5])
    # ---- 链A（冰厚-坡度筛选链，基准链）----
    # Chain A geometry on the native glacier grid; slope is mandatory.
    slope_g = gn["slope"]
    srcA = (thick >= PARAMS["h_thr_m"]) & np.isfinite(slope_g) & (slope_g >= PARAMS["sl_thr_deg"])
    V_src_ice = float(np.sum(thick[srcA]) * cell_ice)
    rho_df = mixture_density(PARAMS["Cv"])           # ρ_df = Cv·ρ_s+(1−Cv)·ρ_w
    V_ice_A = PARAMS["kappa_rel"] * V_src_ice
    V_df_A = PARAMS["gamma_enrich"] * (PARAMS["rho_ice"] / rho_df) * V_ice_A
    # ---- 链B（可动比例链，大体量上界情景）----
    thr = float(np.nanpercentile(thick[thick > 0], 90))
    # 冰崩释放区：冰厚 ≥90% 分位的像元（冰崩只发生在冰体最厚、坡度最陡处）
    rel_mask = thick >= thr
    A_rel = float(rel_mask.sum() * cell_ice) / 1e6
    h_ice = float(np.mean(thick[rel_mask])) if rel_mask.any() else 0.0
    V_ice = A_rel * 1e6 * h_ice                       # = V_b（冰崩体量）
    # 进入沟道的**可动**体积：冰崩体并非全部转化为泥石流。按链式过程
    # 的实际观测，冰崩体量中约 10%–20% 参与形成碎屑流，其余在崩落区
    # 停积或融化。取 mobilizable_fraction 作为显式参数。
    frac_mob = float(PARAMS["mobilizable_fraction"])
    V0_B = V_ice * frac_mob
    # 基准取链A（§4.5.4 报告规则）：链B 作为上界情景进入系综。
    # V0 直接采用派生值，不引入任何与派生链无关的硬常数。
    PARAMS["V0_m3"] = V_df_A
    # 源区合理性旁证（D12）：崩前表面速度场与链A 源区的空间重叠。
    # 速度场不进入运动方程（崩前状态，不是崩后初值），仅作源区选取旁证。
    v_amp = gn.get("v_amp", gn.get("v"))
    d12 = {"可用": False}
    if v_amp is not None:
        vsrc = np.nan_to_num(np.where(np.isfinite(v_amp), v_amp, 0.0), nan=0.0)
        if vsrc.sum() > 0:
            d12 = {
                "可用": True,
                "速度加权占比": float(vsrc[srcA].sum() / vsrc.sum()),
                "口径_速度加权占比": "源区像元的崩前速度 / 全冰川速度总和（速度加权）",
                "P90高速区命中率": float(
                    (srcA & (vsrc >= np.percentile(vsrc[vsrc > 0], 90))).sum()
                    / max((vsrc >= np.percentile(vsrc[vsrc > 0], 90)).sum(), 1)),
                "口径_P90": "速度 P90 以上像元中落入链A 源区的比例",
                "说明": "崩前速度仅为对照；低重叠不支持既定源区，不能称通过验证",
                "源区面积占比": float(srcA.sum()/max(np.isfinite(ice).sum(),1)),
            }
    # 初始沟宽与问题2 起动判据同一口径（B = 9√A），避免同一沟道两处取不同宽度。
    # 集水区面积由本脚本直接算（不读 02_集水区特征.json：该文件由 step02 产出，
    # 而 step02 又依赖本脚本的中心线口径，读它会形成循环依赖）。
    from mmlib import trigger as _tg
    A_catch = _tg.catchment_glacier_fraction()
    save_json(A_catch, "02_集水区特征.json")
    # 静态特征（坡度/土地覆盖/可蚀性）在同一集水区掩膜上一并产出：
    # B1 成果携带 02_静态特征.json 却没有生成代码，step02 冷启动即 KeyError。
    save_json(_tg.catchment_static_features(A_catch), "02_静态特征.json")
    _W_full = float(_tg_kappa_width() * np.sqrt(A_catch["集水区面积_km2"]))
    PARAMS["B0_m"] = _W_full
    # 问题3 的路径 A 需要标定后的起动流量；此处把中心线比降口径一并写入，
    # 使"q_c 用哪条比降"这件事只有一处定义。
    print(f"  链A：源区 {srcA.sum()} 像元（冰厚≥{PARAMS['h_thr_m']:.0f} m 且坡度≥"
          f"{PARAMS['sl_thr_deg']:.0f}°），V_src={V_src_ice:.3e} m³，"
          f"ρ_df={rho_df:.0f} kg/m³ → V_df={V_df_A:.3e} m³（基准）")
    print(f"  链B：冰厚 p90={thr:.1f} m，释放区 {A_rel:.2f} km²，平均冰厚 {h_ice:.1f} m，"
          f"冰崩体量 {V_ice:.2e} m³ × 可动比例 {frac_mob:.2f} = {V0_B:.3e} m³（上界情景）")
    if d12["可用"]:
        print(f"  D12 旁证：速度加权占比 {d12['速度加权占比']:.3f}，"
              f"P90 高速区命中率 {d12['P90高速区命中率']:.3f}")
    print(f"  初始沟宽 B0 = {PARAMS['B0_m']:.1f} m（=9√A；泥石流浆体充满断面，不除 κ）")
    save_json({"冰厚p90_m": thr, "释放区面积_km2": A_rel, "平均冰厚_m": h_ice,
               "冰崩体量_m3": V_ice,
               "可动比例": frac_mob,
               "链B_V0_m3": V0_B,
               "链A": {"h_thr_m": PARAMS["h_thr_m"], "sl_thr_deg": PARAMS["sl_thr_deg"],
                     "源区像元数": int(srcA.sum()),
                     "源区面积_km2": float(srcA.sum() * cell_ice) / 1e6,
                     "V_src_ice_m3": V_src_ice,
                     "kappa_rel": PARAMS["kappa_rel"],
                     "gamma": PARAMS["gamma_enrich"],
                     "rho_ice": PARAMS["rho_ice"], "rho_df": rho_df,
                     "V_ice_A_m3": V_ice_A, "V_df_A_m3": V_df_A},
               "链A_V_df_m3": V_df_A,
               "采用_V0_m3": PARAMS["V0_m3"],
               "体积情景水平_m3": sorted({0.5 * V_df_A, V_df_A, 1.5 * V_df_A, V0_B}),
               "双链差异比": float(V0_B / V_df_A),
               "源区旁证_D12": d12,
               "共同断面": {"注入_rc": cl["start_rc"], "口岸_rc": cl["end_rc"],
                            "注入_lonlat": cl["start_lonlat"], "口岸_lonlat": cl["end_lonlat"],
                            "注入高程_raw_m": cl["injection_z_raw_m"],
                            "注入高程_filled_m": cl["injection_z_filled_m"],
                            "口岸偏移_m": cl["port_offset_m"]},
               "链A源几何": source_geometry(gn, srcA),
               "截断事件": False,
               "初始沟宽_B0_m": PARAMS["B0_m"],
               "说明": ("基准 V0 取链A（冰厚-坡度筛选链）：V_df = γ·(ρ_ice/ρ_df)·κ_rel·"
                       "Σ_src(h_ice·a)，源区为附件4 原生网格上冰厚≥h_thr 且坡度≥sl_thr "
                       "的格网集合。链B（P90 释放区 × 可动比例）作为本情景族大体量对照"
                       "进入场景集合，不是现实严格上界；两链差异是初始条件假设差异。"
                       "κ_rel 与 γ 仅以乘积进入 V_df（框架性质，登记）。"
                       "本文件是一维诊断/D8网络两引擎体积的唯一真源。"
                       "初始沟宽 B0 与问题2 起动判据同口径 B=9√A(km²)，"
                       "使两问的沟道宽度不再是两个互不相干的数。")},
              "04_初始体积.json")

    print("[4/6] 基线仿真…")
    base = run_once(dict(PARAMS), cl=cl)
    payload = {k: v for k, v in base.items() if k not in ("中心线", "强度场")}
    payload["中心线"] = cl
    payload["params"] = dict(PARAMS)
    save_json(payload, "04_仿真基线.json")

    sens = build_sens(PARAMS["V0_m3"], PARAMS["B0_m"], v_upper=V0_B,
                      mu_base=PARAMS["mu"])
    print(f"[5/6] 单参数敏感性（{len(sens)} 参数，含 Cv 三水平）…")
    rows = {}
    n_dup_baseline = 0
    for key, vals in sens.items():
        rows[key] = []
        for v in vals:
            p = dict(PARAMS)
            if key == "mu_prop":
                p["mu"] = PARAMS["mu"] * float(v)
            else:
                p[key] = v
            r = run_once(p, cl=cl, verbose=False)
            capped = int(r["触顶节点数"]["流深"] + r["触顶节点数"]["流速"])
            reached = bool(r["reached_outlet"])
            rows[key].append({
                "取值": v,
                "runout_km": r["runout_km"],
                "reached_outlet": reached,
                # 未到达情境记 None：不外推、不以 0 计入任何统计。
                # hydro.solve_flow 的 arrival_outlet_s=t[stop_idx] 在未到达时
                # 是运动学外推值（非到达时刻），下游统计（step03 系综、
                # 04_结果区间）一律按 reached_outlet 过滤。
                "到达口岸_min": (round(r["arrival_outlet_s"] / 60, 4)
                                if reached else None),
                "v_median_m_s": r["v_median_m_s"],
                "v_max_m_s": r["v_max_m_s"],
                "h_median_m": r["h_median_m"],
                "I_max": r["I_max"],
                "I_median": r["I_median"],
                "Fr_median": r["Fr_median"],
                "波列长度_m": r["波列长度_m"],
                "液型": r["液型"],
                "触顶节点数": capped,
                "params": dict(p),
                "termination_reason": r["termination_reason"],
                "continuity_relative_residual_max": r["continuity_relative_residual_max"],
                # 运程触顶：前锋抵达中心线末端（runout ≥ 0.999×total），
                # Heim 判据未在域内自然停止。与"触顶节点数"（h/v 触发人为
                # 上限的节点数）是两类事件，分开登记、分开解读。
                "运程触顶": bool(r["runout_km"] >= 0.999 * r["total_km"]),
            })
            tags = ((f"（触顶 {capped}）" if capped else "")
                    + ("［运程触顶］" if r["runout_km"] >= 0.999 * r["total_km"] else "")
                    + ("［未到达口岸］" if not reached else ""))
            print(f"  {key}={v:g}: runout {r['runout_km']:.2f} km，"
                  f"v_med {r['v_median_m_s']:.1f} m/s，"
                  f"I_max {r['I_max']/1e3:.0f} kN/m{tags}")
    # 独立情境数：1 + Σ(n_i − 1)，避免把同一基线重复计数（各序列均含基准值）
    unique = {json.dumps(dict(PARAMS), sort_keys=True): {
        "id": "base", "params": dict(PARAMS), "reached_outlet": bool(base["reached_outlet"]),
        "arrival_outlet_s": base["arrival_outlet_s"], "runout_km": base["runout_km"],
        "termination_reason": base["termination_reason"],
        "到达口岸_min": base["arrival_outlet_s"]/60 if base["arrival_outlet_s"] is not None else None,
        **{k:base[k] for k in ("v_median_m_s","v_max_m_s","h_median_m","I_max","I_median")}}}
    for key, group in rows.items():
        for j, record in enumerate(group):
            signature = json.dumps(record["params"], sort_keys=True)
            unique.setdefault(signature, {"id": f"{key}-{j}", **record,
                              "arrival_outlet_s": (record["到达口岸_min"] * 60
                                                   if record["到达口岸_min"] is not None else None)})
    n_scenarios = len(unique)
    unique_runs = list(unique.values())
    n_dup_baseline = sum(len(v) for v in sens.values()) - n_scenarios
    save_json({"基准参数": PARAMS, "扰动设置": sens, "敏感性结果": rows,
               "unique_runs": unique_runs,
               "敏感性情境总数_独立": n_scenarios,
               "敏感性情境总数_含重复基线": sum(len(v) for v in sens.values()),
               "重复基线计数": n_dup_baseline,
               "情境计数口径": "按完整参数字典去重；扫描表可含重复基线",
               "μ水平锚定": (f"μ 水平 = {{0.7×, 0.85×, 1×, 1.2×}}·μb（μb="
                          f"{PARAMS['mu']:.6f}，采用值：大+特大事件的条件中位数）；"
                          "到达口径：未到达情境记 None"),
               "触顶情境": {k: [r["取值"] for r in rows[k] if r["运程触顶"]]
                         for k in rows if any(r["运程触顶"] for r in rows[k])},
               "触顶口径": ("运程触顶：runout_km ≥ 0.999×total_km（前锋抵达"
                        "中心线末端/口岸，Heim 判据未自然停止）；"
                        "与 h/v 截断的触顶节点数分开登记"),
               "未到达情境": {k: [r["取值"] for r in rows[k]
                           if not r["reached_outlet"]]
                          for k in rows
                          if any(not r["reached_outlet"] for r in rows[k])},
               "未到达口径": "reached_outlet=False 的情境；其到达时间不进入任何统计"},
              "04_敏感性.json")

    # 结果区间
    def spread(sub):
        vs = [r[sub] for r in unique_runs if r.get(sub) is not None]
        return {"min": float(np.min(vs)), "p50": float(np.median(vs)),
                "max": float(np.max(vs))} if vs else {}

    n_noreach = sum(not r["reached_outlet"] for r in unique_runs)
    n_reach_arr = sum(r["reached_outlet"] for r in unique_runs)
    save_json({
        "runout_km": spread("runout_km"),
        "到达口岸_min": spread("到达口岸_min"),
        "到达口岸_min_口径": (f"仅统计到达情境（reached_outlet=True，"
                       f"n={n_reach_arr}）；未到达情境（{n_noreach} 个）"
                       "记 None，不以 0 或运动学外推值计入"),
        "未到达情境数": n_noreach,
        "v_median_m_s": spread("v_median_m_s"),
        "v_max_m_s": spread("v_max_m_s"),
        "h_median_m": spread("h_median_m"),
        "I_max": spread("I_max"),
        "I_median": spread("I_median"),
        "基线": {"runout_km": base["runout_km"],
                "到达口岸_min": base["arrival_outlet_s"]/60 if base["arrival_outlet_s"] is not None else None,
                "v_median_m_s": base["v_median_m_s"],
                "v_max_m_s": base["v_max_m_s"],
                "h_median_m": base["h_median_m"],
                "I_max": base["I_max"],
                "I_median": base["I_median"]},
        "敏感性情境总数_独立": n_scenarios,
        "敏感性情境总数_含重复基线": sum(len(v) for v in sens.values()),
    }, "04_结果区间.json")

    print("[6/6] 场量导出与图件…")
    dem, grid = io.load_dem()
    cell_area = abs(grid["dlon"]) * 111320.0 * np.cos(np.radians(28.4)) * \
                abs(grid["dlat"]) * 110574.0
    inten_field = base["强度场"]
    np.savez_compressed(CACHE / "sim_fields.npz",
                        intensity=inten_field,
                        s_km=cl["s_m"] / 1000.0, z_m=cl["z_m"],
                        h_m=base["h_m"], v_m_s=base["v_m_s"],
                        t_s=np.nan_to_num(base["t_s"], nan=-1.0),
                        I=base["intensity"])
    save_json({"走廊像元数": int((inten_field > 0).sum()),
               "走廊面积_km2": base["走廊面积_km2"],
               "最大强度_N_m": float(np.max(inten_field)),
               "Cv": base["Cv"], "rho_m": base["rho_m"],
               "强度指标": "I = ρ_m·h·v²（单位宽度冲击力，N·m⁻¹）",
               "分级（走廊内）": base["强度分级_走廊"],
               "分级（中心线断面）": base["强度分级_中心线"]},
              "04_强度场.json")

    print("完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
