"""问题 4 的图件：仿真环境、过程、结果与敏感性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmlib import io_data as io, hydro  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
# SimHei 缺少上标数字（U+00B2/U+00B3）与上标减号（U+207B），若沿用默认
# mathtext 字体集，$hv^2$ 一类公式会被渲染成方框。改用 DejaVu Sans 渲染
# 数学、中文仍走 SimHei，二者混排正常。
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["font.size"] = 20
plt.rcParams["axes.titlesize"] = 19
plt.rcParams["axes.labelsize"] = 17
plt.rcParams["xtick.labelsize"] = 15
plt.rcParams["ytick.labelsize"] = 15
plt.rcParams["legend.fontsize"] = 13

FIG = io.FIG_OUT
OUT = io.DATA_OUT
CACHE = io.CACHE
C_MAIN = "#1f4e79"
C_WARN = "#c00000"
C_FILL = "#2e75b6"
C_OK = "#2f8f46"        # 原代码在液型比 ≥0.6 的分支引用了未定义的 C_OK，
                        # 因该分支长期未被执行而潜伏；基准体量改为派生值后暴露。


def _load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def fig_profile(base):
    """中心线纵剖面 + 流速/流深/强度包络。"""
    s = np.asarray(base["s_km"])
    z = np.asarray(base["z_m"])
    v = np.asarray(base["v_m_s"])
    h = np.asarray(base["h_m"])
    I = np.asarray(base["intensity"]) / 1000.0
    wet = v > 0
    z0 = base["z_m"][0]
    mu = base["mu"]

    fig, ax = plt.subplots(4, 1, figsize=(9.2, 9.8), sharex=True,
                           gridspec_kw={"height_ratios": [1.15, 1, 1, 1], "hspace": 0.16})

    ax[0].plot(s, z, color="#333333", lw=1.6, label="纵剖面高程")
    ax[0].fill_between(s, z, z0, color="#e8eef5", zorder=0)
    if wet.any():
        i0 = int(np.where(wet)[0][0])
        e_line = z[i0] + base.get("params", {}).get("E0_m2s2", 0.0)/ (2*9.80665) - mu * (s - s[i0])*1000
        ax[0].plot(s, e_line, "--", color=C_WARN, lw=1.4,
                   label=f"Heim 能量线（μ={mu:.3f}, {base['等效摩擦角_deg']:.1f}°）")
        ax[0].scatter([s[i0]], [z[i0]], s=46, color=C_WARN, zorder=5,
                      label=f"起始点 {s[i0]:.2f} km")
        ax[0].axvline(base["runout_km"], color="#555555", ls=":", lw=1.2,
                      label=f"末端/停止位置 {base['runout_km']:.2f} km")
    ax[0].set_ylabel("高程 / m")
    ax[0].set_ylim(z.min() - 150, z.max() + 200)
    ax[0].legend(loc="upper right", fontsize=14, framealpha=0.93)
    ax[0].set_title("吉隆主沟纵剖面与 Heim 能量线（μ 由附件6 反演）", loc="left")

    ax[1].plot(s[wet], v[wet], color=C_MAIN, lw=1.8)
    ax[1].fill_between(s[wet], 0, v[wet], color=C_MAIN, alpha=0.18)
    ax[1].set_ylabel("流速 / (m·s$^{-1}$)")
    ax[1].axhline(base["v_median_m_s"], color=C_WARN, ls=":", lw=1.1)
    ax[1].text(s[wet][-1] * 0.02, base["v_median_m_s"] * 1.07,
               f"中位 {base['v_median_m_s']:.1f} m/s", color=C_WARN, fontsize=12,
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=0.5))
    ax[1].set_title(f"断面平均流速（峰值 {base['v_max_m_s']:.1f} m/s）", loc="left")

    ax[2].plot(s[wet], h[wet], color="#548235", lw=1.8)
    ax[2].fill_between(s[wet], 0, h[wet], color="#548235", alpha=0.18)
    ax[2].set_ylabel("流深 / m")
    ax[2].set_title(f"断面平均流深（中位 {base['h_median_m']:.2f} m）", loc="left")

    ax[3].semilogy(s[wet], np.maximum(I[wet], 1e-2), color="#7030a0", lw=1.8)
    for thr, lab in [(10.0, "低/中"), (100.0, "中/高"), (1000.0, "高/极高")]:
        ax[3].axhline(thr, color="#999999", ls="--", lw=0.8)
        ax[3].text(s[wet][-1] * 0.02, thr * 1.2, lab, fontsize=12, color="#666666")
    ax[3].set_ylabel("冲击力强度 $I$ / (kN·m$^{-1}$)")
    ax[3].set_xlabel("沿程里程 / km")
    ax[3].set_title(rf"强度指标 $I=\rho_m h v^2$（峰值 {base['I_max']/1e3:.0f} kN/m）", loc="left")
    for a in ax:
        a.grid(alpha=0.25)
        a.set_xlim(0, s[-1])
    fig.savefig(FIG / "04_沿程演进.png", bbox_inches="tight")
    plt.close(fig)


def fig_arrival(base):
    """到达时间—里程曲线。"""
    s = np.asarray(base["s_km"])
    t = np.asarray([x if x is not None else np.nan for x in base["t_s"]])
    wet = np.isfinite(t)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.plot(s[wet], t[wet] / 60, color=C_MAIN, lw=2.0, marker="o", ms=2.2)
    ax.fill_between(s[wet], 0, t[wet] / 60, color=C_MAIN, alpha=0.15)
    t_arr = base["arrival_outlet_s"] or base.get("stop_time_s", float(np.nanmax(t)))
    ax.axhline(t_arr / 60, color=C_WARN, ls="--", lw=1.2)
    ax.annotate(f"{'抵达口岸' if base['reached_outlet'] else '自然停止（未到口岸）'} {t_arr / 60:.1f} min",
                xy=(s[wet][-1], t_arr / 60), xytext=(-130, -22),
                textcoords="offset points", color=C_WARN, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.1))
    ax.set_ylim(-1.5, t_arr / 60 * 1.14)
    ax.set_ylabel("自起动起算时间 / min")
    ax.set_title("泥石流前锋到达时间曲线（Manning 均匀流积分）", loc="left")
    ax.grid(alpha=0.25)
    fig.savefig(FIG / "04_到达时间.png", bbox_inches="tight")
    plt.close(fig)


def fig_intensity_map(base):
    """一维走廊强度图；只有抵达口岸后才允许附加非动力学扇形包络。"""
    dem, grid = io.load_dem()
    inten = np.load(CACHE / "sim_fields.npz")["intensity"]
    half_angle = 25.0
    reach = base["runout_km"]
    I0 = base["I_max"]
    cl = base["中心线"]

    reached = bool(base.get("reached_outlet", False))
    if reached:
        spread = hydro.lateral_spread(
            {"x_m": np.array(cl["x_m"]), "y_m": np.array(cl["y_m"]),
             "s_m": np.array(cl["s_m"]), "end_rc": cl["end_rc"]},
            reach_km=min(6.0, reach * 0.28), half_angle_deg=half_angle,
            I0=I0, dem=dem, grid=grid)
        spread.update(status="geometric_envelope",
                      reason="一维情景抵达口岸；扇形仅为非动力学几何包络，不表示守恒输运或实际堆积场",
                      reached_outlet=True)
    else:
        # 口岸是中心线终点，不是未到达情景的实际扇顶；禁止在此凭空生成扇体。
        spread = {"status": "not_applicable",
                  "reason": "一维基线未抵达口岸，不能以中心线口岸终点生成扇体扩散区",
                  "reached_outlet": False,
                  "intensity": np.zeros_like(dem, dtype="float32"),
                  "affected_cells": 0, "affected_area_km2": 0.0,
                  "max_intensity": 0.0, "I0": 0.0,
                  "apex_rc": None, "main_direction_deg": None,
                  "half_angle_deg": None, "reach_km": 0.0}
    fan = spread["intensity"]

    ext = [grid["lon0"], grid["lon0"] + grid["dlon"] * dem.shape[1],
           grid["lat0"] + grid["dlat"] * dem.shape[0], grid["lat0"]]
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem.astype("float64"), vert_exag=2.0, dx=27.2, dy=30.7)

    fig, ax = plt.subplots(figsize=(10.2, 6.5))
    ax.imshow(hs, extent=ext, cmap="gray", alpha=0.9, origin="upper")
    lv = [0, 1e4, 1e5, 1e6, 1e9]
    cmap = ListedColormap(["#7fb3d5", "#f4d03f", "#e67e22", "#a93226"])
    norm = BoundaryNorm(np.arange(5), cmap.N)
    im = ax.imshow(np.ma.masked_where(inten<=0.05, np.digitize(inten,[1e4,1e5,1e6])+.5), extent=ext,
                   cmap=cmap, norm=norm, alpha=0.9, origin="upper")
    if reached and np.any(fan > 1e4):
        ax.contour(np.linspace(ext[0], ext[1], fan.shape[1]),
                   np.linspace(ext[3], ext[2], fan.shape[0]),
                   np.ma.masked_less_equal(fan, 1e4), levels=4,
                   colors="#00e5ff", linewidths=1.1, alpha=0.9)
    cb = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.74, pad=0.015)
    cb.set_ticks([.5,1.5,2.5,3.5]); cb.set_ticklabels(["低 <10", "中 10-100", "高 100-1000", "极高 ≥1000"])
    cb.set_label(r"单位宽动量通量 $I=\rho_m h v^2$ / (kN/m)", fontsize=9)

    unit_x = abs(grid["dlon"]) * 111320.0 * np.cos(np.radians(28.4))
    unit_y = abs(grid["dlat"]) * 110574.0
    clon = grid["lon0"] + (np.array(cl["x_m"]) / unit_x) * grid["dlon"]
    clat = grid["lat0"] - (np.array(cl["y_m"]) / unit_y) * abs(grid["dlat"])
    ax.plot(clon, clat, color="#00ffff", lw=1.3, label="主沟中心线")
    ax.plot(io.EVENT["port_lon"], io.EVENT["port_lat"], "*", ms=18,
            color="#ff0000", mec="white", mew=1.2, label="吉隆口岸", zorder=10)
    ax.set_xlim(clon.min() - 0.03, clon.max() + 0.03)
    ax.set_ylim(clat.min() - 0.02, clat.max() + 0.02)
    ax.set_xlabel("经度 / °E")
    ax.set_ylabel("纬度 / °N")
    title = (f"一维走廊强度与非动力学扇形包络（青色等值线，半角 {half_angle:.0f}°）"
             if reached else "一维走廊强度分布（基线未达口岸，不生成扇体）")
    ax.set_title(title, loc="left")
    ax.legend(loc="lower left", fontsize=8, framealpha=0.92)
    fig.savefig(FIG / "04_强度分级平面.png", bbox_inches="tight")
    plt.close(fig)
    return spread


def fig_tornado(sens, base):
    """单参数敏感性龙卷风图。"""
    names = {"E0_m2s2": "初始激发尺度 $E_0$", "V0_m3": "初始体积 $V_0$", "n_man": "糙率 $n$", "mu": "摩擦系数 $\\mu$",
             "B0_m": "沟宽 $B_0$", "T_pass_s": "过流历时 $T$",
             "mu_prop": "μ 相对扰动", "Cv": "固体体积分数 $C_v$"}
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.4))
    targets = [("v_median_m_s", "中位流速 / (m·s$^{-1}$)"),
               ("h_median_m", "中位流深 / m"),
               ("I_max", "峰值强度 / (kN·m$^{-1}$)")]
    for ax, (key, lab) in zip(axes, targets):
        rows = []
        for k, lst in sens["敏感性结果"].items():
            if k == "mu_prop":
                continue
            vals = [r[key] for r in lst if r.get(key) is not None]
            if not vals:
                continue
            if key == "I_max":                      # 强度用 kPa 显示
                vals = [v / 1e3 for v in vals]
            rows.append((names.get(k, k), min(vals), max(vals)))
        rows.sort(key=lambda r: r[2] - r[1])
        for i, (nm, lo, hi) in enumerate(rows):
            ax.barh(i, max(hi - lo, 1e-9), left=lo, height=0.55,
                    color=C_FILL, alpha=0.78)
            ax.plot([lo, hi], [i, i], color=C_MAIN, lw=0.8)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
        ax.set_xlabel(lab, fontsize=9)
        ax.grid(alpha=0.25, axis="x")
    fig.suptitle("问题4 关键参数的单参数敏感性（各参数在其取值范围内扫描）",
                 fontsize=11.5, y=1.005)
    fig.tight_layout()
    fig.savefig(FIG / "04_参数敏感性.png", bbox_inches="tight")
    plt.close(fig)


def fig_mu_curve(base):
    """μ—运程响应曲线 + 历史沟道 μ 的规模分组分布。"""
    from step04_simulation import channel_centerline
    cl = channel_centerline(100.0)
    mu_used = float(base["mu"])
    # 用敏感性结果里记录的基准参数，而不是模块默认值——模块默认中 V0 为 None
    # （由 main() 按附件派生），直接取它会把 None 传进求解器而崩溃。
    sens = _load("04_敏感性.json")
    p = dict(sens["基准参数"])
    p["mu"] = mu_used
    mus = np.linspace(0.09, 0.34, 26)
    r = hydro.runout_curve(cl["s_m"], cl["z_m"], mus, p)
    hp = io.load_hazard_points().copy()
    hp["muv"] = hp["relief_m"] / (hp["length_m"] + hp["fan_l"])

    fig, ax = plt.subplots(1, 2, figsize=(11.8, 4.4))
    rr = r["曲线"]
    ax[0].plot([c["mu"] for c in rr], [c["runout_km"] for c in rr],
               color=C_MAIN, lw=2.0, marker="o", ms=3)
    ax[0].axvline(mu_used, color=C_WARN, ls="--", lw=1.3,
                  label=f"采用 $\\mu$={mu_used:.3f}")
    ax[0].axvline((cl["z_m"][0] - cl["z_m"][-1]) / cl["s_m"][-1], color="#7f7f7f",
                  ls="-.", lw=1.1, label="临界 $\\mu_{crit}$=Δz/L")
    ax[0].axhline(cl["length_km"], color="#555555", ls=":", lw=1.2,
                  label=f"沟道全长 {cl['length_km']:.2f} km")
    ax[0].set_xlabel("等效摩擦系数 $\\mu$")
    ax[0].set_ylabel("运移距离 / km")
    ax[0].set_title("$\\mu$ 对运移距离的控制（Heim 能量线）", loc="left")
    ax[0].legend(fontsize=8.0, loc="lower left")
    ax[0].grid(alpha=0.25)

    for k, c in zip(["小", "中", "大", "特大"],
                    ["#7fb3d5", "#f4d03f", "#e67e22", "#a93226"]):
        v = hp[hp["规模"] == k]["muv"].dropna()
        if len(v):
            ax[1].hist(v, bins=12, alpha=0.7, label=f"{k}（n={len(v)}）", color=c)
    ax[1].axvline(mu_used, color=C_WARN, ls="--", lw=1.5,
                  label=f"采用 $\\mu$={mu_used:.3f}")
    ax[1].set_xlabel("附件6 反演 $\\mu$ = Δz /(L沟+L扇)")
    ax[1].set_ylabel("沟道条数")
    ax[1].set_title("历史沟道 $\\mu$ 的规模分组分布", loc="left")
    ax[1].legend(fontsize=8, loc="upper right")
    ax[1].grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIG / "04_运移距离标定.png", bbox_inches="tight")
    plt.close(fig)


def fig_budget(base, inv):
    """体积诊断（波列 vs 满槽）+ 质量构成 + 初始体积估算链。"""
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))

    # (a) 波列长度 vs 运移路程：区分脉冲型与满槽型流动
    Ls = float(base.get("波列长度_m", np.nan)) / 1000.0
    run = float(base["runout_km"])
    S = float(base["沟槽储水体积_m3"]) / 1e6
    V0 = float(base["V0"]) / 1e6
    ratio = float(base.get("波列占路程比", np.nan))
    kind = base.get("液型", "")

    ax[0].barh([1], [run], height=0.34, color="#d9d9d9", edgecolor="#7f7f7f",
               label=f"运移路程 {run:.1f} km")
    ax[0].barh([0], [Ls], height=0.34,
               color=C_WARN if ratio < 0.6 else C_OK,
               label=f"诊断波长 {Ls:.1f} km（比值 {Ls/run*100:.0f}%）")
    ax[0].set_yticks([1, 0])
    ax[0].set_yticklabels(["运移路程", "过流波列"], fontsize=9.5)
    ax[0].set_xlabel("沿程距离 / km")
    ax[0].set_xlim(0, max(run, Ls) * 1.18)
    ax[0].set_ylim(-0.7, 1.7)
    ax[0].set_title(f"(a) {kind}：$V_0$ 决定波列长度，$\\mu$ 决定停止位置",
                    loc="left", fontsize=9.6)
    ax[0].legend(fontsize=8.2, loc="lower right")
    ax[0].grid(alpha=0.22, axis="x")
    ax[0].text(0.5, 0.90,
               f"满槽储水 {S:.2f}×10$^6$ m$^3$ / $V_0$ {V0:.2f}×10$^6$ m$^3$\n"
               f"储量/入口体积 = {S / V0:.2f}；波列/过流段 = {Ls/run*100:.0f}%（非守恒）",
               transform=ax[0].transAxes, ha="center", va="top", fontsize=8.2,
               bbox=dict(boxstyle="round,pad=0.3", fc="#f2f2f2", ec="#999999"))

    # (b) 质量构成
    cv = float(base["Cv"])
    rho_s, rho_w = 2650.0, 1000.0
    rho_m = cv * rho_s + (1 - cv) * rho_w
    m_s = base["V0"] * cv * rho_s / 1e9
    m_w = base["V0"] * (1 - cv) * rho_w / 1e9
    bars = ax[1].bar(["固相质量", "液相质量"], [m_s, m_w],
                     color=["#a93226", "#2e75b6"], alpha=0.88, width=0.48)
    for b, v in zip(bars, [m_s, m_w]):
        ax[1].text(b.get_x() + b.get_width() / 2, v * 1.015, f"{v:.2f}",
                   ha="center", fontsize=9.5, fontweight="bold")
    ax[1].set_ylim(0, max(m_s, m_w) * 1.18)
    ax[1].set_ylabel("质量 / $10^{9}$ kg")
    ax[1].set_title(f"(b) 质量构成（$C_v$={cv}，$\\rho_m$={rho_m:.0f} kg·m$^{{-3}}$）",
                    loc="left")
    ax[1].grid(alpha=0.22, axis="y")

    # (c) V0 估算链（双链：链A 冰厚-坡度筛选为基准，链B P90×可动比例为上界）
    chA = inv["链A"]
    labels_c = ["链A\n源区冰体积", "链A\n×$\\kappa_{rel}$", "链A\n×$\\gamma$·$\\rho$比",
                "链B\n厚冰掩膜总体积", "链B\n×可动比例"]
    vals_c = [chA["V_src_ice_m3"] / 1e6, chA["V_ice_A_m3"] / 1e6,
              chA["V_df_A_m3"] / 1e6, inv["冰崩体量_m3"] / 1e6,
              inv["链B_V0_m3"] / 1e6]
    notes_c = [None,
               f"×{chA['kappa_rel']:.2f}",
               f"×{chA['gamma']:.0f}·{chA['rho_ice']:.0f}/{chA['rho_df']:.0f}",
               None, f"×{inv['可动比例']:.2f}"]
    colors_c = ["#c55a11", "#c55a11", "#2e75b6", "#7f7f7f", "#7f7f7f"]
    xs = list(range(5))
    for x, hv, nt, cc in zip(xs, vals_c, notes_c, colors_c):
        adopted = (x == 2)
        ax[2].bar(x, hv, color=cc, alpha=0.9 if adopted else 0.6, width=0.58,
                  edgecolor="#1a1a1a" if adopted else "none",
                  linewidth=1.4 if adopted else 0)
        ax[2].text(x, hv * 1.03, f"{hv:.1f}", ha="center", fontsize=9,
                   fontweight="bold")
        if nt:
            ax[2].text(x, hv * 0.55, nt, ha="center", fontsize=7,
                       color="white", fontweight="bold")
    ax[2].annotate("采用 $V_0$（链A 基准）", xy=(2, vals_c[2]),
                   xytext=(2.6, max(vals_c) * 0.86), fontsize=8.5,
                   arrowprops=dict(arrowstyle="->", lw=1.0), color="#1a1a1a")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(labels_c, fontsize=8)
    ax[2].set_ylabel("体积 / $10^{6}$ m$^{3}$")
    ax[2].set_title("(c) 初始释放体积的双链估算（链A 基准 / 链B 大体量对照）",
                    loc="left")
    ax[2].grid(alpha=0.22, axis="y")
    ax[2].set_ylim(0, max(vals_c) * 1.22)

    fig.tight_layout()
    fig.savefig(FIG / "04_收支校验.png", bbox_inches="tight")
    plt.close(fig)


def fig_process_stages(base):
    """仿真过程：4 个时刻的前锋位置与过流段。"""
    s = np.asarray(base["s_km"])
    t = np.asarray([x if x is not None else np.nan for x in base["t_s"]])
    z = np.asarray(base["z_m"])
    wet = np.isfinite(t)
    tmax = np.nanmax(t)
    stages = [0.15, 0.35, 0.6, 0.85]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.5))
    for ax, frac in zip(axes.ravel(), stages):
        tend = tmax * frac
        front = s[wet][np.searchsorted(t[wet], tend, side="right") - 1]
        ax.plot(s, z, color="#444444", lw=1.4)
        m = s <= front
        ax.plot(s[m], z[m], color=C_FILL, lw=2.8)
        ax.axvline(front, color=C_WARN, ls="--", lw=1.2)
        ax.scatter([front], [np.interp(front, s, z)], s=74, color=C_WARN, zorder=6)
        ax.set_xlim(0, s[-1])
        ax.set_ylim(z.min() - 200, z.max() + 200)
        ax.set_title(f"$t$ = {tend / 60:.1f} min   前锋 {front:.1f} km", fontsize=10)
        ax.set_xlabel("里程 / km")
        ax.set_ylabel("高程 / m")
        ax.grid(alpha=0.22)
    fig.suptitle("一维准恒定传播时间诊断（非守恒时间推进）", fontsize=11.5, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG / "04_演进过程.png", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    base = _load("04_仿真基线.json")
    sens = _load("04_敏感性.json")
    inv = _load("04_初始体积.json")

    fig_profile(base)
    print("  04_沿程演进.png")
    fig_arrival(base)
    print("  04_到达时间.png")
    spread = fig_intensity_map(base)
    print(f"  04_强度分级平面.png（扇体影响区 {spread['affected_area_km2']:.2f} km²）")
    fig_tornado(sens, base)
    print("  04_参数敏感性.png")
    fig_mu_curve(base)
    print("  04_运移距离标定.png")
    fig_budget(base, inv)
    print("  04_收支校验.png")
    fig_process_stages(base)
    print("  04_演进过程.png")

    (OUT / "04_扇体扩散.json").write_text(
        json.dumps({k: (v if not isinstance(v, np.ndarray) else None)
                    for k, v in spread.items() if k != "intensity"},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
