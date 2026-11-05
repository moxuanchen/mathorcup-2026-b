"""论文图件生成：读取既有结果数据，输出到 建模论文/figures 与 求解结果/图表。

图件目标：一图一结论；少面板；事件锚点明确；不可识别/右删失用灰色或红色标出。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LightSource
from matplotlib.patches import Rectangle

PROJECT = Path(__file__).resolve().parents[2]
DATA = PROJECT / "求解结果" / "数据"
CACHE = DATA / "cache"
FIG = PROJECT / "建模论文" / "figures"
OUTFIG = PROJECT / "求解结果" / "图表"

TEAL = "#1f6f68"
TEAL_L = "#a8c3bd"
AMBER = "#c77c2b"
GRAY = "#9aa3a8"
RED = "#b03a2e"
INK = "#2b2b2b"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "dejavusans",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "axes.edgecolor": "#555555",
    "axes.linewidth": 0.8,
})


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def save(fig, name):
    for d in (FIG, OUTFIG):
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def _grid():
    q = load("04b_二维引擎.json")
    return q["grid"]


def _dem_extent():
    g = _grid()
    dem = np.load(CACHE / "dem.npy")
    ny, nx = dem.shape
    return dem, [g["lon0"], g["lon0"] + nx * g["dlon"],
                 g["lat0"] + ny * g["dlat"], g["lat0"]]


def _hillshade(ax, dem, extent, alpha=0.55):
    ls = LightSource(azdeg=315, altdeg=45)
    hs = ls.hillshade(dem, vert_exag=0.7, dx=30.0, dy=30.0)
    ax.imshow(hs, extent=extent, cmap="gray", alpha=alpha, origin="upper")
    ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("经度 / °E"); ax.set_ylabel("纬度 / °N")


def _rc_to_lonlat(rc, g):
    r, c = rc
    return g["lon0"] + c * g["dlon"], g["lat0"] + r * g["dlat"]


# ---------------------------------------------------------------- Figure 1
def fig_study_area():
    dem, extent = _dem_extent()
    g = _grid()
    q = load("03_空间距离范围.json")
    src = q["source_definition"]; inj = q["injection_definition"]; port = q["port_proxy_definition"]
    alt = load("06_坐标敏感性.json")["targets"][1]["lonlat"]
    arrival = np.load(CACHE / "q4b_arrival.npy")
    wet = np.isfinite(arrival)
    rr, cc = np.nonzero(wet)
    pad = 60
    r0, r1 = max(0, rr.min() - pad), min(dem.shape[0], rr.max() + pad)
    c0, c1 = max(0, cc.min() - pad), min(dem.shape[1], cc.max() + pad)
    route_extent = [min(g["lon0"] + c0 * g["dlon"], src[0]) - 0.008,
                    max(g["lon0"] + c1 * g["dlon"], src[0]) + 0.010,
                    min(g["lat0"] + r1 * g["dlat"], src[1]) - 0.008,
                    max(g["lat0"] + r0 * g["dlat"], src[1]) + 0.008]

    hazard = pd.read_parquet(CACHE / "hazard_points.parquet")
    in_dem = ((hazard.lon >= extent[0]) & (hazard.lon <= extent[1]) &
              (hazard.lat >= extent[2]) & (hazard.lat <= extent[3]))

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.0, 1.05]})
    ax = axes[0]
    _hillshade(ax, dem, extent, alpha=0.5)
    ax.scatter(hazard.loc[in_dem, "lon"], hazard.loc[in_dem, "lat"],
               s=16, c=INK, alpha=0.75, label=f"历史灾点（DEM 内 {int(in_dem.sum())}）", zorder=4)
    ax.add_patch(Rectangle((route_extent[0], route_extent[2]),
                           route_extent[1] - route_extent[0],
                           route_extent[3] - route_extent[2],
                           fill=False, ec=RED, lw=1.2, ls="--", zorder=5))
    ax.text(route_extent[1], route_extent[3] + 0.004, "图 (b) 范围", color=RED, fontsize=10,
            va="bottom", ha="right", zorder=7, bbox=dict(fc="white", ec="none", alpha=0.82, pad=1.6))
    ax.set_title("(a) 研究区 DEM 与历史灾点", loc="left")
    ax.legend(loc="lower right", frameon=False)

    ax = axes[1]
    _hillshade(ax, dem, route_extent, alpha=0.55)
    arr = np.where(wet, arrival / 60.0, np.nan)
    im = ax.imshow(arr, extent=extent, cmap="viridis", vmin=0, vmax=np.nanmax(arr),
                   alpha=0.95, origin="upper")
    ax.scatter([src[0]], [src[1]], marker="*", s=260, c=AMBER, edgecolor="white",
               linewidth=0.8, zorder=8, label="源区质心")
    ax.scatter([inj[0]], [inj[1]], marker="s", s=60, c="white", edgecolor=TEAL,
               linewidth=1.5, zorder=8, label="公共注入断面")
    ax.scatter([port[0]], [port[1]], marker="*", s=260, c=RED, edgecolor="white",
               linewidth=0.8, zorder=8, label="口岸工作点")
    ax.scatter([alt[0]], [alt[1]], marker="x", s=55, c=GRAY, linewidth=1.4,
               zorder=8, label="口岸参考候选")
    for txt, xy, xytext, ha in [
        ("源区", (src[0], src[1]), (src[0] - 0.002, src[1] + 0.020), "right"),
        ("注入", (inj[0], inj[1]), (inj[0] + 0.003, inj[1] - 0.015), "left"),
        ("口岸", (port[0], port[1]), (port[0] - 0.020, port[1] + 0.014), "left"),
    ]:
        ax.annotate(txt, xy=xy, xytext=xytext, fontsize=10, color=INK, ha=ha,
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.7))
    cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    cb.set_label("D8 首达时间 / min")
    ax.set_title("(b) 源区—注入—口岸代理的 D8 路由", loc="left")

    fig.text(0.01, -0.02,
             "源区与口岸均为 DEM 近似工作点；84 个历史灾点只有一部分与 DEM 相交，未把域外点夹取进样本。",
             fontsize=9.5, color="#555555")
    save(fig, "01_研究区与链式几何.png")


# ---------------------------------------------------------------- Q1
def fig_q1_top10():
    df = pd.read_csv(DATA / "01c_168h簇完整排序.csv")
    d = df[df["window_length_h"] == 168].head(10)
    A = d["A_abs"].to_numpy(); names = d["name"].tolist()
    colors = [TEAL if i < 5 else TEAL_L for i in range(len(d))]
    fig, ax = plt.subplots(figsize=(9.6, 5.0), constrained_layout=True)
    y = np.arange(len(d))
    ax.barh(y, A, color=colors, height=0.68)
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlabel("绝对稳健异常 $A$（无量纲）")
    ax.set_xlim(0, 4.35)
    for i, v in enumerate(A):
        ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=11, color=INK)
    analog = load("01c_统一关联前五.json")["pseudo_2025_same_calendar"]["max_A"]
    ax.axvline(analog, color=AMBER, ls="--", lw=1.4)
    ax.text(analog + 0.04, 8.9, f"2025 同类窗最大值 {analog:.2f}", color=AMBER, fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("API 是唯一 $A>3$ 的候选；其余四项仅 1.1–2.3", loc="left")
    ax.text(0.99, 0.02, "深色：正式前五；浅色：第 6–10 名", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555555")
    save(fig, "01c_统一候选关联.png")


def fig_q1_full():
    df = pd.read_csv(DATA / "01c_168h簇完整排序.csv")
    d = df[df["window_length_h"] == 168].head(31)
    A = d["A_abs"].to_numpy(); names = d["name"].tolist()
    fam = d["family"].tolist()
    cmap = {"MET": "#4c78a8", "RADM": "#f58518", "SOIL": "#54a24b", "FLUX": "#b279a2"}
    colors = [cmap.get(f, GRAY) for f in fam]
    for i, f in enumerate(fam):
        if i < 5:
            colors[i] = TEAL
    fig, ax = plt.subplots(figsize=(10.5, 9.2), constrained_layout=True)
    y = np.arange(len(d))
    ax.barh(y, A, color=colors, height=0.72)
    ax.set_yticks(y, names); ax.invert_yaxis()
    ax.set_xlabel("绝对稳健异常 $A$（无量纲）")
    for i, v in enumerate(A):
        ax.text(v + 0.03, i, f"{v:.2f}", va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("31 项固定候选的完整体排序（相关簇代表）", loc="left")
    save(fig, "01c_完整31项.png")


# ---------------------------------------------------------------- Q2
def fig_q2_critical():
    sc = pd.read_csv(DATA / "02_状态元组_逐沟道.csv")
    counts = sc["G"].value_counts()
    n_w = int(counts.get("water_controlled", 0))
    n_g = int(counts.get("gravity_unstable", 0))
    n_f = int(counts.get("friction_balanced", 0))
    n_na = int(counts.get("not_evaluable", 0))
    q = load("02_沟道临界条件.json")
    qc = sc.loc[sc["G"] == "water_controlled", "Q_crit_m3s"].dropna().to_numpy()
    qw = 202.650179
    qc_center = 2893.666499

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.9, 1.35]})
    ax = axes[0]
    labels = ["水流控制", "重力失稳", "摩擦平衡"]
    vals = [n_w, n_g, n_f]
    colors = [TEAL, AMBER, GRAY]
    yy = np.arange(3)
    ax.barh(yy, vals, color=colors, height=0.58)
    ax.set_yticks(yy, labels); ax.invert_yaxis()
    ax.set_xlim(0, 90); ax.set_xlabel("条数")
    for y, v in zip(yy, vals):
        ax.text(v + 2, y, str(v), va="center", fontsize=13, color=INK, fontweight="bold")
    ax.set_title("(a) 机械状态分支（84 条）", loc="left")
    ax.grid(axis="x", alpha=0.2)

    ax = axes[1]
    bins = np.logspace(np.log10(max(qc.min(), 1e-5)), np.log10(qc.max() * 1.15), 22)
    ax.hist(qc, bins=bins, color=TEAL, alpha=0.55, edgecolor="white")
    ax.set_xscale("log")
    ax.axvline(qw, color=AMBER, ls="--", lw=1.5)
    ax.axvline(qc_center, color=RED, ls=":", lw=1.5)
    ax.set_xlim(1e-5, 5e3)
    ax.set_xlabel("正门槛起动流量 $Q_c$ / (m$^3$·s$^{-1}$)")
    ax.set_ylabel("沟道数")
    ax.set_title("(b) 78 条正门槛沟道与事件水源", loc="left")
    ax.grid(axis="y", alpha=0.2)
    ax.text(0.02, 0.97, f"$T_{{peak}}={qw/qc_center:.4f}$，未越阈",
            transform=ax.transAxes, va="top", fontsize=11, color=INK,
            bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"))
    ax.annotate(f"事件 $Q_w$={qw:.1f}", xy=(qw, 11.2), xytext=(qw*0.10, 13.0),
                fontsize=9.5, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=0.8))
    ax.annotate(f"中心线 $Q_c$={qc_center:.0f}", xy=(qc_center, 9.0),
                xytext=(qc_center*0.06, 7.0), fontsize=9.5, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    save(fig, "02_临界条件.png")


def fig_q2_id():
    d = load("02_降雨阈值.json")
    band = d["频率曲线自助带"]
    low = d["下包络"]["band"]
    events = pd.DataFrame(d["事件表"])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    for ax, b, a, bb, title, col in [
        (axes[0], band, d["幂律系数a"], d["幂指数b"], "(a) 95% 频率包络", AMBER),
        (axes[1], low, d["下包络"]["alpha"], d["下包络"]["beta"], "(b) 10% 低分位参考", TEAL),
    ]:
        D = np.asarray(b["D_h"]); lo = np.asarray(b["lo"]); hi = np.asarray(b["hi"])
        ax.fill_between(D, lo, hi, color=col, alpha=0.18, lw=0)
        ax.plot(D, a * D ** (-bb), color=col, lw=1.8)
        if "mean_mmh" in events:
            ax.scatter(events["dur_h"], events["mean_mmh"], s=8, color=INK, alpha=0.35)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(1, 40); ax.set_ylim(0.05, 40)
        ax.set_xlabel("历时 $D$ / h"); ax.set_ylabel("强度 $I$ / (mm·h$^{-1}$)")
        ax.set_title(title, loc="left"); ax.grid(alpha=0.2)
    axes[0].text(1.2, 0.08, "描述历史稀有度，不是起动阈值", color=AMBER, fontsize=9)
    axes[1].text(1.2, 0.08, "未作灾害标注的低分位筛查线", color=TEAL, fontsize=9)
    save(fig, "02_I-D包络.png")


def fig_q2_trigger():
    st = pd.read_csv(DATA / "02_状态元组_逐小时.csv", parse_dates=["timestamp"])
    st = st.set_index("timestamp")
    ev = st.loc["2026-08-26 08:00":"2026-08-26 13:00"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1.0]})
    ax = axes[0]
    t = ev.index
    ax.plot(t, ev["Q_water_m3s"], color=TEAL, marker="o", ms=4, lw=1.8, label="$Q_w$")
    ax.set_ylabel("水源流量 $Q_w$ / (m$^3$·s$^{-1}$)", color=TEAL)
    ax.tick_params(axis="y", colors=TEAL)
    ax.set_xlabel("2026-08-26 当地时间")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax2 = ax.twinx()
    ax2.plot(t, ev["T_index"], color=AMBER, marker="s", ms=4, lw=1.8, label="$T$")
    ax2.axhline(1.0, color=RED, ls="--", lw=1.0)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("触发比 $T=Q_w/Q_c$", color=AMBER)
    ax2.tick_params(axis="y", colors=AMBER)
    ax2.text(t[-1], 1.02, "$T=1$", color=RED, fontsize=9, ha="right")
    ax.set_title("(a) 事件窗：$T_{max}=0.070$，六小时未越阈", loc="left")
    ax.grid(alpha=0.2)
    ax.annotate(f"$Q_w$ 峰值 {ev['Q_water_m3s'].max():.1f}", xy=(t[-1], ev["Q_water_m3s"].max()),
                xytext=(-90, -28), textcoords="offset points", fontsize=9, color=TEAL,
                arrowprops=dict(arrowstyle="->", color=TEAL, lw=0.8))

    ax = axes[1]
    st["year"] = st.index.year
    ann = st.groupby("year")["T_index"].max().dropna()
    ax.bar(ann.index.astype(str), ann.values, color=TEAL_L, width=0.62)
    ax.axhline(1.0, color=RED, ls="--", lw=1.0)
    ax.set_ylim(0, 1.1); ax.set_ylabel("年内最大 $T$")
    for x, v in zip(ann.index.astype(str), ann.values):
        ax.text(x, v + 0.025, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_title("(b) 2021–2026 年内最大触发比", loc="left")
    ax.grid(axis="y", alpha=0.2)
    save(fig, "02_触发评估.png")


# ---------------------------------------------------------------- Q3
def fig_q3_forest():
    d = load("03_前向预报.json")
    iv = d["intervals"]
    def local(s):
        return pd.Timestamp(s).tz_convert("Asia/Shanghai").tz_localize(None)
    tm_lo, tm_hi = local(iv["t_m"]["lower_time"]), local(iv["t_m"]["upper_time"])
    ta_lo, ta_hi = local(iv["t_a"]["lower_time"]), local(iv["t_a"]["upper_time"])
    tm_med, ta_med = local(iv["t_m"]["median_time"]), local(iv["t_a"]["median_time"])
    tm_env = (local(iv["t_m"]["support_envelope_lower_time"]), local(iv["t_m"]["support_envelope_upper_time"]))
    ta_env = (local(iv["t_a"]["support_envelope_lower_time"]), local(iv["t_a"]["support_envelope_upper_time"]))
    issue = pd.Timestamp("2026-08-26 08:00")
    anchor = pd.Timestamp("2026-08-26 10:30")

    fig, ax = plt.subplots(figsize=(9.6, 3.9), constrained_layout=True)
    y_tm, y_ta, y_real = 1, 2, 3
    ax.axvspan(issue, pd.Timestamp("2026-08-26 12:00"), color="#f2f2f2", zorder=0)
    ax.axvline(issue, color=INK, lw=1.2, ls="-", label="08:00 发布")
    ax.axvline(anchor, color=RED, lw=1.4, ls=":", label="10:30 口岸冲击锚点")
    for y, lo, hi, med, env, col, lab in [
        (y_tm, tm_lo, tm_hi, tm_med, tm_env, TEAL, "源区融水起始代理 $t_m$"),
        (y_ta, ta_lo, ta_hi, ta_med, ta_env, AMBER, "即时释放条件到达 $t_a^{(0)}$"),
    ]:
        ax.plot([env[0], env[1]], [y, y], color=col, lw=9, alpha=0.18, solid_capstyle="butt")
        ax.plot([lo, hi], [y, y], color=col, lw=8, solid_capstyle="butt")
        ax.plot([med, med], [y - 0.18, y + 0.18], color="white", lw=1.4)
        ax.text(hi + pd.Timedelta(minutes=8), y, f"{lo:%H:%M:%S}–{hi:%H:%M:%S}",
                va="center", fontsize=10, color=col)
    ax.plot([anchor], [y_real], marker="D", ms=8, color=RED, zorder=5)
    ax.text(anchor + pd.Timedelta(minutes=8), y_real, "10:30 落入辅助区间（1/1）",
            va="center", fontsize=10, color=RED)
    y_unknown = 0
    ax.plot([pd.Timestamp("2026-08-26 08:00"), pd.Timestamp("2026-08-26 12:00")],
            [y_unknown, y_unknown], color=GRAY, lw=7, alpha=0.45, solid_capstyle="butt")
    ax.text(pd.Timestamp("2026-08-26 08:05"), y_unknown, "真实 $t_r$、$t_f$、$t_a$：不可识别",
            va="center", fontsize=10, color="#555555")
    ax.set_yticks([y_unknown, y_tm, y_ta, y_real],
                  ["真实释放/形成/到达", "融水起始 $t_m$", "即时释放到达 $t_a^{(0)}$", "口岸锚点"])
    ax.set_xlim(pd.Timestamp("2026-08-26 07:30"), pd.Timestamp("2026-08-26 12:30"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("2026-08-26 当地时间")
    ax.set_ylim(-0.6, 3.35)
    ax.grid(axis="x", alpha=0.2)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    save(fig, "03_时间区间.png")


# ---------------------------------------------------------------- Q4
def fig_q4_profile():
    b = load("04_仿真基线.json")
    s = np.asarray(b["s_km"]); z = np.asarray(b["z_m"])
    v = np.asarray(b["v_m_s"]); h = np.asarray(b["h_m"])
    I = np.asarray(b["intensity"]) / 1000.0
    wet = v > 0
    stop = float(b["runout_km"]); port = 20.882237
    mu = b["mu"]
    fig = plt.figure(figsize=(9.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1.0], wspace=0.10, hspace=0.16)
    a0 = fig.add_subplot(gs[:, 0])
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a3 = fig.add_subplot(gs[2, 1], sharex=a1)

    a0.plot(s, z, color="#333333", lw=1.6, label="纵剖面高程")
    a0.fill_between(s, z, z.max(), color="#e8eef5", zorder=0)
    i0 = int(np.where(wet)[0][0])
    e_line = z[i0] + b.get("params", {}).get("E0_m2s2", 0.0) / (2 * 9.80665) - mu * (s - s[i0]) * 1000
    a0.plot(s, e_line, "--", color=RED, lw=1.5,
            label=f"Heim 能量线（$\\mu$={mu:.3f}）")
    a0.axvline(stop, color="#555555", ls=":", lw=1.2,
               label=f"1D 自然停止 {stop:.2f} km")
    a0.axvline(port, color=AMBER, ls="--", lw=1.2,
               label=f"口岸代理 {port:.2f} km")
    a0.axvspan(stop, port, color=GRAY, alpha=0.12)
    a0.text((stop + port) / 2, z.max() * 0.42, "1D 未到达口岸", ha="center",
            color="#555555", fontsize=10)
    a0.set_ylabel("高程 / m"); a0.set_xlabel("沿程里程 / km")
    a0.legend(loc="upper right", fontsize=8.5, framealpha=0.93)
    a0.set_title("(a) 纵剖面与 Heim 能量线", loc="left")
    a0.set_xlim(0, port + 0.55); a0.grid(alpha=0.22)

    for a, y, col, lab, title in [
        (a1, v, TEAL, "流速 / (m·s$^{-1}$)", f"(b) 断面平均流速（峰值 {b['v_max_m_s']:.1f} m/s）"),
        (a2, h, "#548235", "流深 / m", f"(c) 断面平均流深（中位 {b['h_median_m']:.2f} m）"),
    ]:
        a.plot(s[wet], y[wet], color=col, lw=1.8)
        a.fill_between(s[wet], 0, y[wet], color=col, alpha=0.15)
        a.set_ylabel(lab); a.set_title(title, loc="left", fontsize=11, pad=3)
        a.grid(alpha=0.22); a.set_xlim(0, stop * 1.06)
    a3.semilogy(s[wet], np.maximum(I[wet], 1e-2), color="#7030a0", lw=1.8)
    for thr, lab in [(10.0, "低/中"), (100.0, "中/高"), (1000.0, "高/极高")]:
        a3.axhline(thr, color="#999999", ls="--", lw=0.8)
        a3.text(s[wet][-1] * 0.02, thr * 1.06, lab, fontsize=9, color="#666666", va="bottom")
    a3.set_ylabel("强度 $I=\\rho_m h v^2$ / (kN·m$^{-1}$)")
    a3.set_xlabel("沿程里程 / km")
    a3.text(0.99, 0.93, "(d) 冲击强度", transform=a3.transAxes, fontsize=11, va="top", ha="right")
    a3.text(0.98, 0.07, f"峰值 {b['I_max']/1e3:.0f} kN/m", transform=a3.transAxes,
            ha="right", fontsize=9, color="#7030a0")
    a3.set_ylim(5, 2500)
    a3.grid(alpha=0.22); a3.set_xlim(0, stop * 1.06)
    fig.text(0.01, -0.01, "一维引擎在 9.34 km 自然停止；D8 分析域末端 26.05 km 为右删失，不是自然运程。",
             fontsize=9.5, color="#555555")
    save(fig, "04_沿程演进.png")


def fig_q4_map():
    b = load("04b_二维引擎.json")["runs"]["base"]
    dem, extent = _dem_extent()
    g = _grid()
    arrival = np.load(CACHE / "q4b_arrival.npy")
    final = np.load(CACHE / "q4b_final_h.npy")
    wet = np.isfinite(arrival)
    rr, cc = np.nonzero(wet)
    pad = 35
    r0, r1 = max(0, rr.min() - pad), min(dem.shape[0], rr.max() + pad)
    c0, c1 = max(0, cc.min() - pad), min(dem.shape[1], cc.max() + pad)
    ext = [g["lon0"] + c0 * g["dlon"], g["lon0"] + c1 * g["dlon"],
           g["lat0"] + r1 * g["dlat"], g["lat0"] + r0 * g["dlat"]]
    inj = _rc_to_lonlat(load("04b_二维引擎.json")["断面"]["注入_rc"], g)
    port = _rc_to_lonlat(load("04b_二维引擎.json")["断面"]["口岸投影_rc"], g)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), constrained_layout=True)
    arr_min = np.where(wet & np.isfinite(arrival), arrival / 60.0, np.nan)
    arr_vmax = float(np.nanmax(arr_min))
    for ax, field, title, cmap, vmax in [
        (axes[0], arr_min, "(a) 首达时间", "viridis", arr_vmax),
        (axes[1], np.where(final > 0, final, np.nan), "(b) 终态泥深（域末端右删失）", "magma", float(np.nanmax(final))),
    ]:
        _hillshade(ax, dem, ext, alpha=0.45)
        im = ax.imshow(field, extent=extent, cmap=cmap, vmin=0, vmax=vmax, alpha=0.95, origin="upper")
        ax.scatter([inj[0]], [inj[1]], marker="o", s=50, c="white", edgecolor=TEAL, lw=1.4, zorder=6)
        ax.scatter([port[0]], [port[1]], marker="*", s=240, c=RED, edgecolor="white", lw=0.8, zorder=7)
        ax.set_title(title, loc="left")
        cb = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cb.set_label("min" if "首达" in title else "m")
    axes[0].annotate("注入", xy=inj, xytext=(inj[0] + 0.005, inj[1] - 0.012),
                     fontsize=9, arrowprops=dict(arrowstyle="-", lw=0.7))
    axes[0].annotate("口岸代理", xy=port, xytext=(port[0] - 0.022, port[1] + 0.012),
                     fontsize=9, arrowprops=dict(arrowstyle="-", lw=0.7))
    axes[0].text(0.02, 0.97,
                 f"口岸峰深 {b['peak_depth_port_m']:.2f} m\n峰速 {b['peak_velocity_port_m_s']:.2f} m/s",
                 transform=axes[0].transAxes, va="top", fontsize=9,
                 bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"))
    axes[1].text(0.02, 0.97, "全部 16 配置终态\n位于 26.05 km 域末端",
                 transform=axes[1].transAxes, va="top", fontsize=9, color=RED,
                 bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"))
    save(fig, "04b_到达与停积.png")


def fig_q4_snapshots():
    dem, extent = _dem_extent()
    g = _grid()
    snaps = [np.load(CACHE / f"q4b_snap_t{t}.npy") for t in (600, 1200, 1800, 3600)]
    vmax = max(float(a.max()) for a in snaps)
    arrival = np.load(CACHE / "q4b_arrival.npy")
    wet = np.isfinite(arrival)
    rr, cc = np.nonzero(wet)
    pad = 35
    r0, r1 = max(0, rr.min() - pad), min(dem.shape[0], rr.max() + pad)
    c0, c1 = max(0, cc.min() - pad), min(dem.shape[1], cc.max() + pad)
    ext = [g["lon0"] + c0 * g["dlon"], g["lon0"] + c1 * g["dlon"],
           g["lat0"] + r1 * g["dlat"], g["lat0"] + r0 * g["dlat"]]
    inj = _rc_to_lonlat(load("04b_二维引擎.json")["断面"]["注入_rc"], g)
    port = _rc_to_lonlat(load("04b_二维引擎.json")["断面"]["口岸投影_rc"], g)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    for ax, t, a in zip(axes.ravel(), (600, 1200, 1800, 3600), snaps):
        _hillshade(ax, dem, ext, alpha=0.4)
        im = ax.imshow(np.where(a > 0, a, np.nan), extent=extent, cmap="viridis",
                       vmin=0, vmax=vmax, alpha=0.95, origin="upper")
        ax.scatter([inj[0]], [inj[1]], marker="o", s=40, c="white", edgecolor=TEAL, lw=1.2, zorder=6)
        ax.scatter([port[0]], [port[1]], marker="*", s=180, c=RED, edgecolor="white", lw=0.7, zorder=7)
        ax.set_title(f"t = {t/60:.0f} min", loc="left")
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, pad=0.02)
    cb.set_label("等效泥深 / m")
    fig.text(0.01, -0.01, "D8 单流向路由；只表示给定源项和参数下的条件演进，不含横向二维动力。",
             fontsize=9.5, color="#555555")
    save(fig, "04b_过程快照.png")


def main():
    fig_study_area()
    fig_q1_top10()
    fig_q1_full()
    fig_q2_critical()
    fig_q2_id()
    fig_q2_trigger()
    fig_q3_forest()
    fig_q4_profile()
    fig_q4_map()
    fig_q4_snapshots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
