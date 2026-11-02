"""问题 2 图表：临界条件、触发评估与多尺度可靠性。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmlib import io_data as io, trigger as tg  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams.update({
    "font.size": 23,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

FIG = io.FIG_OUT
OUT = io.DATA_OUT
EVENT_TS = pd.Timestamp("2026-08-26 10:30")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def fig_thresholds(idr, mtl):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    dd = np.array([r["D_h"] for r in idr["逐历时统计"]], dtype=float)
    ii = np.array([r["I_p95"] for r in idr["逐历时统计"]], dtype=float)
    band = idr["频率曲线自助带"]
    Dg = np.asarray(band["D_h"])
    a,b = idr["幂律系数a"],idr["幂指数b"]
    ax.fill_between(Dg,band["lo"],band["hi"],alpha=.2,color="#b64b35",label="按年块自助95%带")
    support=idr["频率包络最大支持历时_h"]
    ax.plot(Dg[Dg<=support],a*Dg[Dg<=support]**(-b),color="#b64b35",lw=2,label=f"频率95%包络 b={b:.2f}")
    ax.plot(Dg[Dg>support],a*Dg[Dg>support]**(-b),color="#b64b35",ls="--",label="拟合历时支持外推")
    ax.scatter(dd,ii,c="black",label="各历时过程最大强度p95")
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel("连续起止跨度 / 滑动窗历时 D (h)");ax.set_ylabel("强度 I (mm/h)")
    ax.set_title("(a) 频率参考，非力学起动门槛")
    ax.legend(fontsize=14);ax.grid(alpha=.25,which="both")

    ax = axes[0,1]
    ev = pd.DataFrame(idr["事件表"])
    lower = idr["下包络"]
    band = lower["band"]
    Dg = np.asarray(band["D_h"])
    ax.scatter(ev.dur_h,ev.mean_mmh,s=12,alpha=.4,c="#405b69",label="完整训练降雨过程")
    ax.fill_between(Dg,band["lo"],band["hi"],alpha=.2,color="#2f766d",label="按年块自助95%带")
    ax.plot(Dg,lower["alpha"]*Dg**(-lower["beta"]),c="#2f766d",lw=2,
            label=f"低分位10%参考 β={lower['beta']:.2f}")
    ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel("过程起止跨度 D (h，含内部干小时)");ax.set_ylabel("总雨量 / 跨度 (mm/h)")
    ax.set_title("(b) 候选水源筛查，非水源充分条件")
    ax.legend(fontsize=14);ax.grid(alpha=.25,which="both")

    ax = axes[1, 0]
    g = pd.read_parquet(io.CACHE / "gully_table.parquet")
    water = g.loc[g["mechanical_regime"].eq("water_controlled")].copy()
    zero_n = int(g["mechanical_regime"].isin(
        ["friction_equilibrium", "gravity_unstable"]
    ).sum())
    invalid_n = int(g["mechanical_regime"].eq("not_evaluable").sum())
    order = ["小", "中", "大", "特大"]
    data = [water.loc[water["规模"] == k, "Q_crit_m3s"].dropna().values for k in order]
    data = [d for d in data if len(d)]
    lab = [k for k in order if (water["规模"] == k).sum() > 0]
    bp = ax.boxplot(data, tick_labels=lab, patch_artist=True, widths=0.55)
    for p_, c in zip(bp["boxes"], ("#4daf4a", "#ff7f00", "#e41a1c", "#6a3d9a")):
        p_.set_facecolor(c); p_.set_alpha(0.7)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(i).normal(0, 0.06, len(d)),
                   d, s=16, c="k", alpha=0.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("起动流量 Q$_c$ (m$^{3}$/s)"); ax.set_xlabel("灾害规模等级")
    ax.set_title("(c) 水流控制分支起动流量")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    typ = water.groupby("类型").agg(
        n=("Q_crit_m3s", "size"),
        q=("Q_crit_m3s", "median"),
        s=("比降", "median"),
        a=("面积_km2", "median"),
    ).sort_values("q")
    sc = ax.scatter(typ["s"], typ["q"], s=np.clip(typ["n"] * 16, 40, 400),
                    c=typ["n"], cmap="viridis", edgecolor="k", linewidth=0.8,
                    alpha=0.88, zorder=4)
    for name, r in typ.iterrows():
        ax.annotate(f"{name}\n(n={int(r['n'])}, A={r['a']:.0f} km$^{{2}}$)",
                    (r["s"], r["q"]), fontsize=12,
                    xytext=(8, 8), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, lw=0))
    ax.set_xlabel("沟道比降中位数")
    ax.set_ylabel("起动流量中位数 $Q_c$ / (m$^{3}$/s)")
    ax.set_title("(d) 成因类型与起动条件")
    # 比降与起动流量的跨度都在一个数量级内，用对数轴会产生难读的
    # 2×10⁻¹、3×10⁻¹ 式刻度；改用线性轴并按数据自适应留白。
    xs = typ["s"].to_numpy(dtype=float)
    ys = typ["q"].to_numpy(dtype=float)
    xpad = max((xs.max() - xs.min()) * 0.45, 0.02)
    ypad = max((ys.max() - ys.min()) * 0.28, 40.0)
    ax.set_xlim(xs.min() - xpad, xs.max() + xpad)
    ax.set_ylim(max(ys.min() - ypad, 0), ys.max() + ypad)
    ax.grid(alpha=0.3)
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("沟道条数", fontsize=12)
    del sc
    p = FIG / "02_临界条件.png"
    fig.savefig(p, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_trigger(tp, rel):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    pt = pd.read_parquet(io.CACHE / "ptrigger.parquet")

    ax = axes[0, 0]
    w = pt.loc["2026-08-23":"2026-08-28"]
    ax.plot(w.index, w["P_trigger"], color="#d62728", lw=1.8, label="事件回顾评分 P")
    ax.plot(w.index, w["T_index"].clip(0, 2) / 2, color="#1f77b4", lw=1.2, ls="--",
            label="触发指数 T/2")
    ax.axhline(0.5, color="gray", ls=":")
    ax.axvline(EVENT_TS, color="k", ls=":", lw=2, label="灾害发生时刻 10:30")
    ax.set_ylabel("评分 / 归一化指数"); ax.set_ylim(0, 1.05)
    ax.set_title("(a) 事件窗口回顾评分（非独立预报）")
    ax.legend(fontsize=14, loc="upper left"); ax.grid(alpha=0.3)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)

    ax = axes[0, 1]
    p = pt["P_trigger"].dropna()
    ax.hist(p, bins=60, color="#4c72b0", alpha=0.8)
    ax.set_yscale("log")
    ax.axvline(0.5, color="r", ls="--", label="预警阈值 0.5")
    ax.set_xlabel("相对强度评分 P"); ax.set_ylabel("小时数（对数）")
    ax.set_title("(b) 全序列回顾评分分布")
    ax.legend(fontsize=14); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 0]
    rows = pd.DataFrame(rel["提前量与报警日率"])
    ax.plot(rows["全序列报警日率"] * 100, rows["提前量_h"], "o-", color="#6a3d9a", ms=7)
    for _, r in rows.iterrows():
        if r["提前量_h"] is not None:
            ax.annotate(f'{r["阈值"]:.1f}', (r["全序列报警日率"] * 100, r["提前量_h"]),
                        fontsize=12, xytext=(4, 4), textcoords="offset points")
    if not rows["提前量_h"].notna().any():
        ax.text(
            0.5,
            0.54,
            "无命中阈值：提前量不可估计\n仅能报告完整观测日报警频度",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=13,
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#b0b0b0"},
        )
    ax.set_xlabel("完整观测日报警频度 (%)"); ax.set_ylabel("相对事件时刻的提前量 (h)")
    ax.set_title("(c) 阈值—提前量—报警频度（非伪警率）")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    sens = rel["空间分辨率敏感性"]
    ks = list(sens.keys())
    slopes = [sens[k]["主沟比降"] for k in ks]
    qcrit = [sens[k]["起动单宽流量_m2s"] for k in ks]
    x = np.arange(len(ks))
    ax.bar(x - 0.2, slopes, 0.4, color="#4c72b0", label="主沟比降")
    ax.set_xticks(x); ax.set_xticklabels([f"{sens[k]['像元_m']:.0f} m" for k in ks])
    ax.set_ylabel("主沟比降", color="#4c72b0")
    ax.set_xlabel("DEM 分辨率")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, qcrit, 0.4, color="#dd8452", label="起动单宽流量")
    ax2.set_ylabel("起动单宽流量 (m$^{2}$/s)", color="#dd8452")
    ax.set_title("(d) 空间分辨率敏感性")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=14)
    ax.grid(alpha=0.3, axis="y")
    p = FIG / "02_触发评估.png"
    fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig)
    return p


def fig_business():
    screen=load("02_水源业务判据.json")
    skill=load("02_气象基线检验.json")
    tab=pd.read_csv(OUT/"02_逐日候选水源.csv",index_col=0,parse_dates=True)
    test=tab.loc["2026-06-01":"2026-08-27"]
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    ax=axes[0,0]
    ax.plot(test.index,test.PDD3_C_h,label="3日逐小时正积温")
    ax.axhline(screen["PDD3"]["threshold"],ls="--",c="#a34533",label="2021—2025暖季95%分位")
    ax.axvline(EVENT_TS,ls=":",c="black")
    ax.set_ylabel("PDD3 (℃·h)");ax.set_title("(a) 冻结训练阈值的融水候选筛查")
    ax.legend(fontsize=8);ax.tick_params(axis="x",rotation=25)
    ax=axes[0,1]
    ax.plot(test.index,test.rise24_C,label="相邻完整日均温差")
    ax.axhline(screen["rise24"]["threshold"],ls="--",c="#a34533",label="训练95%分位")
    ax.set_ylabel("24 h升温 (℃)");ax.set_title("(b) 日尺度升温参考，缺测不补零")
    ax.legend(fontsize=8);ax.tick_params(axis="x",rotation=25)
    ax=axes[1,0]
    for i,(col,lab) in enumerate((("rain_candidate","降雨"),("melt_candidate","融水"),("Tw","任一通道"))):
        dates=test.index[test[col].eq(1)]
        ax.scatter(dates,np.full(len(dates),i),s=12,label=lab)
    ax.set_yticks([0,1,2],["降雨候选","融水候选","联合候选"])
    ax.axvline(EVENT_TS,ls=":",c="black")
    ax.set_title("(c) 仅统计可观测期报警日率，非真实伪警")
    ax.tick_params(axis="x",rotation=25)
    ax=axes[1,1]
    ax.bar([0,1],[skill["BS_climate"],skill["BS"]],color=["#a9b2af","#32766c"])
    ax.set_xticks([0,1],["冻结气候概率","湿小时持续性混合"])
    ax.set_ylabel("2026有效小时 Brier 分数（小者好）")
    ax.set_title(f"(d) 1小时基线 BSS={skill['BSS']:.3f}；不证明灾害预报技巧")
    fig.savefig(FIG/"02_冻结水源筛查与气象技巧.png",dpi=160)
    plt.close(fig)


def main() -> int:
    idr = load("02_降雨阈值.json")
    mtl = load("02_消融阈值.json")
    tp = load("02_触发评估模型.json")
    rel = load("02_多尺度可靠性.json")
    print(fig_thresholds(idr, mtl))
    print(fig_trigger(tp, rel))
    fig_business()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
