"""Chinese figures for the v5.0 forward forecast and paired propagation products.

The figures use only current machine-readable outputs.  They deliberately do
not draw a coverage rate for a true formation time that is not observed.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mmlib import io_data as io

ROOT = Path(__file__).resolve().parents[2]
OUT = io.DATA_OUT
FIG = io.FIG_OUT
LOCAL = "Asia/Shanghai"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 13,
    # DateFormatter 默认按 UTC 解释时区感知坐标；不设此值会把当地时刻
    # 整体左移八小时，而轴标题仍写着“当地时间”。
    "timezone": LOCAL,
})


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def local_time(value: str | None):
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, str) and value.strip().lower() in {
        "", "nan", "none", "null", "<na>", "not_estimable",
    }:
        return None
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert(LOCAL)


def value_missing(value) -> bool:
    """Return an explicit missing-value predicate without bool(np.nan) traps."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "", "nan", "none", "null", "<na>", "not_estimable",
            "not_constructible", "not_constructible_zero_finite",
        }
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def port_coverage_label(row: pd.Series) -> str:
    """Map a single-event coverage row to a conservative display label."""
    if bool(row.get("prediction_missing", False)):
        return "不可估计"
    status = str(row.get("interval_status", "")).strip().lower()
    if status and status != "available":
        return "不可估计"
    if value_missing(row.get("lower_time")) or value_missing(row.get("upper_time")):
        return "不可估计"
    value = row.get("definite_cover")
    if value_missing(value):
        return "不可估计"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "1.0", "true"}:
            return "1/1"
        if normalized in {"0", "0.0", "false"}:
            return "0/1"
        raise ValueError(f"invalid definite_cover label: {value!r}")
    if isinstance(value, (bool, np.bool_)):
        return "1/1" if bool(value) else "0/1"
    number = float(value)
    if not np.isfinite(number) or number not in (0.0, 1.0):
        raise ValueError(f"definite_cover must be 0, 1, or missing; got {value!r}")
    return "1/1" if number == 1.0 else "0/1"


def plot_alarm_marker(axes, alarm) -> bool:
    """Draw a valid alarm or an explicit no-alarm annotation without NaN access."""
    if value_missing(alarm):
        axes.text(
            0.02,
            0.96,
            "未构造报警/未报警",
            transform=axes.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#d0d0d0"},
        )
        return False
    axes.plot([alarm, alarm], [0.25, 0.95], color="#444444", ls="--", lw=1.3)
    axes.text(alarm, 1.0, "08:00发布/报警", ha="center", va="bottom", fontsize=14)
    return True


def done(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def interval_and_states(payload: dict, intervals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    rows = [
        ("源区融化起始代理", intervals.iloc[0]),
        ("假定即时释放后的到达", intervals.iloc[1]),
    ]
    colors = ["#b36b3f", "#276b6a"]
    for y, (label, row) in enumerate(rows):
        lo = local_time(row["lower_time"])
        med = local_time(row["median_time"])
        hi = local_time(row["upper_time"])
        env_lo = local_time(row["support_envelope_lower_time"])
        env_hi = local_time(row["support_envelope_upper_time"])
        axes[0].plot([env_lo, env_hi], [y, y], color=colors[y], lw=18, alpha=.22,
                     solid_capstyle="butt")
        axes[0].plot([lo, hi], [y, y], color=colors[y], lw=9, solid_capstyle="butt",
                     label=label if y == 0 else None)
        axes[0].plot([med, med], [y - .22, y + .22], color="#17252b", lw=2)
        offset, vertical = (.22, "bottom") if y == 0 else (-.24, "top")
        axes[0].text(hi, y + offset, f"{lo:%H:%M}-{hi:%H:%M}", ha="left", va=vertical,
                     fontsize=14, color=colors[y])
    issue = local_time(payload["issue_time_utc"])
    hit = local_time("2026-08-26T02:30:00+00:00")
    axes[0].axvline(issue, color="#333333", ls="--", lw=1.2, label="08:00发布")
    axes[0].axvline(hit, color="#9b2c2c", ls=":", lw=1.5, label="10:30口岸冲击锚点")
    axes[0].set_yticks([0, 1], [row[0] for row in rows])
    axes[0].set_xlim(pd.Timestamp("2026-08-26 07:30", tz=LOCAL),
                     pd.Timestamp("2026-08-26 12:00", tz=LOCAL))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].set_xlabel("2026-08-26 当地时间")
    axes[0].set_title("(a) 代表时刻条件区间（深条）与完整支持包络（浅条）")
    axes[0].grid(axis="x", alpha=.22)
    axes[0].legend(loc="lower right", fontsize=14)

    states = [
        "finite", "no_event", "active_at_issue",
        "active_at_issue_left_censored", "no_arrival",
        "right_censored", "not_evaluable",
    ]
    labels = [
        "未来有限", "窗内未起始", "发布时已活动",
        "口岸到达左删失", "完整输入未到达",
        "轨迹右删失", "输入不可评价",
    ]
    colors_state = [
        "#276b6a", "#9a7a42", "#b36b3f", "#7d5b8c",
        "#4e5c76", "#9b5b45", "#777777",
    ]
    width = .34
    for offset, (target, summary) in enumerate([
        ("源区代理", payload["state_summary_tm"]),
        ("即时释放对照", payload["state_summary_ta"]),
    ]):
        bottom = 0.0
        for state, label, color in zip(states, labels, colors_state):
            value = summary["counts"].get(state, 0) / summary["total_members"]
            axes[1].bar(offset, value, width, bottom=bottom, color=color,
                        label=label if offset == 0 else None)
            if value > .02:
                axes[1].text(offset, bottom + value / 2, f"{value:.0%}",
                             ha="center", va="center", fontsize=14,
                             color="white" if value > .15 else "#202020")
            bottom += value
    axes[1].set_xticks([0, 1], ["源区融化起始代理", "即时释放条件到达"])
    axes[1].set_ylim(0, 1.02)
    axes[1].set_ylabel("辅助模型成员比例")
    axes[1].set_title("(b) 源区五类/口岸六类状态质量分别合计为1；缺测不并入无事件")
    axes[1].grid(axis="y", alpha=.2)
    axes[1].legend(ncol=4, fontsize=12, loc="upper center",
                   bbox_to_anchor=(.5, -.18))
    done(fig, "03_时间区间.png")


def coverage_and_warning(warning: pd.DataFrame, coverage: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    hit = local_time("2026-08-26T02:30:00+00:00")
    lo = local_time(coverage.iloc[1]["lower_time"])
    hi = local_time(coverage.iloc[1]["upper_time"])
    port_cover = port_coverage_label(coverage.iloc[1])
    alarm = local_time(warning.iloc[0]["first_alarm_time"])
    axes[0].errorbar(1, hit, yerr=None, fmt="D", ms=8, color="#9b2c2c",
                     label="题面口岸冲击10:30")
    axes[0].plot([lo, hi], [0.55, 0.55], color="#276b6a", lw=12,
                 solid_capstyle="butt", label="即时释放条件区间")
    plot_alarm_marker(axes[0], alarm)
    axes[0].text(0.01, 0.08, "真实释放、形成及到达预测：不可估计\n本图仅为即时释放假设的辅助对照",
                 transform=axes[0].transAxes, fontsize=14,
                 bbox={"facecolor": "white", "alpha": .88, "edgecolor": "#d0d0d0"})
    axes[0].set_ylim(0, 1.2)
    axes[0].set_xlim(pd.Timestamp("2026-08-26 08:00", tz=LOCAL),
                     pd.Timestamp("2026-08-26 12:00", tz=LOCAL))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].set_yticks([])
    axes[0].set_xlabel("2026-08-26 当地时间")
    axes[0].set_title(f"(a) 即时释放对照单点包含：{port_cover}")
    axes[0].grid(axis="x", alpha=.22)
    axes[0].legend(fontsize=14, loc="upper right")

    delays = [0, 5, 10, 15, 30]
    colnames = [f"lead_port_{d}min" for d in delays]
    values = [float(warning.iloc[0][c]) for c in colnames]
    bars = axes[1].bar([str(d) for d in delays], values,
                       color=["#276b6a", "#4f8581", "#789b96", "#a4b8b4", "#c6d1ce"])
    axes[1].axhline(0, color="black", lw=.8)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + .04,
                     f"{value:.2f} h", ha="center", va="bottom", fontsize=14)
    axes[1].set_ylim(0, max(values) + .45)
    axes[1].set_xlabel("发布后通信与组织延迟 / min")
    axes[1].set_ylabel("融水提示至冲击锚点的时间差 / h")
    axes[1].set_title("(b) 融水提示延迟对照；真实灾害预警不可估计")
    axes[1].grid(axis="y", alpha=.2)
    done(fig, "03_区间覆盖率.png")


def input_availability(payload: dict) -> None:
    runs = payload["ifs_manifest"]["source_runs"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6), constrained_layout=True)
    issue = local_time(payload["issue_time_utc"])
    for index, run in enumerate(runs):
        init = local_time(run["init_time_utc"])
        available = local_time(run["available_at_utc"])
        ax.plot([init, available], [index, index], color="#7b8b87", lw=3)
        ax.scatter([init], [index], color="#9b5b45", s=55, label="起报" if index == 0 else None,
                   zorder=3)
        ax.scatter([available], [index], color="#276b6a", s=55,
                   label="可用上界" if index == 0 else None, zorder=3)
        ax.text(available, index + .18,
                f"{run['available_at_utc'][11:16]} UTC / {available:%H:%M} 当地",
                fontsize=12)
    ax.axvline(issue, color="#9b2c2c", ls="--", lw=1.3, label="08:00发布")
    ax.set_yticks(range(len(runs)),
                  [f"{run['init_time_utc'][11:16]} UTC 起报" for run in runs])
    ax.set_xlim(pd.Timestamp("2026-08-25 10:00", tz=LOCAL),
                pd.Timestamp("2026-08-26 09:00", tz=LOCAL))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for label in ax.get_xticklabels():
        label.set_rotation(25)
        label.set_ha("right")
    ax.set_xlabel("当地时间")
    ax.set_title("三次源区IFS起报均在08:00发布前达到6 h可用上界")
    ax.grid(axis="x", alpha=.22)
    ax.legend(fontsize=14, loc="lower right")
    done(fig, "03_预警指标.png")


def paired_propagation(distance: dict) -> None:
    runout = pd.read_csv(OUT / "runout_ranges.csv")
    member = pd.read_csv(OUT / "03_前向预报成员.csv")
    merged = member.merge(
        runout[["member_id", "first_arrival_time_s", "configuration_weight"]],
        on="member_id",
        how="left",
        validate="one_to_one",
    )
    config_stats = merged.groupby("d8_configuration_id")["first_arrival_time_s"].agg(
        ["min", "median", "max", "count"]
    ).sort_values("median")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.1), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    y = np.arange(len(config_stats))
    axes[0].hlines(y, config_stats["min"] / 60, config_stats["max"] / 60,
                   color="#9ab2ae", lw=4)
    axes[0].plot(config_stats["median"] / 60, y, "o", color="#276b6a", ms=5)
    axes[0].set_yticks(y, config_stats.index, fontsize=12)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("同一成员源区支持至口岸代理的传播时延 / min")
    axes[0].set_title("(a) 16个D8情景与成员源时钟配对")
    axes[0].grid(axis="x", alpha=.2)

    d = distance["route_metrics"]
    labels = ["源区质心→注入\n水平距离", "注入→口岸\n中心线", "源区→口岸\n平面直线"]
    values = [
        d["source_to_injection_horizontal_m"] / 1000,
        d["injection_to_port_centerline_m"] / 1000,
        d["source_to_port_straight_m"] / 1000,
    ]
    bars = axes[1].barh(labels, values, color=["#b36b3f", "#276b6a", "#7a7a7a"])
    for bar, value in zip(bars, values):
        axes[1].text(value + .3, bar.get_y() + bar.get_height() / 2,
                     f"{value:.2f} km", va="center", fontsize=14)
    axes[1].text(.02, .03,
                 f"D8终态{runout['final_runout'].median()/1000:.2f} km仅表示域末端右截断\n"
                 "不是自然停积距离，也不叠加不同路线度量",
                 transform=axes[1].transAxes, fontsize=12,
                 bbox={"facecolor": "white", "alpha": .9, "edgecolor": "#d0d0d0"})
    axes[1].set_xlim(0, max(values) * 1.22)
    axes[1].set_xlabel("距离 / km（分行报告，不相加）")
    axes[1].set_title("(b) 空间范围：路线度量与域末端截断")
    axes[1].grid(axis="x", alpha=.2)
    done(fig, "03_空间距离.png")


def main() -> int:
    payload = load_json("03_前向预报.json")
    intervals = pd.read_csv(OUT / "forecast_intervals.csv")
    warning = pd.read_csv(OUT / "melt_advisory_events.csv")
    coverage = pd.read_csv(OUT / "coverage_events.csv")
    distance = load_json("03_空间距离范围.json")

    interval_and_states(payload, intervals)
    coverage_and_warning(warning, coverage)
    input_availability(payload)
    paired_propagation(distance)
    print("problem 3 figures: 4 written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
