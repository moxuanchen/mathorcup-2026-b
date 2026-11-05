"""问题2：训练期冻结的水源筛查与事件回顾评分，二者不混称预报。

降雨双包络共用缺测不补零、六小时干期切分的过程；各过程历时含内部干小时。
PDD、升温及气象基线只在2021--2025拟合，2026只评估已观测区间。
另报84灾点描述包络；像元空间对照由step02b_spatial.py单独实现。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmlib import io_data as io, meteo as mt, trigger as tg, terrain as tr  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

FIG = io.FIG_OUT
OUT = io.DATA_OUT
EVENT_TS = pd.Timestamp("2026-08-26 10:30")
FORMAL_EVENT_WIN = ("2026-08-26 08:00", "2026-08-26 13:00")
EVENT_WIN = FORMAL_EVENT_WIN
SEED = 20260826
CAL_YEARS = tuple(range(2021, 2026))
WARM_MONTHS = (6, 7, 8, 9)
BOOT = 200
STATE_TUPLE_FIELDS = ["G", "H", "Theta", "F_ice", "U_ice", "C_ice"]


def p_fields(value: float) -> dict:
    """Return a lossless p-value record; reject silent underflow to zero."""
    p = float(value)
    if not np.isfinite(p) or p < 0:
        raise ValueError(f"invalid p-value: {value!r}")
    if p == 0.0:
        raise FloatingPointError("p-value underflowed to zero; use higher precision or report a bound")
    return {"p值": p, "p值科学计数": f"{p:.6e}"}


def daily_valid(series: pd.Series, how="sum") -> pd.Series:
    """自然日有效小时至少20个；短截日与缺测日不伪装成零。"""
    grouped = series.resample("D")
    value = grouped.sum(min_count=20) if how == "sum" else grouped.mean()
    return value.where(grouped.count() >= 20)


def warm_training(index):
    return np.isin(index.year, CAL_YEARS) & np.isin(index.month, WARM_MONTHS)


def qsummary(values):
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy()
    x = x[np.isfinite(x)]
    return {"n": len(x), **{f"q{int(q*100):02d}": float(np.quantile(x, q)) if len(x) else None
                            for q in (.05, .25, .5, .75, .95)}}


def cn_scenario_sensitivity(feat: pd.DataFrame, model) -> dict:
    """Evaluate the declared CN=75/85/95 engineering scenarios."""
    scenarios = {}
    for cn in (75.0, 85.0, 95.0):
        scenario = tg.TriggerModel(
            area_km2=model.area,
            glacier_frac=model.f_gl,
            q_crit_m2s=model.q_crit,
            width_m=model.width,
            cn=cn,
            temp_offset=model.temp_offset,
            mechanical_regime=model.mechanical_regime,
            use_legacy_k=False,
        )
        discharge = scenario.discharge(feat)
        event = discharge.loc[EVENT_WIN[0]:EVENT_WIN[1]]
        q = event["Q_water"].dropna()
        t_index = event["T_index"].dropna()
        peak_time = str(t_index.idxmax()) if len(t_index) else None
        qrain = discharge["Q_rain"].dropna()
        rain_2026 = discharge.loc[discharge.index.year == 2026, "Q_rain"].dropna()
        runoff_2026 = tg.rain_runoff(feat, cn)
        runoff_2026 = runoff_2026.loc[runoff_2026.index.year == 2026].dropna()
        scenarios[f"{cn:.0f}"] = {
            "C_N0": cn,
            "is_formal_scenario": bool(np.isclose(cn, tg.CN0)),
            "formal_event_window_start": FORMAL_EVENT_WIN[0],
            "formal_event_window_end": FORMAL_EVENT_WIN[1],
            "event_Qwater_peak_m3s": float(q.max()) if len(q) else None,
            "event_Qrain_peak_m3s": (
                float(event["Q_rain"].dropna().max())
                if event["Q_rain"].notna().any() else None
            ),
            "event_Qmelt_peak_m3s": (
                float(event["Q_melt"].dropna().max())
                if event["Q_melt"].notna().any() else None
            ),
            "event_T_peak": float(t_index.max()) if len(t_index) else None,
            "event_T_peak_time": peak_time,
            "event_hours_T_ge_1": (
                int((t_index >= 1.0).sum()) if len(t_index) else None
            ),
            "event_valid_hours": int(len(q)),
            "series_Qrain_peak_m3s": float(qrain.max()) if len(qrain) else None,
            "rain_2026_Qrain_peak_m3s": (
                float(rain_2026.max()) if len(rain_2026) else None
            ),
            "rain_2026_runoff_sum_mm": (
                float(runoff_2026.sum()) if len(runoff_2026) else None
            ),
        }
    return {
        "formal_C_N0": float(tg.CN0),
        "formal_event_window": list(FORMAL_EVENT_WIN),
        "event_window_semantics": "inclusive hourly labels on the frozen 08:00--13:00 local event window",
        "scenario_values": [75.0, 85.0, 95.0],
        "AMC_adjustment": "API terciles from 2021-2025 training; factors 0.75/1.00/1.15 and clip [30,98]",
        "provenance": (
            "independent engineering scenario, not derived from unverified "
            "land-cover class codes and not calibrated to the 2026 event"
        ),
        "scenarios": scenarios,
    }


def _ser(x, key: str | None = None):
    if isinstance(x, dict):
        return {str(k): _ser(v, str(k)) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ser(v, key) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if not np.isfinite(v):
            return None
        if key in {"p值", "p_value"}:
            return v
        if 0.0 < abs(v) < 1e-4:
            return round(v, 12)
        return round(v, 6)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _ser(x.tolist())
    return x


def save_json(obj, name: str) -> Path:
    out = OUT / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_ser(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def rank_association_summary(frame: pd.DataFrame, value_col: str) -> dict:
    """Report finite-sample rank tests with the exact flow quantity named."""
    scale_tau = stats.kendalltau(np.log10(frame[value_col]), frame["规模序"])
    frequency_tau = stats.kendalltau(np.log10(frame[value_col]), frame["频率序"])
    scale_rho = stats.spearmanr(frame[value_col], frame["规模序"], nan_policy="omit")
    return {
        "quantity": value_col,
        "units": "m3/s" if value_col == "Q_crit_m3s" else "m2/s",
        "sample_n": int(len(frame)),
        "kendall_vs_scale": {"tau": float(scale_tau.statistic), **p_fields(scale_tau.pvalue)},
        "kendall_vs_frequency": {
            "tau": float(frequency_tau.statistic),
            **p_fields(frequency_tau.pvalue),
        },
        "spearman_vs_scale": {"rho": float(scale_rho.statistic), **p_fields(scale_rho.pvalue)},
    }


# ============================================================ S1 沟道尺度临界条件
def gully_critical_conditions() -> dict:
    """84 条沟道的起动临界流量及其与规模/频率等级的关系。"""
    g = tg.gully_table()
    positive = g.loc[
        (g["mechanical_regime"] == "water_controlled")
        & g["q_crit_m2s"].gt(0)
        & g["Q_crit_m3s"].notna()
    ].copy()
    order_scale = {"小": 1, "中": 2, "大": 3, "特大": 4}
    order_freq = {"低": 1, "中": 2, "高": 3}
    g["规模序"] = g["规模"].map(order_scale)
    g["频率序"] = g["频率"].map(order_freq)
    positive = g.loc[positive.index].copy()
    state_counts = g["mechanical_regime"].value_counts().to_dict()
    state_lists = {
        regime: g.loc[g["mechanical_regime"] == regime, ["Name", "类型", "规模", "比降"]]
        .to_dict(orient="records")
        for regime in ("water_controlled", "friction_equilibrium", "gravity_unstable", "not_evaluable")
    }

    out: dict = {
        "沟道数": int(len(g)),
        "机械状态计数": {str(k): int(v) for k, v in state_counts.items()},
        "机械状态清单": state_lists,
        "water_controlled_n": int(len(positive)),
        "zero_threshold_n": int(g["mechanical_regime"].isin(["friction_equilibrium", "gravity_unstable"]).sum()),
        "not_evaluable_n": int((g["mechanical_regime"] == "not_evaluable").sum()),
        "起动流量_m3s": {"min": float(positive["Q_crit_m3s"].min()), "p25": float(positive["Q_crit_m3s"].quantile(0.25)),
                     "p50": float(positive["Q_crit_m3s"].median()), "p75": float(positive["Q_crit_m3s"].quantile(0.75)),
                     "max": float(positive["Q_crit_m3s"].max())},
        "旧式起动流量_m3s": {"n": int(g["Q_crit_old_m3s"].notna().sum()),
                         "p50": float(g["Q_crit_old_m3s"].median()),
                         "note": "旧式缺少床体沿坡自重项，仅用于新旧对照"},
        "比降": {"min": float(g["比降"].min()), "p50": float(g["比降"].median()),
               "max": float(g["比降"].max())},
        "面积_km2": {"min": float(g["面积_km2"].min()), "p50": float(g["面积_km2"].median()),
                  "max": float(g["面积_km2"].max())},
        "高差_m": {"min": float(g["高差_m"].min()), "p50": float(g["高差_m"].median()),
                 "max": float(g["高差_m"].max())},
    }
    out["床面固相含量敏感性"] = {
        str(cb): {"bed_solid_fraction": cb,
                  "density_kg_m3": 1000.+1650.*cb,
                  "delta_rho_kg_m3": 1650.*cb,
                  "qcrit_m2s": qsummary([tg._q_crit_unit(s,bed_solid_fraction=cb) for s in positive["比降"]]),
                  "Qcrit_m3s": qsummary([tg._q_crit_unit(s,bed_solid_fraction=cb)*width/tg.KAPPA_Q
                                        for s,width in zip(positive["比降"],positive["沟宽_m"])])}
        for cb in (.15,1050./1650.,.8)}

    # 单宽起动流量与实际可能来水的量级对照：用面积×设计雨强估计一次洪水的峰值
    # 设计雨强取 I–D 阈值曲线的 1 h、3 h 值（见 S2），此处先用 p95 小时雨强
    out["按类型统计"] = {}
    for t, sub in g.groupby("类型"):
        positive_sub = positive.loc[positive["类型"] == t]
        out["按类型统计"][t] = {
            "n": int(len(sub)),
            "n_water_controlled": int(len(positive_sub)),
            "zero_or_not_evaluable_n": int(len(sub) - len(positive_sub)),
            "起动流量中位数_m3s": float(positive_sub["Q_crit_m3s"].median()) if len(positive_sub) else None,
            "比降中位数": float(sub["比降"].median()),
            "面积中位数_km2": float(sub["面积_km2"].median()),
            "高差中位数_m": float(sub["高差_m"].median()),
        }

    # 分级检验
    tests = {}
    for col, name in (("规模序", "规模"), ("频率序", "频率")):
        groups = [sub["Q_crit_m3s"].values for _, sub in positive.groupby(col) if len(sub) >= 2]
        labels = [k for k, sub in positive.groupby(col) if len(sub) >= 2]
        if len(groups) >= 2:
            h, p = stats.kruskal(*groups)
            tests[name] = {"等级": [int(v) for v in labels],
                          "各组n": [int(len(x)) for x in groups],
                          "各组起动流量中位数": [float(np.median(x)) for x in groups],
                          "Kruskal_H": float(h), **p_fields(p)}
    out["分级检验"] = tests

    # 序数 logistic：总起动流量能否预测规模等级
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        Xo = np.log10(positive[["Q_crit_m3s"]].values)
        m = OrderedModel(positive["规模序"].values, Xo, distr="logit").fit(method="bfgs", disp=False, maxiter=500)
        out["序数logistic_规模"] = {
            "系数": float(m.params[0]), "标准误": float(m.bse[0]),
            **p_fields(m.pvalues[0]), "对数似然": float(m.llf),
            "解释": "系数符号为已发生灾点的条件规模关联，按实际方向判读，不预设递减",
        }
    except Exception as e:
        out["序数logistic_规模"] = {"状态": f"拟合失败: {type(e).__name__}"}

    def ascii_p_test(entry: dict, statistic: str) -> dict:
        return {
            statistic: entry[statistic],
            "p_value": entry["p值"],
            "p_display": entry["p值科学计数"],
        }

    total_rank = rank_association_summary(positive, "Q_crit_m3s")
    unit_rank = rank_association_summary(positive, "q_crit_m2s")
    out["rank_test_statistics"] = {
        "sample_n": int(len(positive)),
        "zero_or_not_evaluable_n": int(len(g) - len(positive)),
        "total_flow_Q_crit_m3s": {
            "kendall_vs_scale": ascii_p_test(total_rank["kendall_vs_scale"], "tau"),
            "kendall_vs_frequency": ascii_p_test(
                total_rank["kendall_vs_frequency"], "tau"
            ),
            "spearman_vs_scale": ascii_p_test(total_rank["spearman_vs_scale"], "rho"),
        },
        "unit_width_q_crit_m2s": {
            "kendall_vs_scale": ascii_p_test(unit_rank["kendall_vs_scale"], "tau"),
            "kendall_vs_frequency": ascii_p_test(
                unit_rank["kendall_vs_frequency"], "tau"
            ),
            "spearman_vs_scale": ascii_p_test(unit_rank["spearman_vs_scale"], "rho"),
        },
        "note": (
            "总起动流量Q_crit= q_crit B/kappa与单宽起动流量q_crit的秩检验分别保存；"
            "p值以双精度结果和科学计数法并列保存，零表示禁止静默下溢。"
        ),
    }

    # 易发性评分只在有正水流门槛的 water_controlled 分支上定义。
    logq = np.log10(positive["Q_crit_m3s"])
    qn = (logq - logq.min()) / max(logq.max() - logq.min(), 1e-9)
    supply = np.log1p(positive["面积_km2"] * positive["比降"] * 1e3)
    sn = (supply - supply.min()) / max(supply.max() - supply.min(), 1e-9)
    g["易发性"] = np.nan
    g.loc[positive.index, "易发性"] = (1.0 - qn) * (0.5 + 0.5 * sn)
    out["易发性"] = {"min": float(g["易发性"].min()), "p50": float(g["易发性"].median()),
                  "max": float(g["易发性"].max()), "n_defined": int(g["易发性"].notna().sum()),
                  "zero_branch_state": "inapplicable; 不把零水流门槛填成最小易发分数",
                  "公式": "S = (1 − Q̃_c) · (0.5 + 0.5 · S̃)，Q̃_c 与 S̃ 为归一化起动流量与物质供给"}
    top = g.nlargest(10, "易发性").copy()
    top["q_crit_positive"] = top["Q_crit_m3s"].gt(0)
    out["前10易发沟道"] = top[
        [
            "Name",
            "类型",
            "规模",
            "频率",
            "面积_km2",
            "比降",
            "mechanical_regime",
            "q_crit_positive",
            "Q_crit_m3s",
            "易发性",
        ]
    ].to_dict(orient="records")
    g.to_parquet(io.CACHE / "gully_table.parquet")
    return out


# ============================================================ S2 降雨型 I–D 阈值
def rainfall_events(series: pd.Series) -> pd.DataFrame:
    """Missing hours break a record; six *observed* dry hours separate events."""
    p = series.sort_index().asfreq("h")
    rows, start, last_wet, gap = [], None, None, 0
    dry_run, start_complete = 0, False

    def close(reason):
        if start is not None and last_wet is not None:
            segment = p.loc[start:last_wet]
            if segment.notna().all():
                duration = int((last_wet-start).total_seconds()/3600)+1
                assert len(segment) == duration
                rows.append({"start": start, "end": last_wet, "year": start.year,
                             "dur_h": duration, "total_mm": float(segment.sum()),
                             "mean_mmh": float(segment.sum()/duration),
                             "Imax_mmh": float(segment.max()), "curve": segment.to_numpy(),
                             "boundary": reason,
                             "complete": start_complete and reason == "six_observed_dry_hours"})
    for timestamp, value in p.items():
        if not np.isfinite(value):
            close("missing_boundary")
            start, last_wet, gap, dry_run = None, None, 0, 0
            continue
        if value > 0.1:
            if start is None:
                start = timestamp
                start_complete = dry_run >= 6
            last_wet, gap = timestamp, 0
            dry_run = 0
        elif start is not None:
            dry_run += 1
            gap += 1
            if gap >= 6:
                close("six_observed_dry_hours")
                start, last_wet, gap = None, None, 0
        else:
            dry_run += 1
    close("record_end")
    return pd.DataFrame(rows, columns=["start", "end", "year", "dur_h", "total_mm",
                                      "mean_mmh", "Imax_mmh", "curve", "boundary", "complete"])


def _frequency_fit(events, durations):
    rows = []
    for duration in durations:
        values = []
        for curve in events["curve"]:
            if len(curve) >= duration:
                values.append(float(np.convolve(curve, np.ones(duration)/duration, "valid").max()))
        if len(values) >= 8:
            rows.append({"D_h": duration, "n": len(values), "I_p95": float(np.quantile(values, .95)),
                         "I_p50": float(np.median(values)), "I_p99": float(np.quantile(values,.99)),
                         "I_max": float(max(values))})
    if len(rows) < 2:
        return None, rows
    frame = pd.DataFrame(rows)
    slope, intercept = np.polyfit(np.log(frame.D_h), np.log(frame.I_p95), 1)
    return (float(np.exp(intercept)), float(-slope)), rows


def id_threshold(feat: pd.DataFrame) -> dict:
    from sklearn.linear_model import QuantileRegressor
    p = feat["P"].astype(float)
    all_events = rainfall_events(p)
    ev = all_events.loc[all_events.year.isin(CAL_YEARS) &
                        all_events.start.dt.month.isin(WARM_MONTHS) &
                        all_events.complete & (all_events.total_mm >= 1.)].copy()
    if len(ev) < 15:
        raise ValueError(f"insufficient complete training rainfall processes: {len(ev)}")
    durations = (1, 3, 6, 12, 24)
    fit, curve = _frequency_fit(ev, durations)
    if fit is None:
        raise ValueError("fewer than two adequately sampled rainfall durations")
    a, b = fit
    def lower_fit(frame):
        q = QuantileRegressor(quantile=.1, alpha=0, solver="highs")
        q.fit(np.log(frame[["dur_h"]].to_numpy()), np.log(frame.mean_mmh.to_numpy()))
        return float(np.exp(q.intercept_)), float(-q.coef_[0])
    lower = lower_fit(ev)
    rng = np.random.default_rng(SEED)
    freq_boot, lower_boot = [], []
    years = np.array(CAL_YEARS)
    for _ in range(BOOT):
        # Concatenation preserves bootstrap multiplicity (a set would be incorrect).
        pick = rng.choice(years, len(years), replace=True)
        sub = pd.concat([ev.loc[ev.year == year] for year in pick], ignore_index=True)
        pars, _ = _frequency_fit(sub, durations)
        if pars is not None:
            freq_boot.append(pars)
        if len(sub) >= 10 and sub.dur_h.nunique() >= 2:
            lower_boot.append(lower_fit(sub))
    if len(freq_boot) < BOOT*.8 or len(lower_boot) < BOOT*.8:
        raise ValueError("too few successful year-block bootstrap fits")
    fb, lb = np.asarray(freq_boot), np.asarray(lower_boot)
    dg = np.geomspace(1, max(24, ev.dur_h.max()), 80)
    def boot_band(samples):
        lines = samples[:, :1]*dg[None, :]**(-samples[:, 1:2])
        return {"D_h": dg.tolist(), "lo": np.quantile(lines,.025,axis=0).tolist(),
                "hi": np.quantile(lines,.975,axis=0).tolist()}
    def table(frame):
        return frame.drop(columns="curve").assign(start=frame.start.astype(str),
                                                  end=frame.end.astype(str)).to_dict("records")
    save_json({"events": table(all_events), "missing_hours_not_filled": int(p.isna().sum()),
               "wet_threshold_mm_h": .1, "minimum_training_depth_mm": 1.,
               "duration": "inclusive start-to-last-wet span, internal observed dry hours included"},
              "02_降雨过程.json")
    return {"独立暴雨事件数": len(ev), "训练年份": list(CAL_YEARS),
            "训练月份": list(WARM_MONTHS), "分割干期_h": 6, "缺测处理": "缺测断开不补零；边界事件单列",
            "过程边界计数": all_events.boundary.value_counts().to_dict(),
            "历时定义": "起止跨度含内部干小时；强度=总量/跨度", "自助": "按年分块，保留重复抽样权重",
            "自助次数": BOOT, "强度历时样本": sum(r["n"] for r in curve),
            "幂律系数a": a, "幂指数b": b, "公式": "I=a*D^(-b)",
            "语义一": "训练降雨过程内各历时最大平均强度的95%频率包络，非起动临界",
            "分位水平": .95, "参数95pct置信区间_a": np.quantile(fb[:,0],[.025,.975]).tolist(),
            "参数95pct置信区间_b": np.quantile(fb[:,1],[.025,.975]).tolist(),
            "频率曲线自助带": boot_band(fb), "逐历时统计": curve,
            "实测历时阈值": {f'{r["D_h"]}h': r["I_p95"] for r in curve},
            "外推历时阈值": {f"{d}h": float(a*d**(-b)) for d in (24,72) if d > ev.dur_h.max()},
            "频率包络最大支持历时_h": max(r["D_h"] for r in curve),
            "频率外推说明":"超过频率拟合最大有效历时即为外推，即使小于最长过程历时",
            "事件最大历时_h": int(ev.dur_h.max()), "事件最大总量_mm": float(ev.total_mm.max()),
            "事件最大强度_mmh": float(ev.Imax_mmh.max()), "事件表": table(ev),
            "下包络": {"alpha": lower[0], "beta": lower[1], "quantile": .1,
                        "alpha_ci": np.quantile(lb[:,0],[.025,.975]).tolist(),
                        "beta_ci": np.quantile(lb[:,1],[.025,.975]).tolist(),
                        "band": boot_band(lb), "n": len(ev),
                        "解释": "非灾害标注降雨的低分位参考线，仅候选水源筛查，非充分条件"}}


# ============================================================ S3 消融型能量阈值
def melt_threshold(feat: pd.DataFrame, info: dict) -> dict:
    gn = io.load_glacier_nc()
    ok = np.isfinite(gn["thickness"]) & (gn["thickness"] > 1)
    z = float(np.nanmean(gn["dem"][ok]))
    offset = -(z-3980.)*.006
    weighted = feat["cum_melt_energy"]/(1+np.exp(-(feat["Ta_2m"]+offset)/1.5))
    daily = daily_valid(feat["cum_melt_energy"])
    src = daily_valid(weighted)
    train = src.loc[warm_training(src.index)].dropna()
    ec = float(train.quantile(.9))
    frac = float(info["冰川覆盖率"])
    day = pd.Timestamp("2026-08-26")
    src_event = float(src.get(day, np.nan))
    assert np.isclose((ec/.334)*.334, ec)
    ice_depth = ec*1e6/(900.*334000.)
    assert np.isclose(ice_depth*900.*334000./1e6, ec)
    return {"冰川区平均高程_m": z, "冰川平均冰厚_m": float(np.mean(gn["thickness"][ok])),
            "源区气温订正_℃": offset, "湿绝热直减率_K每100m": .6,
            "训练年份": list(CAL_YEARS), "日有效小时门槛": 20,
            "观测地日融化能量_p50": float(daily.loc[warm_training(daily.index)].median()),
            "观测地日融化能量_p90": float(daily.loc[warm_training(daily.index)].quantile(.9)),
            "源区订正日融化能量_p50": float(train.median()), "源区订正日融化能量_p90": ec,
            "事件窗口累计融化能量_MJm-2": src_event,
            "事件窗口定义": "2026-08-26自然日，仅与同日完整性门槛一致的训练日比较",
            "事件窗口化水当量_mm": src_event/.334,
            "源区事件日能量_MJm2": src_event, "集水区事件日化水当量_mm": frac*src_event/.334,
            "临界条件": {"判据": "源区日能量参考90%分位，非已证实物理阈值",
                         "E_s_p90_MJm-2": ec, "M_c_mm": ec/.334,
                         "集水区M_c_mm": frac*ec/.334, "对应冰厚_m": ice_depth},
            "直减率敏感性": {str(l*100): float(daily_valid(feat["cum_melt_energy"] /
                 (1+np.exp(-(feat["Ta_2m"]-(z-3980.)*l)/1.5))).loc[warm_training(src.index)].quantile(.9))
                 for l in (.005,.006,.007)},
            "说明": "能量与水当量同源，不作为两个独立命中证据；缺测不补零"}


def build_object_state_rows(
    gully: pd.DataFrame,
    q_event: float,
    theta_event: str,
    f_ice: int,
    u_ice: int,
) -> list[dict]:
    """Build one uniquely identified state tuple per source object."""
    if "Code" not in gully:
        raise ValueError("gully table must contain the source-unique Code field")
    object_ids = gully["Code"].astype(str).tolist()
    if len(object_ids) != len(set(object_ids)):
        raise ValueError("gully Code values must be unique")
    rows = []
    for object_id, (_, source_row) in zip(object_ids, gully.iterrows()):
        g_state = str(source_row["mechanical_regime"])
        qcrit = (
            float(source_row["Q_crit_m3s"])
            if np.isfinite(source_row["Q_crit_m3s"])
            else np.nan
        )
        if g_state == "not_evaluable" or not np.isfinite(q_event):
            h_state = "not_evaluable"
        elif g_state == "water_controlled":
            if not np.isfinite(qcrit) or qcrit <= 0:
                h_state = "not_evaluable"
            else:
                h_state = (
                    "hydraulic_candidate"
                    if q_event / qcrit >= 1.0
                    else "below_threshold"
                )
        else:
            h_state = "inapplicable"
        rows.append({
            "object_id": object_id,
            "display_name": str(source_row.get("Name", "")),
            "G": g_state,
            "H": h_state,
            "Theta": theta_event,
            "F_ice": f_ice,
            "U_ice": u_ice,
            "C_ice": int(theta_event == "thermal_precondition" and f_ice == 1),
            "ice_mechanics_unresolved": u_ice,
            "Q_water_m3s_at_event": q_event,
            "Q_crit_m3s": qcrit,
            "T_index_at_event": (
                q_event / qcrit
                if np.isfinite(q_event) and np.isfinite(qcrit) and qcrit > 0
                else np.nan
            ),
        })
    return rows


def build_state_tuple(
    feat: pd.DataFrame,
    dis: pd.DataFrame,
    model,
    melt_meta: dict,
    gully: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Persist the theory's non-collapsed G/H/Theta/ice evidence tuple."""
    offset = float(melt_meta["源区气温订正_℃"])
    weighted_hourly = feat["cum_melt_energy"] / (
        1.0 + np.exp(-(feat["Ta_2m"] + offset) / 1.5)
    )
    daily_energy = daily_valid(weighted_hourly)
    threshold = float(melt_meta["临界条件"]["E_s_p90_MJm-2"])
    theta_daily = pd.Series("not_evaluable", index=daily_energy.index, dtype=object)
    valid_daily = daily_energy.notna()
    theta_daily.loc[valid_daily] = np.where(
        daily_energy.loc[valid_daily] >= threshold,
        "thermal_precondition",
        "below_thermal_reference",
    )
    theta = theta_daily.reindex(feat.index.normalize()).to_numpy()

    vol_path = OUT / "04_初始体积.json"
    h_thr, sl_thr = 50.0, 25.0
    if vol_path.exists():
        chain_a = json.loads(vol_path.read_text(encoding="utf-8")).get("链A", {})
        h_thr = float(chain_a.get("h_thr_m", h_thr))
        sl_thr = float(chain_a.get("sl_thr_deg", sl_thr))
    gn = io.load_glacier_nc()
    source_valid = (
        np.isfinite(gn["thickness"])
        & (gn["thickness"] >= h_thr)
        & np.isfinite(gn["slope"])
        & (gn["slope"] >= sl_thr)
    )
    source_n = int(source_valid.sum())
    f_ice = int(source_n > 0)
    u_ice = 1

    hourly = pd.DataFrame({
        "G": model.mechanical_regime,
        "H": dis["hydraulic_state"].to_numpy(),
        "Theta": theta,
        "F_ice": f_ice,
        "U_ice": u_ice,
        "C_ice": ((theta == "thermal_precondition") & (f_ice == 1)).astype(int),
        "ice_mechanics_unresolved": u_ice,
        "Q_water_m3s": dis["Q_water"].to_numpy(),
        "Q_crit_m3s": dis["Q_crit"].to_numpy(),
        "T_index": dis["T_index"].to_numpy(),
    }, index=feat.index)
    hourly.index.name = "timestamp"

    event_time = pd.Timestamp("2026-08-26 10:00")
    event_row = dis.loc[event_time] if event_time in dis.index else dis.iloc[-1]
    q_event = float(event_row["Q_water"]) if np.isfinite(event_row["Q_water"]) else np.nan
    theta_event = theta_daily.get(event_time.normalize(), "not_evaluable")
    gully_rows = build_object_state_rows(
        gully, q_event, theta_event, f_ice, u_ice
    )
    objects = pd.DataFrame(gully_rows)
    summary = {
        "fields": STATE_TUPLE_FIELDS,
        "ice_mechanics_unresolved": True,
        "F_ice_definition": "附件4存在冰厚和坡度同时达到源区阈值的编目像元",
        "source_thresholds": {"h_thr_m": h_thr, "sl_thr_deg": sl_thr},
        "source_pixel_count": source_n,
        "target_catchment_G": model.mechanical_regime,
        "target_catchment_H_counts": hourly["H"].value_counts().to_dict(),
        "target_catchment_Theta_counts": hourly["Theta"].value_counts().to_dict(),
        "gully_G_counts": objects["G"].value_counts().to_dict(),
        "gully_H_counts": objects["H"].value_counts().to_dict(),
        "event_day_Theta": theta_event,
        "hourly_csv": "02_状态元组_逐小时.csv",
        "objects_csv": "02_状态元组_逐沟道.csv",
    }
    return hourly, objects, summary


def block_quantile(series, q=.95):
    train = series.loc[warm_training(series.index)].dropna()
    rng = np.random.default_rng(SEED+1)
    boot = []
    for _ in range(BOOT):
        pick = rng.choice(CAL_YEARS, len(CAL_YEARS), replace=True)
        vals = np.concatenate([train.loc[train.index.year == y].to_numpy() for y in pick])
        if len(vals):
            boot.append(float(np.quantile(vals,q)))
    return {"threshold": float(train.quantile(q)), "q": q, "n_days": len(train),
            "ci": np.quantile(boot,[.025,.975]).tolist(), "bootstrap": "year-block"}


def business_screen(feat, idr):
    positive = feat["Ta_2m"].clip(lower=0).resample("D").sum(min_count=24)
    pdd = positive.rolling(3,min_periods=3).sum()
    temperature = daily_valid(feat["Ta_2m"], "mean")
    rise = temperature-temperature.shift(1)
    rain_day = daily_valid(feat["P"])
    pthr, rthr = block_quantile(pdd), block_quantile(rise)
    tab = pd.DataFrame({"PDD3_C_h": pdd, "rise24_C": rise, "rain_mm": rain_day})
    tab["rain_candidate"] = np.where(rain_day.notna(),0.,np.nan)
    lo = idr["下包络"]
    ev = rainfall_events(feat["P"])
    for _, row in ev.loc[~ev.complete].iterrows():
        dates=pd.date_range(row.start.normalize(),row.end.normalize(),freq="D")
        tab.loc[tab.index.intersection(dates),"rain_candidate"]=np.nan
    for _, row in ev.iterrows():
        if not row.complete or row.total_mm < 1. or row.mean_mmh < lo["alpha"]*row.dur_h**(-lo["beta"]):
            continue
        # Event-end evaluation: no claim of a causal intra-event alarm.
        dates = pd.date_range(row.start.normalize(),row.end.normalize(),freq="D")
        valid = tab.index.intersection(dates)
        tab.loc[valid,"rain_candidate"] = tab.loc[valid,"rain_candidate"].where(
            tab.loc[valid,"rain_mm"].isna(),1.)
    melt_valid = pdd.notna() & rise.notna()
    known_hit=(pdd>=pthr["threshold"]) | (rise>=rthr["threshold"])
    tab["melt_candidate"] = np.where(known_hit,1.,np.where(melt_valid,0.,np.nan))
    tab["PDD3_valid_hours"] = feat["Ta_2m"].resample("D").count().rolling(3,min_periods=3).sum()
    both = tab[["rain_candidate","melt_candidate"]]
    tab["Tw"] = np.where(both.eq(1).any(axis=1),1.,np.where(both.notna().all(axis=1),0.,np.nan))
    tab.to_csv(OUT/"02_逐日候选水源.csv",encoding="utf-8-sig",index_label="date")
    test = tab.loc[tab.index.year == 2026]
    warm = test.loc[test.index.month.isin(WARM_MONTHS)]
    def counts(frame):
        observed = frame.loc[frame.Tw.notna()]
        return {"observed_days": int(frame.Tw.notna().sum()), "candidate_days": int(frame.Tw.eq(1).sum()),
                "candidate_day_rate": float(frame.Tw.dropna().mean()),
                "first_date": str(observed.index.min()), "last_date": str(observed.index.max())}
    return {"training_years": list(CAL_YEARS), "PDD3": pthr, "rise24": rthr,
            "PDD_unit": "sum of positive hourly temperatures over 3 complete natural days, C*h",
            "rise_definition": "difference of adjacent natural-day mean temperature, 24h",
            "2026_observation_period": counts(test), "2026_warm_observed": counts(warm),
            "event_day": tab.loc["2026-08-26"].to_dict(),
            "daily_file": "02_逐日候选水源.csv", "rain_screen": "complete process span/intensity evaluated retrospectively",
            "interpretation": "candidate/alert-day rate is not a true false-alarm rate; unlabelled days are not confirmed negatives"}


def statistical_domains():
    g = tg.gully_table()
    cols = ["比降","主沟长_m","高差_m","面积_km2","沟口高程_m","流域最高高程_m","扇宽_m","扇长_m"]
    def describe(frame):
        return {c:qsummary(frame[c]) for c in cols}
    return {"n_historical_polygons": len(g), "all": describe(g),
            "by_type": {str(k):{"n":len(v),"quantiles":describe(v)} for k,v in g.groupby("类型")},
            "by_size": {str(k):{"n":len(v),"quantiles":describe(v)} for k,v in g.groupby("规模")},
            "interpretation": "case-only descriptive envelope, not occurrence probability or a validated critical boundary"}


def long_temperature_check(feat):
    ext = io.load_temperature()
    station = daily_valid(feat["Ta_2m"],"mean")
    station_pdd = station.clip(lower=0).rolling(3,min_periods=3).sum()
    result = {}
    for col in ext:
        values = pd.to_numeric(ext[col],errors="coerce").asfreq("D")
        values = values.where(values.between(-60,50))
        pdd = values.clip(lower=0).rolling(3,min_periods=3).sum()
        long = pdd.loc[(pdd.index.year <= 2025) & pdd.index.month.isin(WARM_MONTHS)].dropna()
        recent = pdd.loc[warm_training(pdd.index)].dropna()
        same = pd.concat([station.rename("station"),values.rename("era")],axis=1).dropna()
        same = same.loc[warm_training(same.index)]
        matched_pdd = pd.concat([station_pdd.rename("station"),pdd.rename("era")],axis=1).dropna()
        matched_pdd = matched_pdd.loc[warm_training(matched_pdd.index)]
        result[str(col)] = {"long_daily_PDD3_C_day":qsummary(long),
                            "recent_daily_PDD3_C_day":qsummary(recent),
                            "long_available_period":[str(long.index.min()),str(long.index.max())],
                            "overlap_days":len(same),
                            "common_date_PDD3_days":len(matched_pdd),
                            "common_date_station_PDD3":qsummary(matched_pdd.station),
                            "common_date_product_PDD3":qsummary(matched_pdd.era),
                            "common_date_PDD3_correlation":float(matched_pdd.corr().iloc[0,1]) if len(matched_pdd)>2 else None,
                            "temperature_correlation":float(same.corr().iloc[0,1]) if len(same)>2 else None,
                            "station_minus_product_mean_C":float((same.station-same.era).mean())}
    return {"products":result,"station_daily_PDD3_C_day":qsummary(station_pdd.loc[warm_training(station_pdd.index)]),
            "definition":"both products: max(daily mean temperature,0), sum over 3 natural days",
            "limitation":"daily-mean PDD is not hourly-positive PDD; no rainfall or I-D inference from temperature-only attachment"}


def weather_baseline(feat):
    p, temp = feat["P"].asfreq("h"), feat["Ta_2m"].asfreq("h")
    trainmask = np.isin(p.index.year,CAL_YEARS)
    wet = p.gt(.1).astype(float).where(p.notna())
    frequency = float(wet.loc[trainmask].mean())
    pair = pd.concat([wet.shift(1).rename("prev"),wet.rename("now")],axis=1).loc[trainmask].dropna()
    corr = float(pair.prev.corr(pair.now))
    persistence = float(np.clip(corr,0,1)) if np.isfinite(corr) else 0.
    train_t = temp.loc[trainmask].dropna()
    climate = train_t.groupby([train_t.index.dayofyear,train_t.index.hour]).mean().unstack()
    climate = climate.reindex(range(1,367)).interpolate(limit_direction="both").rolling(15,center=True,min_periods=1).mean()
    climate = climate.fillna(train_t.groupby(train_t.index.hour).mean())
    diurnal = np.array([climate.loc[d,h] for d,h in zip(temp.index.dayofyear,temp.index.hour)])
    diurnal = pd.Series(diurnal,index=temp.index)
    table = pd.DataFrame(index=temp.index)
    table["wet_observed"] = wet
    table["wet_probability"] = frequency+persistence*(wet.shift(1)-frequency)
    table["climate_probability"] = frequency
    table["BS_hour"] = (table.wet_probability-wet)**2
    table["BS_climate_hour"] = (frequency-wet)**2
    table["BS_improvement_hour"] = table.BS_climate_hour-table.BS_hour
    table["T_observed_C"] = temp
    table["T_persistence_C"] = temp.shift(1)
    table["T_diurnal_persistence_C"] = temp.shift(1)+diurnal-diurnal.shift(1)
    table["rain_intensity_persistence_mm_h"] = p.shift(1)*persistence
    # Every target is at t, issued at t-1h; only 2026 is an external temporal test.
    table["issued_at"] = table.index-pd.Timedelta(hours=1)
    table["evaluation_2026"] = table.index.year == 2026
    table.to_csv(OUT/"02_逐小时气象基线.csv",encoding="utf-8-sig",index_label="target_time")
    test = table.loc[table.index.year == 2026]
    bsrows = test.dropna(subset=["wet_observed","wet_probability"])
    bs, ref = float(bsrows.BS_hour.mean()),float(bsrows.BS_climate_hour.mean())
    ts = test.dropna(subset=["T_observed_C","T_persistence_C","T_diurnal_persistence_C"])
    def error(col):
        e = ts[col]-ts.T_observed_C
        return {"RMSE_C":float(np.sqrt((e*e).mean())),"MAE_C":float(e.abs().mean()),"n":len(e)}
    return {"training_years":list(CAL_YEARS),"horizon_h":1,"wet_threshold_mm_h":.1,
            "wet_training_frequency":frequency,"persistence_weight":persistence,
            "BS":bs,"BS_climate":ref,"BSS":1-bs/ref if ref>0 else None,"n_test_hours":len(bsrows),
            "T_diurnal_persistence":error("T_diurnal_persistence_C"),"T_plain_persistence":error("T_persistence_C"),
            "hourly_file":"02_逐小时气象基线.csv","missing_not_filled":True,
            "BSS_interpretation":"positive=better than frozen climatology, nonpositive=no skill",
            "limitation":"retrospective one-hour baseline, not an external weather forecast; temperature and rain intensity are not Brier probabilities"}


# ============================================================ S4 触发评估模型
def build_trigger_features(feat: pd.DataFrame, dis: pd.DataFrame, gully: pd.DataFrame) -> pd.DataFrame:
    """合成触发概率所需的物理量与集总参数。"""
    q = dis["Q_water"].astype(float)
    qc = dis["Q_crit"].astype(float)
    T = q / qc.replace(0, np.nan)
    X = pd.DataFrame({
        "lnT": np.log(T.clip(1e-6, 1e3)),
        "lnT_6h": np.log(T.clip(1e-6, 1e3).rolling(6, min_periods=1).mean()),
        "P_3h": feat["P_3h"], "P_24h": feat["P_24h"], "API": feat["API"],
        "cum_melt_6h": feat["cum_melt_energy"].rolling(6, min_periods=1).sum(),
        "Ta_2m": feat["Ta_2m"], "VPD2": feat["VPD2"],
        "dtheta": feat["dtheta_1_20"], "VWC": feat["VWC_0.1m"],
    }, index=feat.index)
    return X


def trigger_probability(feat: pd.DataFrame, dis: pd.DataFrame, static: dict,
                        gully: pd.DataFrame) -> dict:
    """事件回顾指数的logistic相对评分，非发生概率；斜率不可单事件识别。"""
    from sklearn.metrics import roc_auc_score

    T = (dis["Q_water"] / dis["Q_crit"].replace(0, np.nan)).astype(float)
    ev_hour = pd.Timestamp("2026-08-26 10:00")
    T_ev = float(T.loc[ev_hour]) if ev_hour in T.index and np.isfinite(T.loc[ev_hour]) else float("nan")
    kT = 4.0
    P = 1.0 / (1.0 + np.exp(np.clip(-kT * (T - 1.0),-700,700)))
    P.name = "P_trigger"

    # 空间易发性（沟道尺度）与集水区尺度的合成
    erod = static.get("松散物质可蚀性指数")
    s_catch = None if erod is None else float(erod) * (1.0 - 0.5 * static.get("陡坡面积比_gt30deg",np.nan))
    S_gully = float(gully["易发性"].median())

    res = {
        "模型形式": "P(t) = σ[k_T·(T(t) − 1)]，T(t) = Q_w(t)/Q_c",
        "状态变量": "触发指数 T",
        "T_事件时刻_10:00": T_ev,
        "显式先验斜率_kT": float(kT),
        "kT来源": "显式先验4，扫描1/2/4/8；没有用事件窗口拟合或标定",
        "kT敏感性": {str(k): {"score_gt05_hours":int((T>1).sum()),
             "score_gt08_hours":int((T>1+np.log(4)/k).sum())} for k in (1.,2.,4.,8.)},
        "使用边界": "事件窗口回顾系数k已退出正式模型；T/P仅作回顾相对强度评分，单事件不识别发生概率；空间S不与时间评分相乘",
        "集水区易发性": s_catch,
        "集水区评分状态": "unavailable: land-cover class semantics unverified" if s_catch is None else "conditional",
        "沟道易发性中位数": S_gully,
        "P最大值": float(np.nanmax(P)), "P最大时刻": str(P.idxmax()),
        "P超过0.5的小时数": int((P > 0.5).sum()),
        "P超过0.5的小时占比": float((P.dropna() > 0.5).mean()),
    }

    # 事件时刻复现：24 h 内 P 首次超过 0.5 的时刻（预警起点）
    around = P.loc[EVENT_WIN[0]:EVENT_WIN[1]]
    cross = around[around >= 0.5]
    if len(cross):
        res["事件窗口首次超过0.5"] = str(cross.index[0])
        res["相对事件时刻的超前_h"] = float((EVENT_TS - cross.index[0]).total_seconds() / 3600)
        res["事件窗口峰值时刻"] = str(around.idxmax())
        res["峰值相对事件滞后_h"] = float((around.idxmax() - EVENT_TS).total_seconds() / 3600)
    else:
        res["事件窗口首次超过0.5"] = None

    # 已发生灾点内规模排序检验，不是发生/未发生验证。
    # 判决规则在使用之前写定（建模成果/问题2.md §2.4.6）：
    #   AUC>0.5 显著 → 可作排序依据；≈0.5 → 无判别力；<0.5 → 方向错误，判定失效。
    # 任何已执行检验都必须报告，无论方向是否有利。
    g = gully.dropna(subset=["易发性", "规模序"]).copy()
    if len(g) >= 20:
        y = (g["规模序"] >= 3).astype(int).values
        if y.sum() >= 3 and (1 - y).sum() >= 3:
            auc_v = float(roc_auc_score(y, g["易发性"]))
            res["空间排序AUC_易发性预测大规模"] = auc_v
            # 置换检验：AUC 与 0.5 的偏离是否超出随机波动
            rng_s = np.random.default_rng(20260826)
            null = np.array([roc_auc_score(rng_s.permutation(y), g["易发性"])
                             for _ in range(2000)])
            p_two = float(np.mean(np.abs(null - 0.5) >= abs(auc_v - 0.5)))
            res["空间排序AUC_置换检验"] = {
                "AUC": auc_v, "置换次数": 2000, "双侧p值": p_two,
                "判读": ("可作排序依据" if auc_v > 0.5 and p_two < 0.05 else
                         "无判别力（与随机排序无异）" if p_two >= 0.05 else
                         "已发生灾点的大规模排序反向；不能推断发生概率方向"),
            }
        # 不拟合的样本内起动流量—规模序次关联。
        res["样本内规模Kendall_tau"] = float(stats.kendalltau(
            np.log10(g["Q_crit_m3s"]), g["规模序"]).statistic)
        res["规模关联_n"] = int(len(g))

    # 分级预警阈值
    res["分级阈值"] = {"蓝色": [0.2, 0.4], "黄色": [0.4, 0.6],
                    "橙色": [0.6, 0.8], "红色": [0.8, 1.0]}
    res["各级小时数"] = {
        "蓝色(0.2-0.4)": int(((P >= 0.2) & (P < 0.4)).sum()),
        "黄色(0.4-0.6)": int(((P >= 0.4) & (P < 0.6)).sum()),
        "橙色(0.6-0.8)": int(((P >= 0.6) & (P < 0.8)).sum()),
        "红色(≥0.8)": int((P >= 0.8).sum()),
    }
    pd.DataFrame({"P_trigger": P, "T_index": T}).to_parquet(io.CACHE / "ptrigger.parquet")
    return res


# ============================================================ S5 多尺度可靠性
def reliability(feat: pd.DataFrame, dis: pd.DataFrame) -> dict:
    T = (dis["Q_water"] / dis["Q_crit"].replace(0, np.nan)).astype(float)
    res: dict = {}

    # 时间尺度 1：聚合方式对检出率的影响
    res["时间聚合"] = {}
    for tag, rule in (("3h", "3h"), ("6h", "6h"), ("日", "D")):
        agg_max = T.resample(rule).max()
        agg_mean = T.resample(rule).mean()
        needed = 20 if rule=="D" else (3 if rule=="3h" else 6)
        available = T.resample(rule).count()>=needed
        agg_max=agg_max.where(available);agg_mean=agg_mean.where(available)
        res["时间聚合"][tag] = {
            "T≥1的窗口数": int((agg_max >= 1).sum()),
            "按峰值判据检出率": float((agg_max[available] >= 1).mean()),
            "按均值判据检出率": float((agg_mean[available] >= 1).mean()),
            "峰值保持率(T_agg/T_max)": float(agg_mean[available].sum() / max(agg_max[available].sum(), 1e-9)),
            "口径": "3h/6h须全窗有效，日>=20小时；同有效窗mean与max比", "有效窗口数":int(available.sum()),
        }

    # 时间尺度 2：事件回顾提前量与有效观测日报警频度。
    thr_grid = np.arange(0.2, 1.0, 0.1)
    rows = []
    for p0 in thr_grid:
        hit = bool((T.loc[EVENT_WIN[0]:EVENT_WIN[1]] >= p0).any())
        lead = None
        if hit:
            s = T.loc[EVENT_WIN[0]:EVENT_WIN[1]]
            first = s[s >= p0].index[0]
            lead = float((EVENT_TS - first).total_seconds() / 3600)
        day = daily_valid(T,"mean")
        peak = T.resample("D").max().where(day.notna())
        false_alarm = float((peak.dropna() >= p0).mean())
        rows.append({"阈值": float(p0), "命中事件": hit, "提前量_h": lead,
                     "全序列报警日率": false_alarm,"有效日数":int(peak.notna().sum())})
    res["提前量与报警日率"] = rows
    res["报警率说明"] = "仅阈值超越日频度，无确证非灾日不能报真实伪警率"

    # 时间尺度 3：事件回顾指数的年际峰值（不是留年拟合验证）。
    res["年际T峰值"] = {int(y): float(T[T.index.year == y].max()) for y in sorted(set(T.index.year))}

    # 空间尺度 1：DEM 分辨率对主沟比降与起动流量的影响
    # 注意：本项使用**追迹口径**，与 q_c 的正式口径（中心线）不同，两者只在本节
    # 并列出现（建模成果/问题2.md §2.4.2 的三口径表）。此处比降用于比较分辨率
    # 之间的相对偏差，不用于替换模型中的 q_c。
    from step02b_spatial import coarse_dem
    from mmlib.terrain_prep import cell_meters
    dem,grid = io.load_dem()
    sens,areas = {},{}
    for factor in (1,2,3,4,6):
        z,cgrid = coarse_dem(dem,grid,factor)
        dy,dx = cell_meters(cgrid)
        filled=tr.fill_sinks(z)
        padded=np.pad(filled,1,constant_values=np.nan)
        dd=np.zeros(z.shape,dtype=np.int8);steep=np.zeros(z.shape)
        for key,(dr,dc) in tr.DIRS.items():
            neighbour=padded[1+dr:1+dr+z.shape[0],1+dc:1+dc+z.shape[1]]
            slope=(filled-neighbour)/np.hypot(dr*dy,dc*dx)
            good=slope>steep
            dd[good],steep[good]=key,slope[good]
        aa=tr.flow_accumulation(dd)
        r,c=io.lonlat_to_rowcol(cgrid,io.EVENT["port_lon"],io.EVENT["port_lat"])
        candidates=np.argwhere((aa>=max(50.,5e5/factor**2))&np.isfinite(z))
        if not len(candidates):
            raise ValueError("no candidate outlet at coarse resolution")
        distances=np.hypot((candidates[:,0]-r)*dy,(candidates[:,1]-c)*dx)
        outlet=tuple(int(v) for v in candidates[np.argmin(distances)])
        stem=tr.trace_upstream_mainstem(dd,aa,outlet,min_acc=max(50.,2000./factor**2))
        coords=np.asarray(stem)
        length=float(np.hypot(np.diff(coords[:,0])*dy,np.diff(coords[:,1])*dx).sum())
        zz=np.array([z[v] for v in stem])
        ss=float((zz[-1]-zz[0])/length) if length>0 else np.nan
        sens[f"{factor}x"]={"像元_m":float(np.sqrt(dx*dy)),"dy_m":dy,"dx_m":dx,
                            "主沟像元数":len(stem),"主沟长度_km":length/1000,
                            "主沟比降":ss,"起动单宽流量_m2s":float(tg._q_crit_unit(ss)) if ss>0 else None,
                            "口岸投影偏移_m":float(distances.min()),"grid_center":cgrid}
        mask=tr.upstream_mask(dd,outlet)
        areas[f"{factor}x"]={"像元_m":float(np.sqrt(dx*dy)),"集水区像元数":int(mask.sum()),
                             "集水区面积_km2":float(mask.sum()*dx*dy/1e6)}
    base_slope=sens["1x"]["主沟比降"]
    res["空间分辨率敏感性"]=sens
    res["基准主沟比降"]=base_slope
    res["基准起动单宽流量_m2s"]=float(tg._q_crit_unit(base_slope))
    res["集水区面积敏感性"]=areas
    a1=areas["1x"]["集水区面积_km2"]
    res["集水区面积_相对偏差"]={k:(v["集水区面积_km2"]-a1)/a1 for k,v in areas.items() if k!="1x"}

    # 空间尺度 3：沟道属性外推的可迁移性（84 条沟道的比降分布 vs 口岸集水区）
    g = tg.gully_table()
    res["沟道迁移性"] = {
        "历史沟道比降分位": {q: float(g["比降"].quantile(q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "口岸集水区主沟比降": base_slope,
        "在历史分布中的分位": float((g["比降"] < base_slope).mean()),
        "历史沟道面积极值": [float(g["面积_km2"].min()), float(g["面积_km2"].max())],
    }
    return res


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    feat = pd.read_parquet(io.CACHE / "features.parquet")
    api_meta = tg.api_climate_thresholds(feat)
    feat.attrs["api_quantiles"] = (api_meta["q1"], api_meta["q3"])
    save_json(api_meta, "02_API分位.json")
    # 集水区特征由 step04 先行产出（它同时定义中心线口径）；此处若缺失则自行计算。
    if (OUT / "02_集水区特征.json").exists():
        info = json.loads((OUT / "02_集水区特征.json").read_text(encoding="utf-8"))
    else:
        info = tg.catchment_glacier_fraction()
        save_json(info, "02_集水区特征.json")
    model = tg.default_model(feat, calibrate=False)
    dis = model.discharge(feat)
    _event = dis.loc[FORMAL_EVENT_WIN[0]:FORMAL_EVENT_WIN[1]]
    _peak = float(_event["T_index"].max()) if _event["T_index"].notna().any() else np.nan
    legacy_candidates = [
        io.ROOT / "临时产物" / "审稿记录" / "冻结父版v5.1记录" / "v43_旧触发标定审计.json",
        io.ROOT / "临时产物" / "审稿记录" / "v43_旧触发标定审计.json",
    ]
    legacy_path = next((p for p in legacy_candidates if p.exists()), legacy_candidates[0])
    legacy_doc = json.loads(legacy_path.read_text(encoding="utf-8")) if legacy_path.exists() else {}
    legacy = legacy_doc.get("value", {})
    save_json({
        "status": "audit_only",
        "legacy_record": legacy_path.relative_to(io.ROOT).as_posix() if legacy_path.exists() else None,
        "legacy_record_sha256": (
            __import__("hashlib").sha256(legacy_path.read_bytes()).hexdigest()
            if legacy_path.exists() else None
        ),
        "formal_model_uses_legacy_k": False,
        "formal_mechanical_regime": model.mechanical_regime,
        "formal_q_crit_m2s": model.q_crit,
        "formal_C_N0": model.cn0,
        "formal_CN_provenance": (
            "explicit engineering scenario independent of land-cover classes; "
            "75/85/95 sensitivity reported"
        ),
        "formal_temperature_geometry": model.temperature_geometry,
        "legacy_k": legacy.get("标定系数k"),
        "legacy_calibration_note": "v4.3 event-window k remains audit-only and must not be read downstream",
        "event_window_T_peak": _peak,
        "event_window_T_is_formal": bool(model.mechanical_regime == "water_controlled" and _peak == _peak),
    }, "02_旧方法审计.json")
    print(f"触发指数序列 ok（正式状态 {model.mechanical_regime}，事件窗 T 峰值 {_peak}，"
          f"q_c={model.q_crit:.2f} m2/s；旧 k 只审计）")

    gc = gully_critical_conditions()
    save_json(gc, "02_沟道临界条件.json")
    print(f"沟道临界条件 ok：{gc['沟道数']} 条，起动流量中位数 "
          f"{gc['起动流量_m3s']['p50']:.3g} m3/s")

    idr = id_threshold(feat)
    save_json(idr, "02_降雨阈值.json")
    print(f"降雨 I–D: I = {idr['幂律系数a']:.2f} D^(-{idr['幂指数b']:.2f})，"
          f"{idr['独立暴雨事件数']} 场独立暴雨")

    mtl = melt_threshold(feat, info)
    save_json(mtl, "02_消融阈值.json")
    print(f"消融阈值 ok：源区订正 {mtl['源区气温订正_℃']:.1f} ℃，"
          f"源区日融化 p90 {mtl['临界条件']['E_s_p90_MJm-2']:.2f} MJ/m2")

    static = json.loads((OUT / "02_静态特征.json").read_text(encoding="utf-8")) \
        if (OUT / "02_静态特征.json").exists() else {}
    gully = pd.read_parquet(io.CACHE / "gully_table.parquet")

    hourly_tuple, object_tuple, tuple_summary = build_state_tuple(
        feat, dis, model, mtl, gully
    )
    hourly_tuple.to_csv(
        OUT / "02_状态元组_逐小时.csv", encoding="utf-8-sig", index=True
    )
    object_tuple.to_csv(
        OUT / "02_状态元组_逐沟道.csv", encoding="utf-8-sig", index=False
    )
    save_json(tuple_summary, "02_状态元组.json")

    tp = trigger_probability(feat, dis, static, gully)
    tp["曲线数情景"] = cn_scenario_sensitivity(feat, model)
    tp["API_q1_q3"] = api_meta
    tp["状态元组"] = tuple_summary
    save_json(tp, "02_触发评估模型.json")
    print(f"触发评估模型 ok：P 最大 {tp['P最大值']:.3f} @ {tp['P最大时刻']}，"
          f"事件窗口首次超 0.5 = {tp.get('事件窗口首次超过0.5')}")

    rel = reliability(feat, dis)
    save_json(rel, "02_多尺度可靠性.json")
    save_json(business_screen(feat,idr), "02_水源业务判据.json")
    save_json(statistical_domains(), "02_统计临界域.json")
    save_json(long_temperature_check(feat), "02_长期气温对比.json")
    save_json(weather_baseline(feat), "02_气象基线检验.json")
    print("多尺度可靠性 ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
