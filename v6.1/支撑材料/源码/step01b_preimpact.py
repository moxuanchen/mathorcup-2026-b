"""Frozen pre-impact window-anomaly evidence; no water-source or disaster labels."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

PROJECT = Path(__file__).resolve().parents[2]
if "JILONG_ROOT" not in os.environ:
    os.environ["JILONG_ROOT"] = str(PROJECT)
from mmlib import io_data as io, meteo as mt  # noqa: E402

TRAIN_YEARS = (2021, 2022, 2023, 2024)
TARGET = pd.Timestamp("2026-08-26")
LENGTHS = (168, 72)
SEED = 20260826
N_BOOT = 200
COVERAGE = .8
CALENDAR_RADIUS = 15
CANDIDATES = (
    ("WS_10m", "10米风速", "m/s", "未直接入水源式"),
    ("vector_shear_2_20m", "2—20米矢量风切变", "1/s", "未直接入水源式"),
    ("Pressure", "MET气压", "hPa", "未直接入水源式"),
    ("VWC_0.1m", "0.1米土壤含水率", "m3/m3", "未直接入水源式"),
    ("VWC_0.4m", "0.4米土壤含水率", "m3/m3", "未直接入水源式"),
    ("soilT_0.1m", "0.1米土温", "degC", "未直接入水源式"),
    ("ustar", "摩擦速度", "m/s", "未直接入水源式"),
    ("TKE_obs", "观测湍流动能", "m2/s2", "未直接入水源式"),
    ("Ri_b", "订正整体理查森数", "1", "共享温湿廓线"),
    ("dtheta_1_20", "订正1—20米位温差", "K", "共享温湿廓线"),
    ("inv_max", "实际温度逆温强度", "K/100m", "共享温湿廓线"),
    ("RH_2m", "2米相对湿度", "%", "共享温湿廓线"),
)
COLS = [v[0] for v in CANDIDATES]
NAMES = {v[0]: v[1] for v in CANDIDATES}
UNITS = {v[0]: v[2] for v in CANDIDATES}
ORDER = {c: i for i, c in enumerate(COLS)}
LIMITATION = (
    "异常是同季节已选历史窗口中的特殊性，不是灾害关联系数、发生概率或因果效应。"
    "小时标签起止和仪器延迟未知；仅称冲击锚点前可用，不保证早于源区失稳。"
    "每日窗口重叠、参照自参与尺度拟合；配置训练年4年，但2022年因ustar与TKE_obs缺测无共同有效窗，"
    "实际参照年为2021/2023/2024三年；尾秩非显著性p值，年块区间非独立灾害置信区间，2025伪事件不是真实非灾日。"
)


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def calendar_day(dates):
    """A common 365-day calendar; Feb 29 is never a release/query date."""
    dates = pd.DatetimeIndex(dates)
    if ((dates.month == 2) & (dates.day == 29)).any():
        raise ValueError("Feb 29 is not a query date")
    return np.array([pd.Timestamp(2001, d.month, d.day).dayofyear for d in dates])


def nearby(dates, query, radius=CALENDAR_RADIUS):
    delta = np.abs(calendar_day(dates) - calendar_day([query])[0])
    return np.minimum(delta, 365 - delta) <= radius


def all_release_dates():
    dates = pd.date_range("2021-01-01", "2025-12-31", freq="D")
    dates = dates[~((dates.month == 2) & (dates.day == 29))]
    return dates.append(pd.DatetimeIndex([TARGET]))


def candidate_features(features):
    f = features.copy()
    required = ["WS_2m", "WS_20m", "WD_2m", "WD_20m"]
    if not set(required).issubset(f):
        raise ValueError("vector wind shear needs measured wind speed/direction at 2 and 20 m")
    a2, a20 = np.deg2rad(f["WD_2m"]), np.deg2rad(f["WD_20m"])
    du = -f["WS_20m"] * np.sin(a20) + f["WS_2m"] * np.sin(a2)
    dv = -f["WS_20m"] * np.cos(a20) + f["WS_2m"] * np.cos(a2)
    f["vector_shear_2_20m"] = np.hypot(du, dv) / 18.
    missing = [c for c in COLS if c not in f]
    if missing:
        raise ValueError(f"frozen candidate fields missing: {missing}")
    return f[COLS].replace([np.inf, -np.inf], np.nan)


def window_table(features, length, releases):
    releases = pd.DatetimeIndex(releases)
    if (releases.hour != 0).any():
        raise ValueError("release time is fixed at midnight")
    end = releases - pd.Timedelta(hours=2)
    start = end - pd.Timedelta(hours=length - 1)
    grid = pd.date_range(min(start.min(), features.index.min()),
                         max(end.max(), features.index.max()), freq="h")
    f = features.reindex(grid)[COLS]
    means = f.rolling(length, min_periods=1).mean().reindex(end)
    counts = f.notna().rolling(length, min_periods=1).sum().reindex(end).fillna(0)
    means.index, counts.index = releases, releases
    threshold = int(np.ceil(length * COVERAGE))
    out = means.copy()
    out["common_valid"] = (counts >= threshold).all(axis=1)
    out["start"] = start
    out["end"] = end
    out["length_h"] = length
    for c in COLS:
        out[f"n__{c}"] = counts[c].astype(int)
    complete = f.notna().all(axis=1).rolling(length, min_periods=1).sum().reindex(end)
    out["all12_complete_hours"] = complete.to_numpy(dtype=int)
    out.index.name = "release"
    return out


def connected_clusters(features):
    warm = features.loc[features.index.year.isin(TRAIN_YEARS)
                        & features.index.month.isin([6, 7, 8, 9]), COLS].dropna()
    corr = warm.corr(method="spearman", min_periods=200)
    unknown = [(a, b) for i, a in enumerate(COLS) for b in COLS[i + 1:]
               if not np.isfinite(corr.loc[a, b])]
    edges = [(a, b, float(corr.loc[a, b])) for i, a in enumerate(COLS)
             for b in COLS[i + 1:]
             if np.isfinite(corr.loc[a, b]) and abs(corr.loc[a, b]) > .8]
    confirmed = len(warm) >= 200 and not unknown
    groups = components_from_edges(edges) if confirmed else []
    hours_by_year = {int(y): int(n) for y, n in warm.groupby(warm.index.year).size().items()}
    effective = tuple(sorted(hours_by_year))
    excluded = [int(y) for y in TRAIN_YEARS if int(y) not in effective]
    training = (f"相关聚类用2021–2024年6–9月十二项共同有效小时{len(warm)}小时"
                f"（分年{hours_by_year}）；窗口参照年及2022年缺测情况见training_year_audit"
                "（窗口要求80%共同覆盖，与逐时相关聚类不是同一资格）")
    if excluded:
        training += f"；聚类未参与年{','.join(map(str, excluded))}"
    return {"confirmed": confirmed, "common_hours": len(warm),
            "training": training, "hours_by_year": hours_by_year,
            "hourly_cluster_years": list(effective), "excluded_years": excluded,
            "method": "Spearman绝对相关>0.8完整连通簇；至少200共同小时",
            "unknown_pairs": unknown, "edges": edges, "clusters": groups,
            "correlation": corr.to_dict(),
            "missing_policy": "缺相关不填零；结构不可确认时不声称去冗余前五"}


def components_from_edges(edges):
    adjacent = {c: set() for c in COLS}
    for a, b, *_ in edges:
        adjacent[a].add(b)
        adjacent[b].add(a)
    seen, groups = set(), []
    for c in COLS:
        if c in seen:
            continue
        todo, component = [c], []
        while todo:
            v = todo.pop()
            if v not in seen:
                seen.add(v)
                component.append(v)
                todo.extend(adjacent[v] - seen)
        groups.append(sorted(component, key=ORDER.get))
    return groups


def robust_baseline(reference):
    center = reference.median()
    mad = reference.sub(center).abs().median() * 1.4826
    iqr = (reference.quantile(.75) - reference.quantile(.25)) / 1.349
    scale = mad.where(mad > 0, iqr)
    valid = np.isfinite(scale) & (scale > 0)
    method = pd.Series("MAD×1.4826", index=COLS)
    method.loc[~(mad > 0)] = "IQR/1.349回退"
    method.loc[~valid] = "不可排名：MAD及IQR均无有效正尺度"
    return center, scale.where(valid), method


def reference_rows(windows, query, years=TRAIN_YEARS):
    mask = windows.index.year.isin(years) & nearby(windows.index, query)
    return windows.loc[mask]


def effective_years(reference):
    """Training years that actually contribute a common-valid reference window."""
    valid = reference.loc[reference.common_valid]
    return tuple(sorted(int(y) for y in valid.index.year.unique()))


def year_window_audit(windows):
    """Per-length, per-year reference availability and the features that fail coverage."""
    audit = {}
    for length, table in windows.items():
        threshold = int(np.ceil(length * COVERAGE))
        ref = reference_rows(table, TARGET)
        by_year = {}
        for year in TRAIN_YEARS:
            sub = ref.loc[ref.index.year == int(year)]
            failed = sorted({c for _, row in sub.iterrows()
                             for c in COLS if int(row[f"n__{c}"]) < threshold})
            by_year[str(int(year))] = {
                "reference_rows": int(len(sub)),
                "common_valid_rows": int(sub.common_valid.sum()),
                "failed_features": failed,
            }
        audit[str(length)] = by_year
    return audit


def score_window(windows, query, clusters, reference=None):
    query = pd.Timestamp(query)
    row = windows.loc[query]
    rawref = reference_rows(windows, query) if reference is None else reference
    ref = rawref.loc[rawref.common_valid, COLS]
    summary = {"release": query, "start": row.start, "end": row.end,
               "length_h": int(row.length_h), "common_valid": bool(row.common_valid),
               "candidate_reference_windows": len(rawref), "reference_windows": len(ref),
               "reference_counts_by_year": ref.groupby(ref.index.year).size().to_dict(),
               "coverage": {c: int(row[f"n__{c}"]) / int(row.length_h) for c in COLS},
               "all12_complete_hours": int(row.all12_complete_hours)}
    if not row.common_valid or len(ref) < 2:
        summary.update(status="不可评估：目标共同覆盖不足或参照少于2窗", rows=[],
                       full_order=[], display=[], top5=[], maximum_A=np.nan)
        return summary
    center, scale, method = robust_baseline(ref)
    delta = row[COLS].astype(float) - center
    z, amplitude = delta / scale, (delta / scale).abs()
    refa = ref.sub(center).div(scale).abs()
    rows = []
    for c in COLS:
        ok = np.isfinite(amplitude[c])
        tail = (1 + int((refa[c] >= amplitude[c]).sum())) / (len(ref) + 1) if ok else np.nan
        rows.append({"feature": c, "name": NAMES[c], "unit": UNITS[c],
                     "mean": float(row[c]), "baseline_median": center[c],
                     "robust_scale": scale[c], "scale_method": method[c],
                     "delta": delta[c], "Z": z[c], "A": amplitude[c],
                     "tail_rank": tail, "rankable": ok})
    order = sorted([r for r in rows if r["rankable"]],
                   key=lambda r: (-r["A"], ORDER[r["feature"]]))
    place = {r["feature"]: i + 1 for i, r in enumerate(order)}
    for r in rows:
        r["full_rank"] = place.get(r["feature"])
    display = []
    for i, component in enumerate(clusters, 1):
        available = [r for r in order if r["feature"] in component]
        if available:
            winner = available[0]
            display.append({**winner, "cluster_id": i, "members": component,
                            "excluded": [c for c in component if c != winner["feature"]]})
    display.sort(key=lambda r: (-r["A"], ORDER[r["feature"]]))
    summary.update(status="可评估" if clusters else "完整得分可评估；相关簇不可确认",
                   rows=rows, full_order=[r["feature"] for r in order], display=display,
                   top5=[r["feature"] for r in display[:5]],
                   maximum_A=max((r["A"] for r in order), default=np.nan))
    return summary


def compare_scores(left, right):
    common = [c for c in left["full_order"] if c in right["full_order"]]
    tau = float(kendalltau([left["full_order"].index(c) for c in common],
                          [right["full_order"].index(c) for c in common]).statistic) if len(common) > 1 else np.nan
    return {"common_ranked_features": len(common), "full_rank_Kendall_tau": tau,
            "top5_overlap": len(set(left["top5"]) & set(right["top5"]))}


def score_records(result, method, **extra):
    return [{"method": method, "release": result["release"], "length_h": result["length_h"],
             "reference_windows": result["reference_windows"], **extra, **r}
            for r in result["rows"]]


def seven_day_reference(reference):
    selected = []
    for _, yearly in reference.groupby(reference.index.year):
        # Calendar scheduling precedes validity screening, so gaps never shift the schedule.
        selected.extend(pd.date_range(yearly.index.min(), yearly.index.max(), freq="7D"))
    return reference.loc[reference.index.isin(selected)]


def background_sensitivity(windows, target, clusters):
    reference = reference_rows(windows, TARGET)
    valid_reference = reference.loc[reference.common_valid]
    available_years = effective_years(reference)
    excluded_years = {str(int(y)): "该年无共同有效参照窗（ustar/TKE_obs缺测）"
                      for y in TRAIN_YEARS if int(y) not in available_years}
    rng = np.random.default_rng(SEED + int(target["length_h"]))
    replicates, records, choices = [], [], []
    for i in range(N_BOOT):
        if len(available_years) >= 2:
            selected = rng.choice(available_years, len(available_years), replace=True)
            sampled = pd.concat([valid_reference.loc[valid_reference.index.year == y] for y in selected])
            result = score_window(windows, TARGET, clusters, sampled)
            choices.append([int(y) for y in selected])
        else:
            result = {"release": TARGET, "length_h": int(target["length_h"]), "rows": [],
                      "full_order": [], "display": [], "top5": [], "maximum_A": np.nan,
                      "reference_windows": 0, "status": "不可评估：有效训练年少于2"}
            choices.append([])
        replicates.append(result)
        records.extend(score_records(result, "year_block_bootstrap", replicate=i))
    intervals = []
    for c in COLS:
        rows = [next((r for r in v["rows"] if r["feature"] == c), {}) for v in replicates]
        valid = [r for r in rows if r.get("rankable", False)]
        intervals.append({"feature": c, "valid_replicates": len(valid),
                          "intervals_025_50_975": {
                              key: np.quantile([r[key] for r in valid], [.025, .5, .975]).tolist()
                              if valid else [np.nan] * 3
                              for key in ("A", "Z", "delta", "baseline_median", "robust_scale", "tail_rank")},
                          "top5_selection_fraction": sum(c in v["top5"] for v in replicates) / N_BOOT})
    loyo = []
    for year in TRAIN_YEARS:
        year = int(year)
        if year not in available_years:
            loyo.append({"omitted_year": year, "status": "不可评估：该训练年无共同有效参照窗",
                         "reference_windows": 0, "top5": [], "common_ranked_features": 0,
                         "full_rank_Kendall_tau": None, "top5_overlap": None,
                         "reason": excluded_years[str(year)]})
            continue
        r = score_window(windows, TARGET, clusters, reference.loc[reference.index.year != year])
        records.extend(score_records(r, "leave_one_training_year_out", omitted_year=year))
        loyo.append({"omitted_year": year, "status": r["status"],
                     "reference_windows": r["reference_windows"], "top5": r["top5"],
                     **compare_scores(target, r)})
    seven = None
    if int(target["length_h"]) == 168:
        scheduled = seven_day_reference(reference)
        r = score_window(windows, TARGET, clusters, scheduled)
        records.extend(score_records(r, "fixed_seven_day_nonoverlap"))
        seven = {**r, "scheduled_dates_before_validity": scheduled.index.tolist(),
                 "comparison_with_daily": compare_scores(target, r)}
    return {"year_block_bootstrap": {"replicates": N_BOOT, "seed": SEED + int(target["length_h"]),
                                     "status": "可评估" if len(available_years) >= 2 else "不可评估：有效训练年少于2",
                                     "available_training_years": [int(y) for y in available_years],
                                     "excluded_training_years": excluded_years,
                                     "sampled_training_years": choices, "feature_intervals": intervals,
                                     "scope": "整年重抽样仅从有效训练年重估参照中位数与尺度；相关簇不重拟合；非灾害置信区间"},
            "leave_one_training_year_out": loyo, "fixed_seven_day_nonoverlap": seven}, records


def evaluate(features):
    f = candidate_features(features)
    dates = all_release_dates()
    windows = {length: window_table(f, length, dates) for length in LENGTHS}
    effective_by_length = {str(length): [int(y) for y in effective_years(reference_rows(table, TARGET))]
                           for length, table in windows.items()}
    cluster_info = connected_clusters(f)
    window_year_text = "，".join(f"{L}h=" + "/".join(map(str, years))
                                 for L, years in effective_by_length.items())
    cluster_info["window_reference_years_by_length"] = effective_by_length
    cluster_info["training"] += (f"；窗口参照有效年{window_year_text}，"
                                 "2022年因80%共同覆盖为0不进入窗口参照")
    clusters = cluster_info["clusters"]
    pseudo_dates = dates[(dates.year == 2025) & nearby(dates, TARGET)]
    target_results, pseudo, sensitivities, records, audits, used_dates = {}, {}, {}, [], [], {}
    for length, table in windows.items():
        target = score_window(table, TARGET, clusters)
        target_results[str(length)] = target
        pseudo_results = [score_window(table, d, clusters) for d in pseudo_dates]
        records.extend(score_records(target, "target"))
        rows = []
        for r in pseudo_results:
            comparison = compare_scores(target, r)
            records.extend(score_records(r, "pseudo_2025"))
            rows.append({k: r[k] for k in ("release", "status", "common_valid", "reference_windows", "top5", "maximum_A")}
                        | comparison)
        pseudo[str(length)] = {"candidate_days": len(pseudo_dates),
                               "common_valid_days": sum(r["common_valid"] for r in pseudo_results),
                               "evaluable_days": sum(bool(r["rows"]) for r in pseudo_results),
                               "days": rows,
                               "top5_selection_fraction": {c: sum(c in r["top5"] for r in pseudo_results)
                                                              / max(1, sum(bool(r["rows"]) for r in pseudo_results))
                                                           for c in COLS},
                               "scope": "2025只作冻结规则外检；不是有真值的非灾日，不计算伪警率"}
        sensitivity, more = background_sensitivity(table, target, clusters)
        sensitivities[str(length)] = sensitivity
        records.extend(more)
        used = set(pseudo_dates) | {TARGET}
        for d in list(pseudo_dates) + [TARGET]:
            used.update(reference_rows(table, d).index)
        used_dates[length] = used
        selected = table.loc[table.index.isin(used)]
        for release, r in selected.iterrows():
            audits.append({"release": release, "length_h": length, "start": r.start, "end": r.end,
                           "role": "target" if release == TARGET else "pseudo_2025" if release.year == 2025 else "training_reference",
                           "common_valid": bool(r.common_valid), "all12_complete_hours": int(r.all12_complete_hours),
                           "failed_features": ";".join(c for c in COLS if r[f"n__{c}"] < np.ceil(length * COVERAGE)),
                           **{f"n__{c}": int(r[f"n__{c}"]) for c in COLS}})
    common_dates_results = []
    for d in [TARGET] + list(pseudo_dates):
        refs = [reference_rows(windows[length], d) for length in LENGTHS]
        shared = refs[0].index[refs[0].common_valid].intersection(refs[1].index[refs[1].common_valid])
        result = {length: score_window(windows[length], d, clusters, refs[i].loc[shared])
                  for i, length in enumerate(LENGTHS)}
        both = all(r["common_valid"] and bool(r["rows"]) for r in result.values())
        for r in result.values():
            records.extend(score_records(r, "common_reference_dates", both_durations_evaluable=both))
        common_dates_results.append({"release": d, "both_durations_evaluable": both,
                                     "shared_reference_windows": len(shared),
                                     **compare_scores(result[168], result[72]),
                                     "top5_168": result[168]["top5"], "top5_72": result[72]["top5"],
                                     "target_scores" if d == TARGET else "pseudo_scores_omitted": result if d == TARGET else None})
    effective_union = {y for years in effective_by_length.values() for y in years}
    year_audit = year_window_audit(windows)
    summary = {"schema_version": 1, "module": "step01b_preimpact", "seed": SEED,
               "frozen_protocol": {"target_release": TARGET, "last_usable_hour": TARGET - pd.Timedelta(hours=2),
                                   "unused_buffer_label": TARGET - pd.Timedelta(hours=1),
                                   "main_length_h": 168, "sensitivity_length_h": 72,
                                   "training_years": TRAIN_YEARS, "pseudo_year": 2025,
                                   "calendar": "共同非闰年日历±15日；2月29日不作为发布日",
                                   "coverage_per_candidate": COVERAGE, "all_candidates_must_pass": True,
                                   "imputation": "不插补；每项窗口均值仅用该项有效小时，十二项均达80%才接纳",
                                   "scoring": "训练窗口中位数与MAD×1.4826；零MAD仅回退IQR/1.349；A=|Z|降序，固定候选顺序决胜",
                                   "tail_rank": "(1+参照幅度>=查询幅度的窗数)/(Nref+1)，不参与加权",
                                   "labels_or_priors_used": False,
                                   "candidates": [{"feature": c, "name": name, "unit": unit, "relationship": relationship}
                                                  for c, name, unit, relationship in CANDIDATES]},
               "clusters": cluster_info,
               "training_year_audit": {
                   "configured_training_years": [int(y) for y in TRAIN_YEARS],
                   "effective_training_years_by_length": effective_by_length,
                   "excluded_training_years": {
                       str(int(y)): "该年无共同有效参照窗（ustar/TKE_obs缺测）"
                       for y in TRAIN_YEARS if int(y) not in effective_union},
                   "by_length": year_audit,
                   "policy": "留一年仅对有效年份执行；无窗年份标为不可评估，不用空操作结果冒充敏感性；年块自举仅从有效年份重抽样"},
               "target": target_results, "pseudo_2025": pseudo,
               "sensitivity": sensitivities,
               "duration_comparison": {"independently_screened": compare_scores(target_results["168"], target_results["72"]),
                                       "common_reference_dates": common_dates_results,
                                       "scope": "共同日期对照对每个查询分别取两长度均合格的训练发布日交集；2025查询也报告两长度共同资格"},
               "observed_flux_policy": "ustar与TKE_obs来自FLUX观测统计量；仅屏蔽负值，不插补、不作未经元数据支持的上界饱和截尾；测量高度未知，不补为10米或20米",
               "conditional_selection": "窗口审计CSV列出全部实际查询/参照小时起止与各项计数，含被共同覆盖规则排除的窗。FLUX缺测可使可用窗成为条件选样，数量不等于独立事件数。",
               "feature_metadata": serial(features.attrs), "limitations": LIMITATION}
    scores = pd.DataFrame(records)
    for c in ("method", "release", "length_h", "feature", "A", "omitted_year"):
        if c not in scores:
            scores[c] = pd.Series(index=scores.index, dtype="object" if c in ("method", "feature") else "float64")
    return summary, scores, pd.DataFrame(audits), windows


def plot_outputs(summary, records, directory):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    directory.mkdir(parents=True, exist_ok=True)
    labels = [NAMES[c] for c in COLS]
    y = np.arange(len(COLS))
    target = summary["target"]["168"]
    target_rows = {r["feature"]: r for r in target["rows"]}
    if not target_rows:
        paths = []
        for name in ("01b_冲击前窗口异常.png", "01b_伪事件与背景敏感性.png"):
            fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
            ax.axis("off")
            ax.text(.5, .65, "冻结冲击前168小时旁证不可评估", ha="center", fontsize=20)
            ax.text(.5, .45, target["status"], ha="center", fontsize=13)
            ax.text(.5, .25, "不以缺测代表平缓，不改候选或截止时刻；逐窗覆盖详见审计CSV", ha="center", fontsize=12)
            path = directory / name
            fig.savefig(path, dpi=170); plt.close(fig)
            paths.append(path)
        return paths
    a = np.array([target_rows.get(c, {}).get("A", np.nan) for c in COLS])
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    colors = ["#236a72" if c in target["top5"] else "#a9c8cc" for c in COLS]
    axes[0, 0].barh(y, a, color=colors)
    axes[0, 0].set_yticks(y, labels); axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("双侧异常幅度 A=|Z|（无量纲）")
    axes[0, 0].set_title("(a) 冲击前168小时完整十二候选；深色为去冗余前五")
    common = summary["duration_comparison"]["common_reference_dates"][0]["target_scores"]
    for length, offset, color in ((168, -.17, "#236a72"), (72, .17, "#c87243")):
        rows = {r["feature"]: r for r in common[length]["rows"]}
        axes[0, 1].barh(y + offset, [rows.get(c, {}).get("A", np.nan) for c in COLS],
                       height=.32, color=color, label=f"{length}小时")
    axes[0, 1].set_yticks(y, labels); axes[0, 1].invert_yaxis(); axes[0, 1].legend()
    axes[0, 1].set_xlabel("同一训练发布日交集上的 A")
    axes[0, 1].set_title("(b) 时长敏感性：同参照日期，不挑有利窗口")
    corr = pd.DataFrame(summary["clusters"]["correlation"]).reindex(index=COLS, columns=COLS)
    image = axes[1, 0].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[1, 0].set_xticks(y, labels, rotation=65, ha="right", fontsize=8)
    axes[1, 0].set_yticks(y, labels, fontsize=8)
    axes[1, 0].set_title(f"(c) 冻结暖季Spearman簇：共同小时 {summary['clusters']['common_hours']}")
    fig.colorbar(image, ax=axes[1, 0], shrink=.75, label="Spearman相关")
    for length, offset, color in ((168, -.17, "#236a72"), (72, .17, "#c87243")):
        coverage = summary["target"][str(length)]["coverage"]
        axes[1, 1].barh(y + offset, [coverage[c] for c in COLS], height=.32,
                       color=color, label=f"{length}小时")
    axes[1, 1].axvline(.8, color="#9b352d", ls="--", label="每项80%门槛")
    axes[1, 1].set_yticks(y, labels); axes[1, 1].invert_yaxis(); axes[1, 1].set_xlim(0, 1.04)
    axes[1, 1].set_xlabel("目标窗有效小时比例"); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("(d) 十二候选全部合格才接纳；FLUX缺测可能引入选样")
    fig.suptitle("冻结非循环窗口异常旁证｜2026-08-26 00时发布，前日22时截止\n异常特殊性不是灾害关联或概率；共享观测不等于独立证据", fontsize=13)
    p1 = directory / "01b_冲击前窗口异常.png"
    fig.savefig(p1, dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    pseudo = records[(records.method == "pseudo_2025") & (records.length_h == 168)]
    dates = pd.DatetimeIndex([v["release"] for v in summary["pseudo_2025"]["168"]["days"]])
    matrix = pseudo.pivot(index="feature", columns="release", values="A").reindex(index=COLS, columns=dates)
    image = axes[0, 0].imshow(matrix, aspect="auto", cmap="YlOrBr")
    ticks = np.arange(0, len(dates), 5)
    axes[0, 0].set_xticks(ticks, [dates[i].strftime("%m-%d") for i in ticks], rotation=35)
    axes[0, 0].set_yticks(y, labels, fontsize=8)
    axes[0, 0].set_title("(a) 2025同季节伪事件：冻结规则异常幅度")
    fig.colorbar(image, ax=axes[0, 0], shrink=.8, label="A（不是灾害概率）")
    intervals = summary["sensitivity"]["168"]["year_block_bootstrap"]["feature_intervals"]
    by_feature = {r["feature"]: r for r in intervals}
    for i, c in enumerate(COLS):
        lo, mid, hi = by_feature[c]["intervals_025_50_975"]["A"]
        axes[0, 1].hlines(i, lo, hi, color="#236a72", lw=3)
        axes[0, 1].plot(mid, i, "|", color="#236a72", ms=10)
    axes[0, 1].scatter(a, y, c="#bf613d", s=24, label="完整训练背景点值", zorder=3)
    axes[0, 1].set_yticks(y, labels); axes[0, 1].invert_yaxis(); axes[0, 1].legend()
    effective_168 = summary["training_year_audit"]["effective_training_years_by_length"]["168"]
    axes[0, 1].set_title(f"(b) 200次整年块背景敏感性：2.5%—97.5%端点（有效训练年{effective_168}）")
    axes[0, 1].set_xlabel("A；仅有效训练年，不是灾害置信区间")
    axes[1, 0].barh(y, [by_feature[c]["top5_selection_fraction"] for c in COLS],
                    color="#a9c8cc", label="年块重抽样")
    axes[1, 0].scatter([summary["pseudo_2025"]["168"]["top5_selection_fraction"][c] for c in COLS],
                       y, color="#bf613d", s=25, label="2025有效伪事件")
    axes[1, 0].set_yticks(y, labels); axes[1, 0].invert_yaxis(); axes[1, 0].set_xlim(0, 1.05)
    axes[1, 0].set_xlabel("进入去冗余前五的比例（非灾害频率）")
    axes[1, 0].set_title("(c) 前五稳定性而非灾害鉴别力"); axes[1, 0].legend(fontsize=8)
    leave = records[(records.method == "leave_one_training_year_out") & (records.length_h == 168)]
    palette = {2021: "#236a72", 2022: "#c87243", 2023: "#687c45", 2024: "#8e6685"}
    for year in effective_168:
        rows = leave[leave.omitted_year == year].set_index("feature").reindex(COLS)
        axes[1, 1].plot(rows.A, y, ".", color=palette.get(year, "#777777"), label=f"剔除{year}年")
    seven = summary["sensitivity"]["168"]["fixed_seven_day_nonoverlap"]
    rows = {r["feature"]: r for r in seven["rows"]}
    axes[1, 1].scatter([rows.get(c, {}).get("A", np.nan) for c in COLS], y,
                       marker="x", color="black", s=27, label="固定7日非重叠参照")
    axes[1, 1].set_yticks(y, labels); axes[1, 1].invert_yaxis(); axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_title("(d) 逐训练年留一与非重叠参照（2022无共同有效窗，不可评估）")
    axes[1, 1].set_xlabel("A：同一规则，不挑最有利榜单")
    fig.suptitle("伪事件外检与有限历史背景敏感性｜伪事件不是已证实非灾日\n尾秩不是显著性p值；每日窗口重叠，窗口数不是独立事件数", fontsize=13)
    p2 = directory / "01b_伪事件与背景敏感性.png"
    fig.savefig(p2, dpi=170); plt.close(fig)
    return [p1, p2]


def main():
    if io.ROOT.resolve() != PROJECT:
        raise RuntimeError(f"JILONG_ROOT must point to this v2 project, not {io.ROOT}")
    features = mt.build_feature_table(io.load_station())
    summary, records, audit, _ = evaluate(features)
    paths = {
        "summary": io.DATA_OUT / "01b_冲击前窗口异常.json",
        "audit": io.DATA_OUT / "01b_窗口覆盖审计.csv",
        "query_scores": io.DATA_OUT / "01b_查询逐项异常.csv",
        "sensitivity_scores": io.DATA_OUT / "01b_背景敏感性逐项.csv",
    }
    audit.to_csv(paths["audit"], index=False, encoding="utf-8-sig")
    query = records.method.isin(["target", "pseudo_2025", "common_reference_dates"])
    records.loc[query].to_csv(paths["query_scores"], index=False, encoding="utf-8-sig")
    records.loc[~query].to_csv(paths["sensitivity_scores"], index=False, encoding="utf-8-sig")
    figures = plot_outputs(summary, records, io.FIG_OUT)
    summary["outputs"] = {**{k: str(v.relative_to(PROJECT)) for k, v in paths.items()},
                          "figures": [str(p.relative_to(PROJECT)) for p in figures]}
    paths["summary"].write_text(json.dumps(serial(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    for length in LENGTHS:
        target = summary["target"][str(length)]
        print(f"01b {length}h: {target['status']}; reference={target['reference_windows']}; top5={target['top5']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
