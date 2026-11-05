"""问题一正式候选池：冲击前环境关联的冻结单事件排名。

协议在查看本轮排名前冻结：

* 31 个唯一候选键，MET/RADM/SOIL/FLUX = 19/5/3/4；
* 目标窗为事件前 168 h，72 h 只作敏感性；
* 逐候选目标窗和参照窗均至少 80% 有效；
* 评分用参照中位数与 MAD×1.4826，零 MAD 回退 IQR/1.349；
* 相关去冗余用 2021--2024 暖季逐对 Spearman，
  每对至少 200 个共同有效小时，|rho|>0.8 连边后取连通簇。

输出是单事件环境特殊性关联，不是因果效应、发生概率或独立灾害验证。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 23,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 15,
})

PROJECT = Path(__file__).resolve().parents[2]
OUT = PROJECT / "求解结果" / "数据"
FIG = PROJECT / "求解结果" / "图表"
CACHE = OUT / "cache"
EVENT_RELEASE = pd.Timestamp("2026-08-26")
REFERENCE_YEARS = (2021, 2023, 2024)
WINDOW_H = 168
SENSITIVITY_H = 72
MIN_COVERAGE = 0.80
CALENDAR_RADIUS = 15
MIN_PAIR_HOURS = 200
CLUSTER_ABS_RHO = 0.8

# The order is part of the frozen protocol and only resolves exact ties.
CANDIDATES = [
    # MET (19)
    ("Ta_2m", "2 m气温", "MET", "degC", "原始气象观测", ""),
    ("RH_2m", "2 m相对湿度", "MET", "%", "原始气象观测", ""),
    ("VPD2", "2 m水汽压差", "MET", "hPa", "温湿观测派生", "由2 m温度和相对湿度计算"),
    ("WS_10m", "10 m风速", "MET", "m/s", "原始气象观测", ""),
    ("vector_shear_2_20m", "2-20 m矢量风切变", "MET", "1/s", "风廓线派生", "由2/20 m风速和风向计算"),
    ("shear_1_20m", "1-20 m标量风切变", "MET", "1/s", "风廓线派生",
     "由WS_1m与WS_20m计算 (WS_20m-WS_1m)/19 m"),
    ("Pressure", "气压", "MET", "hPa", "原始气象观测", ""),
    ("API", "前期降水指数", "MET", "mm", "降水派生", "由历史小时降水递推"),
    ("P", "小时降水", "MET", "mm/h", "原始降水观测", ""),
    ("P_3h", "3 h累计降水", "MET", "mm", "降水派生", "3 h滚动累计"),
    ("P_24h", "24 h累计降水", "MET", "mm", "降水派生", "24 h滚动累计"),
    ("PDD_3d", "3 d正积温", "MET", "degC d", "温度派生", "3 d正积温"),
    ("PDD_7d", "7 d正积温", "MET", "degC d", "温度派生", "7 d正积温"),
    ("AP_7d", "7 d前期降水", "MET", "mm", "降水派生", "7 d滚动累计"),
    ("rise_24h", "24 h升温", "MET", "degC", "温度派生", "24 h温度变化"),
    ("freeze_thaw", "24 h冻融交替次数", "MET", "次/24h", "温度派生",
     "相邻小时温度乘积跨过0 degC的滚动24 h计数"),
    ("inv_max", "原始逆温强度", "MET", "K/100m", "温度廓线派生",
     "取未订正相邻温度层的原始正温度直减率最大值"),
    ("Ri_b", "整体理查森数", "MET", "1", "静力订正廓线派生",
     "由2 m与20 m静力订正虚位温、水平风速和层厚计算整体Ri_b"),
    ("dtheta_1_20", "1-20 m位温差", "MET", "K", "静力订正廓线派生",
     "由静力订正后的theta_20m减theta_1m计算"),
    # RADM (5)
    ("Rn", "净辐射", "RADM", "W/m2", "辐射观测派生", "四分量净辐射"),
    ("clearness", "晴空指数", "RADM", "1", "辐射派生", "仅白昼物理有效"),
    ("albedo", "反照率", "RADM", "1", "辐射派生", "仅白昼物理有效"),
    ("melt_energy_hourly", "小时消融有效能量", "RADM", "MJ m^-2 h^-1",
     "能量派生", "Rn-G-H-LE截正后乘3600/1e6的逐小时能量，不是历史累计"),
    ("Rsd", "下行短波辐射", "RADM", "W/m2", "原始辐射观测", ""),
    # SOIL (3)
    ("soilT_0.1m", "0.1 m土温", "SOIL", "degC", "土壤观测", ""),
    ("VWC_0.1m", "0.1 m土壤含水率", "SOIL", "m3/m3", "土壤观测", ""),
    ("VWC_0.4m", "0.4 m土壤含水率", "SOIL", "m3/m3", "土壤观测", ""),
    # FLUX (4)
    ("USTAR", "摩擦速度", "FLUX", "m/s", "涡动相关通量观测", ""),
    ("TKE", "观测湍流动能", "FLUX", "m2/s2", "涡动相关通量观测", ""),
    ("H", "感热通量", "FLUX", "W/m2", "涡动相关通量观测", ""),
    ("LE", "潜热通量", "FLUX", "W/m2", "涡动相关通量观测", ""),
]

SOURCE_FIELDS = {
    "Ta_2m": "Ta_2m",
    "RH_2m": "RH_2m",
    "VPD2": "Ta_2m;RH_2m",
    "WS_10m": "WS_10m",
    "vector_shear_2_20m": "WS_2m;WD_2m;WS_20m;WD_20m",
    "shear_1_20m": "WS_1m;WS_20m",
    "Pressure": "Pressure",
    "API": "P",
    "P": "P",
    "P_3h": "P",
    "P_24h": "P",
    "PDD_3d": "Ta_2m",
    "PDD_7d": "Ta_2m",
    "AP_7d": "P",
    "rise_24h": "Ta_2m",
    "freeze_thaw": "Ta_2m",
    "inv_max": "Ta_1m;Ta_2m;Ta_4m;Ta_10m;Ta_20m",
    "Ri_b": "Ta_1m_to_20m;RH_1m_to_20m;WS_1m_to_20m;WD_1m_to_20m;Pressure",
    "dtheta_1_20": "Ta_1m_to_20m;RH_1m_to_20m;Pressure",
    "Rn": "Rsd;Rsu;Rld;Rlu",
    "clearness": "Rsd",
    "albedo": "Rsd;Rsu",
    "melt_energy_hourly": "Rn;G_0.1m;H;LE;Ta_2m",
    "Rsd": "Rsd",
    "soilT_0.1m": "SOIL temperature 0.1 m",
    "VWC_0.1m": "SOIL VWC 0.1 m",
    "VWC_0.4m": "SOIL VWC 0.4 m",
    "USTAR": "FLUX USTAR",
    "TKE": "FLUX TKE",
    "H": "FLUX H",
    "LE": "FLUX LE",
}

DOWNSTREAM_ROLES = {
    "Ta_2m": ["Q2_melt_temperature", "Q3_energy_fit"],
    "RH_2m": ["Q3_energy_fit"],
    "Rsd": ["Q3_energy_fit"],
    "Rn": ["Q2_melt_energy_component"],
    "H": ["Q2_melt_energy_component"],
    "LE": ["Q2_melt_energy_component"],
    "melt_energy_hourly": ["Q2_melt_flux_proxy"],
    "API": ["Q2_rain_proxy"],
    "P": ["Q2_rain_proxy"],
    "P_3h": ["Q2_rain_proxy"],
    "P_24h": ["Q2_rain_proxy"],
}

HOUR_AVAILABILITY_RULE = (
    "hourly label t denotes the observed/derived value available after t+1 h; "
    "the release at 2026-08-26 00:00 uses observations only through 2026-08-25 22:00"
)

COLUMNS = [row[0] for row in CANDIDATES]
COLUMN_ORDER = {column: i for i, column in enumerate(COLUMNS)}


def serial(value):
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [serial(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    return value


def calendar_day(values):
    return np.array([pd.Timestamp(2001, d.month, d.day).dayofyear for d in values])


def nearby(dates, query, radius=CALENDAR_RADIUS):
    delta = np.abs(calendar_day(dates) - calendar_day(pd.DatetimeIndex([query]))[0])
    return np.minimum(delta, 365 - delta) <= radius


def window_mean(feature: pd.DataFrame, release, length: int):
    end = pd.Timestamp(release) - pd.Timedelta(hours=2)
    start = end - pd.Timedelta(hours=length - 1)
    block = feature.loc[start:end]
    return block.mean(), block.count(), start, end


def robust_score(target_mean, reference: pd.Series):
    values = pd.to_numeric(reference, errors="coerce").dropna().to_numpy()
    if len(values) < 5:
        return None
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)) * 1.4826)
    iqr = float((np.quantile(values, .75) - np.quantile(values, .25)) / 1.349)
    scale = mad if mad > 0 else iqr
    if not np.isfinite(scale) or scale <= 0:
        return None
    delta = float(target_mean - center)
    return {
        "delta": delta,
        "A": abs(delta / scale),
        "signed_z": delta / scale,
        "baseline_median": center,
        "robust_scale": scale,
        "n_reference": len(values),
        "scale_method": "MAD*1.4826" if mad > 0 else "IQR/1.349",
        "tail_rank": float((1 + np.sum(np.abs(values - center) >= abs(delta))) / (len(values) + 1)),
    }


def build_reference(features, columns, length, query, years=REFERENCE_YEARS):
    dates = pd.date_range(
        f"{min(years)}-01-01", f"{max(years)}-12-31", freq="D"
    )
    dates = dates[~((dates.month == 2) & (dates.day == 29))]
    dates = dates[np.isin(dates.year, years) & nearby(dates, query)]
    threshold = int(np.ceil(length * MIN_COVERAGE))
    rows = []
    for release in dates:
        mean, count, start, end = window_mean(features[columns], release, length)
        rows.append({
            "release": release,
            "start": start,
            "end": end,
            **{f"mean__{c}": mean[c] for c in columns},
            **{f"count__{c}": count[c] for c in columns},
        })
    return pd.DataFrame(rows), threshold


def pairwise_spearman(features):
    warm = features.loc[
        features.index.year.isin(range(2021, 2025))
        & features.index.month.isin([6, 7, 8, 9]),
        COLUMNS,
    ]
    correlation = pd.DataFrame(np.nan, index=COLUMNS, columns=COLUMNS)
    pair_n = pd.DataFrame(0, index=COLUMNS, columns=COLUMNS, dtype=int)
    edges = []
    for index, left in enumerate(COLUMNS):
        for right in COLUMNS[index + 1:]:
            common = warm[[left, right]].dropna()
            pair_n.loc[left, right] = pair_n.loc[right, left] = len(common)
            rho = np.nan
            if len(common) >= MIN_PAIR_HOURS:
                value = stats.spearmanr(common[left], common[right], nan_policy="omit")
                if np.isfinite(value.statistic):
                    rho = float(value.statistic)
                    correlation.loc[left, right] = correlation.loc[right, left] = rho
            edges.append({
                "left": left,
                "right": right,
                "common_valid_hours": int(len(common)),
                "spearman_rho": rho,
                "edge_abs_rho_gt_0.8": bool(np.isfinite(rho) and abs(rho) > CLUSTER_ABS_RHO),
            })
    return correlation, pair_n, pd.DataFrame(edges)


def connected_components(columns, correlation):
    adjacent = {c: set() for c in columns}
    for index, left in enumerate(columns):
        for right in columns[index + 1:]:
            value = correlation.loc[left, right]
            if np.isfinite(value) and abs(value) > CLUSTER_ABS_RHO:
                adjacent[left].add(right)
                adjacent[right].add(left)
    seen, groups = set(), []
    for column in columns:
        if column in seen:
            continue
        todo, component = [column], []
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            todo.extend(adjacent[current] - seen)
        groups.append(sorted(component, key=COLUMN_ORDER.get))
    return groups


def rank_once(features, clusters, length, query, years=REFERENCE_YEARS):
    target_mean, target_count, target_start, target_end = window_mean(
        features[COLUMNS], query, length
    )
    references, threshold = build_reference(features, COLUMNS, length, query, years)
    detail = {}
    for column in COLUMNS:
        valid_ref = references.loc[
            references[f"count__{column}"] >= threshold,
            f"mean__{column}",
        ]
        if int(target_count[column]) < threshold:
            detail[column] = {
                "rankable": False,
                "reason": "目标窗有效小时不足80%",
                "target_valid_hours": int(target_count[column]),
                "required_hours": threshold,
                "reference_windows": int(len(valid_ref)),
            }
            continue
        score = robust_score(float(target_mean[column]), valid_ref)
        if score is None:
            detail[column] = {
                "rankable": False,
                "reason": "参照窗不足或稳健尺度为零",
                "target_valid_hours": int(target_count[column]),
                "required_hours": threshold,
                "reference_windows": int(len(valid_ref)),
            }
            continue
        detail[column] = {
            "rankable": True,
            "target_mean": float(target_mean[column]),
            "target_valid_hours": int(target_count[column]),
            "required_hours": threshold,
            "reference_windows": int(len(valid_ref)),
            **score,
        }

    cluster_rows = []
    for component in clusters:
        eligible = [column for column in component if detail.get(column, {}).get("rankable")]
        if not eligible:
            cluster_rows.append({"members": component, "representative": None, "status": "不可排名"})
            continue
        representative = max(eligible, key=lambda column: detail[column]["A"])
        cluster_rows.append({
            "members": component,
            "representative": representative,
            "A_abs": detail[representative]["A"],
            "signed_z": detail[representative]["signed_z"],
            "excluded_members": [column for column in component if column != representative],
        })
    ranked = sorted(
        [row for row in cluster_rows if row.get("representative")],
        key=lambda row: row["A_abs"],
        reverse=True,
    )
    for position, row in enumerate(ranked, 1):
        row["rank"] = position
    return {
        "window_length_h": length,
        "target_window": [str(target_start), str(target_end)],
        "reference_window_count": len(references),
        "reference_years": list(years),
        "feature_details": detail,
        "cluster_ranking": cluster_rows,
        "ranked_clusters": ranked,
        "top5": [row["representative"] for row in ranked[:5]],
    }


def overlap(left, right):
    return len(set(left) & set(right))


def candidate_ranking_rows(result: dict, metadata: dict) -> list[dict]:
    """Return every candidate, including an explicit unrankable disposition."""
    rows = []
    for column in sorted(result["feature_details"], key=COLUMN_ORDER.get):
        detail = result["feature_details"][column]
        rows.append({
            "rank": None,
            "window_length_h": result["window_length_h"],
            "target_window_start": result["target_window"][0],
            "target_window_end": result["target_window"][1],
            **metadata[column],
            "rankable": bool(detail.get("rankable", False)),
            "target_valid_hours": detail.get("target_valid_hours"),
            "required_hours": detail.get("required_hours"),
            "reference_windows": detail.get("reference_windows"),
            "target_mean": detail.get("target_mean"),
            "A_abs": detail.get("A"),
            "signed_z": detail.get("signed_z"),
            "delta": detail.get("delta"),
            "baseline_median": detail.get("baseline_median"),
            "robust_scale": detail.get("robust_scale"),
            "scale_method": detail.get("scale_method"),
            "tail_rank": detail.get("tail_rank"),
            "exclusion_reason": detail.get("reason"),
        })
    ranked = sorted(
        [row for row in rows if row["rankable"]],
        key=lambda row: (-float(row["A_abs"]), COLUMN_ORDER[row["variable"]]),
    )
    unrankable = [row for row in rows if not row["rankable"]]
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    return ranked + unrankable


def cluster_ranking_rows(result: dict, metadata: dict) -> list[dict]:
    """Return every correlation component, including non-rankable clusters."""
    ranked_by_representative = {
        row["representative"]: row for row in result["ranked_clusters"]
    }
    rows = []
    for cluster in result["cluster_ranking"]:
        representative = cluster.get("representative")
        ranked = ranked_by_representative.get(representative)
        detail = result["feature_details"].get(representative, {})
        rows.append({
            "rank": ranked.get("rank") if ranked else None,
            "window_length_h": result["window_length_h"],
            "target_window_start": result["target_window"][0],
            "target_window_end": result["target_window"][1],
            "rankable": bool(ranked),
            "cluster_status": "ranked" if ranked else "not_rankable",
            "representative": representative,
            "name": metadata[representative]["name"] if representative else None,
            "family": metadata[representative]["family"] if representative else None,
            "direction": (
                "偏高" if detail.get("signed_z", 0) > 0 else "偏低"
            ) if ranked else None,
            "A_abs": detail.get("A") if ranked else None,
            "signed_z": detail.get("signed_z") if ranked else None,
            "delta": detail.get("delta") if ranked else None,
            "baseline_median": detail.get("baseline_median") if ranked else None,
            "tail_rank": detail.get("tail_rank") if ranked else None,
            "member_count": len(cluster["members"]),
            "cluster_members": "|".join(cluster["members"]),
            "excluded_members": "|".join(cluster.get("excluded_members", [])),
            "display_top5": bool(ranked and ranked["rank"] <= 5),
        })
    return sorted(
        rows,
        key=lambda row: (
            not row["rankable"],
            row["rank"] if row["rankable"] else 10**9,
            row["cluster_members"],
        ),
    )


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(CACHE / "features.parquet")
    # Keep the transport key stable at the source feature layer while the
    # association dictionary exposes its true hourly-energy meaning.
    if "melt_energy_hourly" not in features and "cum_melt_energy" in features:
        features["melt_energy_hourly"] = features["cum_melt_energy"]
    missing = [column for column in COLUMNS if column not in features]
    if missing:
        raise ValueError(f"冻结候选字段缺失: {missing}")
    if len(COLUMNS) != 31 or len(set(COLUMNS)) != 31:
        raise AssertionError("必须冻结31个唯一候选键")
    family_count = pd.Series([row[2] for row in CANDIDATES]).value_counts().to_dict()
    expected = {"MET": 19, "RADM": 5, "SOIL": 3, "FLUX": 4}
    assert family_count == expected, family_count

    correlation, pair_n, edges = pairwise_spearman(features)
    clusters = connected_components(COLUMNS, correlation)
    main_168 = rank_once(features, clusters, WINDOW_H, EVENT_RELEASE)
    sensitivity_72 = rank_once(features, clusters, SENSITIVITY_H, EVENT_RELEASE)
    leave_one_year = {}
    for year in REFERENCE_YEARS:
        kept = tuple(y for y in REFERENCE_YEARS if y != year)
        result = rank_once(features, clusters, WINDOW_H, EVENT_RELEASE, kept)
        leave_one_year[str(year)] = {
            "omitted_year": year,
            "top5": result["top5"],
            "top5_overlap": overlap(main_168["top5"], result["top5"]),
            "ranked_representatives": [
                row["representative"] for row in result["ranked_clusters"]
            ],
        }

    pseudo = pd.Timestamp("2025-08-26")
    pseudo_168 = rank_once(features, clusters, WINDOW_H, pseudo, (2021, 2022, 2023))

    direct_roles = {
        "原始气象观测", "原始降水观测", "原始辐射观测", "土壤观测",
        "涡动相关通量观测",
    }
    metadata = {}
    for row in CANDIDATES:
        if not row[5] and row[4] not in direct_roles:
            raise AssertionError(f"derived candidate lacks derivation: {row[0]}")
        metadata[row[0]] = {
            "variable": row[0],
            "name": row[1],
            "family": row[2],
            "unit": row[3],
            "provenance_role": row[4],
            "derivation": row[5] or "direct field; no additional transformation",
            "source_fields": SOURCE_FIELDS.get(row[0], row[0]),
            "available_time_rule": HOUR_AVAILABILITY_RULE,
            "downstream_model_roles": DOWNSTREAM_ROLES.get(row[0], ["Q1_only"]),
            "shared_with_downstream_model": bool(row[0] in DOWNSTREAM_ROLES),
        }
    top5 = []
    for rank, representative in enumerate(main_168["top5"], 1):
        detail = main_168["feature_details"][representative]
        cluster = next(
            row for row in main_168["ranked_clusters"]
            if row["representative"] == representative
        )
        top5.append({
            "rank": rank,
            "variable": representative,
            "name": metadata[representative]["name"],
            "family": metadata[representative]["family"],
            "direction": "事件窗偏高" if detail["delta"] > 0 else "事件窗偏低",
            "A_abs": detail["A"],
            "signed_z": detail["signed_z"],
            "delta": detail["delta"],
            "baseline_median": detail["baseline_median"],
            "tail_rank": detail["tail_rank"],
            "cluster_members": cluster["members"],
            "cluster_representative_reason": "同簇内绝对稳健异常最大；只在展示层去冗余",
        })

    candidate_dict = []
    for row in CANDIDATES:
        column = row[0]
        detail = main_168["feature_details"].get(column, {})
        candidate_dict.append({
            **metadata[column],
            "candidate_eligible": True,
            "target_valid_hours": detail.get("target_valid_hours"),
            "required_hours": detail.get("required_hours"),
            "reference_windows": detail.get("reference_windows"),
            "rankable": bool(detail.get("rankable", False)),
            "exclusion_reason": detail.get("reason"),
        })
    audit_fields = (
        "variable",
        "name",
        "family",
        "unit",
        "provenance_role",
        "derivation",
        "source_fields",
        "available_time_rule",
        "downstream_model_roles",
        "shared_with_downstream_model",
        "target_valid_hours",
        "required_hours",
        "rankable",
    )
    if len(candidate_dict) != 31 or len({row["variable"] for row in candidate_dict}) != 31:
        raise AssertionError("candidate dictionary must contain 31 unique keys")
    if any(not row.get(field) and row.get(field) is not False for row in candidate_dict for field in audit_fields):
        raise AssertionError("candidate dictionary audit field is empty")
    pd.DataFrame(candidate_dict).to_csv(
        OUT / "01c_候选字典.csv", index=False, encoding="utf-8-sig"
    )
    edges.to_csv(OUT / "01c_相关边表.csv", index=False, encoding="utf-8-sig")

    score_rows = []
    for row in CANDIDATES:
        column = row[0]
        detail = main_168["feature_details"].get(column, {})
        score_rows.append({
            **metadata[column],
            "rankable": bool(detail.get("rankable", False)),
            "target_valid_hours": detail.get("target_valid_hours"),
            "reference_windows": detail.get("reference_windows"),
            "A_abs": detail.get("A"),
            "signed_z": detail.get("signed_z"),
            "delta": detail.get("delta"),
            "baseline_median": detail.get("baseline_median"),
            "tail_rank": detail.get("tail_rank"),
            "exclusion_reason": detail.get("reason"),
        })
    pd.DataFrame(score_rows).to_csv(OUT / "01c_逐候选得分.csv", index=False, encoding="utf-8-sig")

    candidate_168_rows = candidate_ranking_rows(main_168, metadata)
    candidate_72_rows = candidate_ranking_rows(sensitivity_72, metadata)
    cluster_168_rows = cluster_ranking_rows(main_168, metadata)
    cluster_72_rows = cluster_ranking_rows(sensitivity_72, metadata)
    if len(candidate_168_rows) != 31 or len(candidate_72_rows) != 31:
        raise AssertionError("both candidate rankings must contain all 31 candidates")
    if len(cluster_168_rows) != len(main_168["cluster_ranking"]):
        raise AssertionError("168 h cluster ranking must retain every correlation cluster")
    if len(cluster_72_rows) != len(sensitivity_72["cluster_ranking"]):
        raise AssertionError("72 h cluster ranking must retain every correlation cluster")
    pd.DataFrame(candidate_168_rows).to_csv(
        OUT / "01c_168h候选完整排序.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(candidate_72_rows).to_csv(
        OUT / "01c_72h候选完整排序.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cluster_168_rows).to_csv(
        OUT / "01c_完整关联排序.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cluster_168_rows).to_csv(
        OUT / "01c_168h簇完整排序.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(cluster_72_rows).to_csv(
        OUT / "01c_72h簇完整排序.csv", index=False, encoding="utf-8-sig"
    )

    alias_audit = [
        {
            "excluded": "Pmax_1h",
            "included": "P",
            "reason": "1 h滑动最大值与同一小时P的NaN掩膜和数值逐行相同，是别名重复",
        },
        {
            "excluded": "ustar",
            "included": "USTAR",
            "reason": "ustar为USTAR的非负掩膜派生列，不另计物理候选",
        },
        {
            "excluded": "TKE_obs",
            "included": "TKE",
            "reason": "TKE_obs为TKE的非负掩膜派生列，不另计物理候选",
        },
    ]
    pd.DataFrame(alias_audit).to_csv(
        OUT / "01c_别名审计.csv", index=False, encoding="utf-8-sig"
    )

    summary = {
        "schema_version": 2,
        "status": "available",
        "target_definition": "2026-08-26冲击前168小时环境窗相对同积日±15日的2021/2023/2024参照窗的单事件关联描述",
        "event_release": str(EVENT_RELEASE),
        "last_usable_hour": str(EVENT_RELEASE - pd.Timedelta(hours=2)),
        "candidate_count": len(COLUMNS),
        "candidate_families": {family: family_count[family] for family in ("MET", "RADM", "SOIL", "FLUX")},
        "candidate_order": COLUMNS,
        "alias_audit": alias_audit,
        "coverage_rule": "逐候选目标窗与参照窗均至少80%有效；不插值、不把缺测当零",
        "score_rule": "事件窗均值相对参照窗中位数；尺度为MAD×1.4826，零MAD时回退IQR/1.349；A=|signed_z|",
        "cluster_rule": "2021–2024年6–9月逐对Spearman，绝对相关>0.8且共同有效小时>=200才连边；取完整连通簇",
        "cluster_common_hours_policy": {"minimum_pair_hours": MIN_PAIR_HOURS, "abs_rho_threshold": CLUSTER_ABS_RHO},
        "pairwise_valid_pairs": int((pair_n.values > 0).sum() // 2),
        "pairwise_valid_pair_hours_median": float(np.median([
            row.common_valid_hours for row in edges.itertuples() if row.common_valid_hours > 0
        ])) if len(edges) else None,
        "main_168h": main_168,
        "sensitivity_72h": sensitivity_72,
        "leave_one_reference_year_out": leave_one_year,
        "pseudo_2025_same_calendar": {
            "release": str(pseudo),
            "top5": pseudo_168["top5"],
            "max_A": (
                float(pseudo_168["ranked_clusters"][0]["A_abs"])
                if pseudo_168["ranked_clusters"] else None
            ),
            "ranked_representatives": [
                {"variable": row["representative"], "A_abs": row["A_abs"]}
                for row in pseudo_168["ranked_clusters"]
            ],
            "scope": "无灾害真值的同类过程外检，不是确证非灾日",
        },
        "top5": top5,
        "interpretation": "这是单次事件的环境关联证据，不是因果效应、发生概率或普适灾害排序。",
        "findings": [
            "候选池已断言覆盖MET、RADM、SOIL、FLUX四类，家族计数为19/5/3/4。",
            "Pmax_1h、ustar和TKE_obs作为重复/掩膜别名登记而不重复计数。",
            "晴空指数与反照率只在白昼有效；若覆盖不足，保留在字典并标为不可排名。",
        ],
        "outputs": {
            "candidate_dictionary": "01c_候选字典.csv",
            "pairwise_edges": "01c_相关边表.csv",
            "candidate_scores": "01c_逐候选得分.csv",
            "candidate_full_ranking_168h": "01c_168h候选完整排序.csv",
            "candidate_full_ranking_72h": "01c_72h候选完整排序.csv",
            "cluster_full_ranking_168h": "01c_168h簇完整排序.csv",
            "cluster_full_ranking_72h": "01c_72h簇完整排序.csv",
            "full_ranking": "01c_完整关联排序.csv",
            "figure": "01c_统一候选关联.png",
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    rank_rows = main_168["ranked_clusters"][:15]
    labels = [metadata[row["representative"]]["name"] for row in rank_rows]
    values = [row["A_abs"] for row in rank_rows]
    colors = ["#1f6f68" if i < 5 else "#a8c3bd" for i in range(len(rank_rows))]
    axes[0].barh(np.arange(len(rank_rows)), values, color=colors)
    axes[0].set_yticks(np.arange(len(rank_rows)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("绝对稳健异常 A（无量纲）")
    axes[0].set_title("(a) 31项统一候选的相关簇代表")
    for index, value in enumerate(values):
        axes[0].text(value + .03, index, f"{value:.2f}", va="center", fontsize=14)
    axes[0].grid(axis="x", alpha=.2)

    left = {row["representative"]: row["A_abs"] for row in main_168["ranked_clusters"]}
    right = {row["representative"]: row["A_abs"] for row in sensitivity_72["ranked_clusters"]}
    common = [row["representative"] for row in main_168["ranked_clusters"][:12] if row["representative"] in right]
    x = np.arange(len(common))
    axes[1].plot(x, [left[c] for c in common], "o-", label="168小时主窗", color="#1f6f68")
    axes[1].plot(x, [right[c] for c in common], "s--", label="72小时敏感性", color="#b26745")
    axes[1].set_xticks(x, [metadata[c]["name"] for c in common], rotation=55, ha="right", fontsize=14)
    axes[1].set_ylabel("A")
    axes[1].set_title(f"(b) 前五重合 {overlap(main_168['top5'], sensitivity_72['top5'])}/5")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("单事件冲击前环境关联：31项冻结候选与逐对相关去冗余")
    fig.savefig(FIG / "01c_统一候选关联.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    (OUT / "01c_统一关联前五.json").write_text(
        json.dumps(serial(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "01c top5:",
        ", ".join(item["name"] for item in top5),
        f"| 168h refs={main_168['reference_window_count']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
