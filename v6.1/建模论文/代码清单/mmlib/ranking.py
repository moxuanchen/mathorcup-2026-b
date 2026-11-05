"""同源气象资料的两套排序及有时间边界的敏感性检验。"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

SEED = 20260826
EVENT = pd.Timestamp("2026-08-26")
WEIGHTS = {"PI_rank": 0.30, "MI_rank": 0.25, "AN_rank": 0.20, "MP_rank": 0.25}
SOURCE_FAMILY = {"P", "P_24h", "P_3h", "Pmax_1h", "API", "AP_7d", "PDD_3d", "PDD_7d"}
CONSTRUCTIVE = {"P", "P_24h", "P_3h", "Pmax_1h", "API", "AP_7d", "cum_melt_energy", "Rn", "Ta_2m"}
NEGATIVE_DIRECTION = {"RH_2m", "Ri_b", "Pressure", "albedo"}


def _tau(left: list[str], right: list[str]) -> float:
    common = [x for x in left if x in right]
    if len(common) < 2:
        return float("nan")
    return float(stats.kendalltau([left.index(x) for x in common],
                                  [right.index(x) for x in common]).statistic)


def compare(left: list[str], right: list[str]) -> dict:
    return {"前5重合数": len(set(left[:5]) & set(right[:5])),
            "全排序Kendall_tau": _tau(left, right), "共同特征数": len(set(left) & set(right))}


def connected_components(frame: pd.DataFrame, cols: list[str], threshold: float = 0.8) -> tuple[list[list[str]], list[dict]]:
    corr = frame[cols].corr(min_periods=100).abs()
    edges = [{"变量1": a, "变量2": b, "绝对相关": float(corr.loc[a, b])}
             for i, a in enumerate(cols) for b in cols[i + 1:]
             if np.isfinite(corr.loc[a, b]) and corr.loc[a, b] > threshold]
    adjacent = {c: set() for c in cols}
    for edge in edges:
        a, b = edge["变量1"], edge["变量2"]
        adjacent[a].add(b)
        adjacent[b].add(a)
    components, seen = [], set()
    for c in cols:
        if c in seen:
            continue
        todo, component = [c], []
        while todo:
            v = todo.pop()
            if v in seen:
                continue
            seen.add(v)
            component.append(v)
            todo.extend(sorted(adjacent[v] - seen))
        components.append(sorted(component))
    return components, edges


def deduplicate(rows: pd.DataFrame, components: list[list[str]], score: str,
                ascending: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    order = rows.sort_values([score, "变量"], ascending=[ascending, True])
    place = {v: i for i, v in enumerate(order["变量"])}
    keep, disclosure = [], []
    for i, component in enumerate(components, 1):
        members = [v for v in component if v in place]
        if not members:
            continue
        winner = min(members, key=place.get)
        keep.append(winner)
        disclosure.append({"簇": i, "成员": members, "保留变量": winner,
                           "剔除变量": [v for v in members if v != winner]})
    display = order[order["变量"].isin(keep)].copy()
    display["展示次序"] = np.arange(1, len(display) + 1)
    return display, disclosure


def _anomaly(feat: pd.DataFrame, cols: list[str], start: pd.Timestamp,
             end: pd.Timestamp, baseline: pd.DataFrame) -> pd.Series:
    result = {}
    for c in cols:
        s = baseline[c].dropna()
        event = feat.loc[start:end, c].dropna()
        if len(s) < 100 or len(event) < 12:
            result[c] = np.nan
            continue
        value = event.max() if abs(s.skew()) > 1 else event.mean()
        probability = max(2 * min((s >= value).mean(), (s <= value).mean()), 1 / (len(s) + 1))
        result[c] = -np.log10(min(probability, 1.0))
    return pd.Series(result)


def _fit_evidence(feat: pd.DataFrame, qw: pd.Series, cols: list[str],
                  train_mask: np.ndarray, test_mask: np.ndarray, quantile: float = 0.90,
                  repeats: int = 5, iterations: int = 160) -> tuple[pd.Series, pd.Series, dict]:
    from . import trigger
    api_mask = (train_mask & np.isin(feat.index.year, trigger.API_TRAIN_YEARS)
                & np.isin(feat.index.month, trigger.API_WARM_MONTHS))
    fold_api = pd.to_numeric(feat.loc[api_mask, "API"], errors="coerce")
    fold_api = fold_api.where(np.isfinite(fold_api) & fold_api.ge(0)).dropna()
    if len(fold_api) < 100:
        raise ValueError("该折2021--2025暖季有效API样本不足")
    api_thresholds = fold_api.quantile([1 / 3, 2 / 3]).to_numpy().tolist()
    fold_features = feat.copy(deep=False)
    fold_features.attrs = {**feat.attrs, "api_quantiles": api_thresholds}
    model = trigger.default_model(fold_features, calibrate=False)
    qw = model.discharge(fold_features)["Q_water"]
    valid_q = qw.notna() & np.isfinite(qw)
    train = feat.loc[train_mask & valid_q, cols]
    test = feat.loc[test_mask & valid_q, cols]
    threshold = float(qw.loc[train.index].quantile(quantile))
    train = train.iloc[::max(1, int(np.ceil(len(train) / 30000)))]
    test = test.iloc[::max(1, int(np.ceil(len(test) / 12000)))]
    if len(train) < 500 or len(test) < 100:
        raise ValueError("时间留出排序样本不足")
    medians = train.median().fillna(0)
    xtr, xte = train.fillna(medians), test.fillna(medians)
    ytr = (qw.loc[train.index] > threshold).astype(int)
    yte = (qw.loc[test.index] > threshold).astype(int)
    if ytr.nunique() < 2 or yte.nunique() < 2:
        raise ValueError("时间留出标签不足两类")
    clf = HistGradientBoostingClassifier(max_iter=iterations, learning_rate=0.07,
                                         max_depth=5, min_samples_leaf=40, random_state=SEED)
    clf.fit(xtr, ytr)
    pi = pd.Series(permutation_importance(clf, xte, yte, n_repeats=repeats,
                                          scoring="roc_auc", n_jobs=1, random_state=SEED).importances_mean,
                   index=cols)
    mi = pd.Series(mutual_info_classif(xtr, ytr, random_state=SEED), index=cols)
    constant = train.columns[train.nunique(dropna=True) <= 1]
    pi.loc[constant] = 0.0
    mi.loc[constant] = 0.0
    note = {"训练样本": len(train), "测试样本": len(test),
            "训练时间": [str(train.index.min()), str(train.index.max())],
            "测试时间": [str(test.index.min()), str(test.index.max())],
            "标签分位": quantile, "训练标签阈值_m3s": threshold,
            "训练正例率": float(ytr.mean()), "测试正例率": float(yte.mean()),
            "测试AUC": float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1])),
            "训练插补中位数": medians.to_dict(),
            "训练全缺失列": train.columns[train.isna().all()].tolist(),
            "训练常量或全缺失列_重要性置零": constant.tolist(),
            "训练API分位": api_thresholds,
            "标签水文预处理": "各折2021--2025训练暖季API分位冻结后重算整序列Qw；不覆盖共享qwater缓存；未事件标定",
            "口径": "AUC只度量代理水文标签重建，不是灾害发生识别；插补与标签分位只用训练期"}
    return pi, mi, note


def _primary_table(flat: list[tuple[str, str, str]], pi: pd.Series, mi: pd.Series,
                   anomaly: pd.Series, prior: dict) -> pd.DataFrame:
    rows = pd.DataFrame([{"条件组": group, "特征": name, "变量": c,
                          "置换重要性": pi.get(c, np.nan), "互信息": mi.get(c, np.nan),
                          "事件异常度": anomaly.get(c, np.nan), "机理先验": prior.get(c, 0.3),
                          "特征性质": "构造性驱动项" if c in CONSTRUCTIVE else "独立于标签构造的调制项"}
                         for group, name, c in flat])
    for col, tag in (("置换重要性", "PI"), ("互信息", "MI"), ("事件异常度", "AN"), ("机理先验", "MP")):
        rows[f"{tag}_rank"] = rows[col].rank(ascending=False, method="average", na_option="bottom")
    rows["综合得分"] = sum(rows[k] * w for k, w in WEIGHTS.items())
    rows["无先验得分"] = sum(rows[k] * WEIGHTS[k] / 0.75 for k in ("PI_rank", "MI_rank", "AN_rank"))
    rows = rows.sort_values(["综合得分", "变量"]).reset_index(drop=True)
    rows["总次序"] = np.arange(1, len(rows) + 1)
    rows["组内次序"] = rows.groupby("条件组")["综合得分"].rank(method="first").astype(int)
    return rows


def _daily(feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    d = feat[cols].resample("D").mean()
    counts = feat[cols].resample("D").count()
    d = d.where(counts >= 20)
    for c in ("albedo", "clearness"):
        if c in cols:
            eligible = feat["Rsd"] > (50 if c == "albedo" else 20)
            denom = eligible.resample("D").sum()
            valid = feat[c].where(eligible).resample("D").count()
            d[c] = feat[c].where(eligible).resample("D").mean().where((denom >= 4) & (valid >= .8*denom))
    d["P"] = feat["P"].resample("D").sum(min_count=24)
    for c in ("AP_7d", "PDD_3d", "PDD_7d"):
        if c in cols:
            # 23:00缺测不得用更早的“最后有效值”悄悄缩短窗口。
            d[c] = feat[c].where(feat.index.hour == 23).resample("D").max()
    return d


def critic_ranking(feat: pd.DataFrame, flat: list[tuple[str, str, str]],
                   components: list[list[str]]) -> dict:
    cols = [x[2] for x in flat]
    daily = _daily(feat, cols)
    baseline = daily.loc[(daily.index.year <= 2025) & (daily.index.month.isin([6, 7, 8, 9]))]
    if "PDD_3d" not in daily or "AP_7d" not in daily:
        raise KeyError("CRITIC需要PDD_3d与AP_7d")
    rain95, pdd95 = baseline["P"].quantile(.95), baseline["PDD_3d"].quantile(.95)
    label_valid = baseline[["P", "PDD_3d"]].notna().all(axis=1)
    label = ((baseline["P"] >= rain95) | (baseline["PDD_3d"] >= pdd95)).where(label_valid)
    normalized = {}
    norm_params = {}
    for c in ("AP_7d", "PDD_3d"):
        lo, hi = baseline[c].min(), baseline[c].max()
        if not np.isfinite(lo + hi) or hi <= lo:
            raise ValueError(f"CRITIC触发核{c}无法归一化")
        normalized[c] = ((baseline[c] - lo) / (hi - lo)).clip(0, 1)
        norm_params[c] = [float(lo), float(hi)]
    kernel = pd.concat(normalized, axis=1).max(axis=1, skipna=False)
    doy = daily.index.dayofyear
    delta = np.abs(doy - EVENT.dayofyear)
    near = daily.loc[(daily.index.year <= 2025) & (np.minimum(delta, 366 - delta) <= 15)]
    event = daily.loc["2026-08-24":"2026-08-26"]
    evidence, detail = [], []
    for group, name, c in flat:
        direction = -1 if c in NEGATIVE_DIRECTION else 1
        clim, ev = near[c].dropna(), event[c].dropna()
        z = (ev.mean() - clim.mean()) / clim.std(ddof=1) if len(clim) > 10 and len(ev) > 0 and clim.std() > 0 else np.nan
        a = max(direction * z, 0) if np.isfinite(z) else np.nan
        quantile = float((clim <= ev.mean()).mean()) if len(clim) and len(ev) else np.nan
        b = np.nan
        valid = label.notna() & baseline[c].notna()
        if c not in {"P", "Pmax_1h", "PDD_3d"} and valid.sum() >= 30 and label[valid].nunique() == 2:
            b = float(roc_auc_score(label[valid].astype(int), direction * baseline.loc[valid, c]))
        cc = np.nan
        valid_c = kernel.notna() & baseline[c].notna()
        if c not in SOURCE_FAMILY and valid_c.sum() >= 30:
            cc = abs(float(stats.spearmanr(baseline.loc[valid_c, c], kernel[valid_c]).statistic))
        evidence.append([a, b, cc])
        detail.append({"条件组": group, "特征": name, "变量": c, "增益方向假设": direction,
                       "事件标准异常z": z, "事件分位秩": quantile,
                       "证据A": a, "证据B": b, "证据C": cc,
                       "B排除构造变量": c in {"P", "Pmax_1h", "PDD_3d"},
                       "C排除触发族": c in SOURCE_FAMILY,
                       "特征性质": "构造性驱动项" if c in CONSTRUCTIVE else "独立于标签构造的调制项"})
    matrix = pd.DataFrame(evidence, index=cols, columns=["A", "B", "C"])
    scaled = (matrix - matrix.min()) / (matrix.max() - matrix.min()).replace(0, np.nan)
    scaled = scaled.where(matrix.notna())
    # 权重估计采用各证据对的共同可用特征，不把排除项当零值。
    corr = scaled.corr(min_periods=3).fillna(0)
    for col in corr.columns:
        corr.loc[col, col] = 1.0
    information = scaled.std(ddof=0).fillna(0) * (1 - corr).sum(axis=1)
    if information.sum() <= 0:
        raise ValueError("CRITIC证据信息量为零")
    weights = information / information.sum()
    available_weight = scaled.notna().mul(weights, axis=1).sum(axis=1)
    scores = scaled.mul(weights, axis=1).sum(axis=1, min_count=1) / available_weight.replace(0, np.nan)
    table = pd.DataFrame(detail)
    table["CRITIC得分"] = table["变量"].map(scores)
    table["可用证据权重和"] = table["变量"].map(available_weight)
    table = table.sort_values(["CRITIC得分", "变量"], ascending=[False, True])
    table["总次序"] = np.arange(1, len(table) + 1)
    display, clusters = deduplicate(table, components, "CRITIC得分", ascending=False)
    return {"权重": weights.to_dict(), "信息承载量": information.to_dict(),
            "证据相关矩阵": corr.to_dict(), "原始证据矩阵": matrix.to_dict(orient="index"),
            "归一化证据矩阵": scaled.to_dict(orient="index"), "排序表": table.to_dict("records"),
            "去冗余排序表": display.to_dict("records"), "前5特征": display.head(5).to_dict("records"),
            "冗余簇": clusters, "训练阈值": {"日雨量p95_mm": rain95, "PDD3_p95_°Ch": pdd95},
            "触发核归一化范围": norm_params, "事件窗口": ["2026-08-24", "2026-08-26"],
            "气候事件基线": "2021–2025同年积日±15日", "样本覆盖规则": "日均>=20小时，日雨完整24小时；白昼反照率/晴空指数按物理可用时段>=80%且至少4小时；72/168h累积要求整窗有效且23时有效",
            "解释": "与主排序共享原始气象和事件资料，仅为不同证据权重方案的一致性核对；非独立灾害验证"}


def association(feat: pd.DataFrame, qw: pd.Series, groups: dict, prior: dict) -> dict:
    flat = [(g, name, c) for g, names in groups.items() for name, c in names.items() if c in feat]
    cols = [c for _, _, c in flat]
    train_mask = (feat.index.year >= 2021) & (feat.index.year <= 2024)
    test_mask = feat.index.year == 2025
    pi, mi, fit_note = _fit_evidence(feat, qw, cols, train_mask, test_mask)
    start, end = pd.Timestamp("2026-08-20"), pd.Timestamp("2026-08-26 23:00")
    baseline = feat.loc[train_mask]
    anomaly = _anomaly(feat, cols, start, end, baseline)
    table = _primary_table(flat, pi, mi, anomaly, prior)
    components, edges = connected_components(baseline, cols)
    display, clusters = deduplicate(table, components, "综合得分")
    no_prior, np_clusters = deduplicate(table, components, "无先验得分")
    np_full = table.sort_values(["无先验得分", "变量"])["变量"].tolist()
    primary_full = table["变量"].tolist()
    sensitivity = []
    for q in (.75, .80, .85, .95):
        p2, m2, note = _fit_evidence(feat, qw, cols, train_mask, test_mask, q, repeats=2, iterations=120)
        t2 = _primary_table(flat, p2, m2, anomaly, prior)
        d2, _ = deduplicate(t2, components, "综合得分")
        sensitivity.append({"标签分位": q, **note, **compare(primary_full, t2["变量"].tolist()),
                            "去冗余前5重合数": len(set(display.head(5)["变量"]) & set(d2.head(5)["变量"]))})
    loyo = {}
    for year in (2023, 2024, 2025):
        tr, te = (feat.index.year >= 2021) & (feat.index.year < year), feat.index.year == year
        p2, m2, note = _fit_evidence(feat, qw, cols, tr, te, repeats=2, iterations=120)
        # 事件异常只用于固定的回顾性排序部分，不能把整榜称作该年的独立预测重要性。
        t2 = _primary_table(flat, p2, m2, anomaly, prior)
        fold_components, _ = connected_components(feat.loc[tr], cols)
        d2, _ = deduplicate(t2, fold_components, "综合得分")
        loyo[str(year)] = {**note, **compare(primary_full, t2["变量"].tolist()),
                           "前5_去冗余": d2.head(5)["变量"].tolist(),
                           "排序表": t2.to_dict("records"),
                           "限制": "向前留一年仅PI/MI重估；事件异常与机理先验固定，整榜非独立灾前验证"}
    shifted = []
    for year in range(2021, 2026):
        # 控制年从其自身气候参照剔除，减少自身进入经验分布的影响。
        ref = feat.loc[(feat.index.year <= 2025) & (feat.index.year != year)]
        shifted.append(_anomaly(feat, cols, pd.Timestamp(f"{year}-08-20"),
                                pd.Timestamp(f"{year}-08-26 23:00"), ref))
    shifted_anomaly = pd.concat(shifted, axis=1).mean(axis=1)
    real_order = anomaly.sort_values(ascending=False).index.tolist()
    shift_order = shifted_anomaly.sort_values(ascending=False).index.tolist()
    critic = critic_ranking(feat, flat, components)
    cross = compare(primary_full, [r["变量"] for r in critic["排序表"]])
    cross["去冗余前5重合数"] = len(set(display.head(5)["变量"]) & {r["变量"] for r in critic["前5特征"]})
    taus = [x["全排序Kendall_tau"] for x in sensitivity]
    top5 = display.head(5).copy()
    top5["次序"] = np.arange(1, len(top5) + 1)
    return {"候选特征数": len(cols), "标签定义": "Qw超过2021–2024训练期90%分位；Qw缺测不生成负例",
            "标签正例率": fit_note["训练正例率"], **fit_note, "权重": WEIGHTS,
            "排序表": table.to_dict("records"), "去冗余排序表": display.to_dict("records"),
            "前5特征": top5.to_dict("records"), "冗余簇": clusters, "训练相关边": edges,
            "去先验对照": {"权重_无先验": {k: v / .75 for k, v in WEIGHTS.items() if k != "MP_rank"},
                             **compare(primary_full, np_full),
                             "前5_主排序": display.head(5)["特征"].tolist(),
                             "前5_无先验": no_prior.head(5)["特征"].tolist(),
                             "去冗余前5重合数": len(set(display.head(5)["变量"]) & set(no_prior.head(5)["变量"])),
                             "完整排序表": table.sort_values("无先验得分").to_dict("records"), "冗余簇": np_clusters},
            "留一年AUC": {y: r["测试AUC"] for y, r in loyo.items()}, "逐年前向留出": loyo,
            "排序稳健性": {"阈值扰动Kendall_tau": taus, "均值": float(np.nanmean(taus)),
                             "最小值": float(np.nanmin(taus)), "逐阈值": sensitivity},
            "控制组_日历平移": {"排序一致性Kendall_tau": _tau(real_order, shift_order),
                                  "事件年异常度": anomaly.to_dict(), "平移年异常度": shifted_anomaly.to_dict(),
                                  "说明": "同日历窗口平移至2021–2025，控制年不进入其自身参照；只检验季节混杂"},
            "CRITIC": critic, "两方案一致性": cross,
            "共同限制": "排序包含事件后描述资料；代理标签包含能量/降水输入；高AUC不证明真实泥石流因果或识别能力"}
