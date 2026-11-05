"""问题 1：高寒山区大气边界层结构、天气过程、极端气象事件与泥石流关联性特征分析。

方法要点
--------
1. 边界层结构：5 层温/湿/风廓线 → 直减率、逆温强度与频率、整体理查森数、稳定度分类、
   近地层有效厚度、能量闭合与波文比、土壤温湿剖面。
2. 天气过程：廓线标准化后 SVD 提取主导模态（EOF），据降水—消融联合分位划分为四型。
3. 极端事件：POT-GPD 拟合各要素极值，给出 2/5 年再现期与复合极端事件统计。
4. 关联性排序：以水文代理响应状态为标签（训练期Qw的90%分位），
   对候选气象/边界层特征做置换重要性 + 互信息 + 事件异常度 + 机理先验的四准则
   Borda 融合，并按“水源—能量—物质”三条件分组排序。
   标签与降水/能量输入同源；构造性项显式披露，不作独立灾害验证。

输出：求解结果/数据/01_*.json，求解结果/图表/01_*.png
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
_PROJECT = Path(__file__).resolve().parents[2]
if os.environ.get("JILONG_ROOT") and Path(os.environ["JILONG_ROOT"]).resolve() != _PROJECT:
    raise RuntimeError("JILONG_ROOT must point to this project; refusing to write another version")
os.environ["JILONG_ROOT"] = str(_PROJECT)
from mmlib import io_data as io, meteo as mt, trigger as tg  # noqa: E402

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

FIG = io.FIG_OUT
OUT = io.DATA_OUT
EVENT_TS = pd.Timestamp("2026-08-26 10:30")
WIN = (pd.Timestamp("2026-08-20 00:00"), pd.Timestamp("2026-08-26 23:00"))


def _ser(x):
    if isinstance(x, dict):
        return {str(k): _ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_ser(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if not np.isfinite(v) else v
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


# ============================================================ 1. 边界层结构
def boundary_layer_structure(feat: pd.DataFrame) -> dict:
    res: dict = {}
    res["样本"] = {"起": str(feat.index.min()), "止": str(feat.index.max()),
                 "小时数": int(len(feat)), "有效温度小时数": int(feat["Ta_2m"].notna().sum()),
                 "有效降水小时数": int(feat["P"].notna().sum()),
                 "有效辐射小时数": int(feat["Rsd"].notna().sum())}

    prof = {}
    for z in (1, 2, 4, 10, 20):
        for fam, key in (("Ta", "气温_°C"), ("WS", "风速_ms-1"), ("RH", "相对湿度_%")):
            col = f"{fam}_{z}m"
            if col in feat:
                prof.setdefault(key, {})[f"{z}m"] = {
                    "mean": feat[col].mean(), "p10": feat[col].quantile(0.1),
                    "p50": feat[col].median(), "p90": feat[col].quantile(0.9)}
    res["多层次廓线统计"] = prof

    grad_cols = [f"dT_{a}_{b}m" for a, b in [(1, 2), (2, 4), (4, 10), (10, 20)]]
    res["温度直减率_K每100m"] = {c: {"mean": feat[c].mean(), "p50": feat[c].median(),
                                "p05": feat[c].quantile(0.05), "p95": feat[c].quantile(0.95)}
                            for c in grad_cols if c in feat}

    night = (feat.index.hour >= 20) | (feat.index.hour < 8)
    day = (feat.index.hour >= 9) & (feat.index.hour <= 17)
    def frequency(series, threshold):
        valid = series.dropna()
        return float((valid > threshold).mean()) if len(valid) else np.nan

    res["逆温"] = {
        "有效样本": int(feat["inv_max"].notna().sum()),
        "总频率": frequency(feat["inv_max"], 0.05), "操作阈值_K每100m":.05,
        "阈值说明": "预定操作阈值，不是由仪器精度确定的噪声界；逆温采用实际温度而非位温",
        "操作阈值敏感性": {str(t): {"总频率": frequency(feat["inv_max"], t),
                                      "夜间频率": frequency(feat.loc[night, "inv_max"], t),
                                      "白天频率": frequency(feat.loc[day, "inv_max"], t)}
                              for t in (0., .05, .5)},
        "强逆温频率": frequency(feat["inv_max"], 0.5),
        "夜间频率": frequency(feat.loc[night, "inv_max"], 0.05),
        "白天频率": frequency(feat.loc[day, "inv_max"], 0.05),
        "强度中位数_K每100m": feat["inv_max"].median(),
        "强度p95_K每100m": feat["inv_max"].quantile(0.95),
        "逆温状态下2m与20m温差中位数_K": feat.loc[feat["inv_max"] > 0.05, "dT_total_1_20m"].median(),
    }

    stab = feat["stab"].dropna()
    res["稳定度分类频率"] = {f"{int(k)}": float(v) for k, v in stab.value_counts(normalize=True).sort_index().items()}
    res["稳定度分类名称"] = {"1": "强不稳定", "2": "不稳定", "3": "中性", "4": "稳定", "5": "强稳定"}
    res["Ri_b"] = {"mean": feat["Ri_b"].mean(), "p05": feat["Ri_b"].quantile(0.05),
                   "p50": feat["Ri_b"].median(), "p95": feat["Ri_b"].quantile(0.95),
                   "切变平方屏蔽阈值_m2s2": mt.RI_SHEAR_EPS,
                   "切变屏蔽样本": feat.attrs.get("shear_mask_count"),
                   "共同有效原始样本": feat.attrs.get("eligible_count"),
                   "切变屏蔽比例": (feat.attrs.get("shear_mask_count", 0) /
                                 max(feat.attrs.get("eligible_count", 0), 1))}
    crossing = feat["pbl_crossing_height_m"]
    res["近地层有效厚度_m"] = {"mean": crossing.mean(), "p50": crossing.median(),
                          "夜间中位数": crossing.loc[night].median(),
                          "白天中位数": crossing.loc[day].median(),
                          "交点数": int(crossing.notna().sum()),
                          "交点层顶计数": {str(z): int(crossing.eq(z).sum()) for z in (4, 10, 20)},
                          "右删失数":feat.attrs.get("pbl_right_censored_count"),
                          "有效数":int(feat.pblh.notna().sum()),
                          "右删失比例": float(feat.pbl_right_censored.mean()),
                          "口径": "均值和分位只针对已观察到的离散交点，条件统计不是总体厚度；20m交点与20m右删失分开，不是CBL高度"}
    res["静力层压方法"] = feat.attrs.get("hydrostatic", {})

    diurnal = feat.groupby(feat.index.hour)[["dtheta_1_20", "shear_1_20m", "Rn", "melt_energy", "Ri_b"]].mean()
    res["日变化"] = {c: {int(h): diurnal.loc[h, c] for h in diurnal.index} for c in diurnal.columns}

    res["季节循环"] = {
        "气温月均_°C": {int(m): feat.loc[feat.index.month == m, "Ta_2m"].mean() for m in range(1, 13)},
        "降水月合计_mm": {int(m): feat.loc[feat.index.month == m, "P"].sum() for m in range(1, 13)},
        "融化能量月合计_MJm-2": {int(m): feat.loc[feat.index.month == m, "cum_melt_energy"].sum() for m in range(1, 13)},
        "净辐射月均_Wm-2": {int(m): feat.loc[feat.index.month == m, "Rn"].mean() for m in range(1, 13)},
    }

    month_threshold = feat.loc[feat.index.year <= 2025].groupby(feat.loc[feat.index.year <= 2025].index.month)["WS_20m"].quantile(.9)
    thr = pd.Series(feat.index.month, index=feat.index).map(month_threshold)
    llj = (feat["WS_20m"] > thr) & night
    res["近地层夜间强风代理_非低空急流识别"] = {"夜间频率": float(llj[night & feat["WS_20m"].notna()].mean()),
                    "WS_20m月p90均值_ms-1": float(thr.mean()),
                    "急流时WS_20m中位数_ms-1": feat.loc[llj, "WS_20m"].median()}

    ok = feat.loc[feat.index.year <= 2025, ["available", "Rn", "G_0.1m"]].dropna()
    ok = ok[(ok["Rn"] - ok["G_0.1m"]) > 20]
    if len(ok) > 100:
        eff = ok["Rn"] - ok["G_0.1m"]
        slope, intercept = np.polyfit(eff, ok["available"], 1)
        res["能量闭合"] = {"样本": int(len(ok)), "回归斜率": float(slope), "截距": float(intercept),
                       "相关系数": float(np.corrcoef(eff, ok["available"])[0, 1]),
                       "闭合比C_均值比": float(ok["available"].sum() / eff.sum()),
                       "逐时比值中位数_非C": float((ok["available"] / eff).median()),
                       "训练期": "2021–2025，共同有效且Rn-G>20 W/m²",
                       "有效能量中位数_Wm-2": float(eff.median())}
    res["波文比"] = {"中位数": feat["bowen"].median(), "p10": feat["bowen"].quantile(0.1),
                  "p90": feat["bowen"].quantile(0.9)}
    res["土壤"] = {f"{d}m_温度均值_°C": feat[f"soilT_{d}m"].mean() for d in tg.__dict__.get("DEPTHS", ["0.1", "0.2", "0.4", "0.8", "1.6"])}
    res["土壤"].update({f"{d}m_含水率均值": feat[f"VWC_{d}m"].mean() for d in ["0.1", "0.2", "0.4", "0.8", "1.6"]})
    res["土壤"]["浅层含水率p95"] = feat["VWC_0.1m"].quantile(0.95)
    res["土壤"]["浅层含水率年均最大值"] = float(feat.groupby(feat.index.year)["VWC_0.1m"].max().mean())
    res["土壤"]["深层与浅层温差_°C"] = float((feat["soilT_1.6m"] - feat["soilT_0.1m"]).mean())
    res["累计窗口可用性"] = {c: {"有效小时": int(feat[c].notna().sum()),
                                     "缺测比例": float(feat[c].isna().mean())}
                               for c in ("P_3h", "P_24h", "P_168h", "API", "PDD_3d")}
    res["季节循环"]["说明"] = "月合计为全观测期该日历月已观测量之和，不是单个完整年份月总量；缺测不填零"
    return res


def hydrostatic_sensitivity(raw: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    profiles = {"同压对照": mt.hydrostatic_profiles(raw, corrected=False)}
    profiles.update({f"参考{z}m": mt.hydrostatic_profiles(raw, z_ref=z) for z in (2, 1, 10, 20)})
    ris = {name: mt.bulk_richardson(raw, p) for name, p in profiles.items()}
    pbl = {name: mt.pbl_diagnostics(raw, p) for name, p in profiles.items()}
    common_ri = pd.concat(ris, axis=1).notna().all(axis=1)
    common_pbl = pd.concat({name: d.pblh for name, d in pbl.items()}, axis=1).notna().all(axis=1)
    common_thermo = pd.concat({name: p.thermo_valid for name, p in profiles.items()}, axis=1).all(axis=1)
    endpoint = raw[["Ta_2m", "Ta_20m", "RH_2m", "RH_20m", "Pressure"]].notna().all(axis=1)
    old_ri_possible = endpoint & mt.vector_shear_squared(raw).ge(mt.RI_SHEAR_EPS)
    base_class = pd.Series(mt.stability_class(ris["参考2m"]), index=raw.index)
    hourly = pd.DataFrame(index=raw.index)
    hourly["common_thermo"] = common_thermo
    hourly["common_Ri"] = common_ri
    hourly["common_pbl"] = common_pbl
    result = {"基准": "MET Pressure假定2m，五层Ta/RH+p共同有效，逆虚温分段梯形积分三次",
              "限制": "参考高度不是安装元数据；同压对照同样要求全五层有效；FLUX PA不替代；静力不处理非静力扰动",
              "样本": {"小时总数": len(raw), "原端点热力可用": int(endpoint.sum()),
                       "五层热力共同样本": int(common_thermo.sum()),
                       "五层要求新增热力缺失": int((endpoint & ~common_thermo).sum()),
                       "原端点可算Ri小时": int(old_ri_possible.sum()),
                       "各场景Ri共同小时": int(common_ri.sum()),
                       "Ri新增样本损失": int((old_ri_possible & ~common_ri).sum()),
                       "各场景交点或删失共同小时": int(common_pbl.sum())}, "场景": {}}
    for name, profile in profiles.items():
        ri = ris[name].loc[common_ri]
        cls = pd.Series(mt.stability_class(ri), index=ri.index)
        d = pbl[name].loc[common_pbl]
        transition = pd.crosstab(base_class.loc[common_ri], cls).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
        hourly[f"{name}_Ri"] = ris[name]
        hourly[f"{name}_交点_m"] = pbl[name].pbl_crossing_height_m
        hourly[f"{name}_右删失"] = pbl[name].pbl_right_censored
        for field in ("p_1m", "p_2m", "p_20m", "q_2m", "theta_1m", "theta_20m", "thetav_2m", "thetav_20m"):
            hourly[f"{name}_{field}"] = profile[field]
        result["场景"][name] = {"层压审计": profile.attrs,
            "共同Ri中位": ri.median(), "共同Ri_p05_p95": ri.quantile([.05, .95]).tolist(),
            "共同稳定度频率": {str(i): float(cls.eq(i).mean()) for i in range(1, 6)},
            "与2m基准分类不同数": int(cls.ne(base_class.loc[common_ri]).sum()),
            "2m基准到本场景分类计数矩阵": transition.to_numpy().tolist(),
            "共同层压p2减p20中位_hPa": (profile.p_2m - profile.p_20m).loc[common_thermo].median(),
            "共同位温差1至20中位_K": (profile.theta_20m - profile.theta_1m).loc[common_thermo].median(),
            "共同交点计数": {str(z): int(d.pbl_crossing_height_m.eq(z).sum()) for z in (4, 10, 20)},
            "共同右删失数": int(d.pbl_right_censored.eq(1).sum()),
            "共同右删失比例": d.pbl_right_censored.mean()}
    hourly["参考2m_pressure_iteration_residual_hPa"] = profiles["参考2m"].pressure_iteration_residual_hPa
    return result, hourly


def plot_hydrostatic(result: dict) -> Path:
    scenarios = result["场景"]
    names = list(scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    x = np.arange(len(names)); bottom = np.zeros(len(names))
    for i, label in enumerate(("强不稳定", "不稳定", "中性", "稳定", "强稳定"), 1):
        values = np.array([scenarios[n]["共同稳定度频率"][str(i)] for n in names])
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values
    ax.set_xticks(x, names); ax.set_ylabel("共同样本频率"); ax.set_title("(a) 层间压降订正与参考高度分开比较")
    ax.legend(fontsize=14, ncol=3)
    ax = axes[0, 1]
    ax.bar(names, [scenarios[n]["共同Ri中位"] for n in names], color="#3e727b")
    ax.axhline(0, color="black", linewidth=.7); ax.set_ylabel("整体Ri中位数")
    ax.set_title("(b) 所有场景同一有效小时集合")
    ax = axes[1, 0]
    fields = [("4", "4米交点"), ("10", "10米交点"), ("20", "20米交点")]
    bottom = np.zeros(len(names))
    for key, label in fields:
        values = np.array([scenarios[n]["共同交点计数"][key] for n in names])
        ax.bar(x, values, bottom=bottom, label=label); bottom += values
    ax.bar(x, [scenarios[n]["共同右删失数"] for n in names], bottom=bottom,
           label="20米右删失（非交点）", color="#c7c5b6", hatch="//")
    ax.set_xticks(x, names); ax.set_ylabel("小时数"); ax.legend(fontsize=14)
    ax.set_title("(c) 交点与观测上限删失严格分开")
    ax = axes[1, 1]
    matrix = np.asarray(scenarios["同压对照"]["2m基准到本场景分类计数矩阵"])
    image = ax.imshow(np.log10(matrix + 1), cmap="YlGnBu")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=12)
    ax.set_xticks(range(5), range(1, 6)); ax.set_yticks(range(5), range(1, 6))
    ax.set_xlabel("同压对照类别"); ax.set_ylabel("静力2米基准类别")
    ax.set_title("(d) 分类转移计数；颜色为log10(n+1)"); fig.colorbar(image, ax=ax, shrink=.8)
    fig.suptitle("静力订正恢复热力学一致性，不消除传感器与非静力误差", fontsize=13)
    path = FIG / "01_静力订正敏感性.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


# ============================================================ 2. 天气过程模态
def weather_regimes(feat: pd.DataFrame) -> dict:
    cols = [f"{family}_{z}m" for family in ("Ta", "RH", "WS") for z in (1, 2, 4, 10, 20)] + ["Pressure"]
    training = feat.loc[feat.index.year <= 2025, cols].dropna()
    if len(training) < 1000:
        raise ValueError("训练期完整廓线样本不足")
    means, scales = training.mean(), training.std().replace(0, np.nan)
    sub = training.iloc[::max(1, len(training) // 20000)]
    X = (sub - means) / scales
    U, S, Vt = np.linalg.svd(X.to_numpy(), full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    pc = X.to_numpy() @ Vt.T
    sampled = feat.loc[X.index]
    def correlation(left, right):
        pair = pd.DataFrame({"pc": left, "x": right}, index=X.index).dropna()
        return float(pair.corr().iloc[0, 1])
    baseline = feat.loc[feat.index.year <= 2025]
    q_p = float(baseline["P_24h"].quantile(.9))
    q_m = float(baseline["cum_melt_energy"].quantile(.9))
    available = feat[["P_24h", "cum_melt_energy"]].notna().all(axis=1)
    wet = feat.loc[available, "P_24h"] > q_p
    melt = feat.loc[available, "cum_melt_energy"] > q_m
    return {"解释方差比": var[:8].tolist(), "主成分数_85pct": int(np.searchsorted(var.cumsum(), .85) + 1),
            "样本": len(X), "训练期": "2021–2025；标准化/SVD/分型阈值均冻结于该期",
            "标准化均值": means.to_dict(), "标准化标准差": scales.to_dict(),
            "载荷": {f"EOF{k+1}": dict(zip(cols, Vt[k].tolist())) for k in range(min(4, len(Vt)))},
            "主导模态": [{"模态": f"EOF{k+1}", "解释方差": float(var[k]),
                           "主导变量": pd.Series(Vt[k], index=cols).reindex(
                               pd.Series(Vt[k], index=cols).abs().nlargest(4).index).to_dict()}
                          for k in range(min(4, len(Vt)))],
            "PC1与24h降水相关": correlation(pc[:, 0], sampled["P_24h"]),
            "PC1与融化能量相关": correlation(pc[:, 0], sampled["cum_melt_energy"]),
            "PC2与融化能量相关": correlation(pc[:, 1], sampled["cum_melt_energy"]),
            "天气型频率": {"I_强降水主导": float((wet & ~melt).mean()),
                           "II_强消融主导": float((~wet & melt).mean()),
                           "III_降水消融复合": float((wet & melt).mean()),
                           "IV_平缓天气": float((~wet & ~melt).mean())},
            "分型有效小时": int(available.sum()), "分型缺测小时": int((~available).sum()),
            "分型阈值": {"P_24h_p90_mm": q_p, "cum_melt_p90_MJm-2": q_m}}


# ============================================================ 3. 极端事件
def extreme_events(feat: pd.DataFrame) -> dict:
    targets = {"小时降水_mm": "P", "24h降水_mm": "P_24h", "小时融化能量_MJm-2": "cum_melt_energy",
               "2m气温_°C": "Ta_2m", "10m风速_ms-1": "WS_10m", "饱和差_hPa": "VPD2",
               "逆温强度_K每100m": "inv_max", "净辐射_Wm-2": "Rn", "摩擦速度_ms-1": "ustar"}
    baseline = feat.loc[feat.index.year <= 2025]
    res = {"统计口径": "阈值与尾部分布仅训练期2021–2025；有效观测小时/8765.82为年暴露量"}
    for name, col in targets.items():
        if col not in baseline:
            continue
        s = baseline[col].dropna()
        if len(s) < 500 or s.std() < 1e-9:
            continue
        u = float(s.quantile(.97))
        exceed = s[s > u]
        if len(exceed) < 30:
            continue
        exposure = len(s) / (365.2425 * 24)
        # 同一过程内超阈间隔不足12h只留最大值；缺测不创造额外暴露时长。
        cluster_id = exceed.index.to_series().diff().ge(pd.Timedelta(hours=12)).cumsum()
        peaks = exceed.groupby(cluster_id.to_numpy()).max()
        def fit(values):
            if len(values) < 10:
                return {"状态": "去丛集后样本不足", "事件数": len(values)}
            c, _, scale = stats.genpareto.fit(values.to_numpy() - u, floc=0)
            rate = len(values) / exposure
            def level(years):
                probability = 1 / (years * rate)
                if probability >= 1:
                    return None
                value = u + scale / c * (probability ** (-c) - 1) if abs(c) > 1e-7 else u + scale * np.log(1 / probability)
                if c < 0:
                    value = min(value, u - scale / c)
                return float(value)
            return {"状态": "已拟合", "事件数": len(values), "GPD_k": float(c), "GPD_sigma": float(scale),
                    "超阈频率_每年": float(rate), "2年再现期水平": level(2), "5年再现期水平": level(5),
                    "GPD有限上界": float(u - scale / c) if c < 0 else None}
        raw, declustered = fit(exceed), fit(peaks)
        chosen = declustered if declustered.get("状态") == "已拟合" else raw
        maxima = s.groupby(s.index.year).max()
        res[name] = {"阈值u_97pct": u, "超阈次数": len(exceed), "有效观测年暴露量": exposure,
                     "去丛集间隔_h": 12, "原始逐时拟合": raw, "去丛集拟合": declustered,
                     "主报告口径": "去丛集" if chosen is declustered else "逐时（去丛集样本不足）",
                     **chosen, "年最大均值": float(maxima.mean()), "年最大极值": float(maxima.max()),
                     "年最大极值出现年": int(maxima.idxmax())}
    qp, qt, qe = [float(baseline[c].quantile(.9)) for c in ("P_24h", "Ta_2m", "cum_melt_energy")]
    valid = feat[["P_24h", "Ta_2m", "cum_melt_energy"]].notna().all(axis=1)
    comp = ((feat["P_24h"] > qp) & (feat["Ta_2m"] > qt) & (feat["cum_melt_energy"] > qe)) & valid
    groups = comp.ne(comp.shift(fill_value=False)).cumsum()
    segments = comp.groupby(groups).sum()
    segments = segments[segments > 0]
    res["复合极端"] = {"判据": {"P_24h_mm": qp, "Ta_2m_°C": qt, "cum_melt_MJm-2": qe},
                       "阈值期": "2021–2025", "复合小时数": int(comp.sum()), "有效小时数": int(valid.sum()),
                       "事件段数": len(segments), "最长持续_h": int(segments.max()) if len(segments) else 0,
                       "年平均段数": len(segments) / max(valid.sum() / (365.2425 * 24), 1e-9),
                       "发生月份分布": {str(m): int((comp & (feat.index.month == m)).sum()) for m in range(1, 13)}}
    threshold = res.get("24h降水_mm", {}).get("5年再现期水平", np.inf)
    threshold = np.inf if threshold is None else threshold
    big = feat[feat["P_24h"] > threshold]
    res["5年一遇强降水超限小时数"] = len(big)
    res["5年一遇强降水时间"] = big.index[:20].astype(str).tolist()
    return res


# ============================================================ 4. 关联性排序
FEATURE_GROUPS = {
    "水源条件": {
        "24h累积降水": "P_24h", "1h降水强度": "Pmax_1h", "3h累积降水": "P_3h",
        "有效降水指数API": "API", "小时降水": "P", "7日累计降水": "AP_7d",
        "3日正积温": "PDD_3d", "7日正积温": "PDD_7d", "24h升温幅度": "rise_24h",
        "24h冻融交替次数": "freeze_thaw",
        "消融有效能量": "cum_melt_energy", "2m气温": "Ta_2m", "饱和水汽压差": "VPD2",
        "2m相对湿度": "RH_2m",
    },
    "能量与地形条件": {
        "整体理查森数": "Ri_b", "风切变": "shear_1_20m", "位温垂直梯度": "dtheta_1_20",
        "逆温强度": "inv_max", "10m风速": "WS_10m", "净辐射": "Rn", "气压": "Pressure",
    },
    "物质条件": {
        "浅层土壤含水率": "VWC_0.1m", "中层土壤含水率": "VWC_0.4m",
        "浅层土壤温度": "soilT_0.1m", "反照率": "albedo", "晴空指数": "clearness",
    },
}
MECH_PRIOR = {  # 机理先验：是否直接进入三条件驱动项
    "AP_7d": 1.0, "PDD_3d": 0.9, "PDD_7d": 0.9, "rise_24h": 0.7, "freeze_thaw": 0.5,
    "P_24h": 1.0, "Pmax_1h": 1.0, "P_3h": 1.0, "API": 1.0, "P": 0.95,
    "cum_melt_energy": 1.0, "Ta_2m": 0.9, "VPD2": 0.7, "RH_2m": 0.65,
    "Ri_b": 0.6, "shear_1_20m": 0.6, "dtheta_1_20": 0.65, "inv_max": 0.5,
    "WS_10m": 0.45, "Rn": 0.75, "Pressure": 0.2,
    "VWC_0.1m": 0.95, "VWC_0.4m": 0.9, "soilT_0.1m": 0.6, "albedo": 0.35, "clearness": 0.3,
}


def association_ranking(feat: pd.DataFrame, qw: pd.Series) -> dict:
    """主排序及CRITIC同源一致性核对；分类器只用时间留出。"""
    from mmlib.ranking import association
    return association(feat, qw, FEATURE_GROUPS, MECH_PRIOR)


def stability_by_season(feat: pd.DataFrame) -> dict:
    cols = [c for c in ["Ta_2m", "RH_2m", "WS_10m", "dtheta_1_20", "Ri_b", "VPD2", "Pressure"]
            if c in feat.columns]
    width = int((WIN[1]-WIN[0]).total_seconds()/3600)+1
    common = feat[cols].dropna().reindex(feat.index)
    complete = common.notna().all(axis=1).astype(int)
    means = common.rolling(width,min_periods=int(np.ceil(.8*width))).mean()
    good = complete.rolling(width,min_periods=width).sum() >= .8*width
    eligible = ((feat.index.year<=2025)&feat.index.month.isin([6,7,8,9]) &
                (feat.index.hour == 23) & good.to_numpy())
    warm = means.loc[eligible].dropna()
    if len(warm)<100:return {"状态":"同长度窗口不足"}
    mu,cov=warm.mean().to_numpy(),np.cov(warm.to_numpy(),rowvar=False)
    inv=np.linalg.pinv(cov)
    def md(x):
        dd=np.atleast_2d(x)-mu
        return np.sqrt(np.maximum(0,np.einsum("ij,jk,ik->i",dd,inv,dd)))
    reference=md(warm.to_numpy())
    event=common.loc[WIN[0]:WIN[1]]
    if event.notna().all(axis=1).sum()<.8*width:return {"状态":"事件同长度窗口有效率不足"}
    value=float(md(event.mean().to_numpy())[0])
    return {"暖季样本":len(warm),"窗口小时":width,"有效小时门槛":int(np.ceil(.8*width)),
            "事件有效小时":int(event.notna().all(axis=1).sum()),"参照期":"2021–2025暖季同长窗口均值，每日23时取样；窗口重叠非独立",
            "事件窗口Mahalanobis距离":value,"暖季距离p50":float(np.median(reference)),
            "暖季距离p95":float(np.percentile(reference,95)),"事件超越暖季样本比例":float((reference<value).mean()),"参与变量":cols}


# ============================================================ 绘图
def plot_all(feat: pd.DataFrame, qw: pd.Series) -> list[Path]:
    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    for z, c in zip((1, 4, 20), ("#1f77b4", "#ff7f0e", "#d62728")):
        d = feat.groupby(feat.index.hour)[f"Ta_{z}m"].mean()
        ax.plot(d.index, d.values, "o-", ms=3, color=c, label=f"{z} m")
    ax.set_xlabel("地方时 (h)"); ax.set_ylabel("气温 (°C)")
    ax.set_title("(a) 多层气温日变化"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for z, c in zip((1, 4, 20), ("#1f77b4", "#ff7f0e", "#d62728")):
        d = feat.groupby(feat.index.hour)[f"WS_{z}m"].mean()
        ax.plot(d.index, d.values, "s-", ms=3, color=c, label=f"{z} m")
    ax.set_xlabel("地方时 (h)"); ax.set_ylabel("风速 (m/s)")
    ax.set_title("(b) 多层风速日变化"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    mon = feat.groupby(feat.index.month)
    ax.bar(mon["P"].sum().index, mon["P"].sum().values, color="#4c72b0", label="月降水")
    ax.set_xlabel("月份"); ax.set_ylabel("降水 (mm)", color="#4c72b0")
    ax2 = ax.twinx()
    ax2.bar(mon["cum_melt_energy"].sum().index + 0.35,
            mon["cum_melt_energy"].sum().values / 1000.0, width=0.35, color="#dd8452")
    ax2.set_ylabel("未调整能量代理 (10³ MJ/m²)", color="#dd8452")
    ax.set_title("(c) 降水与能量代理的季节循环")
    ax.set_xticks(range(1, 13)); ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    w = feat.loc["2026-08-18":"2026-08-27"]
    ax.plot(w.index, w["Ta_2m"], color="#d62728", label="2 m 气温")
    ax.plot(w.index, w["Ta_20m"], color="#ff9896", ls="--", label="20 m 气温")
    ax.axvline(EVENT_TS, color="k", ls=":", lw=2, label="口岸冲击锚点")
    ax.set_ylabel("气温 (°C)")
    ax3 = ax.twinx()
    ax3.bar(w.index, w["P"], width=0.03, color="#4c72b0", alpha=0.6, label="小时降水")
    ax3.set_ylabel("降水 (mm/h)")
    ax.set_title("(d) 2026-08-26 事件窗口气象过程")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=14)
    ax.grid(alpha=0.3)
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
    p = FIG / "01_边界层结构.png"; fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    ax = axes[0, 0]
    st = feat["stab"].dropna()
    cnt = st.value_counts().sort_index()
    ax.bar(cnt.index, cnt.values / len(st),
           color=["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"])
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["强不稳定", "不稳定", "中性", "稳定", "强稳定"], fontsize=14)
    ax.set_ylabel("频率"); ax.set_title("(a) 近地层稳定度分类频率"); ax.grid(alpha=0.3, axis="y")

    ax = axes[0, 1]
    hh = feat.groupby(feat.index.hour)["inv_max"].apply(lambda s: (s.dropna() > 0.05).mean())
    ax.plot(hh.index, hh.values * 100, "o-", color="#6a3d9a")
    ax.set_xlabel("地方时 (h)"); ax.set_ylabel("逆温出现频率 (%)")
    ax.set_title("(b) 逆温频率日变化"); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ok = feat.loc[feat.index.year <= 2025, ["available", "Rn", "G_0.1m"]].dropna()
    ok = ok[(ok["Rn"] - ok["G_0.1m"]) > 20]
    eff = ok["Rn"] - ok["G_0.1m"]
    ax.scatter(eff, ok["available"], s=2, alpha=0.2, c="#2ca02c")
    lim = [0, float(eff.quantile(.995))]
    ax.plot(lim, lim, "k--", label="1:1 闭合线")
    slope, intercept = np.polyfit(eff, ok["available"], 1)
    ratio = float(ok["available"].sum() / eff.sum())
    ax.plot(lim, [slope * v + intercept for v in lim], "r-", label=f"回归斜率 {slope:.3f}；C={ratio:.3f}")
    ax.set_xlim(lim)
    ax.set_xlabel("可用能量 Rn−G (W/m²)"); ax.set_ylabel("湍流热通量 H+LE (W/m²)")
    ax.set_title("(c) 训练期共同有效样本的能量闭合"); ax.legend(fontsize=14); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    cols = [f"Ta_{z}m" for z in (1, 2, 4, 10, 20)] + [f"RH_{z}m" for z in (1, 2, 4, 10, 20)] + \
           [f"WS_{z}m" for z in (1, 2, 4, 10, 20)] + ["Pressure"]
    modes=json.loads((OUT/"01_天气过程模态.json").read_text(encoding="utf8"))
    Vt=np.array([[modes["载荷"][f"EOF{k}"][c] for c in cols] for k in (1,2)])
    var=np.asarray(modes["解释方差比"])
    ax.plot(Vt[0], np.arange(len(cols)), "o-", color="#1f77b4", ms=4,
            label=f"EOF1 ({var[0] * 100:.0f}%)")
    ax.plot(Vt[1], np.arange(len(cols)), "s-", color="#ff7f0e", ms=3.5,
            label=f"EOF2 ({var[1] * 100:.0f}%)")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels([c.replace("Ta_", "T").replace("RH_", "RH").replace("WS_", "WS")
                        .replace("Pressure", "气压").replace("m", " m") for c in cols], fontsize=12)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("载荷"); ax.set_title("(d) 廓线主导模态载荷")
    ax.legend(fontsize=14); ax.grid(alpha=0.3)
    p = FIG / "01_稳定度与能量.png"; fig.savefig(p, dpi=160, bbox_inches="tight"); plt.close(fig); paths.append(p)
    return paths


def plot_ranking(rank: dict) -> list[Path]:
    full = pd.DataFrame(rank["排序表"])
    display = pd.DataFrame(rank["去冗余排序表"])
    critic = pd.DataFrame(rank["CRITIC"]["去冗余排序表"])
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    colors = {"水源条件": "#4c72b0", "能量与地形条件": "#dd8452", "物质条件": "#55a868"}
    for ax, frame, value, title in (
            (axes[0, 0], display, "综合得分", "(a) 水源代理榜：训练相关连通簇去冗余"),
            (axes[0, 1], critic, "CRITIC得分", "(b) 三证据CRITIC：同源一致性核对")):
        d = frame.iloc[::-1]
        ax.barh(d["特征"], d[value], color=[colors[g] for g in d["条件组"]])
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("融合得分（越小越强）" if value == "综合得分" else "关联得分（越大越强）")
        ax.tick_params(labelsize=8); ax.grid(alpha=.25, axis="x")
    ax = axes[1, 0]
    top = display.head(5)
    positions = np.arange(len(top))
    for j, (column, name) in enumerate((("PI_rank", "置换重要性秩"), ("MI_rank", "互信息秩"),
                                       ("AN_rank", "事件异常秩"), ("MP_rank", "机理先验秩"))):
        ax.bar(positions + (j - 1.5) * .2, top[column], .2, label=name)
    ax.set_xticks(positions); ax.set_xticklabels(top["特征"], rotation=20, fontsize=8)
    ax.set_ylabel("准则秩（不是不同量纲原值）"); ax.set_title("(c) 去冗余前五的分准则秩")
    ax.legend(fontsize=8); ax.grid(alpha=.25, axis="y")
    ax = axes[1, 1]
    crit_full = pd.DataFrame(rank["CRITIC"]["排序表"]).set_index("变量")
    x = full["总次序"].to_numpy()
    y = full["变量"].map(crit_full["总次序"]).to_numpy()
    ax.scatter(x, y, c="#4c72b0", s=30)
    ax.plot([1, len(full)], [1, len(full)], "k--", lw=1)
    for _, row in full[full["变量"].isin(top["变量"])].iterrows():
        ax.annotate(row["特征"], (row["总次序"], crit_full.loc[row["变量"], "总次序"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("代理榜完整名次"); ax.set_ylabel("CRITIC完整名次")
    comparison = rank["两方案一致性"]
    ax.set_title(f"(d) 同源排序对照：τ={comparison['全排序Kendall_tau']:.2f}，去冗余前五重合{comparison['去冗余前5重合数']}")
    ax.grid(alpha=.25)
    path = FIG / "01_特征排序.png"
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return [path]


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    # 必须从当前QC后的原始数据重建，不能消费旧版本派生缓存。
    print("构建特征表（当前QC原始小时观测）…")
    raw = io.load_station()
    feat = mt.build_feature_table(raw)
    hydro, hourly = hydrostatic_sensitivity(raw)
    save_json(hydro, "01_静力层压订正.json")
    hourly.to_csv(OUT / "01_静力层压逐时.csv", encoding="utf-8-sig", index_label="timestamp")
    print("图:", plot_hydrostatic(hydro))
    feat, closure = mt.closure_sensitivity(feat)
    feat.to_parquet(io.CACHE / "features.parquet")
    save_json(closure, "01_能量闭合敏感性.json")
    print(f"特征表 {feat.shape}，{feat.index.min()} ~ {feat.index.max()}")

    save_json(boundary_layer_structure(feat), "01_边界层结构.json")
    print("边界层结构 ok")
    save_json(weather_regimes(feat), "01_天气过程模态.json")
    print("天气过程模态 ok")
    save_json(extreme_events(feat), "01_极端事件.json")
    print("极端事件 ok")

    # 问题一的水源代理榜只消费未标定水源序列；事件窗回顾 k 不得在此重建。
    model = tg.default_model(feat, calibrate=False)
    dis = model.discharge(feat)
    dis.to_parquet(io.CACHE / "qwater.parquet")

    ar = association_ranking(feat, dis["Q_water"])
    ar["暖季稀有度"] = stability_by_season(feat)
    ar["能量基准与C敏感性"] = closure["能量敏感性"]
    save_json(ar, "01_特征关联排序.json")
    save_json(ar["CRITIC"], "01_CRITIC交叉核对.json")
    print("关联排序 ok；前5：")
    for r in ar["前5特征"]:
        print(f"  {r['次序']}. {r['特征']}（{r['条件组']}，{r['变量']}）得分 {r['综合得分']:.2f}")
    print(f"  测试 AUC={ar['测试AUC']:.4f}  留一年AUC={ar['留一年AUC']}")
    print(f"  稳健性 tau 均值={ar['排序稳健性']['均值']}")

    for p in plot_all(feat, dis["Q_water"]) + plot_ranking(ar):
        print("图:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
