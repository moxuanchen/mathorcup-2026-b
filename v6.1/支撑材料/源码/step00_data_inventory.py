"""第 0 步：数据装载、质量评估与空间要素核验，产出统一中间数据。

输出：
  求解结果/数据/cache/*.parquet|npy          清洗后的中间数据
  求解结果/数据/00_数据清单.json              数据本体、范围、缺测与站点元信息
  求解结果/数据/00_栅格分区统计.csv           灾害区 DEM/土地覆盖分区统计
  求解结果/数据/00_冰川dhdt统计.json          四期冰川高程变化率均值与误差
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mmlib import io_data as io  # noqa: E402


def station_meta(df: pd.DataFrame) -> dict:
    """由本站气压反演站点海拔，并统计要素缺测率与观测有效期。"""
    p = df["Pressure"].dropna()
    p0, t0 = 1013.25, 288.15
    lapse, rd, g = 0.0065, 287.05, 9.80665
    elev = t0 / lapse * (1.0 - (p.median() / p0) ** (rd * lapse / g))
    qc_cols = {
        "风速": ["WS_1m", "WS_2m", "WS_4m", "WS_10m", "WS_20m"],
        "气温": ["Ta_1m", "Ta_2m", "Ta_4m", "Ta_10m", "Ta_20m"],
        "相对湿度": ["RH_1m", "RH_2m", "RH_4m", "RH_10m", "RH_20m"],
        "降水": ["Precip"],
        "辐射": ["Rsd", "Rsu", "Rld", "Rlu"],
        "土壤温湿": [c for c in df.columns if c.startswith(("soilT_", "VWC_"))],
        "湍流": ["H", "LE", "USTAR", "TKE", "TAU"],
    }
    miss = {}
    for k, cols in qc_cols.items():
        cols = [c for c in cols if c in df.columns]
        miss[k] = float(df[cols].isna().mean().mean()) if cols else None
    valid = df.notna().mean(axis=1)
    ok = df.index[valid > 0.5]
    return {
        "rows": int(len(df)),
        "valid_first": str(ok[0]) if len(ok) else None,
        "valid_last": str(ok[-1]) if len(ok) else None,
        "valid_hours": int(len(ok)),
        "missing_fraction_by_family": miss,
        "pressure_median_hPa": float(p.median()),
        "elevation_inferred_m": float(elev),
        "precip_annual_sum_mm": {str(k.year): float(v) for k, v in df["Precip"].resample("YS").sum().items()},
        "precip_max_hourly_mm": float(df["Precip"].max()),
        "ta_2m_mean": float(df["Ta_2m"].mean()),
        "ws_10m_mean": float(df["WS_10m"].mean()),
    }


def energy_balance_closure(df: pd.DataFrame) -> dict:
    rn = df["Rsd"] - df["Rsu"] + df["Rld"] - df["Rlu"]
    g0 = df["G_0.1m"] if "G_0.1m" in df else df["G_0.2m"]
    sub = pd.DataFrame({"Rn": rn, "H": df["H"], "LE": df["LE"], "G": g0}).dropna()
    hours = sub.index.hour
    sub = sub[(sub["Rn"] > 0) & (hours >= 9) & (hours <= 17)]
    avail = (sub["Rn"] - sub["G"]).to_numpy()
    turb = (sub["H"] + sub["LE"]).to_numpy()
    slope, icept = np.polyfit(avail, turb, 1)
    r2 = float(np.corrcoef(avail, turb)[0, 1] ** 2)
    return {"n": int(len(sub)), "slope": float(slope), "intercept": float(icept),
            "r2": float(r2), "ratio_sum": float(turb.sum() / avail.sum())}


def terrain_stats(dem: np.ndarray, lc: np.ndarray, grid: dict, lc_grid: dict,
                  hazards: pd.DataFrame) -> pd.DataFrame:
    """对 DEM 覆盖范围内每个历史灾点沟道，统计高程、坡度、土地覆盖构成。"""
    from scipy import ndimage

    z = dem.astype("float64")
    dy = abs(grid["dlat"]) * 110574.0
    dx = grid["dlon"] * 111320.0 * np.cos(np.radians(grid["lat0"]))
    gy, gx = np.gradient(z, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(-gx, gy)) % 360.0
    # 地形湿度指数：基于 D8 汇流累积的近似（多流向简化）
    filled = z.copy()
    filled[np.isnan(filled)] = np.nanmax(filled)
    acc = np.ones_like(filled)
    for _ in range(60):
        acc = np.minimum(1.0, ndimage.uniform_filter(acc, 3)) + 1.0
        acc = ndimage.gaussian_filter(acc, 0.6)
    tpi = z - ndimage.uniform_filter(z, 15)
    rows = []
    for _, r in hazards.iterrows():
        rr, cc = io.lonlat_to_rowcol(grid, r["lon"], r["lat"])
        ri, ci = int(round(rr)), int(round(cc))
        if not (0 <= ri < dem.shape[0] and 0 <= ci < dem.shape[1]):
            continue
        win = (slice(max(ri - 12, 0), ri + 13), slice(max(ci - 12, 0), ci + 13))
        lr, lcn = io.lonlat_to_rowcol(lc_grid, r["lon"], r["lat"])
        lri, lci = int(round(lr)), int(round(lcn))
        lwin = (slice(max(lri - 40, 0), lri + 41), slice(max(lci - 40, 0), lci + 41))
        sub_lc = lc[lwin]
        vals, counts = np.unique(sub_lc, return_counts=True)
        share = {io.LC_NAMES.get(int(v), str(v)): float(c / sub_lc.size) for v, c in zip(vals, counts)}
        rows.append({
            "code": r.get("课题"), "lon": r["lon"], "lat": r["lat"],
            "elev_m": float(z[ri, ci]), "slope_deg": float(np.nanmean(slope[win])),
            "aspect_deg": float(np.nanmean(aspect[win])), "tpi_m": float(np.nanmean(tpi[win])),
            "relief_1km_m": float(np.nanpercentile(z[win], 95) - np.nanpercentile(z[win], 5)),
            "lc_dominant": max(share, key=share.get) if share else None,
            "lc_veg_share": None,
            "lc_bare_share": None,
            "lc_water_ice_share": None,
            "lc_semantics_status": "unavailable: supplied class dictionary not verified; raw codes retained",
        })
    stats = pd.DataFrame(rows)
    stats["grad_obs"] = hazards.set_index("课题").reindex(stats["code"])["grad"].to_numpy()
    return stats


def main() -> int:
    print("[1/6] 装载站点观测 ...")
    station = io.load_station()
    meta = station_meta(station)
    meta["energy_balance_closure"] = energy_balance_closure(station)
    io.write_json(io.DATA_OUT / "00_站点元信息.json", meta)

    print("[2/6] 装载气温序列 ...")
    temp = io.load_temperature(use_cache=False)

    print("[3/6] 装载灾点与人口 ...")
    hazards = io.load_hazard_points(use_cache=False)
    census = io.load_census()

    print("[4/6] 装载 DEM 与土地覆盖 ...")
    dem, dgrid = io.load_dem()
    lc, lgrid = io.load_landcover(2023)

    print("[5/6] 灾点地形/覆被统计 ...")
    stats = terrain_stats(dem, lc, dgrid, lgrid, hazards)
    stats.to_csv(io.DATA_OUT / "00_栅格分区统计.csv", index=False, encoding="utf-8-sig")

    print("[6/6] 冰川高程变化统计 ...")
    dhdt = io.glacier_dhdt_stats()
    io.write_json(io.DATA_OUT / "00_冰川dhdt统计.json", dhdt)

    inventory = {
        "station": {**io.STATION, **meta},
        "temperature": {"county_count": 5, "first": str(temp.index.min().date()),
                        "last": str(temp.index.max().date()), "rows": int(len(temp)),
                        "columns": list(temp.columns)},
        "hazards": {"count": int(len(hazards)),
                    "types": hazards["类型"].value_counts().to_dict(),
                    "scales": hazards["规模"].value_counts().to_dict(),
                    "frequencies": hazards["频率"].value_counts().to_dict(),
                    "lon_range": [float(hazards["lon"].min()), float(hazards["lon"].max())],
                    "lat_range": [float(hazards["lat"].min()), float(hazards["lat"].max())],
                    "grad_range": [float(hazards["grad"].min()), float(hazards["grad"].max())],
                    "relief_range": [float(hazards["relief_m"].min()), float(hazards["relief_m"].max())]},
        "dem": {"shape": list(dem.shape), "grid": dgrid,
                "elev_range": [float(np.nanmin(dem)), float(np.nanmax(dem))],
                "cell_m": [float(dgrid["dlon"] * 111320 * np.cos(np.radians(dgrid["lat0"]))),
                           float(abs(dgrid["dlat"]) * 110574)]},
        "landcover": {"shape": list(lc.shape), "grid": lgrid,
                      "classes": {io.LC_NAMES.get(int(v), str(v)): float(c / lc.size)
                                  for v, c in zip(*np.unique(lc, return_counts=True))}},
        "glacier_dhdt": dhdt,
        "census": census,
        "event": io.EVENT,
    }
    io.write_json(io.DATA_OUT / "00_数据清单.json", inventory)
    print("完成：", io.DATA_OUT / "00_数据清单.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
