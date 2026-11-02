"""DEM 地形派生量的一次性预计算与缓存。

产出（写入 io_data.CACHE）：
    filled.npy     填洼后高程（priority-flood，eps=1e-3 递增梯度）
    dir.npy        D8 流向编码 1–8（0=无下游；与 mmlib.terrain.DIRS 同约定）
    acc.npy        D8 汇流累积（像元数）
    recv.npy       展平的 D8 接收者索引（-1=流出域外，供二维引擎使用）
    slink.npy      接收者链接坡降 (z_i - z_recv)/dist（无量纲）
    distlink.npy   接收者链接距离（m）
    terrain_meta.json  网格与像元尺寸登记

设计说明：fill/d8/acc 与既有脚本（step04/trigger/explore_terrain）读取的
缓存文件名逐一同名兼容；recv/slink/distlink 是二维过程引擎
（step04b_sim2d.py）需要的派生量，与 B2 成果的栅格运动波引擎同构。
全部量由附件7 DEM 唯一确定，属确定性预计算，因此按缓存复用。
"""
from __future__ import annotations

import json
import hashlib

import numpy as np

from . import io_data as io
from . import terrain as tr

LAT_REF = 28.4  # 像元尺寸换算参考纬度（与 step04 场量面积口径一致）
TERRAIN_VERSION = "metric-d8-v3"


def cell_meters(grid: dict) -> tuple[float, float]:
    """经纬网格 → 米制像元尺寸 (dy_m, dx_m)。"""
    dy = abs(float(grid["dlat"])) * 110574.0
    dx = abs(float(grid["dlon"])) * 111320.0 * np.cos(np.radians(LAT_REF))
    return dy, dx


def dir_to_recv(direction: np.ndarray) -> np.ndarray:
    """D8 编码 → 展平接收者索引（流出域外/无下游记 -1）。"""
    ny, nx = direction.shape
    recv = np.full(ny * nx, -1, dtype=np.int64)
    for k, (dr, dc) in tr.DIRS.items():
        rs, cs = np.nonzero(direction == k)
        rr, cc = rs + dr, cs + dc
        ok = (rr >= 0) & (rr < ny) & (cc >= 0) & (cc < nx)
        recv[rs[ok] * nx + cs[ok]] = rr[ok] * nx + cc[ok]
    return recv


def link_geometry(filled: np.ndarray, recv: np.ndarray,
                  cell: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """接收者链接的坡降与距离（米）。无下游处为 NaN。"""
    ny, nx = filled.shape
    dy, dx = cell
    flat = filled.ravel()
    idx = np.arange(ny * nx)
    has = recv >= 0
    srow, scol = np.divmod(idx[has], nx)
    rrow, rcol = np.divmod(recv[has], nx)
    dist = np.hypot((rrow - srow) * dy, (rcol - scol) * dx)
    slink = np.full(ny * nx, np.nan)
    slink[has] = (flat[idx[has]] - flat[recv[has]]) / dist
    return slink.reshape(ny, nx), dist


def prepare(dem: np.ndarray, grid: dict) -> dict:
    """对给定 DEM 网格完成 fill/d8/acc/recv/slink/distlink 计算（不落盘）。"""
    cell = cell_meters(grid)
    filled = tr.fill_sinks(dem.astype("float64"))
    direction = tr.d8(filled, cell)
    acc = tr.flow_accumulation(direction)
    recv = dir_to_recv(direction)
    slink, dist_flat = link_geometry(filled, recv, cell)
    dist = np.full(dem.size, np.nan)
    has = recv >= 0
    dist[has] = dist_flat
    return {"filled": filled, "dir": direction, "acc": acc,
            "recv": recv, "slink": slink, "dist": dist.reshape(dem.shape),
            "cell": cell, "grid": grid}


def ensure_terrain_cache(force: bool = False) -> dict:
    """加载（或首次计算并缓存）30 m 主网格的全部地形派生量。"""
    dem, grid = io.load_dem()
    meta_path = io.CACHE / "terrain_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    dem_hash = hashlib.sha256(dem.tobytes()).hexdigest()
    need = force or meta.get("algorithm_version") != TERRAIN_VERSION \
        or meta.get("dem_sha256") != dem_hash or meta.get("grid") != grid \
        or any(not (io.CACHE / name).exists() for name in
               ("filled.npy", "dir.npy", "acc.npy", "recv.npy", "slink.npy", "dist.npy"))
    if need:
        tp = prepare(dem, grid)
        np.save(io.CACHE / "filled.npy", tp["filled"])
        np.save(io.CACHE / "dir.npy", tp["dir"].astype("int8"))
        np.save(io.CACHE / "acc.npy", tp["acc"])
        np.save(io.CACHE / "recv.npy", tp["recv"])
        np.save(io.CACHE / "slink.npy", tp["slink"])
        np.save(io.CACHE / "dist.npy", tp["dist"])
        io.write_json(io.CACHE / "terrain_meta.json",
                      {"shape": list(dem.shape), "cell_dy_m": tp["cell"][0],
                       "cell_dx_m": tp["cell"][1],
                       "lat_ref": LAT_REF,
                       "fill": "priority-flood eps=1e-3（mmlib.terrain.fill_sinks）",
                       "d8": "metric maximum drop/link length; codes 1-8, 0=no receiver",
                       "algorithm_version": TERRAIN_VERSION,
                       "dem_sha256": dem_hash, "grid": grid})
        return tp
    return {"filled": np.load(io.CACHE / "filled.npy"),
            "dir": np.load(io.CACHE / "dir.npy"),
            "acc": np.load(io.CACHE / "acc.npy"),
            "recv": np.load(io.CACHE / "recv.npy"),
            "slink": np.load(io.CACHE / "slink.npy"),
            "dist": np.load(io.CACHE / "dist.npy"),
            "cell": cell_meters(grid), "grid": grid}


def prepare_coarse(k: int = 3) -> dict:
    """k×k 块均值聚合后的粗网格地形量（90 m 分辨率重演；每次现算不缓存）。"""
    dem, grid = io.load_dem()
    ny, nx = dem.shape
    ny2, nx2 = ny // k * k, nx // k * k
    b = np.where(np.isfinite(dem), dem, np.nan)[:ny2, :nx2]
    with np.errstate(all="ignore"):
        dem9 = np.nanmean(b.reshape(ny2 // k, k, nx2 // k, k), axis=(1, 3))
    grid9 = dict(grid)
    grid9["lon0"] = float(grid["lon0"]) + (k-1)/2*float(grid["dlon"])
    grid9["lat0"] = float(grid["lat0"]) + (k-1)/2*float(grid["dlat"])
    grid9["dlon"] = float(grid["dlon"]) * k
    grid9["dlat"] = float(grid["dlat"]) * k
    return prepare(dem9, grid9)
