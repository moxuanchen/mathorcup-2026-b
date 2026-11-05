"""DEM 地形水文分析：填洼、D8 流向、汇流累积、坡向/曲率、集水区与沟道剖面。"""
from __future__ import annotations

import heapq

import numpy as np

DIRS = {1: (-1, -1), 2: (-1, 0), 3: (-1, 1), 4: (0, -1), 5: (0, 1), 6: (1, -1), 7: (1, 0), 8: (1, 1)}


def fill_sinks(z: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Priority-Flood 洼地填平（Barnes et al. 2014，堆实现，O(N log N)）。

    边界像元视为出口；填平后地形不存在闭合洼地，D8 可连续汇流至边界。
    """
    src = np.asarray(z, dtype="float64")
    h, w = src.shape
    nodata = ~np.isfinite(src)
    out = src.copy()
    closed = nodata.copy()
    heap: list[tuple[float, int, int]] = []
    border_cells = [(r, c) for r in range(h) for c in (0, w - 1)]
    border_cells += [(r, c) for c in range(w) for r in (0, h - 1)]
    for r, c in border_cells:
        if not closed[r, c]:
            closed[r, c] = True
            heapq.heappush(heap, (float(src[r, c]), r, c))
    while heap:
        level, r, c = heapq.heappop(heap)
        for dr, dc in DIRS.values():
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not closed[rr, cc]:
                closed[rr, cc] = True
                if src[rr, cc] <= level:
                    out[rr, cc] = level + eps
                    heapq.heappush(heap, (out[rr, cc], rr, cc))
                else:
                    heapq.heappush(heap, (src[rr, cc], rr, cc))
    out[nodata] = np.nan
    return out


def d8_slope(z: np.ndarray, cell: tuple[float, float] = (1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """D8 流向编码（1–8 顺时针自西北起，0 表示无下游）与最陡坡降（无量纲）。"""
    h, w = z.shape
    pad = np.pad(z, 1, mode="edge")
    best = np.zeros((h, w), dtype=np.int8)
    steep = np.zeros((h, w))
    for k, (dr, dc) in DIRS.items():
        nb = pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        drop = (z - nb) / float(np.hypot(dr * cell[0], dc * cell[1]))
        better = drop > steep
        steep = np.where(better, drop, steep)
        best = np.where(better, k, best)
    return best, steep


def d8(z: np.ndarray, cell: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """D8 流向编码。"""
    return d8_slope(z, cell)[0]


def flow_accumulation(direction: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """按 D8 流向自上游向下游累加（入度拓扑序推进）。"""
    h, w = direction.shape
    acc = np.ones((h, w)) if weights is None else np.asarray(weights, dtype="float64").copy()
    indeg = np.zeros((h, w), dtype="int32")
    rows = np.arange(h)[:, None] * np.ones((1, w), dtype="int64")
    cols = np.ones((h, 1), dtype="int64") * np.arange(w)[None, :]
    for k, (dr, dc) in DIRS.items():
        rr = rows + dr
        cc = cols + dc
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        tgt = np.zeros((h, w), dtype="int32")
        np.add.at(tgt, (rr[ok], cc[ok]), (direction == k)[ok].astype("int32"))
        indeg += tgt
    queue = [tuple(int(v) for v in p) for p in np.argwhere(indeg == 0)]
    while queue:
        r, c = queue.pop()
        k = int(direction[r, c])
        if k == 0:
            continue
        dr, dc = DIRS[k]
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            acc[rr, cc] += acc[r, c]
            indeg[rr, cc] -= 1
            if indeg[rr, cc] == 0:
                queue.append((rr, cc))
    return acc


def upstream_mask(direction: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    """从出口像元沿 D8 反向遍历，返回集水区掩膜。"""
    h, w = direction.shape
    donor: dict[int, list[int]] = {}
    for k, (dr, dc) in DIRS.items():
        rs, cs = np.nonzero(direction == k)
        rr, cc = rs + dr, cs + dc
        ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        for a, b in zip((rs * w + cs)[ok], (rr * w + cc)[ok]):
            donor.setdefault(int(b), []).append(int(a))
    mask = np.zeros(h * w, dtype=bool)
    stack = [int(seed[0] * w + seed[1])]
    while stack:
        cur = stack.pop()
        if mask[cur]:
            continue
        mask[cur] = True
        stack.extend(donor.get(cur, ()))
    return mask.reshape(h, w)


def terrain_geometry(z: np.ndarray, dy: float, dx: float) -> dict:
    gy, gx = np.gradient(z, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.degrees(np.arctan2(-gx, gy)) % 360.0
    curv = np.gradient(gx, dx, axis=1) + np.gradient(gy, dy, axis=0)
    return {"slope": slope, "aspect": aspect, "curv": curv, "gx": gx, "gy": gy}


def trace_downstream(direction: np.ndarray, start: tuple[int, int], max_steps: int = 20000) -> list[tuple[int, int]]:
    """自 start 沿 D8 顺流追踪，返回像元序列。"""
    h, w = direction.shape
    path = [(int(start[0]), int(start[1]))]
    cur = path[0]
    for _ in range(max_steps):
        k = int(direction[cur[0], cur[1]])
        if k == 0:
            break
        dr, dc = DIRS[k]
        nxt = (cur[0] + dr, cur[1] + dc)
        if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
            break
        path.append(nxt)
        cur = nxt
    return path


def trace_upstream_mainstem(direction: np.ndarray, acc: np.ndarray, start: tuple[int, int],
                            min_acc: float = 50.0, max_steps: int = 20000) -> list[tuple[int, int]]:
    """自出口向上游追踪主河道（每次选择汇流面积最大的来水像元）。"""
    h, w = direction.shape
    path = [(int(start[0]), int(start[1]))]
    cur = path[0]
    for _ in range(max_steps):
        best = None
        best_acc = -np.inf
        for k, (dr, dc) in DIRS.items():
            rr, cc = cur[0] - dr, cur[1] - dc
            if 0 <= rr < h and 0 <= cc < w and direction[rr, cc] == k and acc[rr, cc] >= min_acc:
                if acc[rr, cc] > best_acc:
                    best_acc = float(acc[rr, cc])
                    best = (int(rr), int(cc))
        if best is None:
            break
        path.append(best)
        cur = best
    return path


def channel_profile(z: np.ndarray, direction: np.ndarray, acc: np.ndarray,
                    threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """河道栅格（汇流累积 ≥ threshold）及其顺流距离场（以像元为单位）。"""
    river = acc >= threshold
    h, w = z.shape
    dist = np.where(river, 0.0, np.nan)
    queue = [(int(r), int(c)) for r, c in np.argwhere(river)]
    while queue:
        r, c = queue.pop()
        k = int(direction[r, c])
        if k == 0:
            continue
        dr, dc = DIRS[k]
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            step = float(np.hypot(dr, dc))
            if np.isnan(dist[rr, cc]):
                dist[rr, cc] = dist[r, c] + step
                queue.append((rr, cc))
            elif dist[rr, cc] < dist[r, c] + step - 1e-9:
                dist[rr, cc] = dist[r, c] + step
                queue.append((rr, cc))
    return river, dist


def slope_horn(z: np.ndarray, cell: tuple[float, float]) -> np.ndarray:
    """Horn 法坡度（正切值，3×3 加权差分）。

    与二维引擎（B2 移植）的同名函数一致，供集水区坡度统计等使用；
    返回 tanβ，调用方自行 arctan 转角度。
    """
    dy, dx = cell
    a = np.where(np.isfinite(z), z, np.nan)
    zp = np.pad(a, 1, mode="edge")
    dzdx = ((zp[2:, 2:] + 2 * zp[1:-1, 2:] + zp[:-2, 2:]) -
            (zp[2:, :-2] + 2 * zp[1:-1, :-2] + zp[:-2, :-2])) / (8 * dx)
    dzdy = ((zp[2:, 2:] + 2 * zp[2:, 1:-1] + zp[2:, :-2]) -
            (zp[:-2, 2:] + 2 * zp[:-2, 1:-1] + zp[:-2, :-2])) / (8 * dy)
    return np.hypot(dzdx, dzdy)
