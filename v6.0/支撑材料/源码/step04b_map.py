"""Arrival-time field and terminal depth map for the D8 routing baseline."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter
from mmlib import io_data as io

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams.update({
    'font.size': 12.5,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})


def main():
    q = json.loads((io.DATA_OUT / '04b_二维引擎.json').read_text(encoding='utf-8'))
    dem, g = io.load_dem()
    cache, fig = io.CACHE, io.FIG_OUT
    arrival = np.load(cache / 'q4b_arrival.npy')
    final = np.load(cache / 'q4b_final_h.npy')
    recv = np.load(cache / 'recv.npy')
    wet = np.isfinite(arrival)
    rr, cc = np.nonzero(wet)
    pad = 25
    r0, r1 = max(0, rr.min() - pad), min(dem.shape[0], rr.max() + pad)
    c0, c1 = max(0, cc.min() - pad), min(dem.shape[1], cc.max() + pad)
    sl = np.s_[r0:r1, c0:c1]
    extent = [g['lon0'] + c0 * g['dlon'], g['lon0'] + c1 * g['dlon'],
              g['lat0'] + r1 * g['dlat'], g['lat0'] + r0 * g['dlat']]

    inj = tuple(q['断面']['注入_rc'])
    port = tuple(q['断面']['口岸投影_rc'])
    ncol = dem.shape[1]
    path = [inj]
    cur = inj
    reached = False
    for _ in range(20000):
        nxt = int(recv[cur[0] * ncol + cur[1]])
        if nxt < 0:
            break
        cur = (nxt // ncol, nxt % ncol)
        path.append(cur)
        if cur == port:
            reached = True
            break
    lat = [g['lat0'] + r * g['dlat'] for r, _ in path]
    lon = [g['lon0'] + c * g['dlon'] for _, c in path]
    tmin = np.array([arrival[r, c] / 60.0 if wet[r, c] else np.nan for r, c in path])
    segs = [((lon[i], lat[i]), (lon[i + 1], lat[i + 1])) for i in range(len(lon) - 1)]
    seg_t = np.array([np.nanmean(tmin[i:i + 2]) for i in range(len(lon) - 1)])

    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(dem[sl], vert_exag=3.0, dx=g['dlon'], dy=g['dlat'])
    norm = Normalize(vmin=0.0, vmax=float(np.nanmax(tmin)) if np.isfinite(tmin).any() else 1.0)

    f = plt.figure(figsize=(7.6, 3.05), constrained_layout=True)
    gs = f.add_gridspec(1, 4, width_ratios=[1.0, 0.030, 1.0, 0.030])
    ax0 = f.add_subplot(gs[0, 0])
    cax0 = f.add_subplot(gs[0, 1])
    ax1 = f.add_subplot(gs[0, 2])
    cax1 = f.add_subplot(gs[0, 3])
    for ax in (ax0, ax1):
        ax.imshow(shade, extent=extent, cmap='gray', alpha=.35, origin='upper')

    im0 = ax0.imshow(np.ma.masked_less_equal(np.where(wet, arrival / 60.0, 0.0)[sl], 0),
                     extent=extent, cmap='viridis', norm=norm, alpha=.95)
    lc = LineCollection(segs, cmap='viridis', norm=norm, linewidths=1.1)
    lc.set_array(seg_t)
    ax0.add_collection(lc)
    cb0 = f.colorbar(im0, cax=cax0)
    cb0.set_label('到达时间 / min', fontsize=11)
    cb0.ax.tick_params(labelsize=8)
    ax0.set_title('首达时间场与仿真流路', fontsize=13)

    vmax = float(np.nanmax(final))
    lon_w = g['lon0'] + cc * g['dlon']
    lat_w = g['lat0'] + rr * g['dlat']
    im1 = ax1.scatter(lon_w, lat_w, c=final[rr, cc], cmap='inferno', vmin=0, vmax=vmax,
                      s=9, marker='s', linewidths=0, alpha=.95)
    cb1 = f.colorbar(im1, cax=cax1)
    cb1.set_label('终态泥深 / m', fontsize=11)
    cb1.ax.tick_params(labelsize=8)
    ax1.set_title('分析域末端终态泥深（右截断）', fontsize=13)

    for ax in (ax0, ax1):
        ax.plot(g['lon0'] + inj[1] * g['dlon'], g['lat0'] + inj[0] * g['dlat'],
                'o', color='royalblue', ms=5.5, mec='white', mew=.8, label='共同注入断面')
        ax.plot(g['lon0'] + port[1] * g['dlon'], g['lat0'] + port[0] * g['dlat'],
                '*', color='crimson', ms=10, mec='white', mew=.8, label='口岸代理断面')
        ax.set_xlabel('经度 / °E', fontsize=11)
        ax.set_ylabel('纬度 / °N', fontsize=11)
        ax.set_xticks([85.38, 85.42, 85.46, 85.50])
        ax.set_yticks([28.26, 28.28, 28.30, 28.32, 28.34])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=11, loc='upper left', framealpha=.85)
    f.savefig(fig / '04b_到达与停积.png', dpi=220)
    plt.close(f)
    print(json.dumps({'reached_port_proxy': reached, 'path_cells': len(path),
                      'arrival_max_min': float(np.nanmax(tmin)),
                      'final_max_m': vmax}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
