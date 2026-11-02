"""Figures for current D8 routing runs, with distinct peak and final fields."""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mmlib import io_data as io
plt.rcParams['font.sans-serif']=['Microsoft YaHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['mathtext.fontset']='dejavusans'
plt.rcParams.update({
    'font.size': 19,
    'axes.titlesize': 19,
    'axes.labelsize': 17,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 13,
})


def main():
    q=json.loads((io.DATA_OUT/'04b_二维引擎.json').read_text(encoding='utf-8'))
    b=q['runs']['base']
    dem,g=io.load_dem()
    cache,fig=io.CACHE,io.FIG_OUT
    arrival=np.load(cache/'q4b_arrival.npy')
    wet=np.isfinite(arrival)
    rr,cc=np.nonzero(wet)
    pad=25
    r0,r1=max(0,rr.min()-pad),min(dem.shape[0],rr.max()+pad)
    c0,c1=max(0,cc.min()-pad),min(dem.shape[1],cc.max()+pad)
    sl=np.s_[r0:r1,c0:c1]
    extent=[g['lon0']+c0*g['dlon'],g['lon0']+c1*g['dlon'],
            g['lat0']+r1*g['dlat'],g['lat0']+r0*g['dlat']]
    def map_field(ax,a,title,vmax=None):
        ax.imshow(dem[sl],extent=extent,cmap='Greys',alpha=.3)
        im=ax.imshow(np.ma.masked_less_equal(a[sl],0),extent=extent,cmap='YlOrRd',
                     vmin=0,vmax=vmax)
        for name,rc,marker,col in [('注入',q['断面']['注入_rc'],'o','royalblue'),
                                  ('口岸代理',q['断面']['口岸投影_rc'],'*','cyan')]:
            ax.plot(g['lon0']+rc[1]*g['dlon'],g['lat0']+rc[0]*g['dlat'],
                    marker,color=col,ms=7,label=name)
        ax.set_title(title)
        ax.set_xlabel('经度 / °E');ax.set_ylabel('纬度 / °N')
        return im
    snapshots=[np.load(cache/f'q4b_snap_t{t}.npy') for t in (600,1200,1800,3600)]
    maximum=max(float(a.max()) for a in snapshots)
    f,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,t,a in zip(axes.ravel(),(600,1200,1800,3600),snapshots):
        im=map_field(ax,a,f't = {t/60:.0f} min',maximum)
    axes[0,0].legend(fontsize=13)
    f.colorbar(im,ax=axes.ravel().tolist(),label='等效泥深 / m',shrink=.75)
    f.suptitle('D8 单流向路由演进快照（不含横向二维动力）')
    f.savefig(fig/'04b_过程快照.png',dpi=180);plt.close(f)
    final=np.load(cache/'q4b_final_h.npy');peak=np.load(cache/'q4b_peak_h.npy')
    f,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True)
    for ax,a,title in zip(axes,(peak,final),('历史最大泥深','终态残留泥深（分析域末端右截断）')):
        im=map_field(ax,a,title,float(peak.max()))
    f.colorbar(im,ax=axes,label='等效泥深 / m',shrink=.75)
    f.savefig(fig/'04b_峰值与终态.png',dpi=180);plt.close(f)
    f,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    aa=np.where(wet,arrival/60,0.)
    im=map_field(axes[0],aa,'首次达到分析泥深阈值的到达时间')
    f.colorbar(im,ax=axes[0],label='到达时间 / min')
    curve=b['leader']
    axes[1].plot([x['t_s']/60 for x in curve],[x['front_m']/1000 for x in curve])
    axes[1].set(xlabel='自共同注入起算 / min',ylabel='历史到达前锋里程 / km',title='前锋按首次 h≥阈值定义')
    axes[1].grid(alpha=.25)
    f.savefig(fig/'04b_到达与前锋.png',dpi=180);plt.close(f)
    f,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    l=b['ledger'];tt=[x['t_s']/60 for x in l]
    for key,label in [('V_inj_m3','累计注入'),('V_dom_m3','域内'),('V_out_m3','累计出域')]:
        axes[0,0].plot(tt,[x[key]/1e6 for x in l],label=label)
    axes[0,0].axhline(b['V_df_m3']/1e6,ls=':',color='black',label='源项目标')
    axes[0,0].legend(fontsize=14);axes[0,0].set(ylabel=r'体积 / ($10^6$ m$^3$)',xlabel='时间 / min',title='源项目标与质量账本')
    axes[0,1].plot(tt,[x['V_loss_m3']/max(x['V_inj_m3'],1) for x in l])
    axes[0,1].set(
        xlabel='时间 / min',
        ylabel='相对收支残差',
        title=f"max Co={b['actual_cfl_max']:.3g}\n源项注入差={b['source_relative_error']:.2e}（构造）",
    )
    p=b['port_hydrograph'];pt=[x['t_s']/60 for x in p]
    axes[1,0].plot(pt,[x['h_m'] for x in p],label='泥深')
    axes[1,0].axhline(b['peak_depth_port_m'],ls=':',label='历史峰深')
    axes[1,0].legend(fontsize=14);axes[1,0].set(xlabel='时间 / min',ylabel='口岸代理泥深 / m',title='口岸过程线：峰值不等于终态')
    labels=['主网格','粗网格','半 CFL']
    rs=[b,q['scale90_rerun'],q['time_refinement']]
    axes[1,1].bar(labels,[x['arrival_port_s']/60 if x['arrival_port_s'] is not None else np.nan for x in rs])
    axes[1,1].set(ylabel='到达口岸代理 / min',title='尺度及时间步对照（不是空间收敛证明）')
    for ax in axes.ravel():ax.grid(alpha=.2)
    f.savefig(fig/'04b_质量收支.png',dpi=180);plt.close(f)
    names=list(q['runs']);vals=[q['runs'][n]['arrival_port_s'] for n in names]
    f,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    cn={'base':'基准','V_low':'体积低','V_high':'体积高','V_chainB':'体积链B',
        'n_low':'糙率低','n_high':'糙率高','tauy_low':'屈服低','tauy_high':'屈服高',
        'Cv_low':'含固低','Cv_high':'含固高','smin_low':'坡限低','smin_high':'坡限高',
        'duration_low':'源历时短','duration_high':'源历时长','arrival_low':'到达阈值低','arrival_high':'到达阈值高'}
    ax.bar(range(len(names)),[x/60 if x is not None else np.nan for x in vals])
    ax.set_xticks(range(len(names)),[cn.get(n,n) for n in names],rotation=45,ha='right')
    ax.set(ylabel='到达口岸代理 / min',title='唯一定义配置的单因素情景（非概率样本）')
    ax.grid(axis='y',alpha=.2)
    f.savefig(fig/'04b_情景敏感性.png',dpi=180);plt.close(f)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
