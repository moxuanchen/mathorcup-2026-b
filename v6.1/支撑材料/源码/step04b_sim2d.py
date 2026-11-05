"""D8 network routing on a geographic grid; conservative scenario simulation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from mmlib import geodesy as gd, io_data as io, terrain as tr, terrain_prep as tp

OUT, CACHE, FIG = io.DATA_OUT, io.CACHE, io.FIG_OUT
P2D = dict(n_man=.10, tau_y=800., s_min=.02, tau_src=1800., Cv=.45,
           rho_s=2650., rho_w=1000., h_min=.10, T_end=21600., eps_mass=1e-8,
           cfl=.20, dt_max=1.0)
SNAP_TIMES = (600, 1200, 1800, 3600)
SENS2D = {'n_low': {'n_man': .08}, 'n_high': {'n_man': .12},
          'tauy_low': {'tau_y': 400.}, 'tauy_high': {'tau_y': 1600.},
          'Cv_low': {'Cv': .15}, 'Cv_high': {'Cv': .80},
          'smin_low': {'s_min': .01}, 'smin_high': {'s_min': .04},
          'duration_low': {'tau_src': 900.}, 'duration_high': {'tau_src': 2700.},
          'arrival_low': {'h_min': .05}, 'arrival_high': {'h_min': .20}}


def rho_df_of(cv):
    return cv * P2D['rho_s'] + (1-cv) * P2D['rho_w']


def load_volumes():
    return json.loads((OUT / '04_初始体积.json').read_text(encoding='utf-8'))


def glacier_src_region():
    gn = io.load_glacier_nc()
    v = load_volumes()['链A']
    mask = (np.isfinite(gn['thickness']) & (gn['thickness'] >= v['h_thr_m']) &
            np.isfinite(gn['slope']) & (gn['slope'] >= v['sl_thr_deg']))
    gt = gn['geotransform']
    return mask, abs(gt[1]*gt[5]), gt


def reproject_src_cells(mask, gt, grid, shape):
    rr, cc = np.nonzero(mask)
    cells = []
    for r, c in zip(rr, cc):
        lo, la = gd.utm_to_lonlat(gt[0]+(c+.5)*gt[1], gt[3]+(r+.5)*gt[5], 45)
        ri = int(np.floor((la-grid['lat0'])/grid['dlat']+.5))
        ci = int(np.floor((lo-grid['lon0'])/grid['dlon']+.5))
        if 0 <= ri < shape[0] and 0 <= ci < shape[1]:
            cells.append(ri*shape[1]+ci)
    return np.unique(cells).astype('int64'), {
        '源区像元数_冰川网格': int(mask.sum()), '落域像元次': len(cells),
        '落域去重像元数': len(set(cells)), '落域比例': len(cells)/max(mask.sum(), 1)}


def project_sections(terrain, source, target_lonlat=None):
    ny, nx = terrain['filled'].shape
    fin = np.isfinite(terrain['filled']).ravel()
    source = source[fin[source]]
    if not source.size:
        raise ValueError('source geometry unavailable')
    inj = int(source[np.argmax(terrain['acc'].ravel()[source])])
    path = tr.trace_downstream(terrain['dir'], np.unravel_index(inj, (ny, nx)))
    grid, (dy, dx) = terrain['grid'], terrain['cell']
    lon, lat = target_lonlat or (io.EVENT['port_lon'], io.EVENT['port_lat'])
    rp, cp = io.lonlat_to_rowcol(grid, lon, lat)
    offsets = np.hypot((np.array([r for r,c in path])-rp)*dy,
                       (np.array([c for r,c in path])-cp)*dx)
    port_rc = path[int(np.argmin(offsets))]
    border = np.zeros((ny,nx),bool)
    border[[0,-1],:] = True
    border[:,[0,-1]] = True
    out = int(np.argmax(np.where(border & fin.reshape(ny,nx), terrain['acc'], -1)))
    return inj, int(port_rc[0]*nx+port_rc[1]), out, float(offsets.min())


def flow_length_from(source, recv, dist_flat):
    length = np.full(recv.size, np.inf)
    for start in source:
        i, d, seen = int(start), 0., set()
        while i >= 0 and i not in seen:
            seen.add(i)
            length[i] = min(length[i], d)
            nxt = int(recv[i])
            if nxt < 0:
                break
            d += float(dist_flat[i])
            i = nxt
    return length


def cumulative_source(t, volume, duration):
    u = np.clip(t/duration, 0., 1.)
    return volume*(2*u-u*u)


def simulate2d(recv, slink, dist, cell, src_px, inj_px, V_df, over,
               port_idx, out_idx, FL=None, keep_fields=False):
    p = {**P2D, **over}
    dy, dx = cell
    area, width = dy*dx, dx
    N = recv.size
    slope = np.maximum(np.nan_to_num(slink.ravel(), nan=p['s_min']), p['s_min'])
    sqrtS = np.sqrt(slope)
    density = rho_df_of(p['Cv'])
    critical = p['tau_y']/(density*9.81)
    h = np.zeros(N)
    arrival = np.full(N, np.inf)
    hp = np.zeros(N, dtype='float32') if keep_fields else None
    vp = np.zeros(N, dtype='float32') if keep_fields else None
    Ip = np.zeros(N, dtype='float32') if keep_fields else None
    inj_px = np.unique(inj_px)
    active = inj_px.copy()
    t, step, vout, vdom = 0., 0, 0., 0.
    cfl_max, min_dt, min_h, closure_max = 0., np.inf, 0., 0.
    src_err, trunc_steps, volume_check_error = 0., 0, 0.
    front, peak_port, peak_global, peak_port_v, peak_port_I = 0.,0.,0.,0.,0.
    ledger, leader, port_curve, snapshots = [], [], [], {}
    termination = 'time_limit'

    def append_record():
        injected = cumulative_source(t, V_df, p['tau_src'])
        ledger.append(dict(t_s=t, V_inj_m3=float(injected), V_dom_m3=float(vdom),
                           V_out_m3=float(vout), V_loss_m3=float(injected-vdom-vout)))
        leader.append(dict(t_s=t, front_m=front))
        vel = (h[port_idx]**(2/3)*sqrtS[port_idx]/p['n_man']
               if h[port_idx]*slope[port_idx] >= critical else 0.)
        port_curve.append(dict(t_s=t, h_m=float(h[port_idx]), v_m_s=float(vel),
                               I_N_m=float(density*h[port_idx]*vel*vel)))

    while t < p['T_end']-1e-8 and step < 500000:
        active = np.unique(np.concatenate((active,inj_px)))
        remaining_snaps = [s for s in SNAP_TIMES if s > t+1e-8]
        dt = min(p['dt_max'], p['T_end']-t,
                 (remaining_snaps[0]-t) if remaining_snaps else np.inf,
                 (p['tau_src']-t) if t < p['tau_src']-1e-8 else np.inf)
        # Include analytic injection in the state used to select the CFL step.
        for attempt in range(60):
            dv = float(cumulative_source(t+dt,V_df,p['tau_src'])-
                       cumulative_source(t,V_df,p['tau_src']))
            hh = h[active].copy()
            if dv:
                hh[np.searchsorted(active,inj_px)] += dv/(inj_px.size*area)
            vv = hh**(2/3)*sqrtS[active]/p['n_man']
            co = (5/3)*vv*width*dt/area
            maxco = float(co.max()) if co.size else 0.
            if maxco <= p['cfl']*(1+1e-10):
                break
            dt *= max(.1, .90*p['cfl']/maxco)
        else:
            raise RuntimeError('source-aware CFL failed to find a step')
        if dt <= 1e-10:
            raise RuntimeError('CFL step too small')
        min_dt, cfl_max = min(min_dt,dt), max(cfl_max,maxco)
        h[inj_px] += dv/(inj_px.size*area)
        vdom += dv
        hw = h[active]
        canflow = (hw*slope[active] >= critical) & (hw > 0.)
        vel = np.where(canflow,hw**(2/3)*sqrtS[active]/p['n_man'],0.)
        raw = vel*hw*width
        cap = hw*area/dt
        trunc_steps += int(np.any(raw > cap*(1+1e-10)))
        flow = np.minimum(raw,cap)
        inside = recv[active] >= 0
        rout = float(flow[~inside].sum())*dt
        targets = recv[active[inside]]
        touched = np.unique(np.concatenate((active,targets)))
        gain = np.bincount(np.searchsorted(touched,targets), weights=flow[inside],
                           minlength=touched.size)
        loss = np.bincount(np.searchsorted(touched,active), weights=flow,
                           minlength=touched.size)
        if keep_fields:
            hp[active] = np.maximum(hp[active],hw)
            vp[active] = np.maximum(vp[active],vel)
            Ip[active] = np.maximum(Ip[active],density*hw*vel*vel)
        peak_global = max(peak_global,float(hw.max()))
        j = np.searchsorted(active,port_idx)
        if j < active.size and active[j] == port_idx:
            peak_port = max(peak_port,float(hw[j]))
            peak_port_v = max(peak_port_v,float(vel[j]))
            peak_port_I = max(peak_port_I,float(density*hw[j]*vel[j]**2))
        h[touched] += (gain-loss)*dt/area
        vout += rout
        vdom -= rout
        min_h = min(min_h,float(h[touched].min()))
        if min_h < -1e-9:
            raise RuntimeError('negative depth violates positive update')
        t += dt
        step += 1
        newly = touched[(h[touched] >= p['h_min']) & np.isinf(arrival[touched])]
        arrival[newly] = t
        if newly.size and FL is not None:
            front = max(front,float(np.max(FL[newly][np.isfinite(FL[newly])]))
                        if np.isfinite(FL[newly]).any() else 0.)
        active = touched[h[touched] > 1e-12]
        if keep_fields:
            vv = np.where(h[touched]*slope[touched] >= critical,
                          h[touched]**(2/3)*sqrtS[touched]/p['n_man'],0.)
            hp[touched] = np.maximum(hp[touched],h[touched])
            vp[touched] = np.maximum(vp[touched],vv)
            Ip[touched] = np.maximum(Ip[touched],density*h[touched]*vv*vv)
        peak_global=max(peak_global,float(h[touched].max()))
        peak_port=max(peak_port,float(h[port_idx]))
        postv=(h[port_idx]**(2/3)*sqrtS[port_idx]/p['n_man']
               if h[port_idx]*slope[port_idx]>=critical else 0.)
        peak_port_v=max(peak_port_v,float(postv))
        peak_port_I=max(peak_port_I,float(density*h[port_idx]*postv**2))
        at_snap=any(abs(t-st)<1e-6 for st in SNAP_TIMES)
        if step % 250 == 0 or at_snap:
            actual_dom=float(h.sum())*area
            volume_check_error=max(volume_check_error,abs(actual_dom-vdom)/V_df)
            vdom=actual_dom
        injected=float(cumulative_source(t,V_df,p['tau_src']))
        closure_max=max(closure_max,abs(injected-vdom-vout)/max(injected,1.))
        if step % 25 == 0 or at_snap:
            append_record()
        if at_snap and keep_fields:
            snapshots[f't{int(round(t))}']=h.copy()
        if step % 250 == 0 and t >= max(SNAP_TIMES) and t > p['tau_src']:
            mobile=h[active][h[active]*slope[active]>=critical].sum()*area
            if mobile <= 1e-7*V_df:
                termination='stationary_after_injection'
                break
    if step >= 500000:
        termination='step_limit'
    actual_dom=float(h.sum())*area
    volume_check_error=max(volume_check_error,abs(actual_dom-vdom)/V_df)
    vdom=actual_dom
    append_record()
    injected=float(cumulative_source(t,V_df,p['tau_src']))
    closure_max=max(closure_max,abs(injected-vdom-vout)/max(injected,1.))
    src_err=abs(injected-V_df)/V_df if t >= p['tau_src'] else None
    if closure_max>p['eps_mass'] or min_h < -1e-9 or cfl_max > p['cfl']*(1+1e-8):
        raise RuntimeError('numerical acceptance failed')
    if src_err is not None and src_err > 1e-10:
        raise RuntimeError('analytic source volume acceptance failed')
    reached=np.isfinite(arrival[port_idx])
    domain_front_max = (
        float(np.nanmax(FL[np.isfinite(FL)]))
        if FL is not None and np.isfinite(FL).any()
        else None
    )
    domain_censor = bool(
        domain_front_max is not None
        and np.isfinite(front)
        and front >= 0.999 * domain_front_max
    )
    loop_termination = termination
    if loop_termination not in {"stationary_after_injection", "time_limit", "step_limit"}:
        raise RuntimeError(f"unknown terminal reason: {loop_termination}")
    # A domain-boundary endpoint is right-censored regardless of what the
    # mobile-volume check did on the last iteration; it is never a natural stop.
    if domain_censor:
        termination = "domain_censor"
    natural_stop = bool(loop_termination == "stationary_after_injection" and not domain_censor)
    time_limit = loop_termination == "time_limit"
    step_limit = loop_termination == "step_limit"
    runout_right_censored = bool(domain_censor or time_limit or step_limit)
    critical_depth_port = critical / max(float(slope[port_idx]), float(p['s_min']))
    result=dict(arrival_port_s=float(arrival[port_idx]) if reached else None,
                arrival_outlet_s=float(arrival[out_idx]) if np.isfinite(arrival[out_idx]) else None,
                peak_depth_port_m=peak_port,final_depth_port_m=float(h[port_idx]),
                peak_velocity_port_m_s=peak_port_v,peak_intensity_port_N_m=peak_port_I,
                max_depth_m=peak_global,final_max_depth_m=float(h.max()),
                wet_area_km2=float((h>.05).sum())*area/1e6,
                deposit_area_km2=float(((h>.05)&(h*slope<critical)).sum())*area/1e6,
                affected_area_km2=float(np.isfinite(arrival).sum())*area/1e6,
                n_steps=step,t_end_sim_s=t,termination_reason=termination,
                loop_termination_reason=loop_termination,
                natural_stop=natural_stop,time_limit=time_limit,
                step_limit=step_limit,domain_censor=domain_censor,
                reached_port=bool(reached),rho_df=density,
                critical_depth_slope_product_m=float(critical),
                critical_depth_port_m=float(critical_depth_port),
                params=p,V_df_m3=V_df,ledger=ledger,leader=leader,
                port_hydrograph=port_curve,closure_max=closure_max,
                source_relative_error=src_err,actual_cfl_max=cfl_max,min_dt_s=min_dt,
                nonnegative_min=min_h,volume_check_relative_error=volume_check_error,
                flux_trunc_step_frac=trunc_steps/max(step,1),runout_L_m=front,
                runout_right_censored=runout_right_censored,
                domain_front_max_m=domain_front_max,
                runout_semantics=(
                    "historical first-wetting front; at/near domain maximum is a right-censored "
                    "analysis extent, not a natural stopping distance"
                ),
                smin_path_fraction=float(np.mean(np.nan_to_num(slink.ravel()[np.isfinite(FL)],nan=0)<p['s_min']))
                if FL is not None else None)
    if keep_fields:
        shape=slink.shape
        result.update(arrival=arrival.reshape(shape),final_h=h.reshape(shape),
                      peak_h=hp.reshape(shape),peak_v=vp.reshape(shape),peak_I=Ip.reshape(shape),
                      snapshots={k:a.reshape(shape) for k,a in snapshots.items()})
    return result


def main():
    terrain=tp.ensure_terrain_cache()
    dem,grid=io.load_dem()
    terrain['grid']=grid
    vol=load_volumes()
    mask,_,gt=glacier_src_region()
    source,stats=reproject_src_cells(mask,gt,grid,dem.shape)
    inj,port,out,offset=project_sections(terrain,source)
    expected=vol['共同断面']
    if list(np.unravel_index(inj,dem.shape)) != expected['注入_rc'] or list(np.unravel_index(port,dem.shape)) != expected['口岸_rc']:
        raise ValueError('one/two dimensional common sections mismatch')
    distance=flow_length_from(np.array([inj]),terrain['recv'],terrain['dist'].ravel())
    configs={'base': ({},vol['链A_V_df_m3']),
             'V_low': ({},.5*vol['链A_V_df_m3']),
             'V_high': ({},1.5*vol['链A_V_df_m3']),
             'V_chainB': ({},vol['链B_V0_m3'])}
    configs.update({k:(v,vol['链A_V_df_m3']) for k,v in SENS2D.items()})
    runs,seen={},{}
    basefull=None
    for name,(over,V) in configs.items():
        signature=json.dumps({'params':{**P2D,**over},'V_df_m3':V},sort_keys=True)
        if signature in seen:
            continue
        seen[signature]=name
        result=simulate2d(terrain['recv'],terrain['slink'],terrain['dist'].ravel(),terrain['cell'],
                          source,np.array([inj]),V,over,port,out,FL=distance,keep_fields=name=='base')
        if name=='base':
            basefull=result
        runs[name]={k:v for k,v in result.items() if k not in ('arrival','final_h','peak_h','peak_v','peak_I','snapshots')}
        print(name,result['arrival_port_s'],result['closure_max'],flush=True)
    for field in ('arrival','final_h','peak_h','peak_v','peak_I'):
        np.save(CACHE/f'q4b_{field}.npy',basefull[field].astype('float32'))
    for key,value in basefull['snapshots'].items():
        np.save(CACHE/f'q4b_snap_{key}.npy',value.astype('float32'))
    np.save(CACHE/'q4b_cellmeta.npy',np.array([*terrain['cell'],*dem.shape]))
    (CACHE/'q4b_leader.json').write_text(json.dumps(basefull['leader']),encoding='utf-8')
    coarse=tp.prepare_coarse(k=3)
    src9,_=reproject_src_cells(mask,gt,coarse['grid'],coarse['filled'].shape)
    i9,p9,o9,offset9=project_sections(coarse,src9)
    r9=simulate2d(coarse['recv'],coarse['slink'],coarse['dist'].ravel(),coarse['cell'],src9,np.array([i9]),
                  vol['链A_V_df_m3'],{},p9,o9)
    rt=simulate2d(terrain['recv'],terrain['slink'],terrain['dist'].ravel(),terrain['cell'],source,np.array([inj]),
                  vol['链A_V_df_m3'],{'cfl':P2D['cfl']/2},port,out,FL=distance)
    result={'model':'D8 network routing, not lateral 2D dynamics','grid':grid,
            '基准参数':P2D,'源区落域统计':stats,'runs':runs,'scale90_rerun':r9,'time_refinement':rt,
            '断面':{'注入_rc':list(np.unravel_index(inj,dem.shape)),
                    '口岸投影_rc':list(np.unravel_index(port,dem.shape)),
                    '投影偏移_m':offset,'90m口岸偏移_m':offset9,
                    '全域出口_rc':list(np.unravel_index(out,dem.shape))},
            '敏感性情境总数_独立':len(runs),'到达口岸样本_s':[v['arrival_port_s'] for v in runs.values() if v['reached_port']],
            'not_reached':[k for k,v in runs.items() if not v['reached_port']],
            '账本验收':{'eps_mass':P2D['eps_mass'],'全部情景闭合比':{k:v['closure_max'] for k,v in runs.items()}},
            'scale_interpretation':'grid-dependent width and route; not a convergence claim'}
    (OUT/'04b_二维引擎.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item()),encoding='utf-8')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
