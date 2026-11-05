"""Offline repeatable source/time alignment and target-location sensitivity audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from mmlib import io_data as io, terrain as tr, terrain_prep as tp
import step04b_sim2d as routing
ROOT=io.ROOT
RAW=ROOT/'临时产物/外部数据'
(io.DATA_OUT/'外部').mkdir(parents=True,exist_ok=True)
def read(name): return json.loads((RAW/name).read_text(encoding='utf-8'))
def meteo_hourly(name):
 d=read(name);h=d['hourly'];return d,pd.DataFrame({'T':h['temperature_2m'],'P':h['precipitation']},index=pd.to_datetime(h['time'],utc=True)).astype(float)
def nasa(name,hourly=True):
 d=read(name);p=d['properties']['parameter'];f=pd.DataFrame({'T':p['T2M'],'P':p['PRECTOTCORR']}).astype(float)
 f.index=pd.to_datetime(f.index,format='%Y%m%d%H' if hourly else '%Y%m%d',utc=True)
 return d,f.replace(d['header']['fill_value'],np.nan)
def day_summary(frame):
 out=[]
 for day,g in frame.tz_convert('Asia/Shanghai').groupby(lambda t:t.date()):
  if len(g)!=24 or not g.notna().all().all():continue
  out.append({'date':str(day),'n_hours':24,'mean_T_C':float(g['T'].mean()),'max_T_C':float(g['T'].max()),'min_T_C':float(g['T'].min()),'precip_mm':float(g['P'].sum())})
 return out
def external():
 om,a=meteo_hourly('open_meteo_era5_port_utc.json');pw,b=nasa('nasa_power_port_hourly_utc.json');hf,c=meteo_hourly('historical_forecast_ifs.json')
 station=io.load_station()[['Ta_2m','Precip']].rename(columns={'Ta_2m':'T','Precip':'P'}).copy()
 station.index=station.index.tz_localize('Asia/Shanghai').tz_convert('UTC')
 nd,n=nasa('nasa_power_station_daily_utc.json',False)
 count=station.resample('D').count();sd=station.resample('D').agg({'T':'mean','P':'sum'}).where(count==24)
 pair=sd.join(n,lsuffix='_station',rsuffix='_nasa').dropna()
 annual=[]
 for year,g in pair.groupby(pair.index.year):
  annual.append({'year':int(year),'common_complete_UTC_days':len(g),'first':str(g.index.min()),'last':str(g.index.max()),'station_precip_mm':float(g.P_station.sum()),'nasa_precip_mm':float(g.P_nasa.sum()),'ratio_nasa_to_station':float(g.P_nasa.sum()/g.P_station.sum()) if g.P_station.sum()>0 else None,'temperature_bias_nasa_minus_station_C':float((g.T_nasa-g.T_station).mean())})
 pair.to_csv(io.DATA_OUT/'外部/06_同日界逐日站点对照.csv',encoding='utf-8-sig',index_label='UTC_day')
 result={'version':'v5.0','event_day_timezone':'Asia/Shanghai','hour_stamp_assumption':'hour labels treated as hour-start; actual instrument label convention unknown','open_meteo_era5':{'request_model':'era5','elevation_downscaling':'disabled: elevation=nan','returned_grid':{k:om[k] for k in ('latitude','longitude','elevation','timezone')},'days':day_summary(a)},'nasa_power_merra2':{'header':pw['header'],'geometry':pw['geometry'],'days':day_summary(b)},'station_comparison':{'coordinate_status':'assumed representative point, not verified real station','station_timezone_assumption':'UTC+8 converted to UTC','daily_policy':'exactly 24 valid observed hours for BOTH temperature and precipitation; paired common UTC dates only','annual_matched_subsets':annual},'forecast_archive':{'model':'ecmwf_ifs04','returned_grid':{k:hf[k] for k in ('latitude','longitude','elevation','timezone')},'days':day_summary(c),'initialization':None,'lead_time':None,'status':'valid-time archive retrieved; not proof of pre-event issuance or forecast skill'},'interpretation':'Inter-product comparison at coarse grids; no independence guarantee or same-site validation. Old default-product and 18 K altitude-only explanation withdrawn.'}
 io.write_json(io.DATA_OUT/'外部/06_气象同口径核验.json',result)
 return result
def location_sensitivity():
 terrain=tp.ensure_terrain_cache();dem,grid=io.load_dem();terrain['grid']=grid
 mask,_,gt=routing.glacier_src_region();source,_=routing.reproject_src_cells(mask,gt,grid,dem.shape)
 volume=routing.load_volumes()['链A_V_df_m3'];shape=dem.shape;rows=[]
 targets=[('working',(io.EVENT['port_lon'],io.EVENT['port_lat'])),('alternative_reference',(85.3777,28.2797))]
 for name,target in targets:
  inj,port,out,offset=routing.project_sections(terrain,source,target)
  r=routing.simulate2d(terrain['recv'],terrain['slink'],terrain['dist'].ravel(),terrain['cell'],source,np.array([inj]),volume,{},port,out)
  rc=tuple(int(x) for x in np.unravel_index(port,shape));cm=tr.upstream_mask(terrain['dir'],rc)
  rr,cc=io.lonlat_to_rowcol(grid,*target);cand=np.argwhere(terrain['acc']>=5e5);near=tuple(int(x) for x in cand[np.argmin(np.hypot(cand[:,0]-rr,cand[:,1]-cc))]);hydro_mask=tr.upstream_mask(terrain['dir'],near)
  path=tr.trace_downstream(terrain['dir'],np.unravel_index(inj,shape));pi=path.index(rc);p=np.asarray(path[:pi+1]);dist=float(np.hypot(np.diff(p[:,0])*terrain['cell'][0],np.diff(p[:,1])*terrain['cell'][1]).sum())
  rows.append({'target':name,'lonlat':target,'route_snap_rc':rc,'route_snap_offset_m':offset,'route_length_m':dist,'route_section_upstream_area_km2':float(cm.sum()*np.prod(terrain['cell'])/1e6),'water_model_outlet_rc':near,'water_model_upstream_area_km2':float(hydro_mask.sum()*np.prod(terrain['cell'])/1e6),'arrival_s':r['arrival_port_s'],'peak_depth_m':r['peak_depth_port_m'],'peak_velocity_m_s':r['peak_velocity_port_m_s'],'peak_intensity_N_m':r['peak_intensity_port_N_m'],'mass_closure':r['closure_max']})
 result={'version':'v5.0','targets':rows,'alternative_coordinate_status':'inherited public-reference candidate; live Wikidata retrieval failed, exact affected section unresolved','method':'same source, grid, volume and physical parameters; only target section changed. Route and trigger-outlet snapping reported separately.','scope':'location sensitivity, not observed affected-section verification; peak intensity is not observed peak discharge'}
 io.write_json(io.DATA_OUT/'06_坐标敏感性.json',result);return result
def main():
 external();location_sensitivity()
 io.write_json(io.DATA_OUT/'06_数据可用性.json',{'version':'v5.0','actual_disaster_trigger_95CI':None,'actual_disaster_coverage':None,'issued_forecast_validation':None,'landcover_physical_weights':None,'station_metadata':'unverified coordinates, altitude, timezone and hour-label semantics','resolved':['inclusive floating endpoints','training-only constant baseline','RGI6 inventory boundary supply area','unknown landcover physical weights disabled','common UTC complete-day external comparison','target-coordinate sensitivity','source response archive','IFS single-run pre-issue availability at frozen 08:00 issue'],'unresolved':['real multi-event start times and confirmed non-events','real station installation metadata','2026 glacier and snow extent','verified landcover code dictionary','actual affected target section','verification of actual operational distribution latency','real source release and along-route inventory']})
 print('external alignment and target-section sensitivity complete')
 return 0
if __name__=='__main__':raise SystemExit(main())
