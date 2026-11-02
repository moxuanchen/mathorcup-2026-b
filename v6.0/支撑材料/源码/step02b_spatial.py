"""问题2像元采样环境对照：分组嵌套验证、尺度与负样本敏感性。

No rasterio/fiona. Group counts, not the sampled pixel count, delimit inference.
Every resolution rebuilds terrain and polygon labels. No occurrence probability is claimed.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0,str(Path(__file__).resolve().parent))
from mmlib import io_data as io, terrain as tr, geodesy as gd
from mmlib.terrain_prep import cell_meters

SEED = 20260826
MAX_POS_PER_POLYGON = 200
NEG_RATIO = 2
BACKGROUND_BLOCK_M = 2500.
MIN_AREA_M2 = 60.*30.*30.
CS = (.01,.1,1.)
OUTER = 4
INNER = 3
NEG_REPEATS = 4
BOOT = 300
CONTINUOUS = ["Z_m","slope","log10_area_m2","TWI","relief_m","glacier_distance_m"]
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei","Noto Sans SC","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plain(value):
    if isinstance(value,dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value,(np.bool_,bool)):
        return bool(value)
    if isinstance(value,(np.integer,int)):
        return int(value)
    if isinstance(value,(np.floating,float)):
        return float(value) if np.isfinite(value) else None
    return value


def save(obj,name):
    path = io.DATA_OUT/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(plain(obj),ensure_ascii=False,indent=2),encoding="utf-8")


def coarse_dem(dem,grid,k):
    if k == 1:
        return dem.astype(float),dict(grid)
    h,w = dem.shape[0]//k,dem.shape[1]//k
    blocks = dem[:h*k,:w*k].reshape(h,k,w,k)
    count = np.isfinite(blocks).sum(axis=(1,3))
    result = np.nansum(blocks,axis=(1,3))/np.maximum(count,1)
    result[count < .8*k*k] = np.nan
    g = dict(grid)
    # Grid coordinates are pixel CENTERS; coarse centers shift by (k-1)/2.
    g.update(lon0=grid["lon0"]+(k-1)*grid["dlon"]/2,
             lat0=grid["lat0"]+(k-1)*grid["dlat"]/2,
             dlon=k*grid["dlon"],dlat=k*grid["dlat"])
    return result,g


def rasterize(polygons,grid,shape):
    labels = np.zeros(shape,dtype=np.int32)
    ny,nx = shape
    xs = grid["lon0"]+np.arange(nx)*grid["dlon"]
    ys = grid["lat0"]+np.arange(ny)*grid["dlat"]
    overlaps = 0
    for pid,poly in enumerate(polygons,1):
        xmin,ymin,xmax,ymax = poly["bbox"]
        rows = np.flatnonzero((ys>=ymin)&(ys<=ymax))
        cols = np.flatnonzero((xs>=xmin)&(xs<=xmax))
        if not len(rows) or not len(cols):
            continue
        xx,yy = np.meshgrid(xs[cols],ys[rows])
        xy = np.column_stack([xx.ravel(),yy.ravel()])
        # Even-odd ring rule retains polygon holes and disjoint parts.
        inside = np.zeros(len(xy),dtype=bool)
        points = poly["pts"]
        parts = list(poly["parts"])+[len(points)]
        for a,b in zip(parts[:-1],parts[1:]):
            inside ^= PolygonPath(points[a:b]).contains_points(xy,radius=1e-12)
        local = inside.reshape(len(rows),len(cols))
        rr,cc = np.meshgrid(rows,cols,indexing="ij")
        overlaps += int(((labels[rr,cc]>0)&local).sum())
        labels[rr[local],cc[local]] = pid
    return labels,overlaps


def glacier_lonlat():
    gn = io.load_glacier_nc()
    thick = gn["thickness"]
    mask = np.isfinite(thick)&(thick>1)
    gt = gn["geotransform"]
    rr,cc = np.indices(thick.shape)
    xx = gt[0]+(cc+.5)*gt[1]+(rr+.5)*gt[2]
    yy = gt[3]+(cc+.5)*gt[4]+(rr+.5)*gt[5]
    coords = np.array([gd.utm_to_lonlat(float(x),float(y),45)
                       for x,y in zip(xx[mask],yy[mask])])
    return coords


def nearest_lc(grid,shape):
    lc,lg = io.load_landcover(2023)
    xs = grid["lon0"]+np.arange(shape[1])*grid["dlon"]
    ys = grid["lat0"]+np.arange(shape[0])*grid["dlat"]
    ci = np.rint((xs-lg["lon0"])/lg["dlon"]).astype(int)
    ri = np.rint((ys-lg["lat0"])/lg["dlat"]).astype(int)
    valid = (ri[:,None]>=0)&(ri[:,None]<lc.shape[0])&(ci[None,:]>=0)&(ci[None,:]<lc.shape[1])
    result = np.full(shape,-1,dtype=np.int16)
    selected = lc[np.clip(ri,0,lc.shape[0]-1)[:,None],np.clip(ci,0,lc.shape[1]-1)[None,:]]
    result[valid] = selected[valid]
    return result


def build_fields(dem,grid,polygons,glacier):
    dy,dx = cell_meters(grid)
    filled = tr.fill_sinks(dem)
    # Rebuild D8 using metric diagonal lengths, including anisotropic grid spacing.
    pad = np.pad(filled,1,constant_values=np.nan)
    direction = np.zeros(dem.shape,dtype=np.int8)
    steep = np.zeros(dem.shape)
    for key,(dr,dc) in tr.DIRS.items():
        nb = pad[1+dr:1+dr+dem.shape[0],1+dc:1+dc+dem.shape[1]]
        slope = (filled-nb)/np.hypot(dr*dy,dc*dx)
        good = slope>steep
        direction[good],steep[good] = key,slope[good]
    acc = tr.flow_accumulation(direction)
    fin = np.isfinite(dem)
    z = np.where(fin,dem,0.)
    gy,gx = np.gradient(z,dy,dx)
    slope = np.hypot(gx,gy)
    # A gradient touching nodata is unavailable, rather than an artificial cliff.
    valid_gradient = ndimage.binary_erosion(fin,structure=np.ones((3,3)),border_value=0)
    slope[~valid_gradient] = np.nan
    win = max(3,int(round(330/min(dx,dy))))
    if win%2 == 0:
        win += 1
    high = ndimage.maximum_filter(np.where(fin,dem,-np.inf),size=win)
    low = ndimage.minimum_filter(np.where(fin,dem,np.inf),size=win)
    relief = high-low
    area = acc*dx*dy
    twi = np.log(np.maximum(area/dx,1.)/np.maximum(slope,1e-4))
    gmask = np.zeros(dem.shape,dtype=bool)
    rr,cc = io.lonlat_to_rowcol(grid,glacier[:,0],glacier[:,1])
    rr,cc = np.rint(rr).astype(int),np.rint(cc).astype(int)
    ok = (rr>=0)&(rr<dem.shape[0])&(cc>=0)&(cc<dem.shape[1])
    gmask[rr[ok],cc[ok]] = True
    if not gmask.any():
        raise ValueError("attachment4 glacier has no cells inside DEM")
    distance = ndimage.distance_transform_edt(~gmask,sampling=(dy,dx))
    labels,overlaps = rasterize(polygons,grid,dem.shape)
    lc = nearest_lc(grid,dem.shape)
    valid = fin&np.isfinite(slope)&np.isfinite(twi)&np.isfinite(relief)&(lc>=0)
    pos = (labels>0)&valid
    buffer_distance = ndimage.distance_transform_edt(labels==0,sampling=(dy,dx))
    neg = valid&(labels==0)&(buffer_distance>500)&(area>=MIN_AREA_M2)
    fields = dict(Z_m=dem,slope=slope,log10_area_m2=np.log10(np.maximum(area,1)),
                  TWI=twi,relief_m=relief,glacier_distance_m=distance,LC=lc)
    return fields,labels,pos,neg,dict(cell_dy_m=dy,cell_dx_m=dx,
            positive_full_pixels=int(pos.sum()),negative_pool_pixels=int(neg.sum()),
            full_valid_pixels_by_polygon={int(pid):int(((labels==pid)&pos).sum())
                                          for pid in np.unique(labels[pos])},
            polygons_with_valid_pixels=int(np.unique(labels[pos]).size),overlap_pixels_last_polygon_wins=overlaps,
            valid_dem_pixels=int(valid.sum()),relief_window_pixels=win,
            negative_min_area_m2=MIN_AREA_M2,buffer_m=500.)


def sample(fields,labels,pos,neg,meta,seed,positive_indices=None):
    rng = np.random.default_rng(seed)
    if positive_indices is None:
        selected = []
        for pid in np.unique(labels[pos]):
            pool = np.flatnonzero(pos&(labels==pid))
            selected.extend(rng.choice(pool,min(MAX_POS_PER_POLYGON,len(pool)),replace=False).tolist())
        positive_indices = np.asarray(selected,dtype=int)
    target = min(len(positive_indices),int(neg.sum())//NEG_RATIO)
    if target < len(positive_indices):
        positive_indices = rng.choice(positive_indices,target,replace=False)
    if target < 20:
        raise ValueError("insufficient matched sampling pool")
    negative_indices = rng.choice(np.flatnonzero(neg),NEG_RATIO*len(positive_indices),replace=False)
    mask = np.zeros(labels.size,dtype=bool)
    mask[positive_indices],mask[negative_indices] = True,True
    mask = mask.reshape(labels.shape)
    rr,cc = np.where(mask)
    # All arrays are extracted from exactly the same spatial-order mask.
    tab = pd.DataFrame({name:arr[mask] for name,arr in fields.items()})
    tab["row"],tab["col"] = rr,cc
    tab["y"] = pos[mask].astype(int)
    block_r = np.floor(rr*meta["cell_dy_m"]/BACKGROUND_BLOCK_M).astype(int)
    block_c = np.floor(cc*meta["cell_dx_m"]/BACKGROUND_BLOCK_M).astype(int)
    background = 1000000+block_r*10000+block_c
    tab["group"] = np.where(tab.y==1,labels[mask],background)
    assert np.array_equal(tab.y.to_numpy(),(labels[mask]>0).astype(int))
    assert set(tab.loc[tab.y==1,"group"]).isdisjoint(set(tab.loc[tab.y==0,"group"]))
    return tab,positive_indices


def matrix(tab,classes):
    x = tab[CONTINUOUS].to_numpy(dtype=float)
    cats = np.column_stack([(tab.LC.to_numpy()==c).astype(float) for c in classes])
    return np.column_stack([x,cats]),CONTINUOUS+[f"LC_original_code_{c}" for c in classes]


def folds(y,groups,n,seed):
    n = min(n,len(np.unique(groups[y==1])),len(np.unique(groups[y==0])))
    if n<2:
        raise ValueError("fewer than two independent groups per sample class")
    splitter = StratifiedGroupKFold(n_splits=n,shuffle=True,random_state=seed)
    return list(splitter.split(np.zeros(len(y)),y,groups))


def iv_fit(x,y):
    fitted = []
    for k in range(x.shape[1]):
        # Fits both cut points and class frequencies on TRAINING data only.
        bins = np.unique(np.quantile(x[:,k],[.2,.4,.6,.8]))
        edges = np.r_[-np.inf,bins,np.inf]
        idx = np.searchsorted(edges[1:-1],x[:,k],side="right")
        plus = np.bincount(idx[y==1],minlength=len(edges)-1)+.5
        minus = np.bincount(idx[y==0],minlength=len(edges)-1)+.5
        weights = np.log((plus/plus.sum())/(minus/minus.sum()))
        fitted.append((edges,weights))
    return fitted


def iv_predict(x,fitted):
    score = np.zeros(len(x))
    for k,(edges,weights) in enumerate(fitted):
        idx = np.searchsorted(edges[1:-1],x[:,k],side="right")
        score += weights[idx]
    return score


def fit_lr(x,y,c):
    clf = make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=700,
                         class_weight="balanced",solver="lbfgs",random_state=SEED))
    clf.fit(x,y)
    return clf


def nested_cv(x,y,groups,seed=SEED):
    plr,piv = np.full(len(y),np.nan),np.full(len(y),np.nan)
    fold_id = np.full(len(y),-1)
    chosen,records = [],[]
    for fold,(train,test) in enumerate(folds(y,groups,OUTER,seed)):
        assert set(groups[train]).isdisjoint(set(groups[test]))
        inner = folds(y[train],groups[train],INNER,seed+fold+1)
        scores = {}
        for c in CS:
            aucs = []
            for a,b in inner:
                ia,ib = train[a],train[b]
                if len(np.unique(y[ia]))<2 or len(np.unique(y[ib]))<2:
                    continue
                pred = fit_lr(x[ia],y[ia],c).predict_proba(x[ib])[:,1]
                aucs.append(roc_auc_score(y[ib],pred))
            scores[c] = float(np.mean(aucs)) if aucs else -np.inf
        c = max(CS,key=lambda v:scores[v])
        if not np.isfinite(scores[c]):
            raise ValueError("no valid class-balanced inner group fold")
        clf = fit_lr(x[train],y[train],c)
        plr[test] = clf.predict_proba(x[test])[:,1]
        piv[test] = iv_predict(x[test],iv_fit(x[train],y[train]))
        fold_id[test] = fold
        chosen.append(c)
        records.append({"fold":fold,"C":c,"train_groups":len(np.unique(groups[train])),
                        "test_groups":len(np.unique(groups[test])),"test_n":len(test),
                        "standardized_coefficients":clf.named_steps["logisticregression"].coef_[0].tolist()})
    assert np.isfinite(plr).all() and np.isfinite(piv).all() and (fold_id>=0).all()
    return plr,piv,fold_id,chosen,records


def group_bootstrap(y,p,groups,seed):
    rng = np.random.default_rng(seed)
    positive = np.unique(groups[y==1])
    background = np.unique(groups[y==0])
    indices = {g:np.flatnonzero(groups==g) for g in np.unique(groups)}
    aucs = []
    for _ in range(BOOT):
        pick = np.r_[rng.choice(positive,len(positive),replace=True),
                     rng.choice(background,len(background),replace=True)]
        ix = np.concatenate([indices[g] for g in pick])
        aucs.append(roc_auc_score(y[ix],p[ix]))
    return np.quantile(aucs,[.025,.975]).tolist()


def evaluate(tab,classes,seed):
    x,names = matrix(tab,classes)
    y,groups = tab.y.to_numpy(),tab.group.to_numpy()
    plr,piv,fold,chosen,records = nested_cv(x,y,groups,seed)
    c = float(pd.Series(chosen).mode().iloc[0])
    final = fit_lr(x,y,c)
    coefs = final.named_steps["logisticregression"].coef_[0]
    direction = {}
    for i,name in enumerate(CONTINUOUS):
        difference = float(x[y==1,i].mean()-x[y==0,i].mean())
        direction[name] = {"positive_minus_background_mean":difference,
                           "standardized_coefficient":float(coefs[i]),
                           "sign_flip":bool(difference*coefs[i]<0),
                           "univariate_auc_positive_direction":float(roc_auc_score(y,x[:,i])),
                           "outer_fold_coefficient_signs":[int(np.sign(r["standardized_coefficients"][i])) for r in records]}
    result = {"auc_lr_oof":roc_auc_score(y,plr),"auc_iv_oof":roc_auc_score(y,piv),
              "auc_lr_group_bootstrap_ci":group_bootstrap(y,plr,groups,seed+100),
              "auc_iv_group_bootstrap_ci":group_bootstrap(y,piv,groups,seed+101),
              "n_positive_pixels":int(y.sum()),"n_background_pixels":int((y==0).sum()),
              "n_polygon_groups":len(np.unique(groups[y==1])),
              "n_background_groups":len(np.unique(groups[y==0])),
              "sampled_pixels_by_group":tab.groupby("group").size().to_dict(),
              "outer_folds":records,"selected_C_by_fold":chosen,"final_descriptive_C":c,
              "continuous_correlation":pd.DataFrame(x[:,:len(CONTINUOUS)],columns=CONTINUOUS).corr().to_dict(),
              "feature_names":names,"original_landcover_codes":classes,
              "standardized_coefficients":dict(zip(names,coefs)),"direction_diagnostic":direction,
              "direction_note":"sign changes flag collinearity/sampling questions, not proof of bias",
              "group_bootstrap_note":"resamples polygon and background groups with replacement, fixed OOF scores, not a full model-refitting CI"}
    tab = tab.copy()
    tab["lr_oof"],tab["iv_oof"],tab["outer_fold"] = plr,piv,fold
    return result,tab,final,names


def plot_results(base,scales,tab,mapgrid,dem,grid,polygons):
    fig,axes = plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    ax = axes[0,0]
    ext = [grid["lon0"],grid["lon0"]+grid["dlon"]*(dem.shape[1]-1),
           grid["lat0"]+grid["dlat"]*(dem.shape[0]-1),grid["lat0"]]
    image = ax.imshow(mapgrid,extent=ext,cmap="YlOrRd",vmin=0,vmax=1)
    for poly in polygons:
        ax.plot(poly["pts"][:,0],poly["pts"][:,1],lw=.35,c="navy",alpha=.5)
    ax.set_xlim(ext[:2]);ax.set_ylim(ext[2:])
    ax.set_xlabel("经度 (°E)");ax.set_ylabel("纬度 (°N)")
    ax.set_title("(a) 采样环境对比分数（非发生概率）")
    fig.colorbar(image,ax=ax,label="环境对比分数")
    ax = axes[0,1]
    for name,col in (("分组嵌套逻辑回归","lr_oof"),("训练折分箱信息量","iv_oof")):
        fpr,tpr,_ = roc_curve(tab.y,tab[col])
        ax.plot(fpr,tpr,label=f"{name} AUC={roc_auc_score(tab.y,tab[col]):.3f}")
    ax.plot([0,1],[0,1],"--",c="gray");ax.legend(fontsize=8)
    ax.set_xlabel("背景像元高评分率");ax.set_ylabel("灾点面像元高评分率")
    ax.set_title("(b) 外层组留出预测（非真实灾/非灾）")
    ax = axes[1,0]
    labels=list(scales); vals=[scales[k]["auc_lr_oof"] for k in labels]
    ci=np.array([scales[k]["auc_lr_group_bootstrap_ci"] for k in labels])
    ax.errorbar(np.arange(len(vals)),vals,yerr=np.maximum(0,np.vstack([np.array(vals)-ci[:,0],ci[:,1]-vals])),fmt="o-")
    ax.set_xticks(np.arange(len(vals)),["约30米","约90米","约270米"]);ax.set_ylim(0,1)
    ax.set_ylabel("组留出AUC及组自助95%带");ax.set_title("(c) 各尺度重新建模，不证明网格收敛")
    ax=axes[1,1]
    coef=base["standardized_coefficients"]
    names=CONTINUOUS
    ax.barh(np.arange(len(names)),[coef[k] for k in names])
    ax.set_yticks(np.arange(len(names)),["高程","坡度","汇流面积对数","湿润指数","局地高差","距附件4冰川"])
    ax.axvline(0,c="gray",lw=.8);ax.set_xlabel("全样本描述拟合标准化系数")
    ax.set_title("(d) 条件系数，不作因果方向")
    path=io.FIG_OUT/"02b_空间环境对照.png"
    fig.savefig(path,dpi=160);plt.close(fig)
    return str(path)


def main():
    io.DATA_OUT.mkdir(parents=True,exist_ok=True);io.FIG_OUT.mkdir(parents=True,exist_ok=True)
    dem0,grid0=io.load_dem()
    shp=io.HAZARD_DIR/"西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp"
    polygons=io.read_shp_polygons(shp)
    glacier=glacier_lonlat()
    scales,base_tab,base_map,base_fields,base_labels,base_pos,base_neg,base_meta={},None,None,None,None,None,None,None
    for k,name in ((1,"30m_nominal"),(3,"90m_nominal"),(9,"270m_nominal")):
        print("spatial rebuild",name,flush=True)
        dem,grid=coarse_dem(dem0,grid0,k)
        fields,labels,pos,neg,meta=build_fields(dem,grid,polygons,glacier)
        classes=sorted(int(c) for c in np.unique(fields["LC"][fields["LC"]>=0]))
        tab,pidx=sample(fields,labels,pos,neg,meta,SEED+k)
        result,oof,clf,names=evaluate(tab,classes,SEED+k)
        result["sampling_frame"]=meta
        result["landcover_by_sample_class"]={str(y):oof.loc[oof.y==y,"LC"].value_counts().to_dict() for y in (0,1)}
        oof.to_csv(io.DATA_OUT/f"02b_{name}_样本OOF.csv",index=False,encoding="utf-8-sig")
        scales[name]=result
        if k==1:
            base_tab=oof;base_fields=fields;base_labels=labels;base_pos=pos;base_neg=neg;base_meta=meta
            valid=np.isfinite(fields["slope"])&np.isfinite(fields["TWI"])&(fields["LC"]>=0)
            base_map=np.full(dem.shape,np.nan,dtype="float32")
            rr,cc=np.where(valid)
            for start in range(0,len(rr),50000):
                r,c=rr[start:start+50000],cc[start:start+50000]
                local=pd.DataFrame({n:arr[r,c] for n,arr in fields.items()})
                xx,_=matrix(local,classes)
                base_map[r,c]=clf.predict_proba(xx)[:,1]
            np.save(io.DATA_OUT/"02b_环境对比分数.npy",base_map)
            positives=pidx
            base_classes=classes
        print(name,result["auc_lr_oof"],"groups",result["n_polygon_groups"],flush=True)
    repeats=[]
    for i in range(NEG_REPEATS):
        tab,_=sample(base_fields,base_labels,base_pos,base_neg,base_meta,SEED+500+i,positives)
        result,_,_,_=evaluate(tab,base_classes,SEED)
        repeats.append({"repeat":i,"auc_lr_oof":result["auc_lr_oof"],"auc_iv_oof":result["auc_iv_oof"],
                        "n_polygon_groups":result["n_polygon_groups"],"n_background_groups":result["n_background_groups"]})
    background=io.glacier_dhdt_stats()
    fig=plot_results(scales["30m_nominal"],scales,base_tab,base_map,dem0,grid0,polygons)
    save({"seed":SEED,"scales":scales,"negative_resampling":repeats,
          "n_historical_polygon_features":len(polygons),
          "base_grid":grid0,"base_shape":list(dem0.shape),
          "negative_resampling_auc_range":[min(r["auc_lr_oof"] for r in repeats),max(r["auc_lr_oof"] for r in repeats)],
          "protocol":{"positive_cap_per_polygon":MAX_POS_PER_POLYGON,"negative_ratio":NEG_RATIO,
                      "background_block_m":BACKGROUND_BLOCK_M,"bootstrap_replicates":BOOT,
                      "negative_repeats":NEG_REPEATS,"CRS":"geographic grid, pixel-center sampling",
                      "independent_units":"polygons/background blocks, not pixels",
                      "scales_rebuild":"block-mean DEM, metric D8/accumulation/slope, polygon rasterization at new centers"},
          "dhdt_static_background":background,"dhdt_in_coefficients":False,
          "dhdt_note":"regional background has no identified within-domain spatial variation; not interpolated or fitted",
          "interpretation":"sampling environment contrast only; background unlabelled, not confirmed non-disaster; not for zoning or trigger decisions",
          "figure":fig},"02b_空间对照.json")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
