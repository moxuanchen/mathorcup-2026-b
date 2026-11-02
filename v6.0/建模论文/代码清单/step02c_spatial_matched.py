"""Common-channel sampling with shared physical groups and nested spatial purge."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import step02b_spatial as legacy
from mmlib import io_data as io

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "DejaVu Sans"],
    "font.size": 22,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

SEED = 20260826
MIN_AREA_M2 = 50000.
BLOCK_M = 2500.
PURGE_M = 500.
BOOT = 300


class InsufficientDesign(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def shared_groups(labels, meta, polygons=None, grid=None):
    """Union all physical blocks touched by any single historical polygon."""
    rr, cc = np.indices(labels.shape)
    br = np.floor(rr * meta["cell_dy_m"] / BLOCK_M).astype(int)
    bc = np.floor(cc * meta["cell_dx_m"] / BLOCK_M).astype(int)
    ncols = int(bc.max()) + 1
    blocks = br * ncols + bc
    parent = np.arange(int(blocks.max()) + 1)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return int(a)

    touched = {}
    identifiers = range(1,len(polygons)+1) if polygons is not None else np.unique(labels[labels > 0])
    for pid in identifiers:
        # Group from every original polygon before overlap priority can erase a cell.
        support = legacy.rasterize([polygons[pid-1]],grid,labels.shape)[0]>0 if polygons is not None else labels == pid
        bs = np.unique(blocks[support])
        if len(bs)==0:
            continue
        touched[int(pid)] = bs.tolist()
        root = find(int(bs[0]))
        for b in bs[1:]:
            other = find(int(b))
            parent[max(root, other)] = min(root, other)
            root = find(root)
    lookup = np.array([find(i) for i in range(len(parent))])
    result = lookup[blocks]
    assert all(np.unique(lookup[np.asarray(bs)]).size == 1 for bs in touched.values())
    return result, {"physical_blocks": int(np.unique(blocks).size),
                    "connected_groups": int(np.unique(result).size),
                    "blocks_by_polygon": touched,
                    "union_rule": "all rasterized polygon cells before sampling, both sample classes share groups"}


def sample(fields, labels, pos, neg, meta, group_grid, seed, positive=None):
    rng = np.random.default_rng(seed)
    if positive is None:
        indices = []
        for pid in np.unique(labels[pos]):
            pool = np.flatnonzero(pos & (labels == pid))
            indices.extend(rng.choice(pool, min(200, len(pool)), replace=False).tolist())
        positive = np.asarray(indices, dtype=int)
    if len(positive) < 20 or neg.sum() < 2 * len(positive):
        raise InsufficientDesign("fewer than 20 positive cells or insufficient 2:1 background pool")
    negative = rng.choice(np.flatnonzero(neg), 2 * len(positive), replace=False)
    index = np.sort(np.r_[positive, negative])
    rr, cc = np.unravel_index(index, labels.shape)
    tab = pd.DataFrame({name: field.ravel()[index] for name, field in fields.items()})
    tab["row"], tab["col"] = rr, cc
    tab["x_m"], tab["y_m"] = cc * meta["cell_dx_m"], rr * meta["cell_dy_m"]
    tab["y"] = pos.ravel()[index].astype(int)
    tab["polygon"] = labels.ravel()[index]
    tab["group"] = group_grid.ravel()[index]
    assert np.isfinite(tab[legacy.CONTINUOUS].to_numpy()).all()
    assert (10 ** tab.log10_area_m2 >= MIN_AREA_M2 - 1e-6).all()
    assert (tab.groupby("polygon").group.nunique().drop(index=0, errors="ignore") <= 1).all()
    return tab, positive


def purge(train, test, xy):
    distance = cKDTree(xy[test]).query(xy[train], k=1)[0]
    keep = distance > PURGE_M
    retained = train[keep]
    return retained, {"train_before_purge": int(len(train)), "train_after_purge": int(keep.sum()),
                      "purged_n": int((~keep).sum()),
                      "minimum_retained_test_distance_m": float(distance[keep].min()) if keep.any() else None}


def grouped_folds(y, groups, xy, nmax, seed):
    attempts = []
    for n in range(min(nmax, len(np.unique(groups))), 1, -1):
        folds = []
        for train, test in StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed).split(xy, y, groups):
            kept, record = purge(train, test, xy)
            if len(np.unique(y[kept])) < 2 or len(np.unique(y[test])) < 2:
                attempts.append({"n_folds":n,"failed_fold":len(folds),"train_classes":np.unique(y[kept]).tolist(),
                                 "test_classes":np.unique(y[test]).tolist(),**record})
                break
            assert set(groups[kept]).isdisjoint(set(groups[test]))
            assert record["minimum_retained_test_distance_m"] > PURGE_M
            record.update(train_positive=int(y[kept].sum()), test_positive=int(y[test].sum()),
                          train_background=int((y[kept] == 0).sum()), test_background=int((y[test] == 0).sum()),
                          train_groups=int(np.unique(groups[kept]).size), test_groups=int(np.unique(groups[test]).size),
                          train_group_ids=np.unique(groups[kept]).tolist(), test_group_ids=np.unique(groups[test]).tolist())
            folds.append((kept, test, record))
        if len(folds) == n:
            return folds
    raise InsufficientDesign("cannot form >=2 shared-group folds with both classes after 500 m purge",
                             {"n_positive_groups":int(np.unique(groups[y==1]).size),
                              "n_background_groups":int(np.unique(groups[y==0]).size),"attempts":attempts})


def group_bootstrap(y, score, groups, seed):
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    auc = []
    for _ in range(BOOT):
        ix = np.concatenate([indices[g] for g in rng.choice(unique, len(unique), replace=True)])
        if len(np.unique(y[ix])) == 2:
            auc.append(roc_auc_score(y[ix], score[ix]))
    return {"interval": np.quantile(auc, [.025, .975]).tolist() if auc else None,
            "valid": len(auc), "requested": BOOT,
            "scope": "shared connected groups resampled once, both classes retained together; fixed OOF, not refitted-model CI"}


def evaluate(tab, classes, seed):
    x, names = legacy.matrix(tab, classes)
    y, groups = tab.y.to_numpy(), tab.group.to_numpy()
    xy = tab[["x_m", "y_m"]].to_numpy()
    lr, iv, fold_ids = np.full(len(y), np.nan), np.full(len(y), np.nan), np.full(len(y), -1)
    records = []
    outer = grouped_folds(y, groups, xy, 4, seed)
    for fold, (train, test, record) in enumerate(outer):
        try:
            inner = grouped_folds(y[train], groups[train], xy[train], 3, seed + fold + 1)
        except InsufficientDesign as exc:
            raise InsufficientDesign("inner spatial tuning unavailable: " + str(exc),
                                     {"outer_fold":fold,"outer_record":record,"inner":exc.diagnostics}) from exc
        scores = {}
        for c in legacy.CS:
            scores[c] = float(np.mean([
                roc_auc_score(y[train[b]], legacy.fit_lr(x[train[a]], y[train[a]], c).predict_proba(x[train[b]])[:, 1])
                for a, b, _ in inner]))
        c = max(legacy.CS, key=lambda candidate: scores[candidate])
        clf = legacy.fit_lr(x[train], y[train], c)
        lr[test] = clf.predict_proba(x[test])[:, 1]
        iv[test] = legacy.iv_predict(x[test], legacy.iv_fit(x[train], y[train]))
        fold_ids[test] = fold
        record.update(fold=fold, selected_C=c, inner_scores={str(k): v for k, v in scores.items()},
                      inner_folds=[rec for _, _, rec in inner],
                      standardized_coefficients=dict(zip(names, clf.named_steps["logisticregression"].coef_[0])))
        records.append(record)
    assert np.isfinite(lr).all() and np.isfinite(iv).all() and (fold_ids >= 0).all()
    oof = tab.assign(lr_oof=lr, iv_oof=iv, outer_fold=fold_ids)
    result = {"status": "available", "n_positive": int(y.sum()), "n_background": int((y == 0).sum()),
              "n_polygons": int(tab.loc[tab.y == 1, "polygon"].nunique()),
              "n_shared_groups": int(np.unique(groups).size),
              "n_groups_with_both_classes": int((tab.groupby("group").y.nunique() == 2).sum()),
              "auc_lr_oof": float(roc_auc_score(y, lr)), "auc_iv_oof": float(roc_auc_score(y, iv)),
              "lr_group_bootstrap": group_bootstrap(y, lr, groups, seed + 100),
              "iv_group_bootstrap": group_bootstrap(y, iv, groups, seed + 101),
              "outer_folds": records, "feature_names": names,
              "effective_outer_folds": len(outer), "original_landcover_codes": classes}
    return result, oof


def plot_results(scales, old, base_tab):
    fig = plt.figure(figsize=(12.5, 8.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(0.85, 1.15))
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    keys = list(scales)
    xx = np.arange(len(keys))
    ax_a.plot(xx, [old["scales"][k]["auc_lr_oof"] for k in keys], "o--", label="继承采样与分组协议")
    available = [(i, scales[k]) for i, k in enumerate(keys) if scales[k]["status"] == "available"]
    if available:
        ax_a.plot([i for i, _ in available], [r["auc_lr_oof"] for _, r in available], "s-", label="共同沟谷域＋空间清除")
    else:
        ax_a.text(.03,.09,"新协议三尺度均无可用嵌套验证\n不回退随机像元折，不填伪AUC",transform=ax_a.transAxes,fontsize=14)
    ax_a.set(xticks=xx, xticklabels=["约30米", "约90米", "约270米"], ylim=(0, 1), ylabel="外层组留出AUC", title="(a) 复合协议差异，非单项归因")
    ax_a.legend(fontsize=14)
    for i, result in available:
        bs = result["lr_group_bootstrap"]["interval"]
        if bs:
            ax_b.plot([i, i], bs, color="teal", lw=2)
            ax_b.plot(i, result["auc_lr_oof"], "o", color="teal")
    if not available:
        positive_groups = [scales[k]['sampling_frame'].get('n_sampled_positive_groups', 0) for k in keys]
        ax_b.bar(xx, positive_groups, color='teal', label="可用正类共享组")
        ax_b.axhline(8, color="crimson", ls="--", lw=1.5, label="4折最低需求（8组）")
        for i, value in enumerate(positive_groups):
            ax_b.text(i, value + 0.2, f"{value}", ha="center", fontsize=14)
        ax_b.set(xticks=xx, xticklabels=["约30米","约90米","约270米"], ylim=(0, 9),
                 ylabel="含正类的共享连通组数", title="(b) 正类组仅2个，低于4折需求")
        ax_b.legend(fontsize=11, loc="upper right")
    else:
        ax_b.set(xticks=xx, xticklabels=["约30米","约90米","约270米"], ylim=(0, 1), ylabel="AUC与固定OOF组自助带", title="(b) 共享连通组重采样，非像元独立")
    if scales[keys[0]]["status"] == "available":
        records = scales[keys[0]]["outer_folds"]
        before = np.array([r["train_before_purge"] for r in records])
        kept = np.array([r["train_after_purge"] for r in records])
        ax_c.bar(np.arange(len(records)), kept, label="隔离后训练像元")
        ax_c.bar(np.arange(len(records)), before - kept, bottom=kept, label="500米内清除")
        ax_c.legend(fontsize=14)
        ax_c.set(xlabel="外层测试折序号", ylabel="像元数（非独立事件数）", title="(c) 外层清除量；内层同样隔离")
    else:
        ax_c.bar(xx,[scales[k]['sampling_frame']['positive_common_pixels'] for k in keys],label="共同域内历史面像元")
        ax_c.set(xticks=xx,xticklabels=["约30米","约90米","约270米"],ylabel="可用正类像元数",title="(c) 像元充足不等于有独立验证组")
        ax_c.legend(fontsize=14)
    path = io.FIG_OUT / "02c_共同域空间隔离.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path.name


def main():
    dem0, grid0 = io.load_dem()
    polygons = io.read_shp_polygons(io.HAZARD_DIR / "西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp")
    glacier = legacy.glacier_lonlat()
    scales, base_tab, repeated = {}, None, []
    for k, name in [(1, "30m_nominal"), (3, "90m_nominal"), (9, "270m_nominal")]:
        print("matched spatial rebuild", name, flush=True)
        dem, grid = legacy.coarse_dem(dem0, grid0, k)
        fields, labels, valid_pos, old_neg, meta = legacy.build_fields(dem, grid, polygons, glacier)
        channel = 10 ** fields["log10_area_m2"] >= MIN_AREA_M2 - 1e-6
        # Reconstruct the exact common valid domain, not the older negative threshold.
        from scipy.ndimage import distance_transform_edt
        valid = np.isfinite(np.stack([fields[n] for n in legacy.CONTINUOUS])).all(axis=0) & (fields["LC"] >= 0)
        domain = valid & channel
        pos = domain & (labels > 0)
        buffer = distance_transform_edt(labels == 0, sampling=(meta["cell_dy_m"], meta["cell_dx_m"]))
        neg = domain & (labels == 0) & (buffer > PURGE_M)
        groups, grouping = shared_groups(labels, meta, polygons, grid)
        meta.update(common_domain_pixels=int(domain.sum()), positive_common_pixels=int(pos.sum()),
                    background_common_pixels=int(neg.sum()), positive_lost_from_old_domain=int(valid_pos.sum() - pos.sum()),
                    minimum_area_m2=MIN_AREA_M2, group_union=grouping,
                    polygon_buffer_definition='distance to rasterized polygon cell centers, not exact vector boundary',
                    polygon_buffer_half_cell_diagonal_m=float(np.hypot(meta['cell_dy_m'],meta['cell_dx_m'])/2),
                    cv_purge_definition='distance between sampled metric cell centers; >500m',
                    grouping_support='rasterized polygon support; subpixel slivers not guaranteed')
        # build_fields is shared with the legacy 02b design and reports its own 54000 m2
        # background threshold. The 02c background is drawn from the common domain above,
        # so the auditable negative threshold is MIN_AREA_M2; keep the old value explicit.
        legacy_negative = meta.get("negative_min_area_m2")
        meta["negative_min_area_m2"] = MIN_AREA_M2
        if legacy_negative is not None:
            meta["legacy_build_fields_negative_min_area_m2"] = legacy_negative
        meta["negative_domain_definition"] = (
            "common valid DEM/land-cover pixels with ground area >= minimum_area_m2, "
            "outside historical polygons by >500 m; legacy 54000 m2 metadata does not apply")
        classes = sorted(int(c) for c in np.unique(fields["LC"][domain]))
        try:
            tab, positive = sample(fields, labels, pos, neg, meta, groups, SEED + k)
            tab.to_csv(io.DATA_OUT / f"02c_{name}_共享组样本.csv", index=False, encoding="utf-8-sig")
            meta.update(n_sampled_positive_groups=int(tab.loc[tab.y==1,"group"].nunique()),
                        n_sampled_background_groups=int(tab.loc[tab.y==0,"group"].nunique()),
                        n_sampled_positive=int(tab.y.sum()), n_sampled_background=int((tab.y==0).sum()))
            if k == 1:
                base = (fields, labels, pos, neg, meta, groups, positive, classes)
            result, oof = evaluate(tab, classes, SEED + k)
            oof.to_csv(io.DATA_OUT / f"02c_{name}_共享组OOF.csv", index=False, encoding="utf-8-sig")
            if k == 1:
                base_tab = oof
        except InsufficientDesign as exc:
            result = {"status": "insufficient", "reason": str(exc), "diagnostics":exc.diagnostics,"fallback": "none"}
        result["sampling_frame"] = meta
        scales[name] = result
        print(name, result["status"], result.get("auc_lr_oof"), flush=True)
    if 'base' in locals():
        fields, labels, pos, neg, meta, groups, positive, classes = base
        for i in range(4):
            try:
                tab, _ = sample(fields, labels, pos, neg, meta, groups, SEED + 500 + i, positive)
                result, _ = evaluate(tab, classes, SEED + 1)
                repeated.append({"repeat": i, "status": "available", "auc_lr_oof": result["auc_lr_oof"],
                                 "auc_iv_oof": result["auc_iv_oof"], "n_shared_groups": result["n_shared_groups"]})
            except InsufficientDesign as exc:
                repeated.append({"repeat": i, "status": "insufficient", "reason": str(exc),"diagnostics":exc.diagnostics})
    old = json.loads((io.DATA_OUT / "02b_空间对照.json").read_text(encoding="utf8"))
    figure = plot_results(scales, old, base_tab)
    legacy.save({"seed": SEED, "scales": scales, "background_redraws": repeated,
                 "protocol": {"minimum_area_m2": MIN_AREA_M2, "shared_block_m": BLOCK_M, "purge_m": PURGE_M,
                              "outer_max": 4, "inner_max": 3, "positive_cap_per_polygon": 200, "background_ratio": 2,
                              "bootstrap_replicates": BOOT, "insufficient_action": "report unavailable; never random pixel fallback"},
                 "interpretation": "common sampling environmental contrast; unlabelled backgrounds, not occurrence probability or hazard zoning",
                 "comparison": "legacy versus new difference jointly changes candidate domain and validation protocol; no single-factor causal attribution",
                 "figure": figure}, "02c_共同域空间验证.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
