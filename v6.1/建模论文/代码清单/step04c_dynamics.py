"""问题4：有限库存分段动力对照。只读旧04输入，独立写04c结果。"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.isotonic import IsotonicRegression

from mmlib import finite_inventory as fi
from mmlib import io_data as io
from test04c_dynamics import run_report

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "mathtext.fontset": "dejavusans",
                     "font.size": 20, "axes.titlesize": 20, "axes.labelsize": 18,
                     "xtick.labelsize": 15, "ytick.labelsize": 15,
                     "legend.fontsize": 13, "savefig.dpi": 180})

DOMAIN = {"mu_dyn": [.05, .1, .15, .2], "kappa_m_inv": [0., 1e-5, 5e-5, 1e-4],
          "v0_m_s": [math.sqrt(200), 30., 60.], "q_r": [0., .5, 1.], "q_w": [0., .5, 1.],
          "baseline": {"mu": .1, "kappa": 1e-5, "v0": 30.},
          "node_horizontal_fractions": [1/3, 2/3], "node_weights": [.5, .5],
          "M2_mu_multiplier": .6, "target_vreq_m_s": 1.,
          "inverse_relative_perturbation": 1e-6}
LABELS = {"raw": "原始剖面", "filled": "填洼剖面", "isotonic": "单调剖面"}
COLORS = {"raw": "#a14332", "filled": "#27767d", "isotonic": "#283d65"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def save(obj, name):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    p = io.DATA_OUT/name
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=convert,
                            allow_nan=False), encoding="utf-8")
    return p


def summary(result):
    return {k: v for k, v in result.items()
            if k not in ("trajectory", "segment_ledger", "node_ledger")}


def load_geometry():
    required = [io.DATA_OUT/"04_仿真基线.json", io.DATA_OUT/"04_初始体积.json",
                io.CACHE/"dem.npy", io.CACHE/"filled.npy", io.CACHE/"terrain_meta.json"]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"先运行原04与地形缓存，缺少 {p}")
    old = json.loads(required[0].read_text(encoding="utf-8"))
    volume = json.loads(required[1].read_text(encoding="utf-8"))
    meta = json.loads(required[-1].read_text(encoding="utf-8"))
    path = np.asarray(old["中心线"]["path_rc"], int)
    dem, filled = np.load(required[2], mmap_mode="r"), np.load(required[3], mmap_mode="r")
    delta = np.diff(path, axis=0)
    if np.any(np.max(np.abs(delta), axis=1) > 1) or np.any(np.all(delta == 0, axis=1)):
        raise ValueError("原生D8路线不是相邻且非重复的像元序列")
    dh = np.hypot(delta[:, 0]*meta["cell_dy_m"], delta[:, 1]*meta["cell_dx_m"])
    sh = np.r_[0., dh.cumsum()]
    if not math.isclose(sh[-1]/1000, old["total_km"], abs_tol=1e-6):
        raise ValueError("原生水平里程与旧共同路线不一致")
    zr, zf = np.asarray(dem[path[:, 0], path[:, 1]], float), np.asarray(filled[path[:, 0], path[:, 1]], float)
    zi = IsotonicRegression(increasing=False).fit_transform(np.arange(len(zr)), zr)
    profiles = {"raw": zr, "filled": zf, "isotonic": zi}
    if not all(np.all(np.isfinite(z)) for z in profiles.values()):
        raise ValueError("路线存在缺测高程")
    cv, vdf = float(old["Cv"]), float(volume["采用_V0_m3"])
    initial = np.array([cv*vdf, 0., (1-cv)*vdf])
    if not math.isclose(fi.mass(initial)/vdf, old["rho_m"], abs_tol=1e-6):
        raise ValueError("等效岩水初值与旧密度不一致")
    details = {"native_node_count": len(path), "horizontal_length_m": sh[-1],
               "native_path_rc": path.tolist(), "route_s_h_m": sh.tolist(),
               "profile_z_m": {k: z.tolist() for k, z in profiles.items()},
               "horizontal_length_source": "native D8 row/column differences with terrain_meta metric cells",
               "profiles": {k: {"three_dimensional_length_m": float(np.hypot(dh, np.diff(z)).sum()),
                                  "uphill_steps": int(np.sum(np.diff(z) > 0)),
                                  "drop_m": float(z[0]-z[-1])} for k, z in profiles.items()},
               "start_lonlat": old["中心线"]["start_lonlat"],
               "port_proxy_lonlat": old["中心线"]["end_lonlat"],
               "port_offset_m": old["中心线"]["port_offset_m"],
               "original_isotonic_max_adjustment_m": float(np.max(abs(zi-zr))),
               "old_diagnostic": {k: old[k] for k in ["runout_km", "arrival_outlet_s", "stop_time_s", "mu"]},
               "initial_rock_ice_water_m3": initial.tolist(), "initial_Cv": cv,
               "initial_mass_kg": fi.mass(initial),
               "Vi_zero_semantics": "equivalent rock-water representation; not evidence all real ice melted",
               "gamma_account": "Vdf already includes inlet enrichment; synthetic stocks are separate hypothetical downstream accounts",
               "input_sha256": {str(p.relative_to(io.ROOT)): sha(p) for p in required}}
    return sh, profiles, initial, details


def artificial_nodes(sh, initial, qr, qw):
    if qr == 0 and qw == 0:
        return []
    return [{"id": f"artificial_{j+1}", "s_h_m": sh[-1]*fraction,
             "rock_m3": qr*initial[0]*.5, "water_m3": qw*initial[2]*.5}
            for j, fraction in enumerate(DOMAIN["node_horizontal_fractions"])]


def checked_inverse(sh, z, initial, **kwargs):
    inv = fi.inverse_threshold(sh, z, initial, vreq=1., **kwargs)
    v = inv["required_v0_m_s"]
    if v is None or v <= 0:
        inv["forward_check"] = {"status": "zero_or_unrepresentable_threshold_needs_start_rule"}
        return inv
    eps = DOMAIN["inverse_relative_perturbation"]
    checks = {}
    for tag, vv in [("below", v*(1-eps)), ("above", v*(1+eps))]:
        r = fi.simulate(sh, z, initial, v0=vv, **kwargs)
        success = r["reached_target_positive"] and r["final_velocity_m_s"] >= 1.
        checks[tag] = {"v0_m_s": vv, "status": r["status"],
                       "velocity_target_or_stop_m_s": r["final_velocity_m_s"],
                       "meets_vreq": bool(success)}
    inv["forward_check"] = checks
    inv["forward_check_pass"] = not checks["below"]["meets_vreq"] and checks["above"]["meets_vreq"]
    if not inv["forward_check_pass"]:
        raise AssertionError(f"逆向阈值未通过前向核验: {checks}")
    inv["required_v0_within_declared_scan_range"] = DOMAIN["v0_m_s"][0] <= v <= DOMAIN["v0_m_s"][-1]
    return inv


def water_competition(result):
    records = []
    for node in result["node_ledger"]:
        downstream = [s for s in result["segment_ledger"] if s["s0_m"] >= node["s_h_m"]-1e-9]
        if not downstream:
            continue
        seg = downstream[0]
        if seg["a_m_s2"] >= 0:
            records.append({"node_id": node["id"], "status": "immediate_segment_not_decelerating"})
            continue
        b, k, length = -seg["a_m_s2"], seg["kappa_m_inv"], seg["length_m"]
        crit2 = b/k*math.expm1(2*k*length) if k else 2*b*length
        vpre = node["velocity_after_m_s"]*(1+node["lambda"])
        dv = node["consumed_volumes_m3"]
        other = fi.RHO_R*dv[0]+fi.RHO_I*dv[1]
        upper = (node["mass_before_kg"]*(vpre/math.sqrt(crit2)-1)-other)/fi.RHO_W
        records.append({"node_id": node["id"], "status": "fixed_actual_segment_local_necessary_bound",
                        "segment_s1_m": seg["s1_m"], "vcrit_m_s": math.sqrt(crit2),
                        "strict_added_water_upper_m3": upper, "actual_added_water_m3": dv[2],
                        "R_local": vpre*vpre/((1+node["lambda"])**2*crit2),
                        "not_global_or_true_material_boundary": True})
    return records


def real_route_refinement(sh, profiles, initial):
    checks = []
    for name, z in profiles.items():
        for mode in ["M0", "M1", "M2"]:
            nodes = artificial_nodes(sh, initial, .5, .5)
            a = fi.simulate(sh, z, initial, nodes=nodes, mode=mode, **DOMAIN["baseline"])
            b = fi.simulate(sh, z, initial, nodes=nodes, mode=mode,
                            subdivisions=5, **DOMAIN["baseline"])
            entry = {"profile": name, "mode": mode, "refinement_factor": 5,
                     "horizontal_distance_difference_m": b["final_sh_m"]-a["final_sh_m"],
                     "time_difference_s": b["final_time_s"]-a["final_time_s"],
                     "velocity_difference_m_s": b["final_velocity_m_s"]-a["final_velocity_m_s"],
                     "same_event": a["status"] == b["status"],
                     "same_consumed_node_count": len(a["node_ledger"]) == len(b["node_ledger"]),
                     "same_consumed_inventory_mass": a["consumed_inventory_mass_kg"] == b["consumed_inventory_mass_kg"]}
            entry["pass"] = (entry["same_event"] and entry["same_consumed_node_count"] and
                             entry["same_consumed_inventory_mass"] and
                             abs(entry["horizontal_distance_difference_m"]) < 1e-6 and
                             abs(entry["time_difference_s"]) < 1e-6 and
                             abs(entry["velocity_difference_m_s"]) < 1e-6)
            if not entry["pass"]:
                raise AssertionError(entry)
            checks.append(entry)
    return checks


def figures(sh, profiles, baseline, scan, pairs, mechanisms, inverses):
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for name, z in profiles.items():
        axes[0].plot(sh/1000, z, color=COLORS[name], label=LABELS[name], lw=1.25)
        tr = baseline[name]["trajectory"]
        axes[1].plot([p["s_h_m"]/1000 for p in tr], [p["velocity_m_s"] for p in tr],
                     color=COLORS[name], label=LABELS[name])
        axes[2].plot([p["s_h_m"]/1000 for p in tr], [p["time_s"]/60 for p in tr], color=COLORS[name])
    axes[0].set_ylabel("高程 / m")
    axes[1].set_ylabel("代表团块速度 / (m·s$^{-1}$)")
    axes[2].set_ylabel("注入后时间 / min")
    axes[2].set_xlabel("固定路线水平里程 / km")
    axes[0].set_title("无新增节点：三种高程结构的条件动力对照", loc="left")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.axvline(sh[-1]/1000, color=".3", ls=":", lw=.8)
    axes[0].legend(ncol=3, fontsize=9)
    fig.text(.5, .015, "工程参数非率定；曲线终止表示停止或口岸代理事件，不代表真实前锋观测", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .035, 1, 1])
    fig.savefig(io.FIG_OUT/"04c_三剖面动力.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    kappas = DOMAIN["kappa_m_inv"]
    for ax, name in zip(axes, profiles):
        for iv, v in enumerate(DOMAIN["v0_m_s"]):
            points = [r for r in scan if r["profile"] == name and r["parameters"]["v0_m_s"] == v]
            xx = [kappas.index(r["parameters"]["kappa_m_inv"])+(.22*(iv-1)) for r in points]
            yy = [r["parameters"]["mu_dyn"] for r in points]
            cc = [r["final_sh_m"]/sh[-1] for r in points]
            im = ax.scatter(xx, yy, c=cc, cmap="YlGnBu", vmin=0, vmax=1, s=68,
                            marker=["o", "s", "^"][iv], edgecolors=".3", linewidths=.4)
        ax.set_xticks(range(4), ["0", "$10^{-5}$", "$5×10^{-5}$", "$10^{-4}$"])
        ax.set_xlabel("二次阻力 κ / m$^{-1}$")
        ax.set_title(LABELS[name])
        ax.grid(alpha=.15)
    axes[0].set_ylabel("工程动摩擦参数 μ")
    fig.colorbar(im, ax=axes, label="终止/目标水平里程占比", fraction=.025, pad=.02)
    fig.suptitle("预声明无节点工程域：每剖面48配置；符号依次表示入口速度√200、30、60 m/s", fontsize=10)
    fig.subplots_adjust(left=.07, right=.87, bottom=.16, top=.84, wspace=.18)
    fig.savefig(io.FIG_OUT/"04c_无节点工程域.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, name in zip(axes, profiles):
        for mode, marker, color in [("M0", "o", "#4d5967"), ("M1", "s", "#2e7c87"), ("M2", "^", "#b25c33")]:
            subset = [p for p in pairs if p["profile"] == name and p["q_r"] == .5 and p["mode"] == mode]
            ax.plot([p["q_w"] for p in subset], [mechanisms[p["run_id"]]["final_sh_m"]/1000 for p in subset],
                    marker=marker, color=color, label=mode)
        ax.axhline(sh[-1]/1000, color=".4", ls=":", lw=1)
        ax.set_title(LABELS[name])
        ax.set_xlabel("合成新增水幅度 $q_w$（非实测）")
        ax.grid(alpha=.2)
        ax.set_xticks([0, .5, 1])
    axes[0].set_ylabel("首次停止或目标水平里程 / km")
    axes[-1].legend(fontsize=13)
    fig.suptitle("人工1/3、2/3节点，固定$q_r=0.5$：加水混合与唯一假设减阻分离", fontsize=14)
    fig.text(.5, .02, "M2仅假设μ乘0.6；三组均不证明实际补水、实际减阻或真实灾害到达", ha="center", fontsize=12)
    fig.tight_layout(rect=[0, .05, 1, .93])
    fig.savefig(io.FIG_OUT/"04c_合成补水机制.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for name in profiles:
        inv = inverses[f"no_nodes_{name}_0.1_1e-05"]
        rec = inv["records"]
        axes[0].plot([r["s_h_m"]/1000 for r in rec],
                     [math.sqrt(r["required_pre_w_m2_s2"]) for r in rec], color=COLORS[name], label=LABELS[name])
        r = baseline[name]
        terms = [r["coulomb_loss_J"], r["drag_loss_J"], r["mix_loss_J"]]
        axes[1].plot([0, 1, 2], np.asarray(terms)/1e12, marker="o", color=COLORS[name], label=LABELS[name])
    axes[0].set_xlabel("固定路线水平里程 / km")
    axes[0].set_ylabel("逆推局部入口速度需求 / (m·s$^{-1}$)")
    axes[0].set_title("固定阻力瓶颈，目标约定1 m/s")
    axes[1].set_xticks([0, 1, 2], ["摩擦耗散", "二次阻力耗散", "混合耗散"])
    axes[1].set_ylabel("耗散能量 / $10^{12}$ J")
    axes[1].set_title("无节点基线机械能账户")
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(io.FIG_OUT/"04c_瓶颈与能量.png")
    plt.close(fig)


def main():
    tests = run_report()
    save(tests, "04c_合成测试.json")
    if not tests["passed"]:
        raise AssertionError(tests)
    sh, profiles, initial, geometry = load_geometry()
    source = Path(__file__).parent
    sources = [source/"mmlib/finite_inventory.py", source/"step04c_dynamics.py",
               source/"test04c_dynamics.py", io.ROOT/"建模成果/问题4.md"]
    declaration = {"domain": DOMAIN, "geometry": geometry,
                   "density_kg_m3": {"rock": fi.RHO_R, "water": fi.RHO_W, "ice_inactive": fi.RHO_I},
                   "density_basis": "engineering rock-water constants identical to old step04 RHO_S=2650, RHO_W=1000; not site measurements; ice unused",
                   "source_sha256": {str(p.relative_to(io.ROOT)): sha(p) for p in sources},
                   "random_seed": None, "randomness": "none", "ice_initialization": "not executed",
                   "time_limit_s": None, "numerical_zero_speed_threshold": None,
                   "positive_target_arrival": "positive-speed passage; zero-speed contact separate",
                   "boundaries": ["fixed-route representative parcel, not observed front",
                                  "no lateral momentum, no depth, intensity, conversion time or event probability",
                                  "two artificial nodes, not measured geomorphic inventories",
                                  "geometric corners assume constrained contact; no resolved impact or detachment",
                                  "historical Heim mu not pure dynamic friction calibration",
                                  "no restarting after first zero speed"]}
    save(declaration, "04c_参数与路径.json")
    baseline, scan, inverses = {}, [], {}
    for name, z in profiles.items():
        baseline[name] = fi.simulate(sh, z, initial, **DOMAIN["baseline"])
        for mu, k in itertools.product(DOMAIN["mu_dyn"], DOMAIN["kappa_m_inv"]):
            key = f"no_nodes_{name}_{mu}_{k}"
            inverses[key] = checked_inverse(sh, z, initial, mu=mu, kappa=k)
            for v in DOMAIN["v0_m_s"]:
                r = fi.simulate(sh, z, initial, mu=mu, kappa=k, v0=v)
                scan.append({"profile": name, **summary(r)})
        print(name, "48 no-node configurations complete", flush=True)
    save({"baseline": baseline, "scan": scan, "count": len(scan),
          "per_profile": 48, "not_probabilities": True}, "04c_无节点动力.json")
    mechanisms, pairs, details, cache = {}, [], {}, {}
    for name, z in profiles.items():
        for qr, qw, mode in itertools.product(DOMAIN["q_r"], DOMAIN["q_w"], ["M0", "M1", "M2"]):
            water, effective_mode = fi.resolve_mode(mode, qw)
            identity = (name, qr, water, effective_mode)
            if identity not in cache:
                rid = f"{name}_qr{qr}_qw{water}_{effective_mode}"
                cache[identity] = rid
                nodes = artificial_nodes(sh, initial, qr, water)
                r = fi.simulate(sh, z, initial, nodes=nodes, mode=effective_mode, **DOMAIN["baseline"])
                mechanisms[rid] = {"profile": name, "q_r": qr, "q_w": water,
                                   "effective_mode": effective_mode, **summary(r),
                                   "node_ledger": r["node_ledger"], "water_competition": water_competition(r)}
                inverses[rid] = checked_inverse(sh, z, initial, nodes=nodes, mode=effective_mode,
                                               mu=.1, kappa=1e-5)
                if qr == .5 and water in [0., .5]:
                    details[rid] = r
            if mode == "M0":
                alias_reason = "M0关闭新增水，q_w强制为0；与M1在q_w=0时为同一运行"
            elif mode == "M2" and water == 0:
                alias_reason = "M2要求新增水，但q_w=0不触发减阻映射；与M1在q_w=0时为同一运行"
            elif mode == "M2":
                alias_reason = "M2新增水严格大于0，使用预登记的下游mu=0.6mu0映射"
            else:
                alias_reason = "M1直接运行"
            pairs.append({"profile": name, "q_r": qr, "q_w": qw, "mode": mode,
                          "effective_mode": effective_mode, "run_id": cache[identity],
                          "alias_reason": alias_reason})
    requested = {}
    for p in pairs:
        requested.setdefault(p["run_id"], []).append(p["mode"])
    for rid, modes in requested.items():
        mechanisms[rid]["requested_modes"] = sorted(set(modes), key=["M0", "M1", "M2"].index)
    comparisons = []
    for name, qr, qw in itertools.product(profiles, DOMAIN["q_r"], DOMAIN["q_w"]):
        ids = {p["mode"]: p["run_id"] for p in pairs if p["profile"] == name and p["q_r"] == qr and p["q_w"] == qw}
        entry = {"profile": name, "q_r": qr, "q_w": qw, "run_ids": ids}
        for first, second in [("M1", "M0"), ("M2", "M1")]:
            a, b = mechanisms[ids[first]], mechanisms[ids[second]]
            entry[f"{first}_minus_{second}"] = {
                "final_horizontal_distance_difference_m": a["final_sh_m"]-b["final_sh_m"],
                "arrival_difference_s": a["arrival_s"]-b["arrival_s"] if a["arrival_s"] is not None and b["arrival_s"] is not None else None,
                "both_reached": a["reached_target_positive"] and b["reached_target_positive"]}
        comparisons.append(entry)
    mode_resolution = {
        "deduplication_identity": "(profile, q_r, water, effective_mode)",
        "M0": {"water_rule": "每个请求q_w均强制为0", "effective_run_mode": "M1",
               "note": "与M1在q_w=0时数值相同，不另设独立results行"},
        "M1": {"water_rule": "按请求q_w加水", "effective_run_mode": "M1"},
        "M2": {"water_rule": "按请求q_w加水", "effective_run_mode": "M2 if q_w>0 else M1",
               "note": "q_w=0不触发预登记减阻映射，回落到M1"},
    }
    save({"unique_configurations": len(mechanisms),
          "requested_configuration_count": len(pairs),
          "mode_resolution": mode_resolution,
          "deduplication_rule": "M0恒为q_w=0并别名到M1(q_w=0)；M2在q_w=0时也别名到该运行；每个结果的requested_modes列出请求模式",
          "contrast_references": pairs,
          "results": mechanisms, "comparisons": comparisons, "selected_full_ledgers": details,
          "additional_no_node_baseline_duplicates": 3,
          "physical_inventory_evidence": "absent; all nodes and stocks artificial"}, "04c_合成库存机制.json")
    save({"vreq_m_s": 1., "thresholds": inverses,
          "all_positive_threshold_forward_checks_pass": all(v.get("forward_check_pass", True) for v in inverses.values()),
          "outside_speed_domain": [k for k, v in inverses.items() if not v.get("required_v0_within_declared_scan_range", False)],
          "interpretation": "conditional analytic infima, not fitted speeds or event probabilities"}, "04c_逆向瓶颈.json")
    figures(sh, profiles, baseline, scan, pairs, mechanisms, inverses)
    all_results = scan+list(mechanisms.values())
    checks = {"synthetic_tests_pass": tests["passed"], "no_node_scan_count": len(scan),
              "synthetic_unique_count": len(mechanisms), "contrast_reference_count": len(pairs),
              "inverse_count": len(inverses),
              "mass_relative_residual_max": max(abs(r["mass_residual_kg"])/r["initial_mass_kg"] for r in all_results),
              **{k: max(r[k] for r in all_results) for k in ["energy_relative_residual_max",
                    "segment_energy_relative_residual_max", "node_momentum_relative_residual_max", "node_energy_relative_residual_max"]},
              "null_arrival_for_nonpositive_passage": all(r["arrival_s"] is None for r in all_results if not r["reached_target_positive"]),
              "baseline_summary": {n: summary(r) for n, r in baseline.items()},
              "status_counts": {name: {s: sum(r["status"] == s for r in scan if r["profile"] == name)
                                        for s in sorted({r["status"] for r in scan})} for name in profiles},
              "figures": [p.name for p in sorted(io.FIG_OUT.glob("04c_*.png"))],
              "figure_review": "not inspected here; pending central four-up visual QA"}
    checks["real_route_fixed_node_refinement"] = real_route_refinement(sh, profiles, initial)
    checks["acceptance_tolerances"] = {"relative_mass": 1e-10, "relative_energy": 1e-9,
                                       "relative_node_momentum": 1e-10}
    checks["numerical_acceptance_pass"] = (
        checks["mass_relative_residual_max"] < 1e-10 and
        checks["energy_relative_residual_max"] < 1e-9 and
        checks["segment_energy_relative_residual_max"] < 1e-9 and
        checks["node_energy_relative_residual_max"] < 1e-9 and
        checks["node_momentum_relative_residual_max"] < 1e-10 and
        checks["null_arrival_for_nonpositive_passage"])
    save(checks, "04c_数值核验.json")
    if not checks["numerical_acceptance_pass"]:
        raise AssertionError("04c mass/momentum/energy numerical acceptance failed")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
