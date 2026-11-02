"""SI fixed-path, finite stationary-inventory dynamics; not a depth solver."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

G = 9.80665
RHO_R, RHO_W, RHO_I = 2650.0, 1000.0, 900.0


def _phi_psi(u: float) -> tuple[float, float]:
    if abs(u) < 1e-4:
        return (1 - u/2 + u*u/6 - u**3/24 + u**4/120,
                .5 - u/6 + u*u/24 - u**3/120 + u**4/720)
    phi = -math.expm1(-u)/u
    return phi, (1-phi)/u


def speed_squared(w0: float, a: float, kappa: float, x: float) -> float:
    """Analytic segment speed squared, valid only before the first stop."""
    if kappa == 0:
        return w0 + 2*a*x
    phi, _ = _phi_psi(2*kappa*x)
    return w0 + 2*(a-kappa*w0)*x*phi


def stopping_distance(w0: float, a: float, kappa: float) -> float:
    if w0 < 0 or kappa < 0:
        raise ValueError("negative speed squared or drag")
    if w0 == 0:
        return 0.0
    if a >= 0:
        return math.inf
    if kappa == 0:
        return w0/(2*abs(a))
    return math.log1p(kappa*w0/abs(a))/(2*kappa)


def travel_time(w0: float, a: float, kappa: float, x: float,
                w1: float | None = None) -> float:
    if x == 0:
        return 0.0
    if w0 <= 0 or x < 0 or kappa < 0:
        raise ValueError("positive entrance velocity and nonnegative distance/drag required")
    if w1 is None:
        w1 = speed_squared(w0, a, kappa, x)
    if w1 < -1e-9*max(w0, 1):
        raise ValueError("time requested beyond first stop")
    v0, v1 = math.sqrt(w0), math.sqrt(max(w1, 0.0))
    if kappa == 0:
        return 2*x/(v0+v1)
    if a == 0:
        return math.expm1(kappa*x)/(kappa*v0)
    phi, _ = _phi_psi(2*kappa*x)
    dv = 2*(a-kappa*w0)*x*phi/(v0+v1)
    if a < 0:
        b = -a
        angle = math.atan2(-dv*math.sqrt(kappa*b), b+kappa*v0*v1)
        return angle/math.sqrt(b*kappa)
    veq = math.sqrt(a/kappa)
    logpart = math.log1p(dv/(v0+veq))
    numerator = kappa*x + logpart
    if numerator > 1e-10*(abs(kappa*x)+abs(logpart)):
        return numerator/math.sqrt(a*kappa)
    # The analytic antiderivative can subtract close terms above terminal speed.
    return x*quad(lambda u: 1/math.sqrt(speed_squared(w0, a, kappa, x*u)),
                  0, 1, epsabs=1e-11, epsrel=2e-12)[0]


def segment(w0: float, a: float, kappa: float, length: float) -> dict:
    if not all(math.isfinite(x) for x in (w0, a, kappa, length)):
        raise ValueError("finite SI segment inputs required")
    if w0 <= 0 or kappa < 0 or length <= 0:
        raise ValueError("positive entrance speed/length and nonnegative drag required")
    stop = stopping_distance(w0, a, kappa)
    stopped = stop <= length + 8*np.finfo(float).eps*max(length, 1)
    x = min(stop, length) if stopped else length
    w1 = 0.0 if stopped else speed_squared(w0, a, kappa, x)
    if w1 <= 0 and not stopped:
        raise FloatingPointError("unresolved nonpositive velocity")
    phi, psi = _phi_psi(2*kappa*x)
    iw = w0*x*phi + 2*a*x*x*psi
    if iw < -1e-9*max(w0*x, 1):
        raise FloatingPointError("negative squared-speed integral")
    return {"distance_m": x, "w_out_m2_s2": w1,
            "time_s": travel_time(w0, a, kappa, x, w1),
            "stopped": bool(stopped), "integral_w_dell_m3_s2": max(0.0, iw)}


def mass(volumes: np.ndarray) -> float:
    return float(np.dot(volumes, [RHO_R, RHO_I, RHO_W]))


def stationary_mix(volumes, speed: float, added) -> dict:
    before, added = np.asarray(volumes, float), np.asarray(added, float)
    if before.shape != (3,) or added.shape != (3,) or np.any(before < 0) or np.any(added < 0):
        raise ValueError("nonnegative rock, ice, water vectors required")
    m0, dm = mass(before), mass(added)
    if m0 <= 0 or speed < 0:
        raise ValueError("positive mass and nonnegative velocity required")
    after = before + added
    v1 = m0*speed/(m0+dm)
    loss = .5*m0*speed*speed*dm/(m0+dm)
    return {"volumes_after_m3": after, "mass_before_kg": m0,
            "added_mass_kg": dm, "mass_after_kg": m0+dm,
            "velocity_after_m_s": v1, "lambda": dm/m0,
            "mix_loss_J": loss,
            "momentum_residual_kg_m_s": (m0+dm)*v1-m0*speed,
            "kinetic_residual_J": .5*m0*speed**2-.5*(m0+dm)*v1**2-loss}


def resolve_mode(mode, q_w):
    """Map a requested mechanism label to (added water, effective dynamics mode).

    M0 always closes the added-water channel; M2 without added water cannot trigger
    the pre-registered friction mapping and is therefore the same run as M1 at q_w=0.
    """
    if mode not in ("M0", "M1", "M2"):
        raise ValueError("requested mode must be M0, M1 or M2")
    if not np.isfinite(q_w) or q_w < 0:
        raise ValueError("requested water inventory must be finite and nonnegative")
    water = 0.0 if mode == "M0" else float(q_w)
    effective = "M2" if mode == "M2" and water > 0 else "M1"
    return water, effective


def prepare(s_h, z, initial_volumes, nodes=(), target_sh=None, subdivisions=1,
            mu=.1, kappa=1e-5, mode="M1") -> dict:
    """Split geometry at fixed physical nodes/target; never create new stocks."""
    sh, zz = np.asarray(s_h, float), np.asarray(z, float)
    iv = np.asarray(initial_volumes, float)
    if (sh.ndim != 1 or sh.shape != zz.shape or len(sh) < 2 or
            np.any(np.diff(sh) <= 0) or not np.all(np.isfinite(sh)) or
            not np.all(np.isfinite(zz)) or sh[0] != 0):
        raise ValueError("finite increasing horizontal route starting at zero required")
    if iv.shape != (3,) or np.any(iv < 0) or mass(iv) <= 0:
        raise ValueError("invalid initial components")
    if mode not in ("M0", "M1", "M2") or mu < 0 or kappa < 0 or subdivisions < 1:
        raise ValueError("invalid dynamics configuration")
    target = float(sh[-1] if target_sh is None else target_sh)
    if not 0 < target <= sh[-1]:
        raise ValueError("target must lie on route")
    stock = []
    for j, n in enumerate(nodes):
        where = float(n["s_h_m"])
        vv = np.array([n.get("rock_m3", 0), n.get("ice_m3", 0), n.get("water_m3", 0)], float)
        if not 0 < where < target or np.any(vv < 0) or not np.all(np.isfinite(vv)):
            raise ValueError("inventory nodes must be interior, finite, nonnegative")
        if mode == "M0":
            vv[2] = 0
        stock.append({"id": str(n.get("id", j)), "s_h_m": where, "volumes_m3": vv})
    if len({n["id"] for n in stock}) != len(stock):
        raise ValueError("inventory identifiers must be unique")
    points = sorted(set([0.0, target] + list(sh[(sh > 0) & (sh < target)]) +
                        [n["s_h_m"] for n in stock]))
    split = [points[0]]
    for left, right in zip(points[:-1], points[1:]):
        split.extend(np.linspace(left, right, int(subdivisions)+1)[1:])
    split = np.asarray(split)
    elev = np.interp(split, sh, zz)
    dh, dz = np.diff(split), -np.diff(elev)
    ll = np.hypot(dh, dz)
    volumes, added_water = iv.copy(), 0.0
    segments, actions = [], []
    for i in range(len(dh)):
        for n in stock:
            if n["s_h_m"] == split[i]:
                m0 = mass(volumes)
                volumes += n["volumes_m3"]
                added_water += n["volumes_m3"][2]
                actions.append({**n, "before_segment": i,
                                "lambda": (mass(volumes)-m0)/m0})
        muk = mu*(.6 if mode == "M2" and added_water > 0 else 1)
        segments.append({"i": i, "s0_m": float(split[i]), "s1_m": float(split[i+1]),
                         "z0_m": float(elev[i]), "z1_m": float(elev[i+1]),
                         "dh_m": float(dh[i]), "length_m": float(ll[i]),
                         "sin_alpha": float(dz[i]/ll[i]), "cos_alpha": float(dh[i]/ll[i]),
                         "mu": muk, "kappa_m_inv": kappa,
                         "a_m_s2": G*(dz[i]-muk*dh[i])/ll[i]})
    return {"segments": segments, "actions": actions, "nodes": stock,
            "initial_volumes_m3": iv, "target_sh_m": target, "mode": mode}


def simulate(s_h, z, initial_volumes, *, nodes=(), v0=30.0, mu=.1,
             kappa=1e-5, mode="M1", target_sh=None, subdivisions=1,
             time_limit_s=None, vreq=1.0) -> dict:
    if not math.isfinite(v0) or v0 <= 0:
        raise ValueError("positive conditional entrance velocity required; no restart model")
    if time_limit_s is not None and time_limit_s <= 0:
        raise ValueError("positive time limit required")
    plan = prepare(s_h, z, initial_volumes, nodes, target_sh, subdivisions, mu, kappa, mode)
    volumes = plan["initial_volumes_m3"].copy()
    initial_m = mass(volumes)
    initial_inventory = sum(mass(n["volumes_m3"]) for n in plan["nodes"])
    initial_energy = initial_m*(.5*v0*v0+G*float(z[0]))
    w, t, ell = v0*v0, 0.0, 0.0
    loss_c, loss_d, loss_mix, stock_pe, added_mass = 0., 0., 0., 0., 0.
    max_energy, max_segment, max_node_p, max_node_e = 0., 0., 0., 0.
    trajectory, ledger, segments = [], [], []
    consumed = set()

    def point(sh, elev, event):
        m = mass(volumes)
        total = initial_energy+stock_pe-loss_c-loss_d-loss_mix
        residual = total-m*(.5*w+G*elev)
        return {"s_h_m": sh, "ell_m": ell, "z_m": elev, "time_s": t,
                "velocity_m_s": math.sqrt(max(w, 0)), "mass_kg": m,
                "rock_m3": float(volumes[0]), "ice_m3": float(volumes[1]),
                "water_m3": float(volumes[2]), "rock_volume_fraction": float(volumes[0]/volumes.sum()),
                "energy_residual_J": residual, "event": event}

    trajectory.append(point(0., float(z[0]), "injection"))
    status = "positive_target_passage"
    for seg in plan["segments"]:
        for action in plan["actions"]:
            if action["before_segment"] != seg["i"]:
                continue
            if action["id"] in consumed:
                raise AssertionError("duplicate inventory consumption")
            m_before = mass(volumes)
            result = stationary_mix(volumes, math.sqrt(w), action["volumes_m3"])
            volumes = result["volumes_after_m3"]
            w = result["velocity_after_m_s"]**2
            dm = result["added_mass_kg"]
            stock_pe += dm*G*seg["z0_m"]
            added_mass += dm
            loss_mix += result["mix_loss_J"]
            consumed.add(action["id"])
            max_node_p = max(max_node_p, abs(result["momentum_residual_kg_m_s"])/max(m_before*v0, 1))
            max_node_e = max(max_node_e, abs(result["kinetic_residual_J"])/max(.5*m_before*v0*v0, 1))
            ledger.append({"id": action["id"], "s_h_m": action["s_h_m"],
                           "consumed_volumes_m3": action["volumes_m3"].tolist(),
                           "remaining_volumes_m3": [0., 0., 0.],
                           **{k: v for k, v in result.items() if k != "volumes_after_m3"},
                           "stock_potential_transferred_J": dm*G*seg["z0_m"]})
            trajectory.append(point(seg["s0_m"], seg["z0_m"], "stationary_mixing"))
        oldw = w
        out = segment(w, seg["a_m_s2"], seg["kappa_m_inv"], seg["length_m"])
        partial_timeout = time_limit_s is not None and t+out["time_s"] > time_limit_s
        if partial_timeout:
            remaining = time_limit_s-t
            distance = brentq(lambda x: travel_time(oldw, seg["a_m_s2"], kappa, x)-remaining,
                              0, out["distance_m"], xtol=1e-10)
            out = segment(oldw, seg["a_m_s2"], kappa, distance)
        frac = out["distance_m"]/seg["length_m"]
        dh_run = seg["dh_m"]*frac
        znow = seg["z0_m"]+(seg["z1_m"]-seg["z0_m"])*frac
        shnow = seg["s0_m"]+dh_run
        m = mass(volumes)
        wc = seg["mu"]*m*G*dh_run
        wd = m*kappa*out["integral_w_dell_m3_s2"]
        wg = m*G*(seg["z0_m"]-znow)
        residual = .5*m*(out["w_out_m2_s2"]-oldw)-(wg-wc-wd)
        max_segment = max(max_segment, abs(residual)/max(abs(wg)+wc+wd+.5*m*oldw, 1))
        loss_c += wc
        loss_d += wd
        w, t, ell = out["w_out_m2_s2"], t+out["time_s"], ell+out["distance_m"]
        if partial_timeout:
            status = "time_limit_unknown"
        elif out["stopped"]:
            at_target = abs(shnow-plan["target_sh_m"]) <= 1e-9
            status = "zero_speed_target_contact" if at_target else "first_stop_before_target"
        p = point(shnow, znow, status if partial_timeout or out["stopped"] else "segment_end")
        trajectory.append(p)
        max_energy = max(max_energy, abs(p["energy_residual_J"])/max(abs(initial_energy)+abs(stock_pe), 1))
        segments.append({**seg, **out, "gravity_work_J": wg, "coulomb_loss_J": wc,
                         "drag_loss_J": wd, "energy_residual_J": residual})
        if partial_timeout or out["stopped"]:
            break
    reached = status == "positive_target_passage"
    final = trajectory[-1]
    remaining_inventory = initial_inventory-added_mass
    inventory_mass_residual = mass(volumes)+remaining_inventory-initial_m-initial_inventory
    return {"status": status, "reached_target_positive": reached,
            "target_passage_class": ("meets_vreq" if reached and math.sqrt(w) >= vreq else
                                     "positive_below_vreq" if reached else status),
            "arrival_s": t if reached else None,
            "stop_time_s": t if status in ("first_stop_before_target", "zero_speed_target_contact") else None,
            "final_sh_m": final["s_h_m"], "final_ell_m": ell, "final_time_s": t,
            "final_velocity_m_s": math.sqrt(max(w, 0)),
            "initial_mass_kg": initial_m, "final_mass_kg": mass(volumes),
            "declared_inventory_mass_kg": initial_inventory, "consumed_inventory_mass_kg": added_mass,
            "remaining_inventory_mass_kg": remaining_inventory,
            "mass_residual_kg": inventory_mass_residual,
            "initial_mechanical_energy_J": initial_energy,
            "stock_potential_transferred_J": stock_pe, "coulomb_loss_J": loss_c,
            "drag_loss_J": loss_d, "mix_loss_J": loss_mix,
            "final_mechanical_energy_J": mass(volumes)*(.5*w+G*final["z_m"]),
            "energy_relative_residual_max": max_energy,
            "segment_energy_relative_residual_max": max_segment,
            "node_momentum_relative_residual_max": max_node_p,
            "node_energy_relative_residual_max": max_node_e,
            "trajectory": trajectory, "node_ledger": ledger, "segment_ledger": segments,
            "unvisited_node_ids": [n["id"] for n in plan["nodes"] if n["id"] not in consumed],
            "parameters": {"mu_dyn": mu, "kappa_m_inv": kappa, "v0_m_s": v0,
                           "mode": mode, "vreq_m_s": vreq, "subdivisions": subdivisions},
            "clock": "common_injection_to_port_proxy", "restart": "disabled_conditional"}


def inverse_threshold(s_h, z, initial_volumes, *, nodes=(), mu=.1, kappa=1e-5,
                      mode="M1", target_sh=None, vreq=1., subdivisions=1) -> dict:
    if vreq <= 0:
        raise ValueError("positive target convention required")
    plan = prepare(s_h, z, initial_volumes, nodes, target_sh, subdivisions, mu, kappa, mode)
    r = vreq*vreq
    records = []
    for seg in reversed(plan["segments"]):
        u = 2*kappa*seg["length_m"]
        if u > 700:
            return {"status": "required_speed_overflow_outside_engineering_domain",
                    "required_v0_m_s": None, "records": records}
        phi, _ = _phi_psi(u)
        A = math.exp(-u)
        B = 2*seg["a_m_s2"]*seg["length_m"]*phi
        after = r
        r = max(0., (r-B)/A)
        for node in reversed(plan["actions"]):
            if node["before_segment"] == seg["i"]:
                r *= (1+node["lambda"])**2
        records.append({"s_h_m": seg["s0_m"], "required_pre_w_m2_s2": r,
                        "required_out_w_m2_s2": after})
    # Track which intermediate positivity or final-speed constraint controls w0.
    AA, BB, max_required, bottleneck = 1., 0., 0., 0.
    for seg in plan["segments"]:
        for node in plan["actions"]:
            if node["before_segment"] == seg["i"]:
                factor = (1+node["lambda"])**2
                AA, BB = AA/factor, BB/factor
        u = 2*kappa*seg["length_m"]
        phi, _ = _phi_psi(u)
        A = math.exp(-u)
        AA, BB = A*AA, A*BB+2*seg["a_m_s2"]*seg["length_m"]*phi
        need = max(0., -BB/AA)
        if need > max_required:
            max_required, bottleneck = need, seg["s1_m"]
    terminal_requirement = max(0., (vreq*vreq-BB)/AA)
    if terminal_requirement >= max_required:
        bottleneck = plan["target_sh_m"]
    return {"status": "conditional_threshold_requires_forward_check",
            "required_v0_m_s": math.sqrt(r), "required_w0_m2_s2": r,
            "threshold_is_infimum": True, "target_vreq_m_s": vreq,
            "controlling_boundary_sh_m": bottleneck,
            "intermediate_positive_w0_infimum": max_required,
            "terminal_vreq_w0_requirement": terminal_requirement,
            "records": list(reversed(records))}
