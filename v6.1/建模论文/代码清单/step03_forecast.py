"""v5.0 forward forecast: IFS melt-onset proxy, interval support, and D8 propagation.

The main product is deliberately narrower than "true debris-flow initiation":

* issue time: 2026-08-26 08:00 UTC+8 (00:00 UTC);
* target 1: the first new hourly support after issue with forecast melt energy,
  T>0 and SW>20, while the completed pre-issue hour is reported separately;
* target 2: the D8 port-proxy arrival obtained from that source support;
* every hourly source value is a half-open support [t,t+1), never a point;
* event-time quantiles use one fixed uniform draw within each member support
  while the complete support envelope is reported separately;
* finite quantiles are conditional on finite members;
* active-at-issue, no-event, not-evaluable, right-censored and no-arrival are
  distinct states.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from mmlib import io_data as io

ROOT = Path(__file__).resolve().parents[2]
OUT = io.DATA_OUT
EXTERNAL = ROOT / "临时产物" / "外部数据"
ISSUE_UTC = pd.Timestamp("2026-08-26 00:00")
ISSUE_LOCAL = pd.Timestamp("2026-08-26 08:00")
PREISSUE_UTC = ISSUE_UTC - pd.Timedelta(hours=1)
WINDOW_END_UTC = ISSUE_UTC + pd.Timedelta(hours=24)
WINDOW_END_LOCAL = ISSUE_LOCAL + pd.Timedelta(hours=24)
HIT_LOCAL = pd.Timestamp("2026-08-26 10:30")
HIT_UTC = HIT_LOCAL - pd.Timedelta(hours=8)
N_PER_RUN = 1000
SEED = 20260826
MIN_FINITE = 100
SOURCE_STATES = (
    "finite",
    "no_event",
    "active_at_issue",
    "right_censored",
    "not_evaluable",
)
ARRIVAL_STATES = (
    "finite",
    "no_event",
    "active_at_issue_left_censored",
    "no_arrival",
    "right_censored",
    "not_evaluable",
)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    return x


def write_json(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def iso_utc(value) -> str:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.isoformat()


def iso_local(value) -> str:
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    return stamp.tz_convert("Asia/Shanghai").isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(name_or_path):
    path = Path(name_or_path)
    if not path.is_absolute():
        path = OUT / name_or_path
    return json.loads(path.read_text(encoding="utf-8"))


def load_ifs_manifest() -> dict:
    path = EXTERNAL / "ifs_single_run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("actual_file_count") != 4:
        raise AssertionError("IFS manifest must describe four actual files")
    source_files = []
    port_files = []
    for item in manifest["files"]:
        raw = EXTERNAL / item["file"]
        if not raw.exists():
            raise FileNotFoundError(raw)
        actual = sha256(raw)
        if actual != item["sha256"]:
            raise AssertionError(f"IFS hash mismatch: {raw.name}")
        if item.get("role") == "source_primary":
            init = pd.Timestamp(item["init_time_utc"])
            if init + pd.Timedelta(hours=6) > ISSUE_UTC:
                raise AssertionError(f"IFS source run is unavailable at issue: {raw.name}")
            source_files.append(item)
        elif item.get("role") == "independent_port_context":
            port_files.append(item)
    if len(source_files) != 3 or len(port_files) != 1:
        raise AssertionError((len(source_files), len(port_files)))
    return {"path": path, "sha256": sha256(path), "manifest": manifest,
            "source_files": source_files, "port_files": port_files}


def align_ifs_hourly(raw: pd.DataFrame) -> pd.DataFrame:
    """Build [t,t+1h) supports from end-labelled fluxes and point states."""
    if raw.empty or not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("IFS requires a nonempty datetime index")
    if raw.index.hasnans or not raw.index.is_unique or not raw.index.is_monotonic_increasing:
        raise ValueError("IFS times must be valid, sorted and unique")
    if not (raw.index == raw.index.floor("h")).all():
        raise ValueError("IFS labels must be exact hourly endpoints")
    # Reindex first: the next row must be t+1h, never a later surviving record.
    grid = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    raw = raw.reindex(grid).apply(pd.to_numeric, errors="coerce")
    raw = raw.where(np.isfinite(raw))
    end = raw.shift(-1)
    frame = pd.DataFrame({
        "T": (raw["T"] + end["T"]) / 2.0,
        "RH": (raw["RH"] + end["RH"]) / 2.0,
        "SW": end["SW"].where(end["SW"] >= 0.0),
        "P": end["P"].where(end["P"] >= 0.0),
    }, index=grid).iloc[:-1].copy()
    frame.attrs["time_support"] = {
        "model_support": "[t,t+1h)",
        "SW": "raw label t+1h; preceding-hour mean W/m2",
        "P": "raw label t+1h; preceding-hour accumulation mm",
        "T": "trapezoidal mean of instantaneous raw t and t+1h degC",
        "RH": "trapezoidal mean of instantaneous raw t and t+1h percent",
        "nonlinearity_order": "hourly average first, then positive part and gate",
        "missing_endpoint": "missing support; never a one-endpoint mean",
    }
    return frame


def load_if_hourly(item: dict) -> pd.DataFrame:
    raw = json.loads((EXTERNAL / item["file"]).read_text(encoding="utf-8"))
    h = raw["hourly"]
    idx = pd.to_datetime(h["time"], errors="coerce")
    frame = pd.DataFrame({
        "T": pd.to_numeric(h["temperature_2m"], errors="coerce"),
        "P": pd.to_numeric(h["precipitation"], errors="coerce"),
        "SW": pd.to_numeric(h["shortwave_radiation"], errors="coerce"),
        "RH": pd.to_numeric(h["relative_humidity_2m"], errors="coerce"),
    }, index=idx)
    return align_ifs_hourly(frame)


def historical_energy_support(features: pd.DataFrame, met="point",
                              rad_end=False, soil_end=False, flux_end=False) -> pd.DataFrame:
    """Rebuild both inputs and energy target for an explicit station-clock scenario."""
    if met not in {"point", "start_mean", "end_mean"}:
        raise ValueError(f"unknown MET support: {met}")
    if not features.index.is_unique or not features.index.is_monotonic_increasing:
        raise ValueError("station feature times must be sorted and unique")
    frame = features.reindex(pd.date_range(features.index.min(), features.index.max(), freq="h"))
    met_values = frame[["Ta_2m", "RH_2m"]]
    if met == "point":
        met_values = (met_values + met_values.shift(-1)) / 2.0
    elif met == "end_mean":
        met_values = met_values.shift(-1)
    rad = frame[["Rsd", "Rsu", "Rld", "Rlu"]].shift(-int(rad_end))
    soil = frame["G_0.1m"].shift(-int(soil_end))
    flux = frame[["H", "LE"]].shift(-int(flux_end))
    energy = (rad["Rsd"] - rad["Rsu"] + rad["Rld"] - rad["Rlu"]
              - soil - flux["H"] - flux["LE"]).clip(lower=0)
    result = met_values.copy()
    result["Rsd"] = rad["Rsd"]
    result["melt_energy"] = energy
    result = result.where(np.isfinite(result))
    result.attrs["station_support_assumption"] = {
        "MET": met, "RADM": "end_mean" if rad_end else "start_mean",
        "SOIL": "end_mean" if soil_end else "start_mean",
        "FLUX": "end_mean" if flux_end else "start_mean",
        "metadata_verified": False,
        "target_rebuilt_after_alignment": True,
    }
    return result


def mean_energy_proxy(
    beta: np.ndarray,
    t: float,
    sw: float,
    rh: float,
) -> float:
    """Mean-coefficient fitted energy proxy used for the pre-issue state."""
    if not (np.isfinite(t) and np.isfinite(sw) and np.isfinite(rh)):
        return float("nan")
    return float(
        beta[0]
        + beta[1] * max(t, 0.0)
        + beta[2] * max(sw, 0.0)
        + beta[3] * rh
    )


def estimate_ar1_residual(
    residual: np.ndarray,
    index: pd.DatetimeIndex,
) -> dict:
    """Estimate AR(1) only from consecutive observed hourly residuals."""
    residual = np.asarray(residual, dtype=float)
    index = pd.DatetimeIndex(index)
    if residual.shape != (len(index),):
        raise ValueError("residual and index lengths must match")
    valid = np.isfinite(residual)
    residual = residual[valid]
    times = index[valid]
    if len(times) != len(residual) or not times.is_monotonic_increasing:
        raise ValueError("residual time index must be sorted")
    adjacent = (
        times[1:].to_numpy() - times[:-1].to_numpy()
    ) == np.timedelta64(1, "h")
    previous = residual[:-1][adjacent]
    following = residual[1:][adjacent]
    if len(previous) < 2:
        raise ValueError("insufficient adjacent hourly pairs for AR(1)")
    phi = float(np.corrcoef(following, previous)[0, 1])
    if not np.isfinite(phi):
        phi = 0.0
    phi = float(np.clip(phi, -0.99, 0.99))
    marginal_sigma = float(np.std(residual, ddof=1))
    innovation = following - phi * previous
    sigma = float(np.std(innovation, ddof=1))
    if not np.isfinite(sigma) or sigma <= 0 or not np.isfinite(marginal_sigma):
        raise ValueError("invalid residual scale")
    return {
        "phi": phi,
        "innovation_sigma": sigma,
        "marginal_sigma": marginal_sigma,
        "adjacent_pair_n": int(len(previous)),
    }


def fit_energy_model(features: pd.DataFrame) -> dict:
    """Fit the declared historical W/m2 energy proxy and AR(1) residual scale."""
    train = features.loc[
        features.index.year.isin(range(2021, 2026))
        & features.index.month.isin([6, 7, 8, 9]),
        ["melt_energy", "Ta_2m", "Rsd", "RH_2m"],
    ].copy()
    train = train.apply(pd.to_numeric, errors="coerce").dropna()
    if len(train) < 500:
        raise ValueError("insufficient historical energy training rows")
    gn = io.load_glacier_nc()
    valid_ice = np.isfinite(gn["thickness"]) & (gn["thickness"] > 1.0)
    source_elevation = float(np.nanmean(gn["dem"][valid_ice]))
    station_elevation = 3980.0
    lapse = 0.006
    offset = -(source_elevation - station_elevation) * lapse
    t_src = train["Ta_2m"] + offset
    X = np.column_stack([
        np.ones(len(train)),
        np.maximum(t_src.to_numpy(), 0.0),
        np.maximum(train["Rsd"].to_numpy(), 0.0),
        train["RH_2m"].to_numpy(),
    ])
    y = train["melt_energy"].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    residual = y - X @ beta
    valid_residual = np.isfinite(residual)
    residual = residual[valid_residual]
    ar1 = estimate_ar1_residual(residual, train.index[valid_residual])
    phi = ar1["phi"]
    sigma = ar1["innovation_sigma"]
    marginal_sigma = ar1["marginal_sigma"]

    # Year-block coefficient resampling is an explicit model-design uncertainty.
    rng = np.random.default_rng(SEED)
    year_arrays = {}
    for year, sub in train.groupby(train.index.year):
        ts = sub["Ta_2m"] + offset
        x = np.column_stack([
            np.ones(len(sub)),
            np.maximum(ts.to_numpy(), 0.0),
            np.maximum(sub["Rsd"].to_numpy(), 0.0),
            sub["RH_2m"].to_numpy(),
        ])
        year_arrays[int(year)] = (x, sub["melt_energy"].to_numpy(dtype=float))
    years = sorted(year_arrays)
    beta_draws = []
    for _ in range(N_PER_RUN):
        picked = rng.choice(years, size=len(years), replace=True)
        xb = np.vstack([year_arrays[int(y)][0] for y in picked])
        yb = np.concatenate([year_arrays[int(y)][1] for y in picked])
        b, *_ = np.linalg.lstsq(xb, yb, rcond=None)
        beta_draws.append(b if np.all(np.isfinite(b)) else beta)
    beta_draws = np.asarray(beta_draws)
    return {
        "status": "fitted",
        "training_rows": int(len(train)),
        "training_years": years,
        "training_months": [6, 7, 8, 9],
        "station_support_assumption": features.attrs.get("station_support_assumption"),
        "source_elevation_m": source_elevation,
        "station_elevation_m": station_elevation,
        "temperature_offset_C": offset,
        "beta": beta.tolist(),
        "coef_names": ["intercept", "positive_source_temperature", "positive_shortwave", "relative_humidity"],
        "residual_sigma_W_m2": sigma,
        "residual_marginal_sigma_W_m2": marginal_sigma,
        "residual_ar1_phi": phi,
        "residual_adjacent_pair_n": ar1["adjacent_pair_n"],
        "residual_first_step": "zero-mean stationary AR(1) state with variance sigma^2/(1-phi^2)",
        "residual_innovation": "N(0, sigma^2), including the first innovation",
        "residual_mean_W_m2": float(np.mean(residual)),
        "residual_quantiles_W_m2": np.quantile(residual, [0.025, 0.5, 0.975]).tolist(),
        "coefficient_resampling": "5-year block bootstrap with replacement; 1000 draws, seed fixed",
        "coefficient_draws_n": int(len(beta_draws)),
        "beta_draws": beta_draws,
    }


def forecast_trace(item: dict) -> tuple[pd.DataFrame, dict]:
    frame = load_if_hourly(item)
    future_grid = pd.date_range(
        ISSUE_UTC, WINDOW_END_UTC - pd.Timedelta(hours=1), freq="h"
    )
    grid = pd.DatetimeIndex([PREISSUE_UTC, *future_grid])
    trace = frame.reindex(grid)
    available = int(trace[["T", "SW", "RH"]].notna().all(axis=1).sum())
    future_available = int(
        trace.reindex(future_grid)[["T", "SW", "RH"]].notna().all(axis=1).sum()
    )
    preissue_available = bool(
        trace.loc[[PREISSUE_UTC], ["T", "SW", "RH"]].notna().all(axis=1).iloc[0]
    )
    trajectory_end = pd.Timestamp(frame.index.max())
    trace.attrs.update({
        "trajectory_end_utc": trajectory_end,
        "preissue_hour_utc": PREISSUE_UTC,
    })
    return trace, {
        "file": item["file"],
        "sha256": item["sha256"],
        "init_time_utc": item["init_time_utc"],
        "available_at_utc": item["availability"]["available_at_utc"],
        "preissue_hour_utc": iso_utc(PREISSUE_UTC),
        "preissue_hour_available": preissue_available,
        "valid_available_hours_including_preissue": available,
        "required_hours_including_preissue": len(grid),
        "valid_future_hours": future_available,
        "required_future_hours": len(future_grid),
        "trajectory_end_utc": iso_utc(trajectory_end),
        "trajectory_end_semantics": "last complete model-support start; raw final endpoint minus one hour",
        "time_support": frame.attrs["time_support"],
        "all_future_hours_available": future_available == len(future_grid),
    }


def ar1_residual(rng, sigma: float, phi: float, n: int) -> np.ndarray:
    """Simulate the declared stationary AR(1) residual process."""
    out = np.empty(n, dtype=float)
    stationary_sigma = float(sigma) / max(math.sqrt(max(1.0 - float(phi) ** 2, 1e-12)), 1e-12)
    out[0] = float(rng.normal(0.0, stationary_sigma))
    for j in range(1, n):
        out[j] = float(phi) * out[j - 1] + float(sigma) * float(rng.normal())
    return out


def make_members(trace: pd.DataFrame, beta_draws: np.ndarray, model: dict,
                 run_tag: str) -> tuple[list[dict], np.ndarray]:
    # run_tag is "run06"/"run12"/"run18"; use the lead-hour suffix as a
    # deterministic stream offset while preserving the global seed.
    tag_suffix = int(str(run_tag).removeprefix("run"))
    rng = np.random.default_rng(SEED + tag_suffix)
    event_rng = np.random.default_rng(SEED + 10000 + tag_suffix)
    expected_index = pd.DatetimeIndex([
        PREISSUE_UTC,
        *pd.date_range(
            ISSUE_UTC, WINDOW_END_UTC - pd.Timedelta(hours=1), freq="h"
        ),
    ])
    if not trace.index.equals(expected_index):
        raise AssertionError(
            "forecast trace must contain the completed pre-issue hour and "
            "exactly 24 future hourly supports"
        )
    t = trace["T"].to_numpy(dtype=float)
    sw = trace["SW"].to_numpy(dtype=float)
    rh = trace["RH"].to_numpy(dtype=float)
    future_index = trace.index[1:]
    valid_input = np.isfinite(t) & np.isfinite(sw) & np.isfinite(rh)
    mean_beta = np.asarray(model.get("beta", np.mean(beta_draws, axis=0)), dtype=float)
    preissue_energy = mean_energy_proxy(
        mean_beta, float(t[0]), float(sw[0]), float(rh[0])
    )
    preissue_active = bool(
        valid_input[0]
        and t[0] > 0.0
        and sw[0] > 20.0
        and np.isfinite(preissue_energy)
        and preissue_energy > 0.0
    )
    trajectory_end = pd.Timestamp(
        trace.attrs.get("trajectory_end_utc", future_index[-1])
    )
    after_trajectory = future_index > trajectory_end
    records = []
    energy_members = []
    for i in range(N_PER_RUN):
        beta = beta_draws[i]
        eps = ar1_residual(
            rng,
            model["residual_sigma_W_m2"],
            model["residual_ar1_phi"],
            len(t),
        )
        raw = beta[0] + beta[1] * np.maximum(t, 0.0) + beta[2] * np.maximum(sw, 0.0) + beta[3] * rh + eps
        protected = np.where((t > 0.0) & (sw > 20.0), np.maximum(raw, 0.0), 0.0)
        protected = np.where(valid_input, protected, np.nan)
        future_protected = protected[1:]
        future_t = t[1:]
        future_sw = sw[1:]
        crossing = np.where(
            np.isfinite(future_protected)
            & (future_protected > 0.0)
            & (future_t > 0.0)
            & (future_sw > 20.0)
        )[0]
        missing_future = np.where(~np.isfinite(future_protected))[0]
        first_crossing = int(crossing[0]) if len(crossing) else None
        first_invalid = int(missing_future[0]) if len(missing_future) else None
        preissue_missing = not valid_input[0]
        if preissue_missing:
            state = "not_evaluable"
            state_reason = "preissue_input_missing"
            start = end = None
        elif preissue_active:
            state = "active_at_issue"
            state_reason = "completed_preissue_hour_already_active"
            start = end = None
        elif (
            first_crossing is None
            and missing_future.size
            and bool(after_trajectory[missing_future].all())
        ):
            state = "right_censored"
            state_reason = "forecast_trajectory_ends_before_window"
            start = end = None
        elif first_invalid is not None and (
            first_crossing is None or first_invalid < first_crossing
        ):
            # A missing input before the first crossing prevents a unique
            # first-event time; later observations cannot recover that hour.
            state = "not_evaluable"
            state_reason = "future_input_missing_before_first_crossing"
            start = end = None
        elif first_crossing is not None:
            j = int(crossing[0])
            state = "finite"
            state_reason = "future_first_crossing"
            start = ISSUE_UTC + pd.Timedelta(hours=j)
            end = start + pd.Timedelta(hours=1)
        elif missing_future.size:
            # If every unavailable future hour lies beyond the returned IFS
            # trajectory, the outcome is censored rather than no-event.
            state = "not_evaluable"
            state_reason = "future_input_missing_without_first_crossing"
            start = end = None
        else:
            state = "no_event"
            state_reason = "complete_future_trajectory_without_crossing"
            start = end = None
        event_time = None
        if state == "finite":
            # One representative point is sampled only for reporting event-time
            # quantiles. The complete [start,end) support remains the primary
            # object and is propagated unchanged.
            offset_s = 3600.0 * float(event_rng.random())
            event_time = start + pd.Timedelta(seconds=offset_s)
        records.append({
            "member_id": f"{run_tag}-{i:04d}",
            "run": run_tag,
            "state_tm": state,
            "state_tm_reason": state_reason,
            "tm_lower_utc": start,
            "tm_upper_utc": end,
            "tm_event_utc": event_time,
            "first_hour_index": int(crossing[0]) if len(crossing) else None,
            "preissue_T_C": float(t[0]) if np.isfinite(t[0]) else None,
            "preissue_SW_W_m2": float(sw[0]) if np.isfinite(sw[0]) else None,
            "preissue_mean_energy_W_m2": (
                float(preissue_energy) if np.isfinite(preissue_energy) else None
            ),
            "future_missing_hours": int(missing_future.size),
            "trajectory_end_utc": trajectory_end,
        })
        energy_members.append(protected)
    return records, np.asarray(energy_members)


def parse_geometry() -> dict:
    vol = load_json("04_初始体积.json")
    cl = load_json("04_中心线校验.json")
    g = vol["链A源几何"]
    sec = vol["共同断面"]
    lon1, lat1 = g["centroid_lonlat"]
    lon2, lat2 = sec["注入_lonlat"]
    horiz = float(np.hypot((lon1 - lon2) * 111320 * math.cos(math.radians(lat1)),
                           (lat1 - lat2) * 110574))
    options = []
    for tag, z in [
        ("DEM4_surface", g["z_dem_p90_m"]),
        ("DEM4_bed_plus_ice", g["z_dem_plus_ice_p90_m"]),
    ]:
        drop = float(z - sec["注入高程_raw_m"])
        if drop <= 0 or horiz <= 0:
            options.append({"name": tag, "valid": False, "reason": "nonpositive drop or distance"})
            continue
        length = float(np.hypot(drop, horiz))
        fall_s = float(np.sqrt(2.0 * length / (9.80665 * drop / length)))
        options.append({
            "name": tag,
            "valid": True,
            "source_centroid": [lon1, lat1],
            "injection": [lon2, lat2],
            "source_to_injection_horizontal_m": horiz,
            "source_to_injection_route_m": length,
            "drop_m": drop,
            "fall_s": fall_s,
        })
    port_lon, port_lat = sec["口岸_lonlat"]
    straight = float(np.hypot((lon1 - port_lon) * 111320 * math.cos(math.radians(lat1)),
                              (lat1 - port_lat) * 110574))
    route = float(cl["中心线长度_km"]) * 1000.0
    return {
        "source_centroid": [lon1, lat1],
        "injection": [lon2, lat2],
        "port_proxy": [port_lon, port_lat],
        "source_to_injection_horizontal_m": horiz,
        "centerline_injection_to_port_m": route,
        "combined_route_m": horiz + route,
        "source_to_port_straight_m": straight,
        "geometry_options": options,
        "source_semantics": "链A源区质心；不证明真实释放面或冰体失稳时刻",
        "port_semantics": "D8口岸代理断面；与题面地点存在模型吸附误差",
    }


def d8_configs() -> list[dict]:
    q4b = load_json("04b_二维引擎.json")
    rows = []
    for name, result in q4b["runs"].items():
        termination = str(result.get("termination_reason"))
        flags = terminal_flags(
            termination,
            result.get("domain_censor", result.get("runout_right_censored", False)),
            reached_port=bool(result.get("reached_port", False)),
        )
        rows.append({
            "id": name,
            "engine": "D8",
            "arrival_port_s": result.get("arrival_port_s"),
            "reached_port": bool(result.get("reached_port", False)),
            "termination_reason": termination,
            "final_runout_m": result.get("runout_L_m"),
            "params": result.get("params", {}),
            "volume_m3": result.get("V_df_m3"),
            "runout_right_censored": flags["right_censored"],
            "natural_stop": flags["natural_stop"],
            "time_limit": flags["time_limit"],
            "step_limit": flags["step_limit"],
            "domain_censor": flags["domain_censor"],
            "time_limit_nonarrival": flags["time_limit_nonarrival"],
            "step_limit_nonarrival": flags["step_limit_nonarrival"],
            "domain_censor_nonarrival": flags["domain_censor_nonarrival"],
            "weight": 1.0 / len(q4b["runs"]),
        })
    if len(rows) != 16:
        raise AssertionError(f"expected 16 unique D8 configurations, got {len(rows)}")
    signatures = {
        json.dumps(
            {"params": row["params"], "volume_m3": row["volume_m3"]},
            ensure_ascii=False,
            sort_keys=True,
        )
        for row in rows
    }
    if len(signatures) != 16:
        raise AssertionError(f"expected 16 distinct parameter sets, got {len(signatures)}")
    return rows


TERMINATION_REASONS = {
    "stationary_after_injection",
    "time_limit",
    "step_limit",
    "domain_censor",
}


def terminal_flags(
    termination_reason,
    domain_censor: bool = False,
    *,
    reached_port: bool = False,
) -> dict:
    """Map the explicit D8 terminal enum to separate, non-interchangeable flags."""
    reason = str(termination_reason)
    if reason not in TERMINATION_REASONS:
        raise AssertionError(f"unknown D8 termination reason: {reason!r}")
    domain_censor = bool(domain_censor or reason == "domain_censor")
    natural_stop = reason == "stationary_after_injection" and not domain_censor
    time_limit = reason == "time_limit"
    step_limit = reason == "step_limit"
    runout_censored = bool(domain_censor or time_limit or step_limit)
    arrival_censored = bool(not reached_port and runout_censored)
    return {
        "termination_reason": reason,
        "natural_stop": natural_stop,
        "time_limit": time_limit,
        "step_limit": step_limit,
        "domain_censor": domain_censor,
        "right_censored": runout_censored,
        "arrival_right_censored": arrival_censored,
        "time_limit_nonarrival": bool(not reached_port and time_limit),
        "step_limit_nonarrival": bool(not reached_port and step_limit),
        "domain_censor_nonarrival": bool(not reached_port and domain_censor),
    }


def build_legacy_audit(old: dict, source_relative: str, source_sha: str) -> dict:
    """Build a standalone, explicitly non-operative legacy inverse record."""
    return {
        "schema_version": 1,
        "status": "legacy_audit_only_not_used",
        "source_audit": {
            "path": source_relative,
            "sha256": source_sha,
        },
        "legacy_record": {
            "source_file": old.get("source_file"),
            "source_file_sha256": old.get("source_sha256"),
            "physical_source_inverse": old.get("physical_source_inverse"),
            "scope": old.get("scope"),
        },
        "downstream_use": {
            "forward_forecast": False,
            "coverage": False,
            "warning": False,
            "propagation": False,
        },
        "scope": (
            "v4.3 post-event conditional inverse retained only as an audit "
            "record; not an input to v5.0 prediction, coverage, warning or propagation"
        ),
    }


def runout_terminal_fields(config: dict) -> dict:
    """Read explicit terminal meanings from the D8 configuration record."""
    flags = terminal_flags(
        config.get("termination_reason"),
        config.get("domain_censor", config.get("runout_right_censored", False)),
        reached_port=bool(config.get("reached_port", False)),
    )
    return {
        key: bool(config.get(key, flags[key]))
        for key in (
            "natural_stop",
            "time_limit",
            "step_limit",
            "domain_censor",
            "right_censored",
            "time_limit_nonarrival",
            "step_limit_nonarrival",
            "domain_censor_nonarrival",
        )
    }


def propagate(records: list[dict], configs: list[dict], geometry: dict) -> list[dict]:
    rng = np.random.default_rng(SEED + 17)
    out = []
    for i, record in enumerate(records):
        config = configs[int(rng.integers(0, len(configs)))]
        geo = geometry["geometry_options"][i % 2]
        item = dict(record)
        item["d8_configuration_id"] = config["id"]
        item["d8_design_weight"] = config["weight"]
        item["source_geometry"] = geo.get("name")
        item["source_fall_s"] = geo.get("fall_s")
        item["d8_travel_s"] = config.get("arrival_port_s")
        item["ta_censor_upper_utc"] = None
        if record["state_tm"] in {"no_event", "right_censored", "not_evaluable"}:
            item["state_ta"] = record["state_tm"]
            item["ta_lower_utc"] = None
            item["ta_upper_utc"] = None
        elif not geo.get("valid"):
            item["state_ta"] = "not_evaluable"
            item["ta_lower_utc"] = None
            item["ta_upper_utc"] = None
        elif not config["reached_port"]:
            terminal = runout_terminal_fields(config)
            item["state_ta"] = (
                "right_censored" if terminal["right_censored"] else "no_arrival"
            )
            item["ta_lower_utc"] = None
            item["ta_upper_utc"] = None
        elif config["arrival_port_s"] is None:
            item["state_ta"] = "no_arrival"
            item["ta_lower_utc"] = None
            item["ta_upper_utc"] = None
        elif record["state_tm"] == "active_at_issue":
            run = float(config["arrival_port_s"])
            upper = ISSUE_UTC + pd.Timedelta(seconds=geo["fall_s"] + run)
            item["state_ta"] = "active_at_issue_left_censored"
            item["ta_lower_utc"] = None
            item["ta_upper_utc"] = None
            item["ta_censor_upper_utc"] = upper
        elif record["state_tm"] == "finite":
            run = float(config["arrival_port_s"])
            item["state_ta"] = "finite"
            item["ta_lower_utc"] = record["tm_lower_utc"] + pd.Timedelta(seconds=geo["fall_s"] + run)
            item["ta_upper_utc"] = record["tm_upper_utc"] + pd.Timedelta(seconds=geo["fall_s"] + run)
            item["ta_event_utc"] = record["tm_event_utc"] + pd.Timedelta(seconds=geo["fall_s"] + run)
        else:
            raise AssertionError(f"unsupported source state: {record['state_tm']}")
        out.append(item)
    return out


def state_summary(records: list[dict], state_key: str) -> dict:
    if state_key == "state_tm":
        states = SOURCE_STATES
    elif state_key == "state_ta":
        states = ARRIVAL_STATES
    else:
        raise AssertionError(f"unknown state key: {state_key}")
    counts = {s: 0 for s in states}
    for record in records:
        state = record[state_key]
        if state not in counts:
            raise AssertionError(f"unknown state: {state}")
        counts[state] += 1
    total = len(records)
    if sum(counts.values()) != total:
        raise AssertionError((state_key, counts, total))
    return {
        "total_members": total,
        "counts": counts,
        "p_finite": counts["finite"] / total,
        "p_no_event": counts["no_event"] / total,
        "p_right_censored": counts["right_censored"] / total,
        "p_not_evaluable": counts["not_evaluable"] / total,
        **(
            {"p_active_at_issue": counts["active_at_issue"] / total}
            if state_key == "state_tm"
            else {
                "p_active_at_issue_left_censored": (
                    counts["active_at_issue_left_censored"] / total
                ),
                "p_no_arrival": counts["no_arrival"] / total,
            }
        ),
        "p_actionable": (
            (counts.get("finite", 0) + counts.get("active_at_issue", 0)) / total
            if state_key == "state_tm"
            else (
                counts.get("finite", 0)
                + counts.get("active_at_issue_left_censored", 0)
            ) / total
        ),
        "state_sum_check": sum(counts.values()) == total,
    }


def should_alarm(summary: dict) -> bool:
    """Apply the frozen source-risk rule using explicit actionable mass."""
    return bool(summary.get("p_actionable", 0.0) >= 0.5)


def interval_lead_bounds(interval: dict, target_time) -> tuple[float | None, float | None]:
    """Return [target - upper, target - lower] in hours for the arrival interval."""
    if interval.get("status") != "available":
        return None, None
    lower = pd.Timestamp(interval["lower_time"])
    upper = pd.Timestamp(interval["upper_time"])
    target = pd.Timestamp(target_time)
    if lower.tzinfo is None:
        lower = lower.tz_localize("UTC")
    if upper.tzinfo is None:
        upper = upper.tz_localize("UTC")
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    lower = lower.tz_convert("UTC")
    upper = upper.tz_convert("UTC")
    target = target.tz_convert("UTC")
    return (
        float((target - upper).total_seconds() / 3600.0),
        float((target - lower).total_seconds() / 3600.0),
    )


def finite_interval(records: list[dict], lower_key: str, upper_key: str,
                    event_key: str, state_key: str | None = None) -> dict:
    finite = [r for r in records if r.get("state_tm" if state_key is None else state_key) == "finite"]
    if len(finite) == 0:
        return {
            "status": "not_constructible_zero_finite",
            "finite_n": 0,
            "lower_time": None,
            "median_time": None,
            "upper_time": None,
            "width_hours": None,
            "support_envelope_lower_time": None,
            "support_envelope_upper_time": None,
            "support_envelope_width_hours": None,
            "support_endpoint_95_lower_time": None,
            "support_endpoint_95_upper_time": None,
            "support_endpoint_95_width_hours": None,
            "interval_semantics": (
                "no finite member exists; conditional interval, support "
                "envelope and endpoint diagnostics are not constructed"
            ),
        }
    if len(finite) < MIN_FINITE:
        return {
            "status": "insufficient_finite_members",
            "finite_n": len(finite),
            "lower_time": None,
            "median_time": None,
            "upper_time": None,
            "width_hours": None,
            "support_envelope_lower_time": None,
            "support_envelope_upper_time": None,
            "support_envelope_width_hours": None,
            "support_endpoint_95_lower_time": None,
            "support_endpoint_95_upper_time": None,
            "support_endpoint_95_width_hours": None,
            "interval_semantics": (
                f"only {len(finite)} finite members (<{MIN_FINITE}); conditional "
                "interval, support envelope and endpoint diagnostics are not constructed"
            ),
        }
    lo = pd.to_datetime([r[lower_key] for r in finite])
    hi = pd.to_datetime([r[upper_key] for r in finite])
    event = pd.to_datetime([r[event_key] for r in finite])
    if (
        lo.isna().any()
        or hi.isna().any()
        or event.isna().any()
    ):
        raise AssertionError("finite members must carry non-null supports and events")
    lo_ns = lo.astype("int64").to_numpy()
    hi_ns = hi.astype("int64").to_numpy()
    event_ns = event.astype("int64").to_numpy()
    lower_q = np.quantile(event_ns, 0.025)
    upper_q = np.quantile(event_ns, 0.975)
    median_q = np.quantile(event_ns, 0.5)
    endpoint_q_lower = np.quantile(lo_ns, 0.025)
    endpoint_q_upper = np.quantile(hi_ns, 0.975)
    complete_envelope_lower = int(np.min(lo_ns))
    complete_envelope_upper = int(np.max(hi_ns))
    return {
        "status": "available",
        "finite_n": len(finite),
        "lower_time": str(pd.Timestamp(int(lower_q), tz="UTC")),
        "median_time": str(pd.Timestamp(int(median_q), tz="UTC")),
        "upper_time": str(pd.Timestamp(int(upper_q), tz="UTC")),
        "width_hours": (upper_q - lower_q) / 3.6e12,
        "support_envelope_lower_time": str(pd.Timestamp(complete_envelope_lower, tz="UTC")),
        "support_envelope_upper_time": str(pd.Timestamp(complete_envelope_upper, tz="UTC")),
        "support_envelope_width_hours": (complete_envelope_upper - complete_envelope_lower) / 3.6e12,
        "support_endpoint_95_lower_time": str(pd.Timestamp(int(endpoint_q_lower), tz="UTC")),
        "support_endpoint_95_upper_time": str(pd.Timestamp(int(endpoint_q_upper), tz="UTC")),
        "support_endpoint_95_width_hours": (endpoint_q_upper - endpoint_q_lower) / 3.6e12,
        "interval_semantics": (
            "2.5% and 97.5% quantiles of one uniformly sampled event time within "
            "each paired half-open hourly support; conditional on finite members. "
            "The complete finite-member support envelope [min(L),max(U)] and the "
            "separately trimmed endpoint-quantile envelope are reported without "
            "being substituted for the event-time interval. The convex support "
            "envelope may contain gaps and is not the exact union of member supports."
        ),
    }


def main() -> int:
    features = pd.read_parquet(io.CACHE / "features.parquet")
    ifs = load_ifs_manifest()
    source = sorted(ifs["source_files"], key=lambda item: item["init_time_utc"])
    traces = []
    trace_meta = []
    for item in source:
        trace, info = forecast_trace(item)
        traces.append(trace)
        trace_meta.append(info)
    clock_rows = []
    for item, trace in zip(source, traces):
        for t, row in trace.iterrows():
            clock_rows.append({
                "file": item["file"], "file_sha256": item["sha256"],
                "init_time_utc": item["init_time_utc"],
                "support_start_utc": iso_utc(t),
                "support_end_utc": iso_utc(t + pd.Timedelta(hours=1)),
                "raw_SW_P_end_label_utc": iso_utc(t + pd.Timedelta(hours=1)),
                "raw_T_RH_left_time_utc": iso_utc(t),
                "raw_T_RH_right_time_utc": iso_utc(t + pd.Timedelta(hours=1)),
                "T_hour_mean_C": row["T"], "RH_hour_mean_percent": row["RH"],
                "SW_hour_mean_Wm2": row["SW"], "P_hour_total_mm": row["P"],
            })
    pd.DataFrame(clock_rows).to_csv(OUT / "03_IFS逐变量时间支持.csv", index=False, encoding="utf-8-sig")
    if any(not info["all_future_hours_available"] for info in trace_meta):
        # Keep missing values in member states rather than filling them.
        pass
    aligned_features = historical_energy_support(features)
    model = fit_energy_model(aligned_features)
    beta_draws = np.asarray(model.pop("beta_draws"))
    all_tm = []
    for item, trace in zip(source, traces):
        tag = "run" + pd.Timestamp(item["init_time_utc"]).strftime("%H")
        records, _ = make_members(trace, beta_draws, model, tag)
        all_tm.extend(records)
    geometry = parse_geometry()
    configs = d8_configs()
    all_ta = propagate(all_tm, configs, geometry)
    tm_summary = state_summary(all_tm, "state_tm")
    ta_summary = state_summary(all_ta, "state_ta")
    tm_interval = finite_interval(
        all_tm, "tm_lower_utc", "tm_upper_utc", "tm_event_utc", "state_tm"
    )
    ta_interval = finite_interval(
        all_ta, "ta_lower_utc", "ta_upper_utc", "ta_event_utc", "state_ta"
    )
    # A finite-member interval is unconditionally valid only when no member
    # mass is outside the finite state.  A high but sub-unit finite mass is
    # not enough: a mixture distribution would require an explicit point-mass
    # construction, which this model does not make.
    unconditional_tm = (
        tm_interval
        if tm_interval["status"] == "available" and tm_summary["p_finite"] == 1.0
        else None
    )
    unconditional_ta = (
        ta_interval
        if ta_interval["status"] == "available" and ta_summary["p_finite"] == 1.0
        else None
    )
    unconditional_ok = unconditional_tm is not None and unconditional_ta is not None
    p_upper_tm = tm_summary["p_finite"] * 0.95
    p_upper_ta = ta_summary["p_finite"] * 0.95

    # A point port hit is only one event and does not become a frequency estimate.
    hit_utc = HIT_UTC
    port_cover = None
    if ta_interval["status"] == "available":
        port_cover = bool(pd.Timestamp(ta_interval["lower_time"]) <= hit_utc.tz_localize("UTC") < pd.Timestamp(ta_interval["upper_time"]))

    # Materialize the old post-event inverse as its own audit product.  It is
    # never read by the forward-member, interval, coverage, warning or routing
    # calculations above; the explicit downstream-use flags make that boundary
    # machine-checkable rather than relying on prose.
    legacy_path = ROOT / "临时产物" / "审稿记录" / "冻结父版v5.1记录" / "v43_旧反演审计.json"
    legacy_meta = {
        "status": "legacy_audit_only_not_used",
        "source": None,
        "source_sha256": None,
        "output": None,
        "output_sha256": None,
        "downstream_use": {
            "forward_forecast": False,
            "coverage": False,
            "warning": False,
            "propagation": False,
        },
        "scope": None,
    }
    if legacy_path.exists():
        old = json.loads(legacy_path.read_text(encoding="utf-8"))
        legacy_relative = legacy_path.relative_to(ROOT).as_posix()
        legacy_sha = hashlib.sha256(legacy_path.read_bytes()).hexdigest()
        legacy_payload = build_legacy_audit(old, legacy_relative, legacy_sha)
        write_json("03_旧条件反演对照.json", legacy_payload)
        legacy_output = OUT / "03_旧条件反演对照.json"
        legacy_meta.update({
            "source": legacy_relative,
            "source_sha256": legacy_sha,
            "output": legacy_output.relative_to(ROOT).as_posix(),
            "output_sha256": sha256(legacy_output),
            "scope": legacy_payload["scope"],
        })

    interval_rows = []
    for target, interval, summary, unconditional in [
        ("source_melt_onset_proxy", tm_interval, tm_summary, unconditional_tm),
        ("instant_release_scenario_arrival", ta_interval, ta_summary, unconditional_ta),
    ]:
        unconditional_status = "available" if unconditional is not None else "not_constructible"
        row = {
            "event_or_issue_id": "issue_20260826T0800+0800",
            "issue_time": iso_local(ISSUE_UTC),
            "target_definition": target,
            "forecast_horizon": "24h",
            "model_version": "unreleased_repair_workspace",
            "product_role": "auxiliary_only_not_true_debris_flow_forecast",
            "calibration_version": "2021_2025_ols_ar1",
            "interval_method": "uniform_within_support_event_time_quantiles",
            "nominal_level": 0.95,
            "lower_time": interval["lower_time"],
            "median_time": interval["median_time"],
            "upper_time": interval["upper_time"],
            "support_envelope_lower_time": interval.get("support_envelope_lower_time"),
            "support_envelope_upper_time": interval.get("support_envelope_upper_time"),
            "support_envelope_width_hours": interval.get("support_envelope_width_hours"),
            "support_endpoint_95_lower_time": interval.get("support_endpoint_95_lower_time"),
            "support_endpoint_95_upper_time": interval.get("support_endpoint_95_upper_time"),
            "support_endpoint_95_width_hours": interval.get("support_endpoint_95_width_hours"),
            "finite_n": interval["finite_n"],
            "interval_status": interval["status"],
            "interval_condition": "finite_conditional",
            "unconditional_status": unconditional_status,
            "unconditional_lower_time": unconditional["lower_time"] if unconditional else None,
            "unconditional_upper_time": unconditional["upper_time"] if unconditional else None,
            "unconditional_probability_upper_bound": p_upper_tm if target.startswith("source") else p_upper_ta,
            "latest_input_available_time": iso_utc(ISSUE_UTC),
            "status": interval["status"],
        }
        row.update({
            key: value for key, value in summary.items()
            if key.startswith("p_")
        })
        if target.startswith("source"):
            row["p_active_at_issue_left_censored"] = None
            row["p_no_arrival"] = None
        else:
            row["p_active_at_issue"] = None
        interval_rows.append(row)
    pd.DataFrame(interval_rows).to_csv(OUT / "forecast_intervals.csv", index=False, encoding="utf-8-sig")

    true_trigger_row = {
        "event_id": "true_t_trig_20260826",
        "issue_time": iso_local(ISSUE_UTC),
        "target_definition": "true_debris_flow_formation",
        "truth_left": None,
        "truth_right": None,
        "truth_source": None,
        "truth_quality": "not_estimable: no independent source formation time/window",
        "lower_time": None,
        "upper_time": None,
        "definite_cover": None,
        "possible_cover": None,
        "prediction_missing": True,
        "width_hours": None,
        "interval_status": "not_estimable_release_model_unidentified",
        "evaluation_eligible": False,
        "exclusion_reason": "both release/formation forecast and independent truth unresolved; melt proxy is not this target",
    }
    true_trigger_row.update({
        key: value for key, value in tm_summary.items() if key.startswith("p_")
    })
    port_coverage_row = {
        "event_id": "port_hit_20260826_1030",
        "issue_time": iso_local(ISSUE_UTC),
        "target_definition": "instant_release_scenario_arrival",
        "product_role": "auxiliary_only_assumes_release_at_melt_onset",
        "truth_left": iso_local(HIT_UTC),
        "truth_right": iso_local(HIT_UTC),
        "truth_source": "题面单一口岸冲击锚点",
        "truth_quality": "single_point_event",
        "lower_time": ta_interval["lower_time"],
        "upper_time": ta_interval["upper_time"],
        "definite_cover": port_cover,
        "possible_cover": port_cover,
        "prediction_missing": ta_interval["status"] != "available",
        "width_hours": ta_interval.get("width_hours"),
        "interval_status": ta_interval["status"],
        "evaluation_eligible": ta_interval["status"] == "available",
        "exclusion_reason": (
            None if ta_interval["status"] == "available" else ta_interval["status"]
        ),
    }
    port_coverage_row.update({
        key: value for key, value in ta_summary.items() if key.startswith("p_")
    })
    actual_port_row = {**port_coverage_row,
        "target_definition": "actual_port_arrival", "product_role": "requested_real_target",
        "lower_time": None, "upper_time": None, "definite_cover": None, "possible_cover": None,
        "prediction_missing": True, "width_hours": None,
        "interval_status": "not_estimable_release_model_unidentified", "evaluation_eligible": False,
        "exclusion_reason": "no identifiable release state/time; conditional transport cannot substitute",
    }
    for row in (true_trigger_row, actual_port_row):
        for key in list(row):
            if key.startswith("p_"):
                row[key] = None
    coverage_rows = [true_trigger_row, port_coverage_row, actual_port_row]
    pd.DataFrame(coverage_rows).to_csv(OUT / "coverage_events.csv", index=False, encoding="utf-8-sig")

    # The frozen operational rule uses only forecast-state mass and the
    # predicted arrival window; the observed event anchor is evaluation-only.
    alarm = should_alarm(tm_summary)
    alarm_utc = ISSUE_UTC if alarm else None
    lead_port = float((HIT_LOCAL - ISSUE_LOCAL).total_seconds() / 3600.0) if alarm else None
    arrival_lead_lower, arrival_lead_upper = interval_lead_bounds(ta_interval, HIT_UTC)
    latency_lead = {
        f"{minutes}min": (lead_port - minutes / 60.0
                           if lead_port is not None else None)
        for minutes in (0, 5, 10, 15, 30)
    }
    warning_rows = [{
        "event_id": "port_hit_20260826_1030",
        "rule_version": "v5.0_issue_if_p_tm_actionable_ge_0.5",
        "first_alarm_time": iso_local(alarm_utc) if alarm_utc is not None else None,
        "actionable_alarm_time": iso_local(alarm_utc) if alarm_utc is not None else None,
        "source_p_finite": tm_summary["p_finite"],
        "source_p_active_at_issue": tm_summary["p_active_at_issue"],
        "source_p_actionable": tm_summary["p_actionable"],
        "arrival_p_finite": ta_summary["p_finite"],
        "arrival_p_active_at_issue_left_censored": ta_summary[
            "p_active_at_issue_left_censored"
        ],
        "arrival_p_actionable": ta_summary["p_actionable"],
        "trigger_truth_left": None,
        "trigger_truth_right": None,
        "port_hit_time": iso_local(HIT_UTC),
        "lead_trigger_lower": None,
        "lead_trigger_upper": None,
        "model_start_lead_hours": (
            float(
                (
                    pd.Timestamp(tm_interval["median_time"])
                    - ISSUE_UTC.tz_localize("UTC")
                ).total_seconds()
                / 3600.0
            )
            if tm_interval["status"] == "available"
            else None
        ),
        "arrival_interval_lead_lower": arrival_lead_lower,
        "arrival_interval_lead_upper": arrival_lead_upper,
        "lead_port": lead_port,
        "missed_alarm": not alarm,
        "late_alarm": bool(alarm and lead_port is not None and lead_port < 0),
        "latency_assumptions": "no communication delay; 0/5/10/15/30 min treated as post-processing sensitivity in theory",
        **{f"lead_port_{key}": value for key, value in latency_lead.items()},
    }]
    warning_rows[0]["product_role"] = "melt_advisory_only_not_a_disaster_warning"
    pd.DataFrame(warning_rows).to_csv(OUT / "melt_advisory_events.csv", index=False, encoding="utf-8-sig")
    true_warning = {
        "event_id": "port_hit_20260826_1030", "target_definition": "true_debris_flow_warning",
        "status": "not_estimable_release_model_unidentified", "first_alarm_time": None,
        "actionable_alarm_time": None, "port_hit_time": iso_local(HIT_UTC),
        "lead_trigger_lower": None, "lead_trigger_upper": None, "lead_port": None,
        "missed_alarm": None, "late_alarm": None,
        "reason": "melt advisory and instant-release transport do not determine actual release",
    }
    pd.DataFrame([true_warning]).to_csv(OUT / "warning_events.csv", index=False, encoding="utf-8-sig")

    # Keep the source trigger, source-to-injection fall, and D8 arrival in one
    # member row.  A configuration-only table would invite mixing the best
    # source hour with the best travel configuration.
    member_rows = []
    config_by_id = {cfg["id"]: cfg for cfg in configs}
    for record in all_ta:
        cfg = config_by_id[record["d8_configuration_id"]]
        member_rows.append({
            "member_id": record["member_id"],
            "run": record["run"],
            "state_tm": record["state_tm"],
            "state_tm_reason": record.get("state_tm_reason"),
            "tm_lower_utc": iso_utc(record["tm_lower_utc"]) if record["tm_lower_utc"] is not None else None,
            "tm_upper_utc": iso_utc(record["tm_upper_utc"]) if record["tm_upper_utc"] is not None else None,
            "tm_event_utc": iso_utc(record["tm_event_utc"]) if record.get("tm_event_utc") is not None else None,
            "preissue_T_C": record.get("preissue_T_C"),
            "preissue_SW_W_m2": record.get("preissue_SW_W_m2"),
            "preissue_mean_energy_W_m2": record.get("preissue_mean_energy_W_m2"),
            "future_missing_hours": record.get("future_missing_hours"),
            "trajectory_end_utc": (
                iso_utc(record["trajectory_end_utc"])
                if record.get("trajectory_end_utc") is not None else None
            ),
            "state_ta": record["state_ta"],
            "state_release": "not_identified",
            "release_time_utc": None,
            "state_actual_arrival": "not_evaluable_release_unidentified",
            "actual_arrival_time_utc": None,
            "propagation_hypothesis": "instant_release_at_melt_onset_with_prescribed_material",
            "ta_lower_utc": iso_utc(record["ta_lower_utc"]) if record["ta_lower_utc"] is not None else None,
            "ta_upper_utc": iso_utc(record["ta_upper_utc"]) if record["ta_upper_utc"] is not None else None,
            "ta_event_utc": iso_utc(record["ta_event_utc"]) if record.get("ta_event_utc") is not None else None,
            "ta_censor_upper_utc": (
                iso_utc(record["ta_censor_upper_utc"])
                if record.get("ta_censor_upper_utc") is not None else None
            ),
            "d8_configuration_id": record["d8_configuration_id"],
            "source_geometry": record.get("source_geometry"),
            "source_fall_s": record.get("source_fall_s"),
            "d8_travel_s": record.get("d8_travel_s"),
            "d8_design_weight": record["d8_design_weight"],
        })
    pd.DataFrame(member_rows).to_csv(OUT / "03_前向预报成员.csv", index=False, encoding="utf-8-sig")

    runout_rows = []
    for record in all_ta:
        cfg = config_by_id[record["d8_configuration_id"]]
        runout_terminal = runout_terminal_fields(cfg)
        runout_rows.append({
            "member_id": record["member_id"],
            "trigger_member_id": record["member_id"],
            "d8_configuration_id": record["d8_configuration_id"],
            "state_tm": record["state_tm"],
            "state_ta": record["state_ta"],
            "source_definition": "chain_A_source_geometry",
            "target_section": "D8_port_proxy",
            "volume": cfg["volume_m3"],
            "parameter_set": json.dumps(cfg["params"], ensure_ascii=False, sort_keys=True),
            "engine": "D8",
            "distance_metric": "centerline_injection_to_port_proxy",
            "first_arrival_time": record["ta_lower_utc"] if record["state_ta"] == "finite" else None,
            "first_arrival_time_upper": record["ta_upper_utc"] if record["state_ta"] == "finite" else None,
            "first_arrival_censor_upper_time": record.get("ta_censor_upper_utc"),
            "first_arrival_time_s": (record.get("source_fall_s", 0.0) + record.get("d8_travel_s", 0.0)
                                      if record["state_ta"] == "finite" else None),
            "final_runout": cfg["final_runout_m"],
            "d8_configuration_reached_port": cfg["reached_port"],
            "reached_target": record["state_ta"] in {
                "finite", "active_at_issue_left_censored"
            },
            "evaluation_eligible_for_arrival": record["state_ta"] in {
                "finite", "active_at_issue_left_censored"
            },
            "termination_reason": cfg["termination_reason"],
            "natural_stop": runout_terminal["natural_stop"],
            "time_limit": runout_terminal["time_limit"],
            "step_limit": runout_terminal["step_limit"],
            "domain_censor": runout_terminal["domain_censor"],
            "time_limit_nonarrival": runout_terminal["time_limit_nonarrival"],
            "step_limit_nonarrival": runout_terminal["step_limit_nonarrival"],
            "domain_censor_nonarrival": runout_terminal["domain_censor_nonarrival"],
            "right_censored": runout_terminal["right_censored"],
            "weight_or_scenario_type": "equal_member_weight_over_config_design",
            "configuration_weight": record["d8_design_weight"],
            "weight": 1.0 / len(all_ta),
        })
    runout_frame = pd.DataFrame(runout_rows)
    if not np.isclose(float(runout_frame["weight"].sum()), 1.0, atol=1e-12):
        raise AssertionError("runout member weights must sum to one")
    runout_frame.to_csv(OUT / "runout_ranges.csv", index=False, encoding="utf-8-sig")

    d8_arrivals = [r["arrival_port_s"] for r in configs if r["arrival_port_s"] is not None and r["reached_port"]]
    distance_payload = {
        "schema_version": 1,
        "status": "available",
        "source_definition": geometry["source_centroid"],
        "injection_definition": geometry["injection"],
        "port_proxy_definition": geometry["port_proxy"],
        "route_metrics": {
            "source_to_injection_horizontal_m": geometry["source_to_injection_horizontal_m"],
            "injection_to_port_centerline_m": geometry["centerline_injection_to_port_m"],
            "combined_route_m": geometry["combined_route_m"],
            "source_to_port_straight_m": geometry["source_to_port_straight_m"],
            "distances_not_added_across_metrics": True,
        },
        "d8_port_arrival_s_range": [min(d8_arrivals), max(d8_arrivals)] if d8_arrivals else None,
        "d8_final_runout_m_range": [
            min(r["final_runout_m"] for r in configs if r["final_runout_m"] is not None),
            max(r["final_runout_m"] for r in configs if r["final_runout_m"] is not None),
        ],
        "right_censored": all(cfg["runout_right_censored"] for cfg in configs),
        "scope": "D8 network route and model-advected port proxy; not observed hazard extent or natural stopping distance",
    }
    write_json("03_空间距离范围.json", distance_payload)

    onset_support_counts = {}
    for record in all_tm:
        if record["state_tm"] != "finite":
            continue
        offset_hours = int(
            (pd.Timestamp(record["tm_lower_utc"]) - ISSUE_UTC).total_seconds() // 3600
        )
        onset_support_counts[str(offset_hours)] = (
            onset_support_counts.get(str(offset_hours), 0) + 1
        )
    if any(
        pd.Timestamp(record["tm_upper_utc"]) - pd.Timestamp(record["tm_lower_utc"])
        != pd.Timedelta(hours=1)
        for record in all_tm if record["state_tm"] == "finite"
    ):
        raise AssertionError("every finite source member must carry a one-hour support")
    if any(
        pd.Timestamp(record["ta_upper_utc"]) - pd.Timestamp(record["ta_lower_utc"])
        != pd.Timedelta(hours=1)
        for record in all_ta if record["state_ta"] == "finite"
    ):
        raise AssertionError("paired arrival supports must preserve the source support width")
    active_left = [
        record for record in all_ta
        if record["state_ta"] == "active_at_issue_left_censored"
    ]
    if any(record.get("ta_censor_upper_utc") is None for record in active_left):
        raise AssertionError(
            "active-at-issue arrival must carry a left-censoring upper bound"
        )
    if len(runout_frame) != len(all_ta):
        raise AssertionError("runout rows must preserve one-to-one member pairing")
    validation = {
        "member_count": len(all_ta),
        "source_support_counts_by_forecast_hour": onset_support_counts,
        "all_source_supports_one_hour": True,
        "all_paired_arrival_supports_one_hour": True,
        "state_mass_checks": {
            "tm": tm_summary["state_sum_check"],
            "ta": ta_summary["state_sum_check"],
        },
        "source_state_sum": sum(tm_summary["counts"].values()),
        "arrival_state_sum": sum(ta_summary["counts"].values()),
        "p_actionable": {
            "source": tm_summary["p_actionable"],
            "arrival": ta_summary["p_actionable"],
        },
        "active_at_issue_n": tm_summary["counts"]["active_at_issue"],
        "active_at_issue_left_censored_n": len(active_left),
        "active_left_censored_bounds_present": bool(
            all(record.get("ta_censor_upper_utc") is not None for record in active_left)
        ),
        "interval_status": {
            "tm": tm_interval["status"],
            "ta": ta_interval["status"],
        },
        "runout_weight_sum": float(runout_frame["weight"].sum()),
        "runout_member_rows": int(len(runout_frame)),
        "port_single_event_covered": port_cover,
        "true_t_trig_evaluable": False,
    }

    payload = {
        "schema_version": 1,
        "status": "auxiliary_available_real_forecast_blocked",
        "target_definition": "source meltwater onset proxy and conditional D8 port arrival; not true t_trig",
        "release_model": {
            "status": "not_identified", "release_state": "unknown_not_no_release",
            "release_time_interval": None, "formation_time_interval": None,
            "actual_arrival_interval": None,
            "needed": ["causally available source deformation/strength/pressure constraints",
                       "measurement errors and dependence", "release-to-flow transformation model"],
        },
        "task_fulfillment": {"q3_true_trigger_forecast": "not_completed", "release_allowed": False},
        "issue_time_local": iso_local(ISSUE_UTC),
        "issue_time_utc": iso_utc(ISSUE_UTC),
        "window_local": [iso_local(ISSUE_UTC), iso_local(WINDOW_END_UTC)],
        "window_semantics": "half-open prediction window [issue, issue+24h)",
        "ifs_manifest": {
            "path": ifs["path"].relative_to(ROOT).as_posix(),
            "sha256": ifs["sha256"],
            "actual_file_count": ifs["manifest"]["actual_file_count"],
            "source_runs": trace_meta,
            "port_context_not_used_in_main_chain": [x["file"] for x in ifs["port_files"]],
        },
        "fit": model,
        "state_summary_tm": tm_summary,
        "state_summary_ta": ta_summary,
        "validation": validation,
        "intervals": {
            "t_m": tm_interval,
            "t_a": ta_interval,
            "unconditional_interval": {
                "t_m": unconditional_tm,
                "t_a": unconditional_ta,
            } if unconditional_ok else None,
            "unconditional_status": "available" if unconditional_ok else "not_constructible",
            "unconditional_status_by_target": {
                "t_m": (
                    "available" if unconditional_tm is not None
                    else "not_constructible"
                ),
                "t_a": (
                    "available" if unconditional_ta is not None
                    else "not_constructible"
                ),
            },
            "unconditional_probability_upper_bound": {"t_m": p_upper_tm, "t_a": p_upper_ta},
        },
        "coverage": {
            "product_role": "auxiliary_instant_release_scenario_inclusion_only",
            "true_t_trig": "not_estimable",
            "actual_port_arrival": "not_estimable_release_model_unidentified",
            "actual_port_coverage": None,
            "port_hit_event": (
                "not_constructible" if port_cover is None
                else ("1/1" if port_cover else "0/1")
            ),
            "port_hit_covered": port_cover,
            "evaluation_denominator": 1,
            "interval_status": {
                "t_m": tm_interval["status"],
                "t_a": ta_interval["status"],
            },
        },
        "warning": true_warning,
        "melt_advisory": warning_rows[0],
        "geometry": geometry,
        "d8_configuration_ids": [r["id"] for r in configs],
        "legacy_inverse_audit": legacy_meta,
        "outputs": {
            "forecast_intervals": "forecast_intervals.csv",
            "coverage_events": "coverage_events.csv",
            "warning_events": "warning_events.csv",
            "melt_advisory_events": "melt_advisory_events.csv",
            "variable_time_support": "03_IFS逐变量时间支持.csv",
            "runout_ranges": "runout_ranges.csv",
            "forecast_members": "03_前向预报成员.csv",
            "distance": "03_空间距离范围.json",
            "legacy_inverse_audit": "03_旧条件反演对照.json",
        },
        "limitations": [
            "source meltwater onset is an operational proxy, not independent true t_trig",
            "active-at-issue members are left-censored at the port; their pre-issue arrival is not recoverable",
            "IFS station-to-source transfer is structural, not ice-surface validation",
            "D8 configuration sampling is a model-design distribution, not a real probability law",
            "one port hit cannot validate long-run 95% coverage",
        ],
    }
    write_json("03_前向预报.json", payload)
    write_json("03_前向预报成员摘要.json", {
        "tm": tm_summary,
        "ta": ta_summary,
        "tm_interval": tm_interval,
        "ta_interval": ta_interval,
    })
    print(
        "03 forward forecast:",
        f"tm finite={tm_summary['counts']['finite']}/{tm_summary['total_members']}",
        f"tm active={tm_summary['counts']['active_at_issue']}",
        f"p_actionable={tm_summary['p_actionable']:.3f}",
        f"ta finite={ta_summary['counts']['finite']}/{ta_summary['total_members']}",
        f"ta left_censored={ta_summary['counts']['active_at_issue_left_censored']}",
        f"port_cover={port_cover}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
