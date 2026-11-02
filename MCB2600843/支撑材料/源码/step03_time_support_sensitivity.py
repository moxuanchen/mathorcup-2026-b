"""Refit all declared station-clock scenarios without selecting on event coverage."""
from itertools import product
import numpy as np
import pandas as pd
import step03_forecast as f
from mmlib import io_data as io


def main():
    features = pd.read_parquet(io.CACHE / "features.parquet")
    ifs = f.load_ifs_manifest()
    source = sorted(ifs["source_files"], key=lambda item: item["init_time_utc"])
    traces = [f.forecast_trace(item)[0] for item in source]
    geometry, configs = f.parse_geometry(), f.d8_configs()
    results = []
    for met, rad_end, soil_end, flux_end in product(
        ("point", "start_mean", "end_mean"), (False, True), (False, True), (False, True)
    ):
        aligned = f.historical_energy_support(features, met, rad_end, soil_end, flux_end)
        model = f.fit_energy_model(aligned)
        beta = np.asarray(model.pop("beta_draws"))
        members = []
        for item, trace in zip(source, traces):
            tag = "run" + pd.Timestamp(item["init_time_utc"]).strftime("%H")
            records, _ = f.make_members(trace, beta, model, tag)
            members.extend(records)
        arrivals = f.propagate(members, configs, geometry)
        results.append({
            "scenario": aligned.attrs["station_support_assumption"],
            "fit": model, "melt_state": f.state_summary(members, "state_tm"),
            "instant_release_arrival_state": f.state_summary(arrivals, "state_ta"),
            "melt_interval": f.finite_interval(members, "tm_lower_utc", "tm_upper_utc", "tm_event_utc", "state_tm"),
            "instant_release_arrival_interval": f.finite_interval(arrivals, "ta_lower_utc", "ta_upper_utc", "ta_event_utc", "state_ta"),
        })
        print("clock scenario", met, rad_end, soil_end, flux_end,
              "training", model["training_rows"], flush=True)
    assert len(results) == 24
    f.write_json("03_历史家族时间支持敏感性.json", {
        "status": "computed_conditional_sensitivity", "scenario_count": len(results),
        "scenario_selection": "all 24 specified combinations; no event truth used for fitting or selection",
        "main_scenario": {"MET": "point", "RADM": "start_mean", "SOIL": "start_mean", "FLUX": "start_mean"},
        "main_scenario_basis": "unverified working hypothesis, not a metadata finding",
        "release_forecast_status": "not_identified; these are only auxiliary melt and instant-release scenarios",
        "scenarios": results,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
