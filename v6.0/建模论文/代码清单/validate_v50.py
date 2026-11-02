"""Deterministic acceptance checks for the complete v5.0 result chain."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "求解结果" / "数据"
EXTERNAL = ROOT / "临时产物" / "外部数据"
OUT = DATA / "00_v50验收报告.json"


def load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def main() -> int:
    errors: list[str] = []
    checks: dict[str, object] = {}

    q1 = load("01c_统一关联前五.json")
    check(q1.get("candidate_count") == 31, "q1_candidate_count_not_31", errors)
    check(
        q1.get("candidate_families") == {"MET": 19, "RADM": 5, "SOIL": 3, "FLUX": 4},
        "q1_family_counts_mismatch",
        errors,
    )
    check(len(q1.get("top5", [])) == 5, "q1_top5_length_not_5", errors)
    check(
        q1.get("main_168h", {}).get("reference_window_count", 0) >= 80,
        "q1_reference_windows_insufficient",
        errors,
    )
    candidate_168 = pd.read_csv(DATA / "01c_168h候选完整排序.csv")
    candidate_72 = pd.read_csv(DATA / "01c_72h候选完整排序.csv")
    cluster_168 = pd.read_csv(DATA / "01c_168h簇完整排序.csv")
    cluster_72 = pd.read_csv(DATA / "01c_72h簇完整排序.csv")
    for tag, frame in (("168h", candidate_168), ("72h", candidate_72)):
        check(len(frame) == 31, f"q1_{tag}_candidate_rows_not_31", errors)
        check(frame["variable"].nunique() == 31, f"q1_{tag}_candidate_keys_not_unique", errors)
        rankable = frame.loc[frame["rankable"].astype(bool)]
        check(
            rankable["rank"].notna().all()
            and frame.loc[~frame["rankable"].astype(bool), "rank"].isna().all(),
            f"q1_{tag}_candidate_rank_contract_wrong",
            errors,
        )
        check(
            rankable["A_abs"].notna().all(),
            f"q1_{tag}_rankable_candidate_missing_score",
            errors,
        )
    expected_clusters_168 = len(q1["main_168h"]["cluster_ranking"])
    expected_clusters_72 = len(q1["sensitivity_72h"]["cluster_ranking"])
    check(
        len(cluster_168) == expected_clusters_168,
        "q1_168h_cluster_rows_incomplete",
        errors,
    )
    check(
        len(cluster_72) == expected_clusters_72,
        "q1_72h_cluster_rows_incomplete",
        errors,
    )
    checks["q1"] = {
        "candidate_count": q1.get("candidate_count"),
        "top5": [row["variable"] for row in q1.get("top5", [])],
        "reference_windows": q1.get("main_168h", {}).get("reference_window_count"),
        "candidate_full_ranking_rows": {
            "168h": len(candidate_168),
            "72h": len(candidate_72),
        },
        "cluster_full_ranking_rows": {
            "168h": len(cluster_168),
            "72h": len(cluster_72),
        },
    }

    q2 = load("02_沟道临界条件.json")
    states = q2.get("机械状态计数", {})
    allowed = {"water_controlled", "friction_equilibrium", "gravity_unstable", "not_evaluable"}
    check(q2.get("沟道数") == 84, "q2_gully_count_not_84", errors)
    check(sum(states.values()) == 84, "q2_state_counts_do_not_sum_84", errors)
    check(set(states) <= allowed, "q2_unknown_mechanical_state", errors)
    check(
        q2.get("water_controlled_n") + q2.get("zero_threshold_n") + q2.get("not_evaluable_n") == 84,
        "q2_substantive_branch_counts_do_not_sum_84",
        errors,
    )
    rank_tests = q2.get("rank_test_statistics", {})
    check(rank_tests.get("sample_n") == 78, "q2_rank_sample_not_78", errors)
    expected_rank_values = {
        ("total_flow_Q_crit_m3s", "kendall_vs_scale"): (
            "tau", 0.408386, 5.471436410144295e-06,
        ),
        ("total_flow_Q_crit_m3s", "spearman_vs_scale"): (
            "rho", 0.498676, 3.3745699520687622e-06,
        ),
        ("unit_width_q_crit_m2s", "kendall_vs_scale"): (
            "tau", 0.318602187, 0.000390472,
        ),
        ("unit_width_q_crit_m2s", "spearman_vs_scale"): (
            "rho", 0.400619987, 0.000278478,
        ),
    }
    for (quantity, test_name), (statistic_key, expected_stat, expected_p) in (
        expected_rank_values.items()
    ):
        entry = rank_tests.get(quantity, {}).get(test_name, {})
        observed_stat = entry.get(statistic_key)
        observed_p = entry.get("p_value")
        check(
            observed_stat is not None
            and np.isclose(float(observed_stat), expected_stat, rtol=2e-6, atol=1e-9),
            f"q2_rank_stat_mismatch:{quantity}:{test_name}",
            errors,
        )
        check(
            observed_p is not None
            and np.isfinite(float(observed_p))
            and float(observed_p) > 0.0
            and np.isclose(float(observed_p), expected_p, rtol=2e-5, atol=0.0),
            f"q2_rank_p_mismatch:{quantity}:{test_name}",
            errors,
        )
    old = load("02_旧方法审计.json")
    trigger_model = load("02_触发评估模型.json")
    api_quantiles = load("02_API分位.json")
    state_tuple = load("02_状态元组.json")
    hourly_state = pd.read_csv(DATA / "02_状态元组_逐小时.csv")
    object_state = pd.read_csv(DATA / "02_状态元组_逐沟道.csv")
    check(
        api_quantiles.get("training_years") == [2021, 2022, 2023, 2024, 2025],
        "q2_api_training_years_wrong",
        errors,
    )
    check(
        api_quantiles.get("training_months") == [6, 7, 8, 9],
        "q2_api_training_months_wrong",
        errors,
    )
    check(
        api_quantiles.get("training_valid_hours", 0) >= 100
        and api_quantiles.get("q1") <= api_quantiles.get("q3"),
        "q2_api_quantiles_invalid",
        errors,
    )
    tuple_fields = ["G", "H", "Theta", "F_ice", "U_ice", "C_ice"]
    check(
        state_tuple.get("fields") == tuple_fields,
        "q2_state_tuple_fields_wrong",
        errors,
    )
    check(
        state_tuple.get("ice_mechanics_unresolved") is True,
        "q2_ice_mechanics_unresolved_flag_wrong",
        errors,
    )
    check(
        set(tuple_fields + ["ice_mechanics_unresolved"]) <= set(hourly_state.columns),
        "q2_hourly_state_tuple_columns_missing",
        errors,
    )
    check(
        set(tuple_fields + ["ice_mechanics_unresolved"]) <= set(object_state.columns),
        "q2_object_state_tuple_columns_missing",
        errors,
    )
    check(len(object_state) == 84, "q2_object_state_rows_not_84", errors)
    check(
        object_state["object_id"].nunique() == 84,
        "q2_object_state_ids_not_unique",
        errors,
    )
    gully_cache = DATA / "cache" / "gully_table.parquet"
    if gully_cache.is_file():
        source_codes = set(
            pd.read_parquet(gully_cache)["Code"].astype(str)
        )
        check(
            set(object_state["object_id"].astype(str)) == source_codes,
            "q2_object_state_ids_do_not_match_source_codes",
            errors,
        )
    if "Q_water_m3s" in hourly_state and "H" in hourly_state:
        missing_water = hourly_state["Q_water_m3s"].isna()
        check(
            hourly_state.loc[missing_water, "H"].eq("not_evaluable").all(),
            "q2_missing_water_not_marked_not_evaluable",
            errors,
        )
    check(
        trigger_model.get("状态元组", {}).get("fields") == tuple_fields,
        "q2_trigger_model_state_tuple_missing",
        errors,
    )
    density_15 = q2.get("床面固相含量敏感性", {}).get("0.15", {})
    check(
        np.isclose(density_15.get("density_kg_m3", np.nan), 1247.5, atol=1e-12),
        "q2_bed_density_cv015_wrong",
        errors,
    )
    check(
        np.isclose(density_15.get("delta_rho_kg_m3", np.nan), 247.5, atol=1e-12),
        "q2_delta_rho_cv015_wrong",
        errors,
    )
    check(
        density_15.get("qcrit_m2s", {}).get("q50", 0) > 0,
        "q2_qcrit_cv015_not_positive",
        errors,
    )
    check(old.get("formal_model_uses_legacy_k") is False, "q2_formal_model_uses_legacy_k", errors)
    check(old.get("formal_q_crit_m2s", 0) > 0, "q2_formal_qcrit_nonpositive", errors)
    check(old.get("formal_C_N0") == 85.0, "q2_formal_cn0_not_85", errors)
    cn = trigger_model.get("曲线数情景", {})
    check(cn.get("formal_C_N0") == 85.0, "q2_cn_metadata_missing", errors)
    check(
        cn.get("formal_event_window") == ["2026-08-26 08:00", "2026-08-26 13:00"],
        "q2_cn_event_window_wrong",
        errors,
    )
    check(set(cn.get("scenarios", {})) == {"75", "85", "95"}, "q2_cn_scenarios_not_75_85_95", errors)
    check(
        all(
            row.get("event_Qwater_peak_m3s") is not None
            for row in cn.get("scenarios", {}).values()
        ),
        "q2_cn_scenario_result_missing",
        errors,
    )
    check(
        all(
            row.get("formal_event_window_start") == "2026-08-26 08:00"
            and row.get("formal_event_window_end") == "2026-08-26 13:00"
            and row.get("event_T_peak_time") is not None
            for row in cn.get("scenarios", {}).values()
        ),
        "q2_cn_window_or_peak_time_missing",
        errors,
    )
    top_gullies = q2.get("前10易发沟道", [])
    name_256 = next((row for row in top_gullies if str(row.get("Name")) == "256"), None)
    check(name_256 is not None, "q2_name_256_missing_from_top10", errors)
    if name_256 is not None:
        check(
            name_256.get("Q_crit_m3s", 0) > 0,
            "q2_name_256_positive_threshold_rounded_to_zero",
            errors,
        )
        check(
            name_256.get("mechanical_regime") == "water_controlled",
            "q2_name_256_mechanical_regime_missing",
            errors,
        )
        check(
            name_256.get("q_crit_positive") is True,
            "q2_name_256_positive_flag_missing",
            errors,
        )
    checks["q2"] = {
        "mechanical_states": states,
        "rank_tests": rank_tests,
        "formal_qcrit_m2s": old.get("formal_q_crit_m2s"),
        "event_window_T_peak": old.get("event_window_T_peak"),
        "formal_C_N0": old.get("formal_C_N0"),
        "cn_scenarios": cn.get("scenarios"),
        "legacy_k_audit_only": old.get("formal_model_uses_legacy_k") is False,
    }

    q3 = load("03_前向预报.json")
    intervals = pd.read_csv(DATA / "forecast_intervals.csv")
    coverage = pd.read_csv(DATA / "coverage_events.csv")
    warning = pd.read_csv(DATA / "warning_events.csv")
    advisory = pd.read_csv(DATA / "melt_advisory_events.csv")
    members = pd.read_csv(DATA / "03_前向预报成员.csv")
    runout = pd.read_csv(DATA / "runout_ranges.csv")
    legacy = load("03_旧条件反演对照.json")
    check(
        legacy.get("status") == "legacy_audit_only_not_used",
        "q3_legacy_status_wrong",
        errors,
    )
    check(
        legacy.get("downstream_use") == {
            "forward_forecast": False,
            "coverage": False,
            "warning": False,
            "propagation": False,
        },
        "q3_legacy_downstream_use_not_disabled",
        errors,
    )
    source_audit = legacy.get("source_audit", {})
    source_path = ROOT / str(source_audit.get("path", ""))
    check(source_path.is_file(), "q3_legacy_source_missing", errors)
    if source_path.is_file():
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        check(
            source_audit.get("sha256") == source_hash,
            "q3_legacy_source_hash_mismatch",
            errors,
        )
        check(
            legacy.get("legacy_record", {}).get("source_file_sha256")
            == json.loads(source_path.read_text(encoding="utf-8")).get("source_sha256"),
            "q3_legacy_original_source_hash_mismatch",
            errors,
        )
    check(
        q3.get("legacy_inverse_audit", {}).get("output")
        == "求解结果/数据/03_旧条件反演对照.json",
        "q3_legacy_output_pointer_missing",
        errors,
    )
    check(q3.get("issue_time_local") == "2026-08-26T08:00:00+08:00", "q3_issue_time_wrong", errors)
    check(len(q3.get("ifs_manifest", {}).get("source_runs", [])) == 3, "q3_source_run_count_not_3", errors)
    check(
        all(row.get("all_future_hours_available") for row in q3["ifs_manifest"]["source_runs"]),
        "q3_future_hours_missing",
        errors,
    )
    check(len(intervals) == 2, "q3_interval_rows_not_2", errors)
    check(
        intervals["interval_method"].eq("uniform_within_support_event_time_quantiles").all(),
        "q3_event_time_interval_method_wrong",
        errors,
    )
    check(
        intervals["interval_status"].isin([
            "available", "not_constructible_zero_finite",
            "insufficient_finite_members",
        ]).all(),
        "q3_interval_status_unknown",
        errors,
    )
    endpoint_columns = [
        "lower_time", "median_time", "upper_time", "width_hours",
        "support_envelope_lower_time", "support_envelope_upper_time",
        "support_envelope_width_hours",
        "support_endpoint_95_lower_time", "support_endpoint_95_upper_time",
        "support_endpoint_95_width_hours",
    ]
    unavailable = intervals.loc[intervals["interval_status"].ne("available")]
    if len(unavailable):
        check(
            unavailable[endpoint_columns].isna().all().all(),
            "q3_nonconstructible_interval_has_endpoint_values",
            errors,
        )
    available_intervals = intervals.loc[intervals["interval_status"].eq("available")]
    if len(available_intervals):
        check(
            available_intervals[
                ["support_envelope_lower_time", "support_envelope_upper_time"]
            ].notna().all().all(),
            "q3_support_envelope_missing",
            errors,
        )
    check(len(coverage) == 3, "q3_coverage_rows_not_3", errors)
    check(set(coverage["target_definition"]) == {
        "true_debris_flow_formation", "instant_release_scenario_arrival",
        "actual_port_arrival",
    }, "q3_coverage_target_contract_wrong", errors)
    check(len(members) == 3000, "q3_member_count_not_3000", errors)
    check(len(runout) == 3000, "q3_runout_rows_not_3000", errors)
    check(
        np.isclose(float(runout["weight"].sum()), 1.0, atol=1e-12),
        "q3_runout_weights_do_not_sum_one",
        errors,
    )
    q4_configs = load("04b_二维引擎.json").get("runs", {})
    check(
        set(runout["d8_configuration_id"].astype(str)) == set(q4_configs),
        "q3_runout_configuration_ids_mismatch",
        errors,
    )
    for config_id, group in runout.groupby("d8_configuration_id", sort=True):
        config = q4_configs[str(config_id)]
        expected_right = bool(config.get("runout_right_censored"))
        expected_natural = bool(config.get("natural_stop"))
        expected_time_limit = bool(config.get("time_limit", False))
        expected_step_limit = bool(config.get("step_limit", False))
        expected_domain_censor = bool(config.get("domain_censor", False))
        check(
            group["right_censored"].astype(bool).eq(expected_right).all(),
            f"q3_runout_right_censored_mismatch:{config_id}",
            errors,
        )
        check(
            group["natural_stop"].astype(bool).eq(expected_natural).all(),
            f"q3_runout_natural_stop_mismatch:{config_id}",
            errors,
        )
        check(
            group["time_limit"].astype(bool).eq(expected_time_limit).all(),
            f"q3_runout_time_limit_flag_mismatch:{config_id}",
            errors,
        )
        check(
            group["step_limit"].astype(bool).eq(expected_step_limit).all(),
            f"q3_runout_step_limit_flag_mismatch:{config_id}",
            errors,
        )
        check(
            group["domain_censor"].astype(bool).eq(expected_domain_censor).all(),
            f"q3_runout_domain_censor_flag_mismatch:{config_id}",
            errors,
        )
        expected_time_nonarrival = bool(
            expected_time_limit
            and not bool(config.get("reached_port", False))
        )
        expected_step_nonarrival = bool(
            expected_step_limit
            and not bool(config.get("reached_port", False))
        )
        expected_domain_nonarrival = bool(
            expected_domain_censor
            and not bool(config.get("reached_port", False))
        )
        check(
            group["time_limit_nonarrival"].astype(bool).eq(expected_time_nonarrival).all(),
            f"q3_runout_time_limit_mismatch:{config_id}",
            errors,
        )
        check(
            group["step_limit_nonarrival"].astype(bool).eq(expected_step_nonarrival).all(),
            f"q3_runout_step_limit_nonarrival_mismatch:{config_id}",
            errors,
        )
        check(
            group["domain_censor_nonarrival"].astype(bool).eq(expected_domain_nonarrival).all(),
            f"q3_runout_domain_censor_nonarrival_mismatch:{config_id}",
            errors,
        )
    expected_tm = {"finite", "no_event", "active_at_issue", "right_censored", "not_evaluable"}
    expected_ta = {
        "finite", "no_event", "active_at_issue_left_censored",
        "no_arrival", "right_censored", "not_evaluable",
    }
    for summary_key, expected_states in (
        ("state_summary_tm", expected_tm),
        ("state_summary_ta", expected_ta),
    ):
        summary = q3[summary_key]
        check(set(summary["counts"]) == expected_states, f"{summary_key}_keys_wrong", errors)
        check(sum(summary["counts"].values()) == summary["total_members"], f"{summary_key}_sum_failed", errors)
        check(
            all(
                np.isclose(
                    summary[f"p_{state}"],
                    summary["counts"][state] / summary["total_members"],
                    atol=1e-12,
                )
                for state in expected_states
            ),
            f"{summary_key}_masses_wrong",
            errors,
        )
    tm_summary = q3["state_summary_tm"]
    ta_summary = q3["state_summary_ta"]
    check(
        np.isclose(
            tm_summary["p_actionable"],
            tm_summary["p_finite"] + tm_summary["p_active_at_issue"],
            atol=1e-12,
        ),
        "q3_source_p_actionable_wrong",
        errors,
    )
    check(
        np.isclose(
            ta_summary["p_actionable"],
            ta_summary["p_finite"] + ta_summary["p_active_at_issue_left_censored"],
            atol=1e-12,
        ),
        "q3_arrival_p_actionable_wrong",
        errors,
    )
    finite_members = members.loc[members["state_tm"].eq("finite")]
    finite_runout = members.loc[members["state_ta"].eq("finite")]
    source_lower = pd.to_datetime(finite_members["tm_lower_utc"], utc=True)
    source_upper = pd.to_datetime(finite_members["tm_upper_utc"], utc=True)
    arrival_lower = pd.to_datetime(finite_runout["ta_lower_utc"], utc=True)
    arrival_upper = pd.to_datetime(finite_runout["ta_upper_utc"], utc=True)
    check(
        bool(((source_upper - source_lower) == pd.Timedelta(hours=1)).all()),
        "q3_source_support_not_one_hour",
        errors,
    )
    check(
        bool(((arrival_upper - arrival_lower) == pd.Timedelta(hours=1)).all()),
        "q3_arrival_support_not_one_hour",
        errors,
    )
    true_row = coverage.loc[coverage["target_definition"].eq("true_debris_flow_formation")].iloc[0]
    port_row = coverage.loc[coverage["target_definition"].eq("instant_release_scenario_arrival")].iloc[0]
    actual_row = coverage.loc[coverage["target_definition"].eq("actual_port_arrival")].iloc[0]
    check(bool(true_row["evaluation_eligible"]) is False, "q3_true_trigger_marked_evaluable", errors)
    for row in (true_row, actual_row):
        check(row["interval_status"] == "not_estimable_release_model_unidentified",
              "q3_unidentified_release_status_wrong", errors)
        check(not bool(row["evaluation_eligible"]), "q3_actual_target_marked_evaluable", errors)
        fields = ["lower_time", "upper_time", "definite_cover", "possible_cover", "width_hours"]
        fields += [key for key in row.index if key.startswith("p_")]
        check(row[fields].isna().all(), "q3_actual_target_contains_auxiliary_prediction", errors)
    check(members["state_release"].eq("not_identified").all(),
          "q3_release_falsely_identified", errors)
    check(members[["release_time_utc", "actual_arrival_time_utc"]].isna().all().all(),
          "q3_actual_times_filled_from_auxiliary_scenario", errors)
    if port_row["interval_status"] == "available":
        check(isinstance(bool(port_row["definite_cover"]), bool), "q3_port_cover_not_boolean", errors)
    else:
        check(pd.isna(port_row["definite_cover"]), "q3_nonconstructible_port_cover_not_null", errors)
    check(warning.iloc[0]["port_hit_time"] == "2026-08-26T10:30:00+08:00", "q3_port_hit_time_wrong", errors)
    check(warning.iloc[0]["status"] == "not_estimable_release_model_unidentified",
          "q3_actual_warning_status_wrong", errors)
    check(warning[["first_alarm_time", "actionable_alarm_time", "lead_trigger_lower",
                   "lead_trigger_upper", "lead_port", "missed_alarm", "late_alarm"]].isna().all().all(),
          "q3_actual_warning_contains_auxiliary_alarm", errors)
    check(advisory["product_role"].eq("melt_advisory_only_not_a_disaster_warning").all(),
          "q3_advisory_role_wrong", errors)
    alarm = bool(float(advisory.iloc[0]["source_p_actionable"]) >= 0.5)
    if alarm:
        check(float(advisory.iloc[0]["lead_port"]) == 2.5, "q3_advisory_information_lead_not_2_5h", errors)
    else:
        check(pd.isna(advisory.iloc[0]["lead_port"]), "q3_missed_advisory_has_lead", errors)
    active_left = members.loc[members["state_ta"].eq("active_at_issue_left_censored")]
    check(
        active_left.empty or active_left["ta_censor_upper_utc"].notna().all(),
        "q3_active_left_without_upper_bound",
        errors,
    )
    check(
        set(members["state_tm"]).issubset(expected_tm),
        "q3_member_source_state_unknown",
        errors,
    )
    check(
        set(members["state_ta"]).issubset(expected_ta),
        "q3_member_arrival_state_unknown",
        errors,
    )
    checks["q3"] = {
        "issue_time_local": q3["issue_time_local"],
        "legacy_audit_output": q3["legacy_inverse_audit"]["output"],
        "legacy_audit_sha256": q3["legacy_inverse_audit"]["output_sha256"],
        "source_state_counts": q3["state_summary_tm"]["counts"],
        "arrival_state_counts": q3["state_summary_ta"]["counts"],
        "source_interval_utc": q3["intervals"]["t_m"],
        "instant_release_scenario_interval_utc": q3["intervals"]["t_a"],
        "instant_release_scenario_single_event_inclusion": (
            bool(port_row["definite_cover"])
            if port_row["interval_status"] == "available" else None
        ),
        "runout_terminal_flag_counts": {
            column: int(runout[column].astype(bool).sum())
            for column in ("natural_stop", "time_limit_nonarrival", "right_censored")
        },
        "true_trigger_coverage": "not_estimable",
        "actual_port_coverage": None,
        "actual_disaster_warning_lead_h": None,
        "melt_advisory_information_lead_h": (
            float(advisory.iloc[0]["lead_port"]) if alarm else None
        ),
    }

    q4 = load("04b_二维引擎.json")
    runs = q4.get("runs", {})
    check(len(runs) == 16, "q4_d8_config_count_not_16", errors)
    required_terminal = {
        "termination_reason",
        "natural_stop",
        "time_limit",
        "step_limit",
        "domain_censor",
        "runout_right_censored",
        "reached_port",
    }
    check(
        all(required_terminal <= set(run) for run in runs.values()),
        "q4_terminal_flags_missing",
        errors,
    )
    check(
        all(
            "critical_depth_slope_product_m" in run
            and "critical_depth_port_m" in run
            and "h_stop_m" not in run
            for run in runs.values()
        ),
        "q4_critical_depth_fields_mislabeled",
        errors,
    )
    check(
        all(bool(run.get("domain_censor")) for run in runs.values()),
        "q4_domain_censor_not_recorded",
        errors,
    )
    check(
        all(
            not (bool(run.get("domain_censor")) and bool(run.get("natural_stop")))
            for run in runs.values()
        ),
        "q4_domain_censor_conflicts_with_natural_stop",
        errors,
    )
    check(
        all(bool(run.get("runout_right_censored")) for run in runs.values()),
        "q4_runout_not_marked_censored",
        errors,
    )
    baseline = runs["base"]
    check(abs(float(baseline["arrival_port_s"]) - 1965.996174) < 1e-5, "q4_baseline_arrival_changed", errors)
    checks["q4"] = {
        "d8_config_count": len(runs),
        "baseline_arrival_port_s": baseline["arrival_port_s"],
        "runout_L_m": baseline["runout_L_m"],
        "runout_right_censored": baseline["runout_right_censored"],
        "domain_front_max_m": baseline.get("domain_front_max_m"),
        "terminal_flags": {
            key: baseline.get(key)
            for key in ("termination_reason", "natural_stop", "time_limit",
                        "step_limit", "domain_censor")
        },
    }

    external_manifest = EXTERNAL / "ifs_single_run_manifest.json"
    check(external_manifest.is_file(), "ifs_manifest_missing", errors)
    report = {
        "schema_version": 1,
        "status": "pass" if not errors else "blocked",
        "errors": errors,
        "checks": checks,
        "scope": "deterministic result-chain integrity only; not physical validation or task completion",
        "task_completion": {
            "status": "blocked", "release_eligible": False,
            "unmet_requirements": [
                "Q3 identifiable real release/formation-time forecast and its 95% interval",
                "Q3 real disaster warning and independently evaluable coverage",
            ],
            "reason": "melt onset and instant-release transport are auxiliary scenarios, not the requested target",
        },
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "errors": errors, "output": OUT.as_posix()},
                     ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
