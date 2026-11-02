"""Synthetic and protocol checks for the v5.0 trigger and forecast chain."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import step01c_unified_association as association
import step02_trigger_threshold as threshold
import step02c_spatial_matched as spatial
import step03_figures as figures
import step03_forecast as forecast
from mmlib import trigger as tg


class SpatialProtocolTests(unittest.TestCase):
    def test_polygon_block_union_and_shared_background(self):
        labels = np.zeros((8, 15), dtype=int)
        labels[1, 1:10] = 1
        labels[2, 8:14] = 2
        grid, _ = spatial.shared_groups(labels, {"cell_dy_m": 1000., "cell_dx_m": 1000.})
        self.assertEqual(np.unique(grid[labels > 0]).size, 1)
        self.assertEqual(grid[0, 0], grid[1, 1])
        self.assertNotEqual(grid[-1, -1], grid[1, 1])

    def test_purge_includes_exact_boundary(self):
        xy = np.array([[0., 0.], [300., 400.], [501., 0.], [900., 900.]])
        kept, record = spatial.purge(np.array([1, 2, 3]), np.array([0]), xy)
        self.assertEqual(kept.tolist(), [2, 3])
        self.assertEqual(record["purged_n"], 1)

    def test_fully_overwritten_polygon_grouping(self):
        polygon = {
            "bbox": (0., 0., 4., 4.),
            "pts": np.array([[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]], dtype=float),
            "parts": [0],
        }
        grid = {"lon0": 0., "lat0": 4., "dlon": 1., "dlat": -1.}
        labels, _ = spatial.legacy.rasterize([polygon, polygon], grid, (5, 5))
        self.assertNotIn(1, np.unique(labels))
        groups, meta = spatial.shared_groups(
            labels, {"cell_dy_m": 1000., "cell_dx_m": 1000.},
            [polygon, polygon], grid,
        )
        self.assertEqual(len(meta["blocks_by_polygon"]), 2)
        self.assertEqual(np.unique(groups[labels > 0]).size, 1)

    def test_shared_group_bootstrap_not_class_split(self):
        y = np.tile([0, 1], 10)
        g = np.repeat(np.arange(10), 2)
        score = y * .8 + .1
        r = spatial.group_bootstrap(y, score, g, 11)
        self.assertEqual(r["valid"], spatial.BOOT)
        np.testing.assert_allclose(r["interval"], [1., 1.])

    def test_insufficient_does_not_random_pixel_fallback(self):
        y = np.array([0, 0, 1, 1])
        g = np.array([0, 0, 1, 1])
        xy = np.array([[0, 0], [1, 0], [2000, 0], [2001, 0]])
        with self.assertRaises(spatial.InsufficientDesign):
            spatial.grouped_folds(y, g, xy, 4, 1)


class AssociationProtocolTests(unittest.TestCase):
    def test_frozen_candidate_dictionary_has_31_unique_keys(self):
        self.assertEqual(len(association.CANDIDATES), 31)
        self.assertEqual(len(association.COLUMNS), 31)
        self.assertEqual(len(set(association.COLUMNS)), 31)
        counts = pd.Series([row[2] for row in association.CANDIDATES]).value_counts().to_dict()
        self.assertEqual(counts, {"MET": 19, "RADM": 5, "SOIL": 3, "FLUX": 4})

    def test_aliases_are_not_independent_candidates(self):
        self.assertNotIn("Pmax_1h", association.COLUMNS)
        self.assertIn("P", association.COLUMNS)
        self.assertIn("H", association.COLUMNS)
        self.assertIn("LE", association.COLUMNS)
        self.assertIn("Rsd", association.COLUMNS)

    def test_candidate_ranking_retains_unrankable_candidates(self):
        metadata = {
            column: {"variable": column, "name": column, "family": "TEST"}
            for column in association.COLUMNS[:3]
        }
        result = {
            "window_length_h": 168,
            "target_window": ["start", "end"],
            "feature_details": {
                association.COLUMNS[0]: {
                    "rankable": True, "A": 1.0, "signed_z": 1.0,
                    "target_valid_hours": 168, "required_hours": 135,
                    "reference_windows": 90,
                },
                association.COLUMNS[1]: {
                    "rankable": False, "reason": "insufficient coverage",
                    "target_valid_hours": 10, "required_hours": 135,
                    "reference_windows": 90,
                },
                association.COLUMNS[2]: {
                    "rankable": True, "A": 2.0, "signed_z": -2.0,
                    "target_valid_hours": 168, "required_hours": 135,
                    "reference_windows": 90,
                },
            },
        }
        rows = association.candidate_ranking_rows(result, metadata)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["rank"] for row in rows], [1, 2, None])
        self.assertEqual(rows[-1]["exclusion_reason"], "insufficient coverage")

    def test_full_cluster_ranking_retains_unrankable_cluster(self):
        metadata = {
            column: {"variable": column, "name": column, "family": "TEST"}
            for column in association.COLUMNS[:2]
        }
        result = {
            "window_length_h": 168,
            "target_window": ["start", "end"],
            "feature_details": {
                association.COLUMNS[0]: {"signed_z": 1.0, "A": 1.0},
                association.COLUMNS[1]: {"rankable": False},
            },
            "ranked_clusters": [{
                "rank": 1,
                "representative": association.COLUMNS[0],
                "members": [association.COLUMNS[0]],
            }],
            "cluster_ranking": [
                {
                    "representative": association.COLUMNS[0],
                    "members": [association.COLUMNS[0]],
                },
                {
                    "representative": None,
                    "members": [association.COLUMNS[1]],
                    "status": "不可排名",
                },
            ],
        }
        rows = association.cluster_ranking_rows(result, metadata)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["rank"], 1)
        self.assertIsNone(rows[1]["rank"])
        self.assertEqual(rows[1]["cluster_status"], "not_rankable")


class TriggerMechanicsTests(unittest.TestCase):
    def test_formal_event_window_is_frozen_and_positive_threshold_retains_precision(self):
        self.assertEqual(
            threshold.FORMAL_EVENT_WIN,
            ("2026-08-26 08:00", "2026-08-26 13:00"),
        )
        self.assertEqual(
            threshold._ser({"Q_crit_m3s": 7.180069e-05})["Q_crit_m3s"],
            7.180069e-05,
        )

    def test_water_controlled_positive_threshold(self):
        result = tg._mechanics(0.30)
        self.assertEqual(result["regime"], "water_controlled")
        self.assertGreater(result["tau_w_c_Pa"], 0)
        self.assertGreater(result["q_crit_m2s"], 0)
        self.assertGreater(result["tau_r_Pa"], result["tau_g_parallel_Pa"])

    def test_friction_equilibrium_has_zero_water_threshold(self):
        result = tg._mechanics(tg.TAN_PHI)
        self.assertEqual(result["regime"], "friction_equilibrium")
        self.assertEqual(result["q_crit_m2s"], 0.0)
        self.assertEqual(result["h_crit_m"], 0.0)

    def test_gravity_unstable_has_zero_water_threshold(self):
        result = tg._mechanics(0.72)
        self.assertEqual(result["regime"], "gravity_unstable")
        self.assertEqual(result["q_crit_m2s"], 0.0)
        self.assertGreater(result["tau_g_parallel_Pa"], result["tau_r_Pa"])

    def test_invalid_geometry_is_not_evaluable(self):
        for slope in (0.0, -0.1, np.nan):
            self.assertEqual(tg._mechanics(slope)["regime"], "not_evaluable")

    def test_bed_density_sensitivity_is_monotone(self):
        low = tg._mechanics(0.30, bed_solid_fraction=0.15)["q_crit_m2s"]
        high = tg._mechanics(0.30, bed_solid_fraction=0.80)["q_crit_m2s"]
        self.assertGreater(high, low)

    def test_density_difference_matches_declared_density_law(self):
        self.assertAlmostEqual(tg._density_difference(1050.0 / 1650.0), 1050.0, places=12)
        self.assertAlmostEqual(tg._density_difference(0.15), 247.5, places=12)
        result = tg._mechanics(0.30, bed_solid_fraction=0.15)
        self.assertAlmostEqual(result["delta_rho_kg_m3"], 247.5, places=12)
        self.assertGreater(result["q_crit_m2s"], 0.0)

    def test_cn0_is_frozen_and_runoff_scenarios_are_ordered(self):
        self.assertEqual(tg.CN0, 85.0)
        index = pd.date_range("2024-07-01", periods=10, freq="h")
        feat = pd.DataFrame({
            "P": [0.0] * 6 + [12.0, 12.0, 0.0, 0.0],
            "API": [10.0] * 6 + [5.0, 10.0, 20.0, 25.0],
        }, index=index)
        feat.attrs["api_quantiles"] = (8.0, 18.0)
        low = tg.rain_runoff(feat, 75.0)
        base = tg.rain_runoff(feat, 85.0)
        high = tg.rain_runoff(feat, 95.0)
        self.assertTrue((low <= base).all())
        self.assertTrue((base <= high).all())
        self.assertGreater(float(high.sum()), float(low.sum()))

    def test_api_terciles_use_only_2021_2025_warm_season(self):
        warm = pd.date_range("2024-06-01", periods=120, freq="h")
        winter = pd.date_range("2024-01-01", periods=120, freq="h")
        feat = pd.DataFrame(
            {"P": np.zeros(240), "API": np.r_[np.linspace(0.0, 119.0, 120),
                                               np.full(120, 1e6)]},
            index=warm.append(winter),
        )
        meta = tg.api_climate_thresholds(feat)
        expected_q1, expected_q3 = np.quantile(np.arange(120.0), [1 / 3, 2 / 3])
        self.assertAlmostEqual(meta["q1"], expected_q1)
        self.assertAlmostEqual(meta["q3"], expected_q3)
        self.assertEqual(meta["training_valid_hours"], 120)
        self.assertEqual(meta["training_months"], [6, 7, 8, 9])

    def test_zero_or_invalid_solid_fraction_is_not_evaluable(self):
        for fraction in (0.0, -0.1, 1.01, np.nan):
            with self.subTest(fraction=fraction):
                result = tg._mechanics(0.30, bed_solid_fraction=fraction)
                self.assertEqual(result["regime"], "not_evaluable")
                self.assertIsNone(result["q_crit_m2s"])

    def test_trigger_constructor_uses_zero_branch_without_division(self):
        feat = pd.DataFrame(
            {"P": [0.0], "API": [10.0], "Ta_2m": [2.0], "melt_energy": [10.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )
        feat.attrs["api_quantiles"] = (8.0, 18.0)
        model = tg.TriggerModel(
            area_km2=1.0,
            glacier_frac=0.2,
            q_crit_m2s=0.0,
            width_m=10.0,
            mechanical_regime="gravity_unstable",
        )
        discharge = model.discharge(feat)
        self.assertTrue(np.isnan(discharge.loc[feat.index[0], "T_index"]))

    def test_missing_water_is_not_a_below_threshold_observation(self):
        feat = pd.DataFrame(
            {"P": [np.nan], "API": [10.0], "Ta_2m": [2.0], "melt_energy": [10.0]},
            index=pd.date_range("2024-01-01", periods=1, freq="h"),
        )
        feat.attrs["api_quantiles"] = (8.0, 18.0)
        model = tg.TriggerModel(
            area_km2=1.0,
            glacier_frac=0.2,
            q_crit_m2s=0.0,
            width_m=10.0,
            mechanical_regime="gravity_unstable",
        )
        discharge = model.discharge(feat)
        self.assertEqual(discharge.iloc[0]["hydraulic_state"], "not_evaluable")

    def test_object_state_tuple_has_unique_ids_and_missing_water_semantics(self):
        count = 84
        gully = pd.DataFrame({
            "Code": [f"g{i:03d}" for i in range(count)],
            "Name": ["same display name"] * count,
            "mechanical_regime": ["water_controlled"] * 78
            + ["gravity_unstable"] * 6,
            "Q_crit_m3s": [10.0] * count,
        })
        missing = pd.DataFrame(threshold.build_object_state_rows(
            gully, np.nan, "below_thermal_reference", 1, 1
        ))
        self.assertEqual(len(missing), count)
        self.assertEqual(missing["object_id"].nunique(), count)
        self.assertEqual(set(missing["object_id"]), set(gully["Code"]))
        self.assertTrue(missing["H"].eq("not_evaluable").all())
        self.assertTrue(missing["ice_mechanics_unresolved"].eq(1).all())
        self.assertTrue(set(threshold.STATE_TUPLE_FIELDS).issubset(missing.columns))

        present = pd.DataFrame(threshold.build_object_state_rows(
            gully, 20.0, "below_thermal_reference", 1, 1
        ))
        self.assertTrue(
            present.loc[:77, "H"].eq("hydraulic_candidate").all()
        )
        self.assertTrue(
            present.loc[78:, "H"].eq("inapplicable").all()
        )

    def test_rank_tests_name_total_and_unit_width_quantities(self):
        frame = pd.DataFrame({
            "Q_crit_m3s": [1., 2., 4., 8., 16., 32., 64., 128.],
            "q_crit_m2s": [1., 4., 2., 8., 4., 16., 8., 32.],
            "规模序": [1., 1., 2., 2., 3., 3., 4., 4.],
            "频率序": [1., 2., 1., 2., 1., 2., 1., 2.],
        })
        total = threshold.rank_association_summary(frame, "Q_crit_m3s")
        unit = threshold.rank_association_summary(frame, "q_crit_m2s")
        self.assertEqual(total["quantity"], "Q_crit_m3s")
        self.assertEqual(unit["quantity"], "q_crit_m2s")
        for result in (total, unit):
            self.assertGreater(result["kendall_vs_scale"]["p值"], 0)
            self.assertGreater(result["spearman_vs_scale"]["p值"], 0)


class ForecastProtocolTests(unittest.TestCase):
    def make_trace(self) -> pd.DataFrame:
        index = pd.date_range(
            forecast.PREISSUE_UTC, periods=25, freq="h"
        )
        frame = pd.DataFrame(
            {"T": 2.0, "SW": 100.0, "RH": 80.0, "P": 0.0},
            index=index,
        )
        frame.loc[forecast.PREISSUE_UTC, ["T", "SW"]] = [-2.0, 0.0]
        return frame

    def make_model(self) -> dict:
        return {
            "residual_sigma_W_m2": 0.0,
            "residual_ar1_phi": 0.0,
            "residual_quantiles_W_m2": [0.0, 0.0, 0.0],
            "beta": np.array([0.0, 1.0, 0.01, 0.0]),
        }

    def test_ar1_uses_only_consecutive_hour_pairs(self):
        index = pd.DatetimeIndex([
            "2024-06-01 00:00",
            "2024-06-01 01:00",
            "2024-06-01 02:00",
            "2024-07-01 00:00",
            "2024-07-01 01:00",
            "2024-07-01 02:00",
        ])
        residual = np.array([1.0, 2.0, 4.0, 100.0, 200.0, 400.0])
        result = forecast.estimate_ar1_residual(residual, index)
        self.assertEqual(result["adjacent_pair_n"], 4)
        self.assertAlmostEqual(result["phi"], 0.99, places=12)
        innovation = np.array([
            2.0 - 0.99 * 1.0,
            4.0 - 0.99 * 2.0,
            200.0 - 0.99 * 100.0,
            400.0 - 0.99 * 200.0,
        ])
        self.assertAlmostEqual(
            result["innovation_sigma"], float(np.std(innovation, ddof=1)), places=12
        )
        self.assertGreater(result["marginal_sigma"], result["innovation_sigma"])

    def test_ar1_first_sample_uses_stationary_variance(self):
        class FixedRNG:
            def __init__(self, draws):
                self.draws = iter(draws)

            def normal(self, loc=0.0, scale=1.0):
                return loc + scale * next(self.draws)

        sigma = 2.0
        phi = 0.5
        out = forecast.ar1_residual(FixedRNG([1.0, 2.0, 3.0]), sigma, phi, 3)
        stationary = sigma / np.sqrt(1.0 - phi**2)
        expected = np.array([
            stationary,
            phi * stationary + sigma * 2.0,
            phi * (phi * stationary + sigma * 2.0) + sigma * 3.0,
        ])
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-14)

    def test_source_states_are_mutually_exclusive_and_sum(self):
        records = [
            {"state_tm": "finite"},
            {"state_tm": "no_event"},
            {"state_tm": "active_at_issue"},
            {"state_tm": "not_evaluable"},
            {"state_tm": "right_censored"},
        ]
        summary = forecast.state_summary(records, "state_tm")
        self.assertEqual(summary["total_members"], 5)
        self.assertTrue(summary["state_sum_check"])
        self.assertEqual(sum(summary["counts"].values()), 5)
        self.assertEqual(summary["p_active_at_issue"], 0.2)
        self.assertEqual(summary["p_actionable"], 0.4)

    def test_arrival_states_are_mutually_exclusive_and_sum(self):
        records = [
            {"state_ta": "finite"},
            {"state_ta": "no_event"},
            {"state_ta": "active_at_issue_left_censored"},
            {"state_ta": "no_arrival"},
            {"state_ta": "not_evaluable"},
            {"state_ta": "right_censored"},
        ]
        summary = forecast.state_summary(records, "state_ta")
        self.assertEqual(summary["total_members"], 6)
        self.assertTrue(summary["state_sum_check"])
        self.assertEqual(sum(summary["counts"].values()), 6)
        self.assertEqual(summary["p_active_at_issue_left_censored"], 1 / 6)
        self.assertEqual(summary["p_actionable"], 2 / 6)

    def test_hour_support_is_exactly_one_hour(self):
        old_n = forecast.N_PER_RUN
        forecast.N_PER_RUN = 2
        try:
            beta = np.array([[0., 1., 0.01, 0.], [0., 1., 0.01, 0.]])
            records, _ = forecast.make_members(
                self.make_trace(), beta, self.make_model(), "run06"
            )
        finally:
            forecast.N_PER_RUN = old_n
        self.assertEqual(records[0]["state_tm"], "finite")
        lower = pd.Timestamp(records[0]["tm_lower_utc"])
        upper = pd.Timestamp(records[0]["tm_upper_utc"])
        self.assertEqual(upper - lower, pd.Timedelta(hours=1))
        self.assertEqual(lower, forecast.ISSUE_UTC)

    def test_active_preissue_hour_is_not_reported_as_future_finite(self):
        old_n = forecast.N_PER_RUN
        forecast.N_PER_RUN = 2
        trace = self.make_trace()
        trace.loc[forecast.PREISSUE_UTC, ["T", "SW"]] = [2.0, 100.0]
        trace.loc[trace.index[1:], ["T", "SW"]] = [-2.0, 0.0]
        try:
            beta = np.array([[0., 1., 0.01, 0.], [0., 1., 0.01, 0.]])
            model = self.make_model()
            model["beta"] = beta.mean(axis=0)
            records, _ = forecast.make_members(trace, beta, model, "run18")
        finally:
            forecast.N_PER_RUN = old_n
        self.assertEqual(records[0]["state_tm"], "active_at_issue")
        self.assertIsNone(records[0]["tm_lower_utc"])

    def test_truncated_future_trajectory_is_right_censored(self):
        old_n = forecast.N_PER_RUN
        forecast.N_PER_RUN = 2
        trace = self.make_trace()
        trace.loc[forecast.PREISSUE_UTC, ["T", "SW"]] = [-2.0, 0.0]
        trace.loc[trace.index[1:], ["T", "SW"]] = [-2.0, 0.0]
        trace.loc[trace.index[10:], ["T", "SW", "RH"]] = np.nan
        trace.attrs["trajectory_end_utc"] = trace.index[9]
        try:
            beta = np.array([[0., 1., 0.01, 0.], [0., 1., 0.01, 0.]])
            records, _ = forecast.make_members(
                trace, beta, self.make_model(), "run18"
            )
        finally:
            forecast.N_PER_RUN = old_n
        self.assertEqual(records[0]["state_tm"], "right_censored")

    def test_missing_before_first_crossing_is_not_evaluable(self):
        old_n = forecast.N_PER_RUN
        forecast.N_PER_RUN = 2
        trace = self.make_trace()
        trace.loc[forecast.PREISSUE_UTC, ["T", "SW"]] = [-2.0, 0.0]
        trace.loc[trace.index[1], "T"] = np.nan
        trace.loc[trace.index[1:3], "SW"] = 0.0
        try:
            beta = np.array([[0., 1., 0.01, 0.], [0., 1., 0.01, 0.]])
            records, _ = forecast.make_members(
                trace, beta, self.make_model(), "run12"
            )
        finally:
            forecast.N_PER_RUN = old_n
        self.assertEqual(records[0]["state_tm"], "not_evaluable")
        self.assertIsNone(records[0]["tm_lower_utc"])

    def test_finite_interval_uses_only_finite_members(self):
        start = forecast.ISSUE_UTC + pd.Timedelta(hours=2)
        records = [
            {
                "state": "finite",
                "lower": start,
                "upper": start + pd.Timedelta(hours=1),
                "event": start + pd.Timedelta(minutes=15),
            }
            for i in range(100)
        ] + [{"state": "no_event", "lower": None, "upper": None, "event": None}]
        result = forecast.finite_interval(records, "lower", "upper", "event", "state")
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["finite_n"], 100)
        self.assertEqual(result["width_hours"], 0.0)

    def test_zero_finite_members_return_all_null_interval_fields(self):
        records = [{"state": "no_event", "lower": None, "upper": None, "event": None}]
        result = forecast.finite_interval(records, "lower", "upper", "event", "state")
        self.assertEqual(result["status"], "not_constructible_zero_finite")
        self.assertEqual(result["finite_n"], 0)
        for field in (
            "lower_time", "median_time", "upper_time", "width_hours",
            "support_envelope_lower_time", "support_envelope_upper_time",
            "support_envelope_width_hours",
            "support_endpoint_95_lower_time", "support_endpoint_95_upper_time",
            "support_endpoint_95_width_hours",
        ):
            self.assertIsNone(result[field])

    def test_one_and_ninety_nine_finite_members_are_insufficient(self):
        start = forecast.ISSUE_UTC + pd.Timedelta(hours=2)
        for count in (1, 99):
            with self.subTest(count=count):
                records = [
                    {
                        "state": "finite",
                        "lower": start,
                        "upper": start + pd.Timedelta(hours=1),
                        "event": start + pd.Timedelta(minutes=30),
                    }
                    for _ in range(count)
                ]
                result = forecast.finite_interval(
                    records, "lower", "upper", "event", "state"
                )
                self.assertEqual(result["status"], "insufficient_finite_members")
                self.assertEqual(result["finite_n"], count)
                self.assertIsNone(result["lower_time"])
                self.assertIsNone(result["support_envelope_lower_time"])
                self.assertIsNone(result["width_hours"])

    def test_event_time_quantiles_are_not_support_endpoint_envelope(self):
        start = forecast.ISSUE_UTC + pd.Timedelta(hours=2)
        records = [
            {
                "state": "finite",
                "lower": start,
                "upper": start + pd.Timedelta(hours=1),
                "event": start + pd.Timedelta(hours=i / 100.0),
            }
            for i in range(100)
        ]
        result = forecast.finite_interval(records, "lower", "upper", "event", "state")
        self.assertGreater(result["width_hours"], 0.9)
        self.assertLess(result["width_hours"], 1.0)
        self.assertEqual(result["support_envelope_width_hours"], 1.0)

    def test_complete_support_envelope_contains_endpoint_outliers(self):
        start = forecast.ISSUE_UTC + pd.Timedelta(hours=2)
        records = [
            {
                "state": "finite",
                "lower": start,
                "upper": start + pd.Timedelta(hours=1),
                "event": start + pd.Timedelta(minutes=30),
            }
            for _ in range(100)
        ]
        records[0].update(
            lower=start - pd.Timedelta(hours=10),
            upper=start + pd.Timedelta(hours=20),
            event=start + pd.Timedelta(hours=5),
        )
        result = forecast.finite_interval(records, "lower", "upper", "event", "state")
        self.assertEqual(
            pd.Timestamp(result["support_envelope_lower_time"]),
            (start - pd.Timedelta(hours=10)).tz_localize("UTC"),
        )
        self.assertEqual(
            pd.Timestamp(result["support_envelope_upper_time"]),
            (start + pd.Timedelta(hours=20)).tz_localize("UTC"),
        )
        self.assertLess(
            pd.Timestamp(result["support_endpoint_95_upper_time"]),
            (start + pd.Timedelta(hours=2)).tz_localize("UTC"),
        )

    def test_support_envelope_covers_gap_without_calling_it_exact_union(self):
        first = forecast.ISSUE_UTC + pd.Timedelta(hours=2)
        second = forecast.ISSUE_UTC + pd.Timedelta(hours=4)
        records = []
        for start in (first, second):
            records.extend([
                {
                    "state": "finite",
                    "lower": start,
                    "upper": start + pd.Timedelta(hours=1),
                    "event": start + pd.Timedelta(minutes=30),
                }
                for _ in range(50)
            ])
        result = forecast.finite_interval(records, "lower", "upper", "event", "state")
        self.assertEqual(
            pd.Timestamp(result["support_envelope_lower_time"]),
            first.tz_localize("UTC"),
        )
        self.assertEqual(
            pd.Timestamp(result["support_envelope_upper_time"]),
            (second + pd.Timedelta(hours=1)).tz_localize("UTC"),
        )
        self.assertEqual(result["support_envelope_width_hours"], 3.0)
        self.assertIn("not", result["interval_semantics"].lower())

    def test_active_at_issue_arrival_is_left_censored_with_upper_bound(self):
        record = {
            "member_id": "run18-0000",
            "run": "run18",
            "state_tm": "active_at_issue",
            "tm_lower_utc": None,
            "tm_upper_utc": None,
            "tm_event_utc": None,
        }
        config = {
            "id": "reached",
            "arrival_port_s": 600.0,
            "reached_port": True,
            "termination_reason": "stationary_after_injection",
            "final_runout_m": 1000.0,
            "params": {},
            "volume_m3": 1.0,
            "runout_right_censored": False,
            "natural_stop": False,
            "weight": 1.0,
        }
        geometry = {
            "geometry_options": [
                {"name": "g", "valid": True, "fall_s": 120.0}
            ]
        }
        result = forecast.propagate([record], [config], geometry)[0]
        self.assertEqual(result["state_ta"], "active_at_issue_left_censored")
        self.assertIsNone(result["ta_lower_utc"])
        self.assertEqual(
            pd.Timestamp(result["ta_censor_upper_utc"]),
            forecast.ISSUE_UTC + pd.Timedelta(seconds=720),
        )

    def test_arrival_state_precedence_no_arrival_before_left_censoring(self):
        record = {
            "member_id": "run18-0000",
            "run": "run18",
            "state_tm": "active_at_issue",
            "tm_lower_utc": None,
            "tm_upper_utc": None,
            "tm_event_utc": None,
        }
        config = {
            "id": "stop",
            "arrival_port_s": None,
            "reached_port": False,
            "termination_reason": "stationary_after_injection",
            "final_runout_m": 1000.0,
            "params": {},
            "volume_m3": 1.0,
            "runout_right_censored": False,
            "natural_stop": True,
            "weight": 1.0,
        }
        geometry = {
            "geometry_options": [
                {"name": "g", "valid": True, "fall_s": 120.0}
            ]
        }
        result = forecast.propagate([record], [config], geometry)[0]
        self.assertEqual(result["state_ta"], "no_arrival")
        self.assertIsNone(result["ta_censor_upper_utc"])

    def test_terminal_enum_separates_six_arrival_cases(self):
        base_geometry = {
            "geometry_options": [
                {"name": "g", "valid": True, "fall_s": 120.0}
            ]
        }
        cases = [
            ("stationary_after_injection", False, False, "no_arrival"),
            ("time_limit", False, False, "right_censored"),
            ("step_limit", False, False, "right_censored"),
            ("stationary_after_injection", False, True, "right_censored"),
            ("time_limit", True, True, "finite"),
            ("step_limit", True, True, "finite"),
        ]
        for reason, reached, domain, expected in cases:
            with self.subTest(reason=reason, reached=reached, domain=domain):
                record = {
                    "member_id": "m",
                    "state_tm": "finite",
                    "tm_lower_utc": pd.Timestamp("2026-08-26T01:00:00+00:00"),
                    "tm_upper_utc": pd.Timestamp("2026-08-26T02:00:00+00:00"),
                    "tm_event_utc": pd.Timestamp("2026-08-26T01:30:00+00:00"),
                }
                config = {
                    "id": "case",
                    "arrival_port_s": 600.0 if reached else None,
                    "reached_port": reached,
                    "termination_reason": reason,
                    "domain_censor": domain,
                    "runout_right_censored": domain or reason in {
                        "time_limit", "step_limit"
                    },
                    "natural_stop": reason == "stationary_after_injection",
                    "time_limit": reason == "time_limit",
                    "step_limit": reason == "step_limit",
                    "time_limit_nonarrival": (
                        reason == "time_limit" and not reached
                    ),
                    "step_limit_nonarrival": (
                        reason == "step_limit" and not reached
                    ),
                    "domain_censor_nonarrival": (
                        domain and not reached
                    ),
                    "params": {},
                    "volume_m3": 1.0,
                    "weight": 1.0,
                }
                result = forecast.propagate([record], [config], base_geometry)[0]
                self.assertEqual(result["state_ta"], expected)

    def test_legacy_audit_payload_is_standalone_and_nonoperative(self):
        payload = forecast.build_legacy_audit(
            {
                "source_file": "求解结果/数据/old.json",
                "source_sha256": "abc",
                "physical_source_inverse": {"conditional": [1, 2]},
                "scope": "old",
            },
            "临时产物/审计.json",
            "def",
        )
        self.assertEqual(payload["status"], "legacy_audit_only_not_used")
        self.assertEqual(payload["source_audit"]["sha256"], "def")
        self.assertEqual(payload["source_audit"]["path"], "临时产物/审计.json")
        self.assertFalse(any(payload["downstream_use"].values()))

    def test_local_time_of_port_hit_is_not_double_shifted(self):
        self.assertEqual(
            forecast.iso_local(forecast.HIT_UTC),
            "2026-08-26T10:30:00+08:00",
        )

    def test_arrival_interval_lead_bounds_preserve_negative_lead(self):
        interval = {
            "status": "available",
            "lower_time": "2026-08-26T02:30:36.821000+00:00",
            "upper_time": "2026-08-26T03:33:09.275000+00:00",
        }
        lower, upper = forecast.interval_lead_bounds(interval, forecast.HIT_UTC)
        self.assertAlmostEqual(lower, -(1 + 3 / 60 + 9.275 / 3600), places=9)
        self.assertAlmostEqual(upper, -(36.821 / 3600), places=9)
        self.assertLessEqual(lower, upper)

    def test_arrival_interval_lead_is_unavailable_without_interval(self):
        lower, upper = forecast.interval_lead_bounds(
            {"status": "not_constructible_zero_finite"},
            forecast.HIT_UTC,
        )
        self.assertIsNone(lower)
        self.assertIsNone(upper)

    def test_d8_terminal_flags_separate_natural_and_censored_states(self):
        natural = forecast.terminal_flags("stationary_after_injection", False)
        limited = forecast.terminal_flags("time_limit", False)
        self.assertTrue(natural["natural_stop"])
        self.assertFalse(natural["right_censored"])
        self.assertFalse(limited["natural_stop"])
        self.assertTrue(limited["right_censored"])
        self.assertTrue(limited["time_limit_nonarrival"])
        reached_after_time_limit = forecast.terminal_flags(
            "time_limit", True, reached_port=True
        )
        self.assertTrue(reached_after_time_limit["right_censored"])
        self.assertFalse(reached_after_time_limit["arrival_right_censored"])

    def test_runout_terminal_fields_come_from_configuration(self):
        natural = forecast.runout_terminal_fields({
            "termination_reason": "stationary_after_injection",
            "runout_right_censored": False,
            "natural_stop": True,
            "time_limit_nonarrival": False,
        })
        censored = forecast.runout_terminal_fields({
            "termination_reason": "stationary_after_injection",
            "runout_right_censored": True,
            "natural_stop": False,
            "time_limit_nonarrival": False,
        })
        limited = forecast.runout_terminal_fields({
            "termination_reason": "time_limit",
            "runout_right_censored": True,
            "natural_stop": False,
            "time_limit_nonarrival": True,
        })
        self.assertEqual(natural, {
            "natural_stop": True,
            "time_limit": False,
            "step_limit": False,
            "domain_censor": False,
            "time_limit_nonarrival": False,
            "step_limit_nonarrival": False,
            "domain_censor_nonarrival": False,
            "right_censored": False,
        })
        self.assertTrue(censored["right_censored"])
        self.assertFalse(censored["natural_stop"])
        self.assertTrue(limited["time_limit_nonarrival"])

    def test_port_coverage_label_does_not_coerce_nan_to_one(self):
        common = {
            "interval_status": "available",
            "lower_time": "2026-08-26T02:30:00+00:00",
            "upper_time": "2026-08-26T03:30:00+00:00",
            "prediction_missing": False,
        }
        self.assertEqual(
            figures.port_coverage_label(pd.Series({
                **common, "definite_cover": np.nan,
            })),
            "不可估计",
        )
        self.assertEqual(
            figures.port_coverage_label(pd.Series({
                **common, "definite_cover": None,
            })),
            "不可估计",
        )
        self.assertEqual(
            figures.port_coverage_label(pd.Series({
                **common, "definite_cover": False,
            })),
            "0/1",
        )
        self.assertEqual(
            figures.port_coverage_label(pd.Series({
                **common, "definite_cover": True,
            })),
            "1/1",
        )

    def test_no_alarm_branch_is_annotated_without_plotting_nan(self):
        fig, axes = figures.plt.subplots()
        try:
            self.assertFalse(figures.plot_alarm_marker(axes, None))
            self.assertTrue(
                any("未报警" in text.get_text() for text in axes.texts)
            )
        finally:
            figures.plt.close(fig)

    def test_alarm_rule_uses_actionable_mass(self):
        self.assertTrue(forecast.should_alarm({
            "p_finite": 0.3, "p_active_at_issue": 0.2, "p_actionable": 0.5,
        }))
        self.assertFalse(forecast.should_alarm({
            "p_finite": 0.49, "p_active_at_issue": 0.0, "p_actionable": 0.49,
        }))
        self.assertFalse(forecast.should_alarm({
            "p_finite": 0.0, "p_active_at_issue": 0.0,
            "p_right_censored": 1.0, "p_actionable": 0.0,
        }))

    def test_ifs_runs_are_available_before_issue(self):
        manifest = forecast.load_ifs_manifest()
        self.assertEqual(len(manifest["source_files"]), 3)
        self.assertEqual(len(manifest["port_files"]), 1)
        for item in manifest["source_files"]:
            init = pd.Timestamp(item["init_time_utc"])
            self.assertLessEqual(init + pd.Timedelta(hours=6), forecast.ISSUE_UTC)


if __name__ == "__main__":
    unittest.main()
