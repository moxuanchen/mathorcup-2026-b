"""Independent numeric and boundary checks for event-cumulative SCS-CN."""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from mmlib import ranking, trigger as tg


Q20_CN85 = 2.1800981218098032


def features(rain, api=10.0, start="2024-07-01"):
    frame = pd.DataFrame({"P": rain, "API": api},
                         index=pd.date_range(start, periods=len(rain), freq="h"))
    frame.attrs["api_quantiles"] = (8.0, 18.0)
    return frame


def known_event(rain, **kwargs):
    return features([0.0] * 6 + list(rain), **kwargs)


class EventRunoffTests(unittest.TestCase):
    def test_independent_twenty_mm_reference_and_hourly_increments(self):
        out = tg.event_rain_runoff(known_event([5.0] * 4))
        np.testing.assert_allclose(out.runoff_mm.iloc[6:],
                                   [0.0, 0.02337246853590084,
                                    0.6928213511154855, 1.4639043021584168],
                                   rtol=0, atol=1e-12)
        self.assertAlmostEqual(out.Q.iloc[-1], Q20_CN85, places=12)
        self.assertEqual(out.cumulative_P.iloc[-1], 20.0)

    def test_same_event_total_is_invariant_to_rainfall_partition(self):
        for rain in ([20.0], [5.0] * 4, [1.0] * 20):
            with self.subTest(rain=rain):
                out = tg.event_rain_runoff(known_event(rain))
                self.assertAlmostEqual(out.runoff_mm.sum(), Q20_CN85, places=12)
                self.assertTrue(out.runoff_mm.ge(0).all())
                self.assertTrue(out.runoff_mm.le(out.cumulative_P.diff().fillna(0)).all())

    def test_five_dry_hours_keep_event_six_start_new_event(self):
        for dry in (5, 6):
            with self.subTest(dry=dry):
                out = tg.event_rain_runoff(known_event([20.0] + [0.0] * dry + [20.0]))
                first, last = out.iloc[6], out.iloc[-1]
                if dry == 5:
                    self.assertEqual(first.event_id, last.event_id)
                    merged = tg.rain_runoff(known_event([40.0])).sum()
                    self.assertAlmostEqual(out.runoff_mm.sum(), merged, places=12)
                    self.assertEqual(last.cumulative_P, 40.0)
                else:
                    self.assertNotEqual(first.event_id, last.event_id)
                    self.assertAlmostEqual(out.runoff_mm.sum(), 2 * Q20_CN85, places=12)
                    self.assertTrue(out.boundary_confirmed.iloc[-2])
                    self.assertTrue(pd.isna(out.event_id.iloc[-2]))

    def test_only_strict_zero_counts_towards_a_boundary(self):
        for drizzle in (0.1, 1e-12):
            with self.subTest(drizzle=drizzle):
                out = tg.event_rain_runoff(features([0.] * 3 + [drizzle] + [0.] * 3 + [20.]))
                self.assertTrue(np.isnan(out.runoff_mm.iloc[-1]))
                self.assertFalse(out.boundary_confirmed.any())
                self.assertTrue(out.event_id.isna().all())
        out = tg.event_rain_runoff(known_event([0.1] + [0.] * 5 + [20.]))
        self.assertEqual(out.event_id.iloc[6], out.event_id.iloc[-1])
        self.assertAlmostEqual(out.cumulative_P.iloc[-1], 20.1)
        self.assertGreater(out.runoff_mm.sum(), Q20_CN85)

    def test_initial_unknown_is_not_mistaken_for_a_new_event(self):
        out = tg.event_rain_runoff(features([20.] + [0.] * 5 + [20.] + [0.] * 6 + [20.]))
        self.assertTrue(out.runoff_mm.iloc[[0, 6]].isna().all())
        self.assertTrue(out.event_id.iloc[:13].isna().all())
        self.assertTrue(out.state.iloc[:12].eq("unknown_boundary").all())
        self.assertTrue(out.runoff_mm.iloc[1:6].eq(0).all())
        self.assertFalse(out.boundary_known.iloc[5])
        self.assertAlmostEqual(out.runoff_mm.iloc[-1], Q20_CN85, places=12)

    def test_invalid_rain_breaks_even_an_almost_complete_dry_period(self):
        for invalid in (np.nan, -0.1, np.inf, -np.inf):
            with self.subTest(invalid=invalid):
                rain = [0.] * 5 + [invalid] + [0.] * 5 + [20.] + [0.] * 6 + [20.]
                out = tg.event_rain_runoff(features(rain))
                self.assertEqual(out.state.iloc[5], "invalid_rain")
                self.assertTrue(np.isnan(out.runoff_mm.iloc[5]))
                self.assertTrue(np.isnan(out.runoff_mm.iloc[11]))
                self.assertTrue(pd.isna(out.event_id.iloc[11]))
                self.assertAlmostEqual(out.runoff_mm.iloc[-1], Q20_CN85, places=12)

    def test_missing_rain_invalidates_continuation_without_erasing_known_prefix(self):
        out = tg.event_rain_runoff(known_event([20., np.nan, 20., 0., 0.]))
        self.assertAlmostEqual(out.runoff_mm.iloc[6], Q20_CN85, places=12)
        self.assertTrue(out.runoff_mm.iloc[7:9].isna().all())
        self.assertTrue(out.event_id.iloc[7:].isna().all())
        self.assertTrue(out.cumulative_P.iloc[7:].isna().all())
        self.assertTrue(out.runoff_mm.iloc[9:].eq(0).all())

    def test_cn_uses_previous_hour_api_and_is_frozen_within_event(self):
        for prior_api, expected_cn in ((1., 63.75), (10., 85.), (25., 97.75)):
            with self.subTest(prior_api=prior_api):
                frame = known_event([5., 5., 0., 5., 5.])
                frame.loc[frame.index[5], "API"] = prior_api
                frame.loc[frame.index[6:], "API"] = [100., np.nan, -1., np.inf, 0.]
                out = tg.event_rain_runoff(frame)
                np.testing.assert_allclose(out.event_CN.iloc[6:], expected_cn, rtol=0, atol=1e-12)
                self.assertTrue(out.event_API.iloc[6:].eq(prior_api).all())
                constant = known_event([5., 5., 0., 5., 5.], api=prior_api)
                np.testing.assert_allclose(out.runoff_mm, tg.rain_runoff(constant))

    def test_api_class_boundaries_are_inclusive_and_cn_is_clipped(self):
        for api, cn, expected in ((8., 85., 85.), (18., 85., 85.),
                                  (25., 95., 98.), (1., 20., 30.)):
            with self.subTest(api=api, cn=cn):
                out = tg.event_rain_runoff(known_event([20.], api=api), cn)
                self.assertEqual(out.event_CN.iloc[-1], expected)

    def test_missing_antecedent_api_invalidates_whole_event_then_recovers(self):
        for invalid in (np.nan, np.inf, -np.inf, -1.):
            with self.subTest(invalid=invalid):
                frame = known_event([5., 0., 15.] + [0.] * 6 + [20.])
                frame.loc[frame.index[5], "API"] = invalid
                out = tg.event_rain_runoff(frame)
                self.assertTrue(out.runoff_mm.iloc[[6, 8]].isna().all())
                self.assertTrue(out.state.iloc[6:9].eq("event_api_missing").all())
                self.assertTrue(out.event_CN.iloc[6:9].isna().all())
                self.assertEqual(out.runoff_mm.iloc[7], 0.)
                self.assertEqual(out.cumulative_P.iloc[8], 20.)
                self.assertTrue(np.isnan(out.Q.iloc[8]))
                self.assertAlmostEqual(out.runoff_mm.iloc[-1], Q20_CN85, places=12)
                self.assertNotEqual(out.event_id.iloc[6], out.event_id.iloc[-1])

    def test_year_boundary_and_unclosed_record_do_not_reset_event(self):
        frame = known_event([20., 0., 20.], start="2024-12-31 17:00")
        out = tg.event_rain_runoff(frame)
        self.assertEqual(out.index[6].year, 2024)
        self.assertEqual(out.index[-1].year, 2025)
        self.assertEqual(out.event_id.iloc[6], out.event_id.iloc[-1])
        self.assertEqual(out.cumulative_P.iloc[-1], 40.)
        reference = tg.rain_runoff(known_event([40.])).sum()
        self.assertAlmostEqual(out.runoff_mm.sum(), reference, places=12)
        closed = tg.event_rain_runoff(pd.concat([frame, features([0.] * 6, start="2025-01-01 02:00")]))
        np.testing.assert_allclose(out.runoff_mm, closed.runoff_mm.iloc[:len(out)])

    def test_time_gap_during_dry_hours_restarts_confirmation(self):
        frame = features([0.] * 6 + [20.] + [0.] * 6 + [20.])
        frame.index = frame.index[:3].append(frame.index[3:] + pd.Timedelta(hours=1))
        out = tg.event_rain_runoff(frame)
        self.assertTrue(out.time_gap.iloc[3])
        self.assertTrue(np.isnan(out.runoff_mm.iloc[6]))
        self.assertTrue(pd.isna(out.event_id.iloc[6]))
        self.assertAlmostEqual(out.runoff_mm.iloc[-1], Q20_CN85, places=12)
        pd.testing.assert_index_equal(out.index, frame.index)

    def test_time_gap_during_rain_invalidates_event_and_antecedent(self):
        frame = known_event([20., 20.])
        frame.index = frame.index[:-1].append(pd.DatetimeIndex([frame.index[-1] + pd.Timedelta(hours=1)]))
        out = tg.event_rain_runoff(frame)
        self.assertAlmostEqual(out.runoff_mm.iloc[6], Q20_CN85, places=12)
        self.assertEqual(out.state.iloc[-1], "unknown_boundary")
        self.assertTrue(np.isnan(out.runoff_mm.iloc[-1]))
        self.assertTrue(np.isnan(out.Q.iloc[-1]))

    def test_boundary_sensitivity_three_six_twelve_hours(self):
        frame = features([0.] * 12 + [20.] + [0.] * 6 + [20.])
        for dry in (3, 6, 12):
            with self.subTest(dry=dry):
                out = tg.event_rain_runoff(frame, dry_hours=dry)
                expected = 2 * Q20_CN85 if dry <= 6 else tg.rain_runoff(known_event([40.])).sum()
                self.assertAlmostEqual(out.runoff_mm.sum(), expected, places=12)
                self.assertEqual(out.attrs["rainfall_event_protocol"]["dry_hours"], dry)

    def test_wrapper_diagnostics_and_no_input_mutation(self):
        frame = known_event([20., 0.])
        before = frame.copy(deep=True)
        out = tg.event_rain_runoff(frame)
        pd.testing.assert_series_equal(tg.rain_runoff(frame), out.runoff_mm.rename(None))
        pd.testing.assert_frame_equal(frame, before)
        self.assertEqual(frame.attrs, before.attrs)
        self.assertEqual(str(out.event_id.dtype), "Int64")
        self.assertEqual(out.event_start.sum(), 1)
        self.assertEqual(out.boundary_confirmed.sum(), 1)
        self.assertTrue(out.runoff_mm.iloc[:6].eq(0).all())
        self.assertTrue(out.Q.iloc[:5].isna().all())
        self.assertEqual(out.Q.iloc[5], 0.)

    def test_invalid_index_and_parameters_are_rejected(self):
        base = known_event([20.])
        indexes = [base.index[::-1], pd.DatetimeIndex([base.index[0]] * len(base)),
                   pd.RangeIndex(len(base)), base.index + pd.Timedelta(minutes=30),
                   pd.DatetimeIndex([pd.NaT, *base.index[1:]])]
        for index in indexes:
            with self.subTest(index=index):
                frame = base.copy()
                frame.index = index
                with self.assertRaises(ValueError):
                    tg.event_rain_runoff(frame)
        for cn in (0., -1., 101., np.inf, np.nan):
            with self.subTest(cn=cn), self.assertRaises(ValueError):
                tg.event_rain_runoff(base, cn=cn)
        for dry in (0, -1, 1.5, True):
            with self.subTest(dry=dry), self.assertRaises(ValueError):
                tg.event_rain_runoff(base, dry_hours=dry)

    def test_timezone_aware_index_and_empty_input(self):
        frame = known_event([20.])
        frame.index = frame.index.tz_localize("Asia/Shanghai")
        out = tg.event_rain_runoff(frame)
        pd.testing.assert_index_equal(frame.index, out.index)
        self.assertAlmostEqual(out.runoff_mm.iloc[-1], Q20_CN85, places=12)
        empty = tg.event_rain_runoff(frame.iloc[:0])
        self.assertTrue(empty.empty)
        self.assertIn("boundary_confirmed", empty.columns)

    def test_routing_keeps_missing_water_and_preserves_known_runoff_volume(self):
        model = tg.TriggerModel(1., 0., 1., 1.)
        frame = known_event([20., np.nan, 0., 0., 0.])
        frame["melt_energy"], frame["Ta_2m"] = 0., 10.
        discharge = model.discharge(frame)
        self.assertTrue(discharge.Q_rain.iloc[7:10].isna().all())
        self.assertTrue(discharge.Q_water.iloc[7:10].isna().all())
        self.assertTrue(discharge.hydraulic_state.iloc[7:10].eq("not_evaluable").all())
        self.assertEqual(discharge.Q_rain.iloc[10], 0.)
        frame = known_event([20., 0., 0., 0.])
        frame["melt_energy"], frame["Ta_2m"] = 0., 10.
        discharge = model.discharge(frame)
        self.assertAlmostEqual(discharge.Q_rain.sum() * 3600 / 1000, Q20_CN85, places=12)

    def test_routing_uses_clock_hours_across_a_missing_index_row(self):
        frame = known_event([20., 0., 0., 0.])
        frame["melt_energy"], frame["Ta_2m"] = 10., 10.
        frame.index = frame.index[:7].append(frame.index[7:] + pd.Timedelta(hours=1))
        discharge = tg.TriggerModel(1., 0.2, 1., 1.).discharge(frame)
        pd.testing.assert_index_equal(discharge.index, frame.index)
        self.assertTrue(discharge.Q_rain.iloc[7:9].isna().all())
        self.assertTrue(discharge.Q_melt.iloc[7:9].isna().all())
        self.assertEqual(discharge.Q_rain.iloc[9], 0.)
        self.assertTrue(np.isfinite(discharge.Q_melt.iloc[9]))


class FoldAPIThresholdTests(unittest.TestCase):
    def test_only_this_folds_2021_2025_warm_season_enters_terciles(self):
        groups = [("2020-07-01", np.full(120, 1e9)),
                  ("2024-01-01", np.full(120, 1e6)),
                  ("2024-07-01", np.arange(120.)),
                  ("2025-07-01", np.full(120, 1e7)),
                  ("2026-07-01", np.full(120, 1e8))]
        frame = pd.concat([pd.DataFrame({"API": values}, index=pd.date_range(start, periods=120, freq="h"))
                           for start, values in groups])
        frame.attrs["api_quantiles"] = (1e12, 1e13)
        train = frame.index.year != 2025
        test = ~train
        class ThresholdsCaptured(Exception):
            pass
        with mock.patch.object(tg, "default_model", side_effect=ThresholdsCaptured) as model:
            with self.assertRaises(ThresholdsCaptured):
                ranking._fit_evidence(frame, pd.Series(dtype=float), [], train, test)
        frozen = model.call_args.args[0]
        np.testing.assert_allclose(frozen.attrs["api_quantiles"], [119 / 3, 238 / 3])
        self.assertEqual(frame.attrs["api_quantiles"], (1e12, 1e13))
        pd.testing.assert_index_equal(frozen.index, frame.index)

    def test_insufficient_valid_warm_sample_cannot_fall_back_to_winter(self):
        frame = features([0.] * 120)
        frame["API"] = np.r_[np.arange(99.), np.full(21, np.inf)]
        winter = features([0.] * 120, api=1e6, start="2024-01-01")
        frame = pd.concat([winter, frame])
        train = np.ones(len(frame), dtype=bool)
        with mock.patch.object(tg, "default_model") as model:
            with self.assertRaisesRegex(ValueError, "暖季"):
                ranking._fit_evidence(frame, pd.Series(dtype=float), [], train, ~train)
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
