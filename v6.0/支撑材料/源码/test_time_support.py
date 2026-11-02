"""Checks at the raw-clock boundary, including independent energy reconstruction."""
import unittest
import numpy as np
import pandas as pd
import step03_forecast as f


class ClockTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame({"T": [2., 6., 10.], "RH": [40., 60., 80.],
                             "SW": [999., 100., 200.], "P": [99., 1., 2.]},
                            index=pd.date_range("2026-08-26", periods=3, freq="h"))

    def test_variable_sentinels_and_final_support(self):
        raw = self.raw()
        result = f.align_ifs_hourly(raw)
        np.testing.assert_allclose(result[["T", "RH", "SW", "P"]],
                                   [[4., 50., 100., 1.], [8., 70., 200., 2.]])
        self.assertEqual(result.index[-1] + pd.Timedelta(hours=1), raw.index[-1])
        self.assertIn("preceding-hour", result.attrs["time_support"]["SW"])

    def test_missing_hour_does_not_jump_to_next_surviving_row(self):
        result = f.align_ifs_hourly(self.raw().iloc[[0, 2]])
        self.assertTrue(result["T"].isna().all())
        self.assertTrue(pd.isna(result["SW"].iloc[0]))
        self.assertEqual(result["SW"].iloc[1], 200.)

    def test_one_missing_endpoint_is_not_an_hourly_average(self):
        raw = self.raw()
        raw.iloc[1, raw.columns.get_loc("RH")] = np.nan
        self.assertTrue(f.align_ifs_hourly(raw)["RH"].isna().all())

    def test_preissue_hour_uses_issue_end_flux_and_two_states(self):
        raw = self.raw()
        raw.index = pd.date_range(f.PREISSUE_UTC, periods=3, freq="h")
        result = f.align_ifs_hourly(raw)
        self.assertEqual(result.loc[f.PREISSUE_UTC, "SW"], 100.)
        self.assertEqual(result.loc[f.PREISSUE_UTC, "T"], 4.)

    def test_duplicate_and_unsorted_times_rejected(self):
        raw = self.raw()
        for broken in (raw.iloc[[0, 0, 1]], raw.iloc[::-1]):
            with self.assertRaises(ValueError):
                f.align_ifs_hourly(broken)

    def test_invalid_flux_is_missing_not_zero(self):
        raw = self.raw()
        raw.loc[raw.index[1], ["SW", "P"]] = [-1., np.inf]
        result = f.align_ifs_hourly(raw)
        self.assertTrue(result.iloc[0][["SW", "P"]].isna().all())


class HistoricalClockTests(unittest.TestCase):
    def features(self):
        return pd.DataFrame({"Ta_2m": [2., 6., 10.], "RH_2m": [40., 60., 80.],
                             "Rsd": [100., 300., 500.], "Rsu": [0., 0., 0.],
                             "Rld": [0., 0., 0.], "Rlu": [0., 0., 0.],
                             "G_0.1m": [10., 20., 30.], "H": [30., 40., 50.],
                             "LE": [5., 10., 15.], "melt_energy": [9999.]*3},
                            index=pd.date_range("2024-07-01", periods=3, freq="h"))

    def test_energy_target_rebuilt_after_independent_family_alignment(self):
        raw = self.features()
        mixed = f.historical_energy_support(raw, rad_end=True)
        self.assertEqual(mixed.iloc[0]["melt_energy"], 255.)
        all_end = f.historical_energy_support(raw, rad_end=True, soil_end=True, flux_end=True)
        self.assertEqual(all_end.iloc[0]["melt_energy"], 230.)
        self.assertEqual(mixed.iloc[0]["Ta_2m"], 4.)
        self.assertNotEqual(mixed.iloc[0]["melt_energy"], raw.iloc[0]["melt_energy"])

    def test_met_mean_and_point_are_different_declared_scenarios(self):
        raw = self.features()
        self.assertEqual(f.historical_energy_support(raw, met="start_mean").iloc[0]["Ta_2m"], 2.)
        self.assertEqual(f.historical_energy_support(raw, met="end_mean").iloc[0]["Ta_2m"], 6.)
        self.assertEqual(f.historical_energy_support(raw).iloc[0]["Ta_2m"], 4.)
        self.assertFalse(f.historical_energy_support(raw).attrs["station_support_assumption"]["metadata_verified"])

    def test_family_endpoint_missing_propagates_to_energy_target(self):
        raw = self.features()
        raw.iloc[1, raw.columns.get_loc("H")] = np.nan
        result = f.historical_energy_support(raw, flux_end=True)
        self.assertTrue(pd.isna(result.iloc[0]["melt_energy"]))


if __name__ == "__main__":
    unittest.main()
