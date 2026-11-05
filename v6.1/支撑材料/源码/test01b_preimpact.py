"""Small deterministic checks for the frozen label-free window protocol."""
import unittest

import numpy as np
import pandas as pd

import step01b_preimpact as p


class PreimpactTests(unittest.TestCase):
    def test_nonleap_calendar(self):
        self.assertEqual(p.calendar_day(["2024-08-26"])[0], p.calendar_day(["2025-08-26"])[0])
        self.assertTrue(p.nearby(pd.DatetimeIndex(["2024-08-11", "2023-09-10"]), p.TARGET).all())
        self.assertFalse(p.nearby(pd.DatetimeIndex(["2024-08-10"]), p.TARGET)[0])
        with self.assertRaises(ValueError):
            p.calendar_day(["2024-02-29"])

    def test_clock_and_shared_qualification(self):
        release = pd.Timestamp("2026-08-26")
        grid = pd.date_range("2026-08-18", "2026-08-26", freq="h")
        f = pd.DataFrame(1., index=grid, columns=p.COLS)
        table = p.window_table(f, 168, [release])
        self.assertEqual(table.loc[release, "end"], pd.Timestamp("2026-08-25 22:00"))
        self.assertEqual(table.loc[release, "start"], pd.Timestamp("2026-08-18 23:00"))
        self.assertEqual(table.loc[release, "n__WS_10m"], 168)
        start = table.loc[release, "start"]
        f.loc[pd.date_range(start, periods=33, freq="h"), "TKE_obs"] = np.nan
        self.assertTrue(p.window_table(f, 168, [release]).loc[release, "common_valid"])
        f.loc[start + pd.Timedelta(hours=33), "TKE_obs"] = np.nan
        self.assertFalse(p.window_table(f, 168, [release]).loc[release, "common_valid"])

    def test_scale_fallback_and_no_epsilon(self):
        f = pd.DataFrame({c: [0., 0., 0., 1.] for c in p.COLS})
        f[p.COLS[-1]] = 0.
        center, scale, method = p.robust_baseline(f)
        self.assertEqual(center.iloc[0], 0.)
        self.assertAlmostEqual(scale.iloc[0], .25 / 1.349)
        self.assertEqual(method.iloc[0], "IQR/1.349回退")
        self.assertTrue(np.isnan(scale.iloc[-1]))

    def test_complete_connected_components(self):
        a, b, c = p.COLS[:3]
        groups = p.components_from_edges([(a, b, .9), (b, c, -.9)])
        self.assertEqual(groups[0], [a, b, c])
        self.assertEqual(sum(map(len, groups)), 12)
        short = pd.DataFrame(np.arange(12 * 199).reshape(199, 12),
                             index=pd.date_range("2021-06-01", periods=199, freq="h"), columns=p.COLS)
        result = p.connected_clusters(short)
        self.assertFalse(result["confirmed"])
        self.assertEqual(result["clusters"], [])

    def test_seven_day_schedule_precedes_missing_screen(self):
        dates = pd.date_range("2021-08-11", "2021-09-10", freq="D")
        table = pd.DataFrame({"common_valid": True}, index=dates)
        table.loc[dates[0], "common_valid"] = False
        selected = p.seven_day_reference(table)
        self.assertEqual(selected.index[0], dates[0])
        self.assertEqual(selected.index[1], dates[7])
        self.assertFalse(selected.common_valid.iloc[0])

    def test_score_uses_window_distribution_and_frozen_years(self):
        dates = pd.DatetimeIndex(["2021-08-26", "2022-08-26", "2023-08-26", "2024-08-26", "2025-08-26", "2026-08-26"])
        table = pd.DataFrame({c: [1., 2., 3., 4., 999., 5.] for c in p.COLS}, index=dates)
        table["common_valid"] = True
        table["start"] = dates - pd.Timedelta(hours=169)
        table["end"] = dates - pd.Timedelta(hours=2)
        table["length_h"] = 168
        table["all12_complete_hours"] = 168
        for c in p.COLS:
            table[f"n__{c}"] = 168
        result = p.score_window(table, p.TARGET, [[c] for c in p.COLS])
        row = result["rows"][0]
        self.assertEqual(result["reference_windows"], 4)
        self.assertEqual(row["baseline_median"], 2.5)
        self.assertEqual(row["tail_rank"], .2)
        self.assertEqual(result["top5"], p.COLS[:5])
        self.assertNotIn(2025, result["reference_counts_by_year"])

    def test_missing_training_year_is_not_a_noop_sensitivity(self):
        dates = pd.DatetimeIndex(["2021-08-26", "2022-08-26", "2023-08-26",
                                  "2024-08-26", p.TARGET])
        table = pd.DataFrame({c: [1., 2., 3., 4., 2.5] for c in p.COLS}, index=dates)
        table["common_valid"] = [True, False, True, True, True]
        table["start"] = dates - pd.Timedelta(hours=169)
        table["end"] = dates - pd.Timedelta(hours=2)
        table["length_h"] = 168
        table["all12_complete_hours"] = 168
        for c in p.COLS:
            table[f"n__{c}"] = 168
        clusters = [[c] for c in p.COLS]
        target = p.score_window(table, p.TARGET, clusters)
        result, _ = p.background_sensitivity(table, target, clusters)
        bootstrap = result["year_block_bootstrap"]
        self.assertEqual(bootstrap["available_training_years"], [2021, 2023, 2024])
        self.assertEqual(set(bootstrap["excluded_training_years"]), {"2022"})
        self.assertTrue(all(2022 not in draw for draw in bootstrap["sampled_training_years"]))
        loyo = {row["omitted_year"]: row for row in result["leave_one_training_year_out"]}
        self.assertEqual(loyo[2022]["status"], "不可评估：该训练年无共同有效参照窗")
        self.assertEqual(loyo[2022]["reference_windows"], 0)
        self.assertIsNone(loyo[2022]["full_rank_Kendall_tau"])


if __name__ == "__main__":
    unittest.main()
