"""Synthetic checks for the frozen five-layer hydrostatic protocol."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mmlib import meteo as mt


def observations(n=4):
    frame = pd.DataFrame(index=pd.date_range("2024-08-01", periods=n, freq="h"))
    for z in mt.LEVELS_T:
        frame[f"Ta_{z}m"] = 5.
        frame[f"RH_{z}m"] = 30.
        frame[f"WS_{z}m"] = 1. + .1 * z
        frame[f"WD_{z}m"] = 0.
    frame["Pressure"] = 600.
    return frame


class HydrostaticTests(unittest.TestCase):
    def test_pressure_anchor_and_monotonicity(self):
        df = observations()
        for ref in (1, 2, 10, 20):
            profile = mt.hydrostatic_profiles(df, ref)
            np.testing.assert_allclose(profile[f"p_{ref}m"], df.Pressure)
            self.assertTrue((np.diff(profile[[f"p_{z}m" for z in mt.LEVELS_T]], axis=1) < 0).all())
            self.assertLess(profile.attrs["iteration_3_minus_2_max_hPa"], 1e-7)
            self.assertEqual(profile.attrs["iterations"], 3)

    def test_isothermal_analytic_dry_column(self):
        df = observations(); df[[f"RH_{z}m" for z in mt.LEVELS_T]] = 0.
        profile = mt.hydrostatic_profiles(df)
        for z in mt.LEVELS_T:
            expected = 600 * np.exp(-mt.G * (z - 2) / (mt.R_D * 278.15))
            np.testing.assert_allclose(profile[f"p_{z}m"], expected, rtol=1e-12)

    def test_neutral_dry_adiabatic_column(self):
        df = observations()
        for z in mt.LEVELS_T:
            df[f"RH_{z}m"] = 0.
            df[f"Ta_{z}m"] = 5 - mt.G / mt.CP * (z - 2)
        profile = mt.hydrostatic_profiles(df)
        self.assertLess(float((profile.theta_20m - profile.theta_2m).abs().max()), 2e-5)
        self.assertLess(float(mt.bulk_richardson(df, profile).abs().max()), 1e-4)

    def test_missing_any_layer_masks_thermodynamics(self):
        for family in ("Ta", "RH"):
            for z in mt.LEVELS_T:
                df = observations(); df.loc[df.index[1], f"{family}_{z}m"] = np.nan
                corrected = mt.hydrostatic_profiles(df)
                old = mt.hydrostatic_profiles(df, corrected=False)
                self.assertFalse(corrected.thermo_valid.iloc[1])
                self.assertTrue(corrected.filter(regex=r"^(p|q|theta|thetav)_").iloc[1].isna().all())
                self.assertTrue(mt.bulk_richardson(df, corrected).isna().iloc[1])
                np.testing.assert_array_equal(corrected.thermo_valid, old.thermo_valid)

    def test_vector_shear_uses_direction(self):
        df = observations(); df[[f"WS_{z}m" for z in mt.LEVELS_T]] = 2.
        df["WD_20m"] = 180.
        np.testing.assert_allclose(mt.vector_shear_squared(df), 16.)
        np.testing.assert_allclose(mt.wind_shear(df).vector_shear_2_20m, 4 / 18)
        df["WD_20m"] = 0.
        self.assertTrue(mt.bulk_richardson(df).isna().all())

    def test_crossing_at_20_is_not_right_censoring(self):
        df = observations()
        profile = mt.hydrostatic_profiles(df)
        for z in mt.LEVELS_T:
            profile[f"thetav_{z}m"] = 300.
        profile.loc[df.index[0], "thetav_20m"] = 302.
        result = mt.pbl_diagnostics(df, profile)
        self.assertEqual(result.pbl_crossing_height_m.iloc[0], 20.)
        self.assertEqual(result.pbl_right_censored.iloc[0], 0.)
        self.assertTrue(pd.isna(result.pbl_crossing_height_m.iloc[1]))
        self.assertEqual(result.pbl_right_censored.iloc[1], 1.)
        self.assertEqual(result.pblh.iloc[0], result.pblh.iloc[1])

    def test_missing_lower_wind_pair_cannot_be_skipped(self):
        df = observations(); df.loc[df.index[0], "WD_4m"] = np.nan
        profile = mt.hydrostatic_profiles(df)
        profile["thetav_20m"] += 10.
        result = mt.pbl_diagnostics(df, profile)
        self.assertTrue(result.iloc[0].isna().all())

    def test_pressure_is_shared_by_all_derived_quantities(self):
        df = observations()
        for name in ("Precip", "Rsd", "Rsu", "Rld", "Rlu", "H", "LE", "G_0.1m"):
            df[name] = 0.
        feat = mt.build_feature_table(df)
        np.testing.assert_allclose(feat.q, feat.q_2m)
        np.testing.assert_allclose(feat.dtheta_1_20, feat.theta_20m - feat.theta_1m)
        np.testing.assert_allclose(feat.Ri_b, mt.bulk_richardson(df))
        self.assertEqual(feat.attrs["feature_version"], "v2-hydrostatic-five-layer-3iter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
