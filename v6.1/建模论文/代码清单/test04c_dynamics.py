"""Deterministic synthetic checks; no GIS writes or observed-event calibration."""
from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.integrate import solve_ivp

from mmlib import finite_inventory as fi


class DynamicsTests(unittest.TestCase):
    def assertClose(self, left, right, rtol=1e-9, atol=1e-9):
        self.assertTrue(math.isclose(left, right, rel_tol=rtol, abs_tol=atol),
                        f"{left!r} != {right!r}")

    def test_requested_mode_resolution(self):
        self.assertEqual(fi.resolve_mode("M0", 0.7), (0.0, "M1"))
        self.assertEqual(fi.resolve_mode("M1", 0.5), (0.5, "M1"))
        self.assertEqual(fi.resolve_mode("M2", 0.5), (0.5, "M2"))
        self.assertEqual(fi.resolve_mode("M2", 0.0), (0.0, "M1"))
        with self.assertRaises(ValueError):
            fi.resolve_mode("M3", 0.1)
        with self.assertRaises(ValueError):
            fi.resolve_mode("M1", -0.1)

    def test_kappa_zero_energy_line(self):
        sh = np.array([0., 30., 80., 200.])
        z = np.array([100., 95., 89., 70.])
        r = fi.simulate(sh, z, [4., 0., 6.], v0=20, mu=.05, kappa=0)
        for p in r["trajectory"]:
            expected = 400+2*fi.G*(100-p["z_m"]-.05*p["s_h_m"])
            self.assertClose(p["velocity_m_s"]**2, expected)

    def test_ode_all_signs_and_terminal_sides(self):
        for a, k, v, length in [(2., .01, 4., 30.), (2., .01, 30., 30.),
                                 (-2., .01, 20., 30.), (0., .01, 20., 30.),
                                 (2., 0., 4., 30.), (-2., 0., 20., 30.),
                                 (0., 0., 20., 30.), (1e-12, .01, 20., 30.)]:
            with self.subTest(a=a, k=k, v=v):
                ref = fi.segment(v*v, a, k, length)
                def rhs(t, y):
                    return [y[1], a-k*y[1]**2]
                def target(t, y):
                    return y[0]-ref["distance_m"]
                target.terminal = True
                target.direction = 1
                ode = solve_ivp(rhs, [0., 2*ref["time_s"]+1], [0., v],
                                events=target, rtol=2e-11, atol=2e-12, max_step=.02)
                self.assertTrue(len(ode.t_events[0]) == 1)
                self.assertClose(ode.t_events[0][0], ref["time_s"], 2e-9)
                self.assertClose(ode.y_events[0][0, 1]**2, ref["w_out_m2_s2"], 2e-9)

    def test_stop_ode_and_finite_time(self):
        a, k, v = -2., .01, 10.
        ref = fi.segment(v*v, a, k, 1000.)
        def rhs(t, y):
            return [y[1], a-k*y[1]**2]
        def stop(t, y):
            return y[1]
        stop.terminal, stop.direction = True, -1
        ode = solve_ivp(rhs, [0, 100.], [0., v], events=stop,
                        rtol=1e-11, atol=1e-12, max_step=.02)
        self.assertTrue(ref["stopped"])
        self.assertClose(ode.t_events[0][0], ref["time_s"])
        self.assertClose(ode.y_events[0][0, 0], ref["distance_m"])

    def test_terminal_exact(self):
        a, k, w = 2., .01, 200.
        r = fi.segment(w, a, k, 50.)
        self.assertClose(r["w_out_m2_s2"], w)
        self.assertClose(r["time_s"], 50/math.sqrt(w))

    def test_degenerate_limits(self):
        base = fi.segment(400., .2, 0., 20.)
        tiny = fi.segment(400., .2, 1e-14, 20.)
        self.assertClose(base["w_out_m2_s2"], tiny["w_out_m2_s2"])
        self.assertClose(base["time_s"], tiny["time_s"])
        zero = fi.segment(400., 0., .01, 20.)
        for a in [-1e-12, 1e-12]:
            r = fi.segment(400., a, .01, 20.)
            self.assertClose(r["w_out_m2_s2"], zero["w_out_m2_s2"])
            self.assertClose(r["time_s"], zero["time_s"])

    def test_near_terminal(self):
        for eps in [-1e-11, 1e-11]:
            r = fi.segment(200*(1+eps), 2., .01, 50.)
            self.assertClose(r["time_s"], 50/math.sqrt(200), 1e-9)

    def test_flat_never_finite_stop(self):
        r = fi.segment(400., 0., .01, 200.)
        self.assertFalse(r["stopped"])
        self.assertGreater(r["w_out_m2_s2"], 0)

    def test_signed_uphill_and_first_stop(self):
        r = fi.simulate([0., 100., 200.], [0., 10., -100.], [4., 0., 6.],
                        v0=10., mu=.1, kappa=0)
        self.assertEqual(r["status"], "first_stop_before_target")
        self.assertLess(r["final_sh_m"], 100)
        self.assertIsNone(r["arrival_s"])
        self.assertClose(r["final_velocity_m_s"], 0)

    def test_zero_speed_target_contact(self):
        L, mu, v = 100., .1, 10.
        sh = v*v/(2*fi.G*mu)
        r = fi.simulate([0., L], [0., 0.], [4., 0., 6.], v0=v, mu=mu,
                        kappa=0, target_sh=sh)
        self.assertEqual(r["status"], "zero_speed_target_contact")
        self.assertIsNone(r["arrival_s"])

    def test_target_inside_segment(self):
        r = fi.simulate([0., 100.], [0., 0.], [4., 0., 6.],
                        v0=10., mu=0., kappa=0., target_sh=37.)
        self.assertClose(r["arrival_s"], 3.7)
        self.assertClose(r["final_sh_m"], 37.)

    def test_stationary_mixing(self):
        r = fi.stationary_mix([4., 2., 6.], 10., [1., 0., 3.])
        self.assertClose(r["mass_before_kg"]*10,
                         r["mass_after_kg"]*r["velocity_after_m_s"])
        self.assertGreater(r["mix_loss_J"], 0)
        self.assertLess(r["velocity_after_m_s"], 10)
        self.assertClose(r["kinetic_residual_J"], 0, atol=1e-8)

    def test_zero_stock_identity(self):
        n = [{"id": "zero", "s_h_m": 50., "rock_m3": 0., "water_m3": 0.}]
        args = ([0., 100.], [20., 0.], [4., 0., 6.])
        a = fi.simulate(*args, mu=.05, kappa=.001)
        b = fi.simulate(*args, nodes=n, mu=.05, kappa=.001)
        self.assertClose(a["arrival_s"], b["arrival_s"])
        self.assertClose(a["final_velocity_m_s"], b["final_velocity_m_s"])

    def test_inventory_once_and_energy(self):
        n = [{"id": "a", "s_h_m": 40., "rock_m3": 2., "water_m3": 3.},
             {"id": "b", "s_h_m": 70., "rock_m3": 1., "water_m3": 2.}]
        r = fi.simulate([0., 100.], [40., 0.], [4., 0., 6.], nodes=n,
                        mu=.05, kappa=.001)
        self.assertEqual(len(r["node_ledger"]), 2)
        self.assertClose(r["remaining_inventory_mass_kg"], 0)
        self.assertClose(r["mass_residual_kg"], 0)
        self.assertLess(r["energy_relative_residual_max"], 1e-12)
        self.assertLess(r["segment_energy_relative_residual_max"], 1e-12)

    def test_subdivision_fixed_nodes(self):
        n = [{"id": "a", "s_h_m": 33.333333333, "rock_m3": 2., "water_m3": 3.},
             {"id": "b", "s_h_m": 66.666666666, "rock_m3": 1., "water_m3": 2.}]
        args = ([0., 50., 100.], [40., 20., 0.], [4., 0., 6.])
        for mode in ["M0", "M1", "M2"]:
            a = fi.simulate(*args, nodes=n, mode=mode, mu=.05, kappa=.001)
            b = fi.simulate(*args, nodes=n, mode=mode, mu=.05, kappa=.001, subdivisions=7)
            self.assertClose(a["arrival_s"], b["arrival_s"])
            self.assertClose(a["final_velocity_m_s"], b["final_velocity_m_s"])
            self.assertEqual(len(b["node_ledger"]), 2)
            self.assertClose(a["consumed_inventory_mass_kg"], b["consumed_inventory_mass_kg"])

    def test_subdivision_stop(self):
        args = ([0., 100.], [0., 0.], [4., 0., 6.])
        a = fi.simulate(*args, v0=10., mu=.1, kappa=.001)
        b = fi.simulate(*args, v0=10., mu=.1, kappa=.001, subdivisions=13)
        self.assertClose(a["final_sh_m"], b["final_sh_m"])
        self.assertClose(a["stop_time_s"], b["stop_time_s"])

    def test_inverse_terminal_forward(self):
        args = ([0., 40., 100.], [0., 0., 0.], [4., 0., 6.])
        for k in [0., .001]:
            inv = fi.inverse_threshold(*args, mu=.1, kappa=k, vreq=1.)
            v = inv["required_v0_m_s"]
            high = fi.simulate(*args, v0=v*(1+1e-6), mu=.1, kappa=k)
            low = fi.simulate(*args, v0=v*(1-1e-6), mu=.1, kappa=k)
            self.assertTrue(high["reached_target_positive"])
            self.assertGreater(high["final_velocity_m_s"], 1.)
            self.assertTrue(not low["reached_target_positive"] or low["final_velocity_m_s"] < 1.)

    def test_inverse_internal_bottleneck(self):
        args = ([0., 50., 100.], [0., 10., -100.], [4., 0., 6.])
        inv = fi.inverse_threshold(*args, mu=.1, kappa=.001)
        self.assertClose(inv["controlling_boundary_sh_m"], 50.)
        v = inv["required_v0_m_s"]
        a = fi.simulate(*args, v0=v*(1-1e-6), mu=.1, kappa=.001)
        b = fi.simulate(*args, v0=v*(1+1e-6), mu=.1, kappa=.001)
        self.assertFalse(a["reached_target_positive"])
        self.assertTrue(b["reached_target_positive"])

    def test_inverse_with_nodes_and_M2(self):
        args = ([0., 100.], [0., 0.], [4., 0., 6.])
        n = [{"id": "a", "s_h_m": 40., "rock_m3": 2., "water_m3": 3.}]
        for mode in ["M0", "M1", "M2"]:
            inv = fi.inverse_threshold(*args, nodes=n, mu=.1, kappa=.001, mode=mode)
            v = inv["required_v0_m_s"]
            r = fi.simulate(*args, nodes=n, mu=.1, kappa=.001, mode=mode, v0=v)
            self.assertClose(r["final_velocity_m_s"], 1., atol=2e-10)

    def test_timeout_unknown(self):
        r = fi.simulate([0., 100.], [0., 0.], [4., 0., 6.], v0=10,
                        mu=0, kappa=0, time_limit_s=2)
        self.assertEqual(r["status"], "time_limit_unknown")
        self.assertIsNone(r["arrival_s"])
        self.assertClose(r["final_sh_m"], 20.)
        self.assertClose(r["final_time_s"], 2.)

    def test_M2_unique_mapping(self):
        args = ([0., 100.], [20., 0.], [4., 0., 6.])
        n = [{"id": "a", "s_h_m": 50., "water_m3": 3.}]
        p = fi.prepare(*args, nodes=n, mode="M2", mu=.1)
        self.assertClose(p["segments"][0]["mu"], .1)
        self.assertClose(p["segments"][1]["mu"], .06)
        n[0]["water_m3"] = 0.
        p = fi.prepare(*args, nodes=n, mode="M2", mu=.1)
        self.assertTrue(all(s["mu"] == .1 for s in p["segments"]))

    def test_unvisited_inventory_remains(self):
        n = [{"id": "a", "s_h_m": 90., "rock_m3": 2.}]
        r = fi.simulate([0., 100.], [0., 0.], [4., 0., 6.], nodes=n,
                        v0=2, mu=.1, kappa=0)
        self.assertEqual(r["unvisited_node_ids"], ["a"])
        self.assertClose(r["remaining_inventory_mass_kg"], 5300.)
        self.assertClose(r["mass_residual_kg"], 0.)

    def test_no_zero_velocity_restart(self):
        with self.assertRaises(ValueError):
            fi.simulate([0., 100.], [100., 0.], [4., 0., 6.], v0=0)


def run_report():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DynamicsTests)
    result = unittest.TestResult()
    suite.run(result)
    return {"tests_run": result.testsRun, "passed": result.wasSuccessful(),
            "failures": [{"test": str(t), "trace": msg} for t, msg in result.failures],
            "errors": [{"test": str(t), "trace": msg} for t, msg in result.errors],
            "test_names": sorted(n for n in dir(DynamicsTests) if n.startswith("test_")),
            "randomness": "none; deterministic synthetic cases"}


if __name__ == "__main__":
    unittest.main(verbosity=2)
