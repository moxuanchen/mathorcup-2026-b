import unittest
import numpy as np
from mmlib.interval_metrics import canonical_endpoint,covered,interval_score

class IntervalRegressionTests(unittest.TestCase):
 def test_integer_boundary_roundoff_is_covered(self):
  self.assertTrue(bool(covered(13,13.000000000000002,16)))
  self.assertFalse(bool(covered(12.99,13.000000000000002,16)))
 def test_canonicalization_does_not_round_meaningful_fraction(self):
  np.testing.assert_allclose(canonical_endpoint([13+2e-15,12.3912]),[13,12.3912],rtol=0,atol=0)
 def test_interval_score_penalizes_misses_and_width(self):
  self.assertAlmostEqual(float(interval_score(14,13,16)),3)
  self.assertAlmostEqual(float(interval_score(17,13,16)),43,places=6)
 def test_labels_outside_day_remain_outside(self):
  self.assertFalse(bool(covered(-1,0,24)))
  self.assertFalse(bool(covered(25,0,24)))
if __name__=='__main__':unittest.main()
