"""Interval diagnostics with explicit numerical and label-resolution semantics."""
import numpy as np

TOLERANCE_H = 1e-9

def canonical_endpoint(value):
    a = np.asarray(value, dtype=float)
    nearest = np.rint(a)
    return np.where(np.abs(a-nearest) <= TOLERANCE_H, nearest, a)

def covered(y, lower, upper):
    y, lower, upper = map(lambda a: np.asarray(a, dtype=float), (y, lower, upper))
    return (y >= lower-TOLERANCE_H) & (y <= upper+TOLERANCE_H)

def interval_score(y, lower, upper, alpha=.05):
    y, lower, upper = map(lambda a: np.asarray(a, dtype=float), (y, lower, upper))
    left = np.maximum(lower-y-TOLERANCE_H, 0)
    right = np.maximum(y-upper-TOLERANCE_H, 0)
    return upper-lower + 2/alpha*(left+right)
