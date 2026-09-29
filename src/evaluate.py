"""Validation harness: fixed grouped folds + test-weighted MASE (the number that tracks the LB).

eda/12-13 showed per-scale-bucket error is stable between train-sim and the test period, while the
test has 4x more small pairs. So the LB-relevant score is the OOF error re-weighted to the test's
scale composition (TW-MASE), not the plain OOF mean.
"""
import numpy as np
import pandas as pd

from common import base_title

BINS = [0, 5, 20, 50, 100, 200, 500, 1e9]


def folds(X, k=5, seed=2026):
    g = base_title(X.movie_title).values
    ug = np.array(sorted(set(g)))
    f = dict(zip(ug, np.random.RandomState(seed).permutation(len(ug)) % k))
    return np.array([f[x] for x in g])


def test_weights(X, Xte):
    """Row weights that give each scale bucket its test share (mean 1)."""
    bt = pd.cut(X.scale, BINS)
    share_te = pd.cut(Xte.scale, BINS).value_counts(normalize=True)
    share_tr = bt.value_counts(normalize=True)
    w = bt.map(share_te / share_tr).astype(float).values
    return w / w.mean()


def report(X, pred, w, name=""):
    e = np.abs(X.total_ticket.values - pred) / X.scale.values
    tw = float(np.average(e, weights=w))
    small = X.scale.values <= 20
    print(f"  {name:42s} OOF {e.mean():.4f} | TW-MASE {tw:.4f} | s<=20 {e[small].mean():.4f} | s>200 {e[X.scale.values > 200].mean():.4f}")
    return tw
