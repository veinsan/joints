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


def test_weights_fd(X, Xte):
    """Like test_weights but matches the test share of (scale bucket x first sale day), eda/42."""
    k = lambda D: pd.cut(D.scale, BINS).astype(str) + "|" + D.first_day.astype(str)
    share_te, share_tr = k(Xte).value_counts(normalize=True), k(X).value_counts(normalize=True)
    w = k(X).map(share_te / share_tr).fillna(0).astype(float).values
    return w / w.mean()


def kappa_worlds(y, s, cm, p0, Q, a=-1.32, kappa=0.5, seeds=3, qs=np.round(np.arange(0.05, 1.0, 0.05), 2)):
    """LB-calibrated validation worlds (eda/26): zero targets are un-pulled with prob 1 - p0'/p0,
    p0' = sigmoid(logit p0 + kappa * a), and replaced by a draw from the OOF positive-part quantiles."""
    lg = np.log(np.clip(p0, 1e-4, 1 - 1e-4) / (1 - np.clip(p0, 1e-4, 1 - 1e-4)))
    ps = 1 / (1 + np.exp(-(lg + kappa * a)))
    out = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        un = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(p0, 1e-6, None))
        dr = np.array([np.interp(u, qs, q) for u, q in zip(rng.random(len(y)), Q)])
        out.append(np.where(un, np.clip(dr, 0, None) * cm * s, y))
    return out
