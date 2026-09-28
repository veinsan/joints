"""LightGBM on the MASE-normalised target, shared by local CV and the notebook."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from common import base_title

LEVEL = ["y1", "y2", "y3", "sh1", "sh2", "sh3", "occ1", "occ2", "occ3", "occ_mean", "tps3", "f_occ", "f_tps3",
         "f_per_cin", "fT1", "fT2", "fT3", "f_logT", "base_logT", "log_s", "cohort_logT", "comp_logT", "comp_maxT",
         "mean3"]
NEVER = ["d1_month"]  # months of train (Apr-Sep) and test (Oct-Mar) never overlap

PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=1500, verbose=-1)


def target(X, use_cal=True):
    """r = y / s (/ cal_mult). MASE == mean |r - r_hat| (x cal_mult weight), so L1 on r matches the metric."""
    t = X.total_ticket / X.scale
    return t / X.cal_mult if use_cal else t


def fit_predict(Xa, Xb, cols, use_cal=True, seeds=(2026,), params=None, n_estimators=None):
    p = {**PARAMS, **(params or {})}
    if n_estimators:
        p["n_estimators"] = n_estimators
    ya = target(Xa, use_cal)
    # with cal-normalised target each row's MASE weight is cal_mult -> pass it as sample weight
    w = Xa.cal_mult.values if use_cal else None
    out = np.zeros(len(Xb))
    for s in seeds:
        m = lgb.LGBMRegressor(**p, random_state=s).fit(Xa[cols], ya, sample_weight=w)
        out += m.predict(Xb[cols]) / len(seeds)
    out = np.clip(out, 0, None) * Xb.scale.values
    return out * Xb.cal_mult.values if use_cal else out, m


def groups(X):
    return base_title(X.movie_title)
