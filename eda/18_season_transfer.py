"""18 - Busy -> quiet transfer inside train: the closest full-horizon analogue of train -> test.

Apr-Jul releases (busy, median occupancy 20-42%) -> Aug-Sep releases (quiet, 13-19%, like test).
1. Does the 'small means pulled' pattern weaken in the quiet months at equal absolute scale?
2. Feature variants on the busy->quiet split, scored with TW-MASE (test composition weights):
   abs  = absolute levels (log_s, f_logT, ...)       rel = market/film-relative levels only
   both = both                                        GroupKFold numbers shown for contrast
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import fig_dir, style
from evaluate import BINS, folds, test_weights
from features import dataset, feature_cols
from model import LEVEL, NEVER, PARAMS

plt = style()
F = fig_dir("18_season")
Xtr, Xte, Xlim = dataset()
quiet = (Xtr.d1 >= "2025-08-01").values
print(f"busy films {Xtr[~quiet].movie_title.nunique()} rows {(~quiet).sum()} | quiet films {Xtr[quiet].movie_title.nunique()} rows {quiet.sum()}")

z = (Xtr.total_ticket == 0)
b = pd.cut(Xtr.scale, BINS)
t = pd.DataFrame({"busy_zero": z[~quiet].groupby(b[~quiet], observed=False).mean(),
                  "quiet_zero": z[quiet].groupby(b[quiet], observed=False).mean(),
                  "busy_rows": b[~quiet].value_counts(), "quiet_rows": b[quiet].value_counts()}).sort_index()
print("=== 1. D4-D10 zero rate at equal absolute scale ===\n", t.round(3))
rq = pd.qcut(Xtr.pair_vs_mkt, 5)
print("zero rate by market-relative scale quintile:\n",
      pd.DataFrame({"busy": z[~quiet].groupby(rq[~quiet], observed=False).mean(),
                    "quiet": z[quiet].groupby(rq[quiet], observed=False).mean()}).round(3))

ABS = ["log_s", "f_logT", "base_logT", "y1", "y2", "y3", "sh1", "sh2", "sh3", "tps3", "f_per_cin", "fT1", "fT2", "fT3",
       "occ1", "occ2", "occ3", "occ_mean", "f_occ", "f_tps3", "cohort_logT", "comp_logT", "comp_maxT", "c_new_tx",
       "mk_occ", "mk_tps", "mk_logT", "mk_lpc", "cin_size"]
REL = ["pair_vs_mkt", "film_pc_vs_mkt", "logT_rel", "occ_rel", "tps_rel"]
core = [c for c in feature_cols(Xtr) if c not in NEVER + ABS + REL + LEVEL]
V = {"abs (v1 set)": core + ["log_s", "f_logT"],
     "rel": core + REL,
     "both": core + REL + ["log_s", "f_logT"]}
w_q = test_weights(Xtr[quiet], Xte)


def fit(Xa, Xb, cs):
    m = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": 500}, random_state=2026).fit(
        Xa[cs], Xa.total_ticket / Xa.scale / Xa.cal_mult, sample_weight=Xa.cal_mult)
    return np.clip(m.predict(Xb[cs]), 0, None) * Xb.cal_mult.values * Xb.scale.values


print("\n=== 2. busy -> quiet (full horizon) vs GroupKFold ===")
rows = {}
fold = folds(Xtr)
for n, cs in V.items():
    p = fit(Xtr[~quiet], Xtr[quiet], cs)
    e = np.abs(Xtr.total_ticket.values[quiet] - p) / Xtr.scale.values[quiet]
    o = np.zeros(len(Xtr))
    for k in range(5):
        o[fold == k] = fit(Xtr[fold != k], Xtr[fold == k], cs)
    eo = np.abs(Xtr.total_ticket.values - o) / Xtr.scale.values
    sm = Xtr.scale.values[quiet] <= 50
    rows[n] = dict(quiet_MASE=e.mean(), quiet_TW=np.average(e, weights=w_q), quiet_small=e[sm].mean(),
                   quiet_pred0=(p / Xtr.scale.values[quiet] < .05).mean(), quiet_true0=(Xtr.total_ticket.values[quiet] == 0).mean(),
                   gkf_TW=np.average(eo, weights=test_weights(Xtr, Xte)))
R = pd.DataFrame(rows).T
print(R.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
t[["busy_zero", "quiet_zero"]].plot.bar(ax=ax[0], rot=30)
ax[0].set(title="D4-D10 zero rate at equal absolute scale", xlabel="pair scale")
x = np.arange(len(R))
ax[1].bar(x - .2, R.gkf_TW, .4, label="GroupKFold TW"); ax[1].bar(x + .2, R.quiet_TW, .4, label="busy->quiet TW")
ax[1].set_xticks(x, R.index); ax[1].set(title="Feature variants", ylim=(0.3, None)); ax[1].legend()
fig.savefig(F / "season.png")
print(f"figures -> {F}")
