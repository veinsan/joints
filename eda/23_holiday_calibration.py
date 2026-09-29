"""23 - Is the structural calendar under-stating holidays? Calibrate it where train CAN validate it.

The Lebaran week (6% of test rows, D1-D3 in the last Ramadan days, D4-D10 in the biggest cinema week of the
year) cannot be validated. What can be validated: single holidays, cuti bersama and the Jun-Jul school break
that exist in train. If OOF predictions are biased low on holiday target rows, the calendar is too timid and
the Lebaran week (which is built from the same parameters) inherits it.

1. OOF signed error on target rows by calendar type (holiday weekday, school weekday, normal weekday, weekend).
2. Grid over HOL_LEVEL (level of a weekday holiday) x SCHOOL_WD (school-break weekday uplift); each point
   rebuilds cal_mult, refits LightGBM with the same folds, and scores TW-MASE + holiday-row MASE and bias.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
import features as FT
from common import fig_dir, load, style
from evaluate import folds, test_weights
from features import dataset
from model import PARAMS

plt = style()
F = fig_dir("23_holiday")
d = load()
Xtr, Xte, Xlim = dataset()
BASE = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
        'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
        'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
        'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
        'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)


def with_cal(X, hol_level, school_wd):
    FT.HOL_LEVEL, FT.SCHOOL_WEEKDAY = hol_level, school_wd
    cal = FT.calendar(d["hol"])
    X = X.copy()
    ch = np.stack([cal.cal.reindex(X.d1 + pd.Timedelta(days=k)).values for k in range(3)], 1).mean(1)
    X["cal_mult"] = cal.cal.reindex(X.date_show).values / ch
    return X


def run(hol_level, school_wd):
    X, L = with_cal(Xtr, hol_level, school_wd), with_cal(Xlim, hol_level, school_wd)
    o = np.zeros(len(X))
    for k in range(5):
        Xa = pd.concat([X[fold != k], L], ignore_index=True)
        m = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": 500}, random_state=2026).fit(
            Xa[BASE], Xa.total_ticket / Xa.scale / Xa.cal_mult, sample_weight=Xa.cal_mult)
        v = fold == k
        o[v] = np.clip(m.predict(X[v][BASE]), 0, None) * X.cal_mult.values[v] * X.scale.values[v]
    return o


kind = np.select([(Xtr.t_hol == 1) & (Xtr.dow < 5), (Xtr.t_school == 1) & (Xtr.dow < 4), Xtr.dow >= 5],
                 ["holiday weekday", "school-break weekday", "weekend"], "normal weekday")
y, s = Xtr.total_ticket.values, Xtr.scale.values
rows, keep = [], {}
for hl in (1.29, 1.45, 1.6, 1.8):
    for sw in (1.0, 1.15, 1.3):
        o = run(hl, sw)
        e, bias = np.abs(y - o) / s, (o - y) / s
        hol = kind == "holiday weekday"
        rows.append(dict(HOL_LEVEL=hl, SCHOOL_WD=sw, TW=np.average(e, weights=w_te), MASE=e.mean(),
                         hol_MASE=e[hol].mean(), hol_bias=np.median(bias[hol]),
                         school_MASE=e[kind == "school-break weekday"].mean(), school_bias=np.median(bias[kind == "school-break weekday"])))
        keep[(hl, sw)] = o
        print({k: round(v, 4) for k, v in rows[-1].items()})
R = pd.DataFrame(rows)
print("\n", R.round(4).to_string(index=False))
best = R.sort_values("TW").iloc[0]
print(f"\nbest by TW-MASE: HOL_LEVEL={best.HOL_LEVEL}, SCHOOL_WD={best.SCHOOL_WD}")
o0 = keep[(1.29, 1.15)]
print("current calendar (1.29, 1.15): signed error by target-day kind (median (pred-y)/s, MASE, rows):")
print(pd.DataFrame({"bias": pd.Series((o0 - y) / s).groupby(kind).median(), "MASE": pd.Series(np.abs(y - o0) / s).groupby(kind).mean(),
                    "rows": pd.Series(kind).value_counts()}).round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for sw, g in R.groupby("SCHOOL_WD"):
    ax[0].plot(g.HOL_LEVEL, g.TW, marker="o", label=f"SCHOOL_WD={sw}")
ax[0].set(title="TW-MASE vs holiday level", xlabel="HOL_LEVEL"); ax[0].legend()
for sw, g in R.groupby("SCHOOL_WD"):
    ax[1].plot(g.HOL_LEVEL, g.hol_bias, marker="o", label=f"SCHOOL_WD={sw}")
ax[1].axhline(0, color="k", lw=.6); ax[1].set(title="Median signed error on holiday-weekday targets", xlabel="HOL_LEVEL"); ax[1].legend()
fig.savefig(F / "holiday.png")
print(f"figures -> {F}")
