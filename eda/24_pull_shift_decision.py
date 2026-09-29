"""24 - Decision analysis for the pull-policy correction (cannot be validated on D4-D10 labels).

eda/21: in the test period the logit of P(stop selling at D3) is shifted by a = -1.27 (stable by month).
eda/22: in train, D3 drop propensity barely predicts D4-D10 pulls cross-sectionally -> transfer is unproven.
So instead of trusting it, quantify both sides on train-sim OOF rows:
  world REAL    : targets as observed (the shift does not carry to D4-D10)
  world SHIFTED : each zero target is 'un-pulled' with probability 1 - p0'/p0, where p0 is the OOF zero
                  probability and p0' = sigmoid(logit p0 + a); an un-pulled row gets a draw from its own
                  OOF positive-part distribution (quantile models of r | r > 0)
Predictors: L1 median model (current) vs hurdle median with shift fraction lam in {0, 0.25, 0.5, 1} of a.
Expected gain = MASE(world SHIFTED, L1) - MASE(world SHIFTED, hurdle lam); risk = same on world REAL.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from evaluate import folds, test_weights
from features import dataset

plt = style()
F = fig_dir("24_decision")
Xtr, Xte, Xlim = dataset()
A_SHIFT = float(pd.read_csv(OUT / "cache" / "pull_recal_ab.csv", index_col=0).iloc[0, 0])
BASE = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
        'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
        'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
        'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
        'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         verbose=-1, deterministic=True, force_row_wise=True, random_state=2026)
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
rr = lambda X: (X.total_ticket / X.scale / X.cal_mult).values

l1, p0, Q = np.zeros(len(Xtr)), np.zeros(len(Xtr)), np.zeros((len(Xtr), len(QS)))
for k in range(5):
    Xa, v = pd.concat([Xtr[fold != k], Xlim], ignore_index=True), fold == k
    l1[v] = lgb.LGBMRegressor(objective="l1", **P).fit(Xa[BASE], rr(Xa), sample_weight=Xa.cal_mult).predict(Xtr[v][BASE])
    p0[v] = lgb.LGBMClassifier(**P).fit(Xa[BASE], Xa.total_ticket == 0).predict_proba(Xtr[v][BASE])[:, 1]
    pos = Xa[Xa.total_ticket > 0]
    Q[v] = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **P).fit(pos[BASE], rr(pos))
                                    .predict(Xtr[v][BASE]) for q in QS]), axis=1)
    print(f"fold {k} done")
l1 = np.clip(l1, 0, None)
np.savez(OUT / "cache" / "hurdle_oof.npz", l1=l1, p0=p0, Q=Q)
sig = lambda z: 1 / (1 + np.exp(-z))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def hurdle(p):
    t = np.clip((0.5 - p) / (1 - p + 1e-9), QS[0], QS[-1])
    val = np.array([np.interp(ti, QS, qi) for ti, qi in zip(t, Q)])
    return np.where(p >= 0.5, 0.0, np.clip(val, 0, None))


cm, s = Xtr.cal_mult.values, Xtr.scale.values
y_real = Xtr.total_ticket.values
rng = np.random.default_rng(2026)
p_shift = sig(lg(p0) + A_SHIFT)
unpull = (y_real == 0) & (rng.random(len(Xtr)) < 1 - p_shift / np.clip(p0, 1e-6, None))
draw = np.array([np.interp(u, QS, q) for u, q in zip(rng.random(len(Xtr)), Q)])
y_shift = np.where(unpull, np.clip(draw, 0, None) * cm * s, y_real)
print(f"zero rate real {np.mean(y_real == 0):.3f} -> shifted world {np.mean(y_shift == 0):.3f} (un-pulled {unpull.mean():.3f} of rows)")

score = lambda y, pr: (np.mean(np.abs(y - pr) / s), np.average(np.abs(y - pr) / s, weights=w_te))
rows = [dict(predictor="L1 median (current)", **dict(zip(["real_MASE", "real_TW"], score(y_real, l1 * cm * s))),
             **dict(zip(["shift_MASE", "shift_TW"], score(y_shift, l1 * cm * s))))]
for lam in (0.0, 0.25, 0.5, 1.0):
    pr = hurdle(sig(lg(p0) + lam * A_SHIFT)) * cm * s
    rows.append(dict(predictor=f"hurdle, shift x{lam}", **dict(zip(["real_MASE", "real_TW"], score(y_real, pr))),
                     **dict(zip(["shift_MASE", "shift_TW"], score(y_shift, pr)))))
R = pd.DataFrame(rows).set_index("predictor")
R["gain_if_shift_real"] = R.loc["L1 median (current)", "shift_TW"] - R.shift_TW
R["loss_if_not"] = R.real_TW - R.loc["L1 median (current)", "real_TW"]
print(R.round(4))

fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(R))
ax.bar(x - .2, R.real_TW, .4, label="world REAL (no transfer)"); ax.bar(x + .2, R.shift_TW, .4, label="world SHIFTED (a carries)")
ax.set_xticks(x, R.index, rotation=20); ax.set(title="TW-MASE under both worlds", ylim=(R[["real_TW", "shift_TW"]].min().min() - .02, None)); ax.legend()
fig.savefig(F / "decision.png")
print(f"figures -> {F}")
