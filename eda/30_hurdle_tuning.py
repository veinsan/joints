"""30 - Tune the LightGBM hurdle on the LB-calibrated metric (TW-MASE in the kappa=0.5 world).

v5 showed this metric now tracks the leaderboard (0.4066 vs LB 0.4087), so it is the decision metric.
Two independent searches (each reuses the other component from the cached OOF so a grid point is cheap):
  A. pull classifier P(zero): leaves x trees x min_child_samples
  B. positive-part quantile models: leaves x trees x min_child_samples, and grid density (19 vs 39 levels)
Every point is scored with the same simulated shifted world (3 seeds) and lambda = 0.5.
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
F = fig_dir("30_tuning")
Xtr, Xte, Xlim = dataset()
BASE = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
        'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
        'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
        'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
        'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
ZC = BASE + ["c_new_n", "c_new_sh", "c_new_sh_rel", "c_new_sh_vs_own", "c_new_tx_vs_own"]
A_SHIFT = -1.32
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
rr = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
z0 = np.load(OUT / "cache" / "hurdle_oof.npz")
QS0 = np.round(np.arange(0.05, 1.0, 0.05), 2)
worlds = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    ps = sig(lg(z0["p0"]) + 0.5 * A_SHIFT)
    un = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(z0["p0"], 1e-6, None))
    dr = np.array([np.interp(u, QS0, q) for u, q in zip(rng.random(len(y)), z0["Q"])])
    worlds.append(np.where(un, np.clip(dr, 0, None) * cm * s, y))


def score(p0, Q, qs, lam=0.5):
    p = sig(lg(p0) + lam * A_SHIFT)
    t = np.clip((0.5 - p) / (1 - p + 1e-9), qs[0], qs[-1])
    pred = np.where(p >= .5, 0, np.clip([np.interp(ti, qs, qi) for ti, qi in zip(t, Q)], 0, None)) * cm * s
    return float(np.average(np.abs(y - pred) / s, weights=w_te)), float(np.mean([np.average(np.abs(w - pred) / s, weights=w_te) for w in worlds]))


def base_params(**kw):
    return dict(learning_rate=0.05, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                deterministic=True, force_row_wise=True, random_state=2026, **kw)


# A. classifier grid (quantiles fixed = cached)
p0_ref = None
rowsA = []
for nl, ne, mc in [(63, 300, 100), (31, 300, 100), (127, 300, 100), (63, 600, 100), (63, 300, 300), (63, 600, 300), (31, 800, 200)]:
    p0 = np.zeros(len(Xtr))
    for k in range(5):
        Xa = pd.concat([Xtr[fold != k], Xlim], ignore_index=True)
        p0[fold == k] = lgb.LGBMClassifier(**base_params(num_leaves=nl, n_estimators=ne, min_child_samples=mc)).fit(
            Xa[ZC], Xa.total_ticket == 0).predict_proba(Xtr[fold == k][ZC])[:, 1]
    real, kap = score(p0, z0["Q"], QS0)
    rowsA.append(dict(leaves=nl, trees=ne, min_child=mc, real=real, kappa05=kap))
    print("A", rowsA[-1])
    if p0_ref is None or kap < min(r["kappa05"] for r in rowsA[:-1]):
        p0_ref = p0
RA = pd.DataFrame(rowsA).sort_values("kappa05")
print("\nA. classifier grid:\n", RA.round(4).to_string(index=False))

# B. quantile grid (classifier fixed = best of A)
rowsB = []
for nl, ne, mc, step in [(63, 300, 100, .05), (31, 400, 100, .05), (127, 300, 200, .05), (63, 600, 200, .05), (63, 300, 100, .025)]:
    qs = np.round(np.arange(step, 1.0, step), 3)
    Q = np.zeros((len(Xtr), len(qs)))
    for k in range(5):
        Xa = pd.concat([Xtr[fold != k], Xlim], ignore_index=True)
        pos = Xa[Xa.total_ticket > 0]
        Q[fold == k] = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=float(q), **base_params(
            num_leaves=nl, n_estimators=ne, min_child_samples=mc)).fit(pos[BASE], rr(pos)).predict(Xtr[fold == k][BASE]) for q in qs]), axis=1)
    real, kap = score(p0_ref, Q, qs)
    rowsB.append(dict(leaves=nl, trees=ne, min_child=mc, levels=len(qs), real=real, kappa05=kap))
    print("B", rowsB[-1])
RB = pd.DataFrame(rowsB).sort_values("kappa05")
print("\nB. quantile grid (best classifier):\n", RB.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
la = [f"{r.leaves}/{r.trees}/{r.min_child}" for r in RA.itertuples()]
ax[0].barh(la, RA.kappa05, color="#2a78d6"); ax[0].set(title="Classifier grid: TW-MASE kappa=0.5 (leaves/trees/min_child)", xlim=(RA.kappa05.min() - .003, RA.kappa05.max() + .001))
lb_ = [f"{r.leaves}/{r.trees}/{r.min_child}/{r.levels}q" for r in RB.itertuples()]
ax[1].barh(lb_, RB.kappa05, color="#1baf7a"); ax[1].set(title="Quantile grid: TW-MASE kappa=0.5", xlim=(RB.kappa05.min() - .003, RB.kappa05.max() + .001))
fig.savefig(F / "tuning.png")
print(f"figures -> {F}")
