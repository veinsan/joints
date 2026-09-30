"""27 - Feature groups for the PULL classifier P(y = 0) specifically.

eda/16 judged feature groups on the L1 model (all within noise). With the hurdle, 'is the film pulled?'
is a separate model, and the visible schedule is its natural driver: new titles opening at the same
cluster on the target date take screens. Here each group is added to the zero classifier only; the
positive-part quantiles are reused from the cached OOF (eda/24), so only the classifier changes.
Scores: AUC / log-loss of P(zero), and the hurdle TW-MASE in the real world and in the world where half
of the D3 pull-odds shift applies (kappa = 0.5, the level consistent with the v3 -> v4 leaderboard change).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from common import OUT, fig_dir, style
from evaluate import folds, test_weights
from features import dataset

plt = style()
F = fig_dir("27_zero_clf")
Xtr, Xte, Xlim = dataset()
z = np.load(OUT / "cache" / "hurdle_oof.npz")
p0_base, Q = z["p0"], z["Q"]
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
A_SHIFT, KAPPA = -1.32, 0.5
BASE = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
        'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
        'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
        'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
        'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
GROUPS = {
    "cinema competition": ["c_new_n", "c_new_sh", "c_new_sh_rel", "c_new_sh_vs_own", "c_new_tx_vs_own"],
    "programme change": ["n_wed_passed", "n_thu_passed"],
    "first day / shows": ["first_day", "sh_r31", "f_sh_r31", "sh3", "tps3"],
    "market relative": ["pair_vs_mkt", "film_pc_vs_mkt", "occ_rel", "tps_rel", "logT_rel"],
}
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         verbose=-1, deterministic=True, force_row_wise=True, random_state=2026)
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
zt = (y == 0).astype(int)
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
rng = np.random.default_rng(2026)
ps = sig(lg(p0_base) + KAPPA * A_SHIFT)
unpull = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(p0_base, 1e-6, None))
draw = np.array([np.interp(u, QS, q) for u, q in zip(rng.random(len(y)), Q)])
y_k = np.where(unpull, np.clip(draw, 0, None) * cm * s, y)


def hurdle(p, lam):
    pp = sig(lg(p) + lam * A_SHIFT)
    t = np.clip((0.5 - pp) / (1 - pp + 1e-9), QS[0], QS[-1])
    return np.where(pp >= 0.5, 0.0, np.clip([np.interp(ti, QS, qi) for ti, qi in zip(t, Q)], 0, None)) * cm * s


def clf_oof(cols):
    p = np.zeros(len(Xtr))
    for k in range(5):
        Xa = pd.concat([Xtr[fold != k], Xlim], ignore_index=True)
        p[fold == k] = lgb.LGBMClassifier(**P).fit(Xa[cols], Xa.total_ticket == 0).predict_proba(Xtr[fold == k][cols])[:, 1]
    return p


tw = lambda yy, pr: float(np.average(np.abs(yy - pr) / s, weights=w_te))
rows, keep = [], {}
for name, extra in [("base", [])] + list(GROUPS.items()) + [("all groups", sum(GROUPS.values(), []))]:
    p = clf_oof(BASE + extra)
    keep[name] = p
    rows.append(dict(set=name, auc=roc_auc_score(zt, p), logloss=log_loss(zt, p),
                     TW_real_lam0=tw(y, hurdle(p, 0.0)), TW_real_lam05=tw(y, hurdle(p, 0.5)),
                     TW_kappa_lam05=tw(y_k, hurdle(p, 0.5))))
    print({k: (round(v, 4) if not isinstance(v, str) else v) for k, v in rows[-1].items()})
R = pd.DataFrame(rows).set_index("set")
print("\n", R.round(4))
print("\ndelta vs base:\n", R.sub(R.loc["base"]).round(4))
np.save(OUT / "cache" / "p0_allgroups.npy", keep["all groups"])

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
d = R.sub(R.loc["base"]).drop("base")
ax[0].barh(d.index, d.auc, color="#2a78d6"); ax[0].set(title="AUC gain of the pull classifier vs base")
ax[1].barh(d.index, d.TW_kappa_lam05, color=["#1baf7a" if v < 0 else "#e34948" for v in d.TW_kappa_lam05])
ax[1].axvline(0, color="k", lw=.6); ax[1].set(title="Hurdle TW-MASE change (world kappa=0.5, lambda=0.5)")
fig.savefig(F / "zero_clf.png")
print(f"figures -> {F}")
