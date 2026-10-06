"""48 - Seed bagging of the hurdle part (classifier + 19 quantile models): 1 seed (v6) vs 3 seeds.

The L1 model already averages 5 seeds; the hurdle (weight 0.75 in the LightGBM mix) uses one seed, although
bagging_fraction 0.8 / feature_fraction 0.7 make it seed-dependent. Same params, folds and training rows as
the notebook; scored with composition-correct weights (bucket x first day) in kappa worlds.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
from common import OUT, fig_dir, style
from evaluate import folds, kappa_worlds, test_weights_fd
from features import GENRES, dataset

plt = style()
F = fig_dir("48_seed_bagging")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
w = test_weights_fd(Xtr, Xte)
z0, z6 = np.load(C / "hurdle_oof.npz"), np.load(C / "lgb_parts_v6.npz")
W = kappa_worlds(y, s, cm, z0["p0"], z0["Q"])
FEATS = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT', 'f_nc_trend', 'fmt',
         'fmt_share', 'd1_dow', 'rating'] + [f'g_{g}' for g in GENRES] + [
         'n_genre', 'n_cast', 'comp_n', 'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist',
         'n_comp_open', 'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
ZFEATS = FEATS + ['c_new_n', 'c_new_sh', 'c_new_sh_rel', 'c_new_sh_vs_own', 'c_new_tx_vs_own']
FAST = dict(learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1,
            lambda_l2=1.0, n_estimators=300, deterministic=True, force_row_wise=True, n_jobs=8, verbose=-1)
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
SEEDS = [2026, 2027, 2028]
fold = folds(Xtr)
rt = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))

P0 = np.zeros((len(SEEDS), len(y)))
QQ = np.zeros((len(SEEDS), len(y), len(QS)))
t0 = time.time()
for k in range(5):
    Xa = __import__("pandas").concat([Xtr[fold != k], Xlim], ignore_index=True)
    Xb, v = Xtr[fold == k], fold == k
    pos = Xa[Xa.total_ticket > 0]
    for i, sd in enumerate(SEEDS):
        P0[i, v] = lgb.LGBMClassifier(objective="binary", **{**FAST, "num_leaves": 31}, random_state=sd).fit(Xa[ZFEATS], Xa.total_ticket == 0).predict_proba(Xb[ZFEATS])[:, 1]
        QQ[i, v] = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=float(q), **FAST, random_state=sd)
                                            .fit(pos[FEATS], rt(pos)).predict(Xb[FEATS]) for q in QS]), axis=1)
    print(f"fold {k} done ({time.time() - t0:.0f}s)", flush=True)


def mixmed(p, Qm, lam=0.5):
    pp = sig(lg(p) - 1.32 * lam)
    t = np.clip((0.5 - pp) / (1 - pp + 1e-9), QS[0], QS[-1])
    return np.where(pp >= 0.5, 0.0, np.clip([np.interp(ti, QS, qi) for ti, qi in zip(t, Qm)], 0, None))


sc = lambda p: (np.average(np.abs(y - p) / s, weights=w), np.mean([np.average(np.abs(t - p) / s, weights=w) for t in W]))
for name, p0, Q in [("seed 2026 only", P0[0], QQ[0]), ("seed 2027 only", P0[1], QQ[1]), ("3-seed average", P0.mean(0), QQ.mean(0))]:
    mix = (0.25 * z6["l1"] + 0.75 * mixmed(p0, Q)) * cm * s
    print(f"{name:16s} LightGBM mix TW-fd real {sc(mix)[0]:.4f} | kappa {sc(mix)[1]:.4f}")
