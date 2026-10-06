"""34 - Calibrate the pull classifier before the mixture median.

The mixture median flips from 0 to positive at p0 = 0.5, so p0 must be CALIBRATED, not just ranked well.
eda/21 found the D3 classifier over-confident even on train OOF (logit slope 0.57), and eda/33 shows the
test-period shift concentrates on high-p0 rows. Here, on the MAIN task (D4-D10):
  1. reliability of the OOF p0 (predicted vs observed zero rate by bin) and the Platt fit (a, b)
  2. cross-fitted Platt recalibration (fit on 4 folds' OOF, apply to the 5th) -> hurdle TW-MASE real & kappa
  3. on the test-period proxy: after calibrating the train model, what shift remains (intercept vs slope)?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
from common import OUT, base_title, fig_dir, style
from evaluate import folds, test_weights
from features import dataset

plt = style()
F = fig_dir("34_calibration")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
z6 = np.load(C / "lgb_parts_v6.npz")
l1, p0, Q = z6["l1"], z6["p0"], z6["Q"]
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
A_SHIFT, LAM, HW = -1.32, 0.5, 0.75
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
zt = (y == 0).astype(float)
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def platt(p, zz):
    nll = lambda ab: -np.mean(zz * np.log(sig(ab[0] + ab[1] * lg(p)) + 1e-9) + (1 - zz) * np.log(1 - sig(ab[0] + ab[1] * lg(p)) + 1e-9))
    return minimize(nll, [0.0, 1.0], method="Nelder-Mead").x


print("=== 1. reliability of OOF p0 (main task) ===")
bins = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]
cb = pd.cut(p0, bins)
rel = pd.DataFrame({"pred": pd.Series(p0).groupby(cb, observed=True).mean(), "obs": pd.Series(zt).groupby(cb, observed=True).mean(),
                    "n": pd.Series(zt).groupby(cb, observed=True).size()})
print(rel.round(3))
ab = platt(p0, zt)
print(f"Platt on all OOF: a={ab[0]:.3f}, b={ab[1]:.3f}")

p_cal = np.zeros(len(p0))
for k in range(5):
    abk = platt(p0[fold != k], zt[fold != k])
    p_cal[fold == k] = sig(abk[0] + abk[1] * lg(p0[fold == k]))
rel["cal_pred"] = pd.Series(p_cal).groupby(cb, observed=True).mean()

worlds = []
z0 = np.load(C / "hurdle_oof.npz")
for seed in range(3):
    rng = np.random.default_rng(seed)
    ps = sig(lg(z0["p0"]) + 0.5 * A_SHIFT)
    un = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(z0["p0"], 1e-6, None))
    dr = np.array([np.interp(u, QS, q) for u, q in zip(rng.random(len(y)), z0["Q"])])
    worlds.append(np.where(un, np.clip(dr, 0, None) * cm * s, y))


def score(pp, lam=LAM):
    p = sig(lg(pp) + lam * A_SHIFT)
    t = np.clip((0.5 - p) / (1 - p + 1e-9), QS[0], QS[-1])
    hu = np.where(p >= .5, 0, np.clip([np.interp(ti, QS, qi) for ti, qi in zip(t, Q)], 0, None))
    pred = ((1 - HW) * l1 + HW * hu) * cm * s
    return (float(np.average(np.abs(y - pred) / s, weights=w_te)),
            float(np.mean([np.average(np.abs(w - pred) / s, weights=w_te) for w in worlds])))


print("\n=== 2. LightGBM mix with raw vs calibrated p0 (TW real, TW kappa0.5) ===")
res = {}
for lam in (0.0, 0.5):
    res[f"raw p0, lambda {lam}"] = score(p0, lam)
    res[f"Platt p0, lambda {lam}"] = score(p_cal, lam)
R = pd.DataFrame(res, index=["TW_real", "TW_kappa0.5"]).T
print(R.round(4))

print("\n=== 3. test-period proxy: shift that remains after calibrating the train model ===")
A = pd.read_parquet(C / "proxy_A.parquet").reset_index(drop=True)
B = pd.read_parquet(C / "proxy_B.parquet").reset_index(drop=True)
pc = ["q1", "q2", "log_s", "sh1", "sh2", "sh_tr", "occ2", "f_tr", "f_log", "f_nc", "cm", "d1_dow"]
PP = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, verbose=-1, deterministic=True,
          force_row_wise=True, random_state=2026)
pa = np.zeros(len(A))
for a_, b_ in GroupKFold(5).split(A, groups=base_title(A.movie_title)):
    pa[b_] = lgb.LGBMClassifier(**PP).fit(A.iloc[a_][pc], A.y.iloc[a_] == 0).predict_proba(A.iloc[b_][pc])[:, 1]
abA = platt(pa, (A.y == 0).values.astype(float))
pb = lgb.LGBMClassifier(**PP).fit(A[pc], A.y == 0).predict_proba(B[pc])[:, 1]
pb_cal = sig(abA[0] + abA[1] * lg(pb))
abB = platt(pb_cal, (B.y == 0).values.astype(float))
aB_only = minimize(lambda v: -np.mean((B.y == 0) * np.log(sig(lg(pb_cal) + v[0]) + 1e-9) + (B.y != 0) * np.log(1 - sig(lg(pb_cal) + v[0]) + 1e-9)),
                   [0.0], method="Nelder-Mead").x[0]
print(f"proxy train-OOF Platt (a, b) = ({abA[0]:.3f}, {abA[1]:.3f})")
print(f"test-period recalibration of the CALIBRATED model: (a, b) = ({abB[0]:.3f}, {abB[1]:.3f}) | intercept-only a = {aB_only:.3f}")
cbB = pd.cut(pb_cal, bins)
print(pd.DataFrame({"pred_cal": pd.Series(pb_cal).groupby(cbB, observed=True).mean(),
                    "obs_test": (B.y == 0).groupby(cbB, observed=True).mean(), "n": pd.Series(pb_cal).groupby(cbB, observed=True).size()}).round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(rel.pred, rel.obs, marker="o", label="raw OOF p0"); ax[0].plot(rel.cal_pred, rel.obs, marker="s", label="Platt (cross-fitted)")
ax[0].plot([0, 1], [0, 1], color="k", lw=.6); ax[0].set(title="Reliability of P(zero), main task", xlabel="predicted", ylabel="observed"); ax[0].legend()
ax[1].barh(R.index, R["TW_kappa0.5"], color="#2a78d6"); ax[1].set(title="TW-MASE kappa=0.5", xlim=(R["TW_kappa0.5"].min() - .004, R["TW_kappa0.5"].max() + .002))
fig.savefig(F / "calibration.png")
np.save(C / "p0_platt.npy", p_cal)
print(f"figures -> {F}")
