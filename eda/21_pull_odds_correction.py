"""21 - Estimate the pull-odds shift in the test period and validate a correction ON TEST-PERIOD LABELS.

Hurdle view of the target r = y/s:  P(r = 0) = p  (pair pulled / no sale)  and  r | r > 0 ~ F.
The MASE-optimal prediction is the median of the mixture:
    p >= 0.5 -> 0,   else  F^-1((0.5 - p) / (1 - p)).
If cinemas in the test period pull less, the TRAIN-fitted p is too high. A logistic recalibration
    logit p_test = a + b * logit p_train_model
fitted on labelled test-period rows (proxy D1,D2 -> D3) measures the shift (a < 0 = fewer pulls).

Validation (no leaderboard): split test-period FILMS in two halves; fit (a, b) on one half, score MASE on
the other half; swap; average. Compared against (i) plain L1 model, (ii) hurdle without correction.
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

plt = style()
F = fig_dir("21_pull_odds")
A = pd.read_parquet(OUT / "cache" / "proxy_A.parquet").reset_index(drop=True)
B = pd.read_parquet(OUT / "cache" / "proxy_B.parquet").reset_index(drop=True)
cols = ["q1", "q2", "log_s", "sh1", "sh2", "sh_tr", "occ2", "f_tr", "f_log", "f_nc", "cm", "d1_dow"]
QS = np.round(np.arange(0.02, 1.0, 0.02), 2)
base = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, verbose=-1,
            deterministic=True, force_row_wise=True, random_state=2026)
r = lambda X: X.y / X.s / X.cm


def fit_parts(Xa):
    """Zero classifier, L1 median model, and quantile models of r | r > 0 (all on r/cm)."""
    z = lgb.LGBMClassifier(**base).fit(Xa[cols], (Xa.y == 0).astype(int))
    l1 = lgb.LGBMRegressor(objective="l1", **base).fit(Xa[cols], r(Xa), sample_weight=Xa.cm)
    pos = Xa[Xa.y > 0]
    qm = {q: lgb.LGBMRegressor(objective="quantile", alpha=q, **base).fit(pos[cols], r(pos)) for q in QS}
    return z, l1, qm


def predict_parts(models, Xb):
    z, l1, qm = models
    p0 = z.predict_proba(Xb[cols])[:, 1]
    Q = np.sort(np.column_stack([qm[q].predict(Xb[cols]) for q in QS]), axis=1)
    med = np.clip(l1.predict(Xb[cols]), 0, None)
    return p0, Q, med


def hurdle_median(p0, Q):
    t = np.clip((0.5 - p0) / (1 - p0 + 1e-9), QS[0], QS[-1])
    val = np.array([np.interp(ti, QS, qi) for ti, qi in zip(t, Q)])
    return np.where(p0 >= 0.5, 0.0, np.clip(val, 0, None))


def recal(p, a, b):
    lg = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    return 1 / (1 + np.exp(-(a + b * lg)))


def fit_ab(p, z):
    nll = lambda ab: -np.mean(z * np.log(recal(p, *ab) + 1e-9) + (1 - z) * np.log(1 - recal(p, *ab) + 1e-9))
    return minimize(nll, [0.0, 1.0], method="Nelder-Mead").x


mase = lambda X, pr: float(np.mean(np.abs(X.y.values - pr * X.cm.values * X.s.values) / X.s.values))

# train-sim OOF (calibration check: a ~ 0, b ~ 1 expected)
oof_p = np.zeros(len(A))
for a, b in GroupKFold(5).split(A, groups=base_title(A.movie_title)):
    oof_p[b] = fit_parts(A.iloc[a])[0].predict_proba(A.iloc[b][cols])[:, 1]
print("train-sim OOF recalibration (a, b):", np.round(fit_ab(oof_p, (A.y == 0).values), 3))

models = fit_parts(A)
p0, Q, med = predict_parts(models, B)
zB = (B.y == 0).values
ab_all = fit_ab(p0, zB)
print("test-period recalibration (a, b):  ", np.round(ab_all, 3),
      f"| mean predicted p0 {p0.mean():.3f} -> recalibrated {recal(p0, *ab_all).mean():.3f} | true {zB.mean():.3f}")
by_m = {m: np.round(fit_ab(p0[B.month == m], zB[B.month == m]), 2) for m in sorted(B.month.unique())}
print("per test month (a, b):", by_m)

print("\n=== split-half validation on test-period films (fit on one half, score the other) ===")
films = np.array(sorted(base_title(B.movie_title).unique()))
res = []
for seed in range(5):
    rng = np.random.RandomState(seed)
    half = set(rng.permutation(films)[: len(films) // 2])
    h1 = base_title(B.movie_title).isin(half).values
    for fit_m, ev_m in [(h1, ~h1), (~h1, h1)]:
        ab = fit_ab(p0[fit_m], zB[fit_m])
        Be = B[ev_m]
        res.append(dict(seed=seed, a=ab[0], b=ab[1],
                        l1=mase(Be, med[ev_m]),
                        hurdle=mase(Be, hurdle_median(p0[ev_m], Q[ev_m])),
                        hurdle_recal=mase(Be, hurdle_median(recal(p0[ev_m], *ab), Q[ev_m]))))
R = pd.DataFrame(res)
print(R.round(4).to_string(index=False))
print("mean:", R[["l1", "hurdle", "hurdle_recal"]].mean().round(4).to_dict(), "| sd a:", round(R.a.std(), 3))

print("\n=== same models on train-sim OOF (sanity: hurdle must not be worse than L1 there) ===")
oof_med, oof_h = np.zeros(len(A)), np.zeros(len(A))
for a, b in GroupKFold(5).split(A, groups=base_title(A.movie_title)):
    pp, QQ, mm = predict_parts(fit_parts(A.iloc[a]), A.iloc[b])
    oof_med[b], oof_h[b] = mm, hurdle_median(pp, QQ)
print(f"train-sim OOF MASE: L1 {mase(A, oof_med):.4f} | hurdle {mase(A, oof_h):.4f}")
pd.Series({"a": ab_all[0], "b": ab_all[1]}).to_csv(OUT / "cache" / "pull_recal_ab.csv")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
bins = np.linspace(0, 1, 11)
cb = pd.cut(p0, bins)
cal = pd.DataFrame({"pred": pd.Series(p0).groupby(cb, observed=True).mean(), "true": pd.Series(zB).groupby(cb, observed=True).mean(),
                    "recal": pd.Series(recal(p0, *ab_all)).groupby(cb, observed=True).mean()})
ax[0].plot(cal.pred, cal.true, marker="o", label="train model on test period")
ax[0].plot(cal.recal, cal.true, marker="s", label="after recalibration")
ax[0].plot([0, 1], [0, 1], color="k", lw=.6); ax[0].set(title="P(D3 = 0): predicted vs observed (test period)", xlabel="predicted", ylabel="observed"); ax[0].legend()
m = R[["l1", "hurdle", "hurdle_recal"]].mean()
ax[1].bar(m.index, m.values, color=["#2a78d6", "#eda100", "#1baf7a"]); ax[1].set(title="Held-out test-period MASE (split-half)", ylim=(m.min() - .02, m.max() + .01))
fig.savefig(F / "pull_odds.png")
print(f"figures -> {F}")
