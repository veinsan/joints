"""25 - Choose the shift fraction lambda with TEST-PERIOD labels (proxy D1,D2 -> D3), split-half by film.

eda/24: hurdle beats L1 by 0.021 TW-MASE on real train validation; the pull shift lambda is a trade-off
(lambda=0 best if the D3 shift does not carry, lambda=1 best if it does; minimax regret -> 0.5).
eda/21: on the test-period proxy the un-recalibrated hurdle LOSES to L1 (0.482 vs 0.473), recalibrated ~ L1.
Here: lambda curve on the test-period proxy, with 'a' fitted on one half of the test films and scored on the
other half (5 random splits x 2 directions), next to the same curve on train-sim OOF.
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
F = fig_dir("25_lambda")
A = pd.read_parquet(OUT / "cache" / "proxy_A.parquet").reset_index(drop=True)
B = pd.read_parquet(OUT / "cache" / "proxy_B.parquet").reset_index(drop=True)
cols = ["q1", "q2", "log_s", "sh1", "sh2", "sh_tr", "occ2", "f_tr", "f_log", "f_nc", "cm", "d1_dow"]
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, verbose=-1,
         deterministic=True, force_row_wise=True, random_state=2026)
rr = lambda X: (X.y / X.s / X.cm).values
sig = lambda z: 1 / (1 + np.exp(-z))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
LAMS = [0, 0.25, 0.5, 0.75, 1.0]


def parts(Xa, Xb):
    l1 = np.clip(lgb.LGBMRegressor(objective="l1", **P).fit(Xa[cols], rr(Xa), sample_weight=Xa.cm).predict(Xb[cols]), 0, None)
    p0 = lgb.LGBMClassifier(**P).fit(Xa[cols], Xa.y == 0).predict_proba(Xb[cols])[:, 1]
    pos = Xa[Xa.y > 0]
    Q = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **P).fit(pos[cols], rr(pos)).predict(Xb[cols])
                                 for q in QS]), axis=1)
    return l1, p0, Q


def hurdle(p, Q):
    t = np.clip((0.5 - p) / (1 - p + 1e-9), QS[0], QS[-1])
    return np.where(p >= 0.5, 0.0, np.clip([np.interp(ti, QS, qi) for ti, qi in zip(t, Q)], 0, None))


def fit_a(p, z):
    nll = lambda a: -np.mean(z * np.log(sig(lg(p) + a) + 1e-9) + (1 - z) * np.log(1 - sig(lg(p) + a) + 1e-9))
    return minimize(lambda v: nll(v[0]), [0.0], method="Nelder-Mead").x[0]


mase = lambda X, pr: float(np.mean(np.abs(X.y.values - pr * X.cm.values * X.s.values) / X.s.values))

# train-sim OOF curve (a fixed from the full test period, as it would be applied)
l1A, p0A, QA = np.zeros(len(A)), np.zeros(len(A)), np.zeros((len(A), len(QS)))
for a, b in GroupKFold(5).split(A, groups=base_title(A.movie_title)):
    l1A[b], p0A[b], QA[b] = parts(A.iloc[a], A.iloc[b])
l1B, p0B, QB = parts(A, B)
a_full = fit_a(p0B, (B.y == 0).values)
print(f"intercept-only shift a on the full test period: {a_full:.3f}")
curve_A = {lam: mase(A, hurdle(sig(lg(p0A) + lam * a_full), QA)) for lam in LAMS}
print(f"train-sim OOF: L1 {mase(A, l1A):.4f} | hurdle by lambda {({k: round(v, 4) for k, v in curve_A.items()})}")

films = np.array(sorted(base_title(B.movie_title).unique()))
rows = []
for seed in range(5):
    half = set(np.random.RandomState(seed).permutation(films)[: len(films) // 2])
    h1 = base_title(B.movie_title).isin(half).values
    for fm, em in [(h1, ~h1), (~h1, h1)]:
        a = fit_a(p0B[fm], (B.y.values == 0)[fm])
        r = dict(seed=seed, a=a, L1=mase(B[em], l1B[em]))
        for lam in LAMS:
            r[f"lam{lam}"] = mase(B[em], hurdle(sig(lg(p0B[em]) + lam * a), QB[em]))
        rows.append(r)
R = pd.DataFrame(rows)
m = R.drop(columns=["seed"]).mean()
print("test-period held-out halves (mean of 10):\n", m.round(4).to_string())
print("share of splits where lambda beats L1:", {c: float((R[c] < R.L1).mean()) for c in R.columns if c.startswith("lam")})

fig, ax = plt.subplots(figsize=(7, 3.8))
ax.plot(LAMS, [curve_A[l] for l in LAMS], marker="o", label="train-sim OOF (hurdle)")
ax.axhline(mase(A, l1A), color="#2a78d6", ls="--", lw=1, label="train-sim OOF (L1)")
ax.plot(LAMS, [m[f"lam{l}"] for l in LAMS], marker="s", color="#e34948", label="test period held-out (hurdle)")
ax.axhline(m.L1, color="#e34948", ls="--", lw=1, label="test period held-out (L1)")
ax.set(title="Proxy D3 MASE vs pull-shift fraction lambda", xlabel="lambda"); ax.legend(fontsize=7)
fig.savefig(F / "lambda.png")
print(f"figures -> {F}")
