"""33 - Is the test-period pull shift uniform? (v4-v6 apply ONE logit shift a for every row)

Small pairs (scale <= 20) are 13% of test rows but ~0.11 of the ~0.40 MASE, and at D3 they stop selling far
less often in the test period. If the logit shift is stronger for small pairs (or for some weekday / film
size), a single 'a' under-corrects exactly where the error mass is. Measured on the test-period proxy
(D1,D2 -> D3, labelled), with split-half stability, per:
  scale bucket | D1 weekday | film national size | train-model p0 bucket
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from common import OUT, base_title, fig_dir, style

plt = style()
F = fig_dir("33_shift_het")
A = pd.read_parquet(OUT / "cache" / "proxy_A.parquet").reset_index(drop=True)
B = pd.read_parquet(OUT / "cache" / "proxy_B.parquet").reset_index(drop=True)
cols = ["q1", "q2", "log_s", "sh1", "sh2", "sh_tr", "occ2", "f_tr", "f_log", "f_nc", "cm", "d1_dow"]
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, verbose=-1,
         deterministic=True, force_row_wise=True, random_state=2026)
p0 = np.clip(lgb.LGBMClassifier(**P).fit(A[cols], A.y == 0).predict_proba(B[cols])[:, 1], 1e-4, 1 - 1e-4)
z = (B.y == 0).values
lg = lambda p: np.log(p / (1 - p))
sig = lambda t: 1 / (1 + np.exp(-t))


def fit_a(m):
    nll = lambda a: -np.mean(z[m] * np.log(sig(lg(p0[m]) + a[0])) + (1 - z[m]) * np.log(1 - sig(lg(p0[m]) + a[0])))
    return float(minimize(nll, [0.0], method="Nelder-Mead").x[0]) if m.sum() > 50 else np.nan


a_all = fit_a(np.ones(len(B), bool))
print(f"global a = {a_all:.3f}")
films = np.array(sorted(base_title(B.movie_title).unique()))
halves = [set(np.random.RandomState(sd).permutation(films)[: len(films) // 2]) for sd in range(5)]
B["half"] = 0
groups = {
    "scale": pd.cut(B.s, [0, 5, 20, 50, 100, 200, 1e9]).astype(str),
    "D1 weekday": B.d1_dow.map(dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]))),
    "film size": pd.qcut(B.f_log, 4, labels=["Q1 small", "Q2", "Q3", "Q4 big"]).astype(str),
    "train p0": pd.Series(pd.cut(p0, [0, .02, .05, .1, .2, .4, 1]).astype(str)),
}
out = {}
for gname, g in groups.items():
    rows = []
    for lev in sorted(g.unique()):
        m = (g == lev).values
        hs = [fit_a(m & base_title(B.movie_title).isin(h).values) for h in halves]
        rows.append(dict(level=lev, rows=int(m.sum()), zero_rate=z[m].mean(), mean_p0=p0[m].mean(), a=fit_a(m),
                         a_half_sd=np.nanstd(hs)))
    out[gname] = pd.DataFrame(rows)
    print(f"\n=== by {gname} ===\n", out[gname].round(3).to_string(index=False))

fig, ax = plt.subplots(1, 4, figsize=(20, 3.8))
for a_, (gname, T) in zip(ax, out.items()):
    a_.bar(T.level.astype(str), T.a, yerr=T.a_half_sd, color="#2a78d6", capsize=3)
    a_.axhline(a_all, color="#e34948", ls="--", label="global a")
    a_.set(title=f"logit pull shift a by {gname}"); a_.tick_params(axis="x", rotation=35); a_.legend()
fig.savefig(F / "shift_het.png")
print(f"figures -> {F}")
