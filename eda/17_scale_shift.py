"""17 - 'Small' means different things in train and test: absolute level features are season-confounded.

In train (Apr-Sep, busy) a small pair is a film failing at that cluster -> pulled -> zeros.
In test (Oct-Mar, quiet, Ramadan) a small pair is often a normal film in a quiet market.
The D3 zero rate at equal absolute scale is 2-4x lower in the test period (proxy task, eda/13).

Experiment: proxy task (D1,D2 -> D3) with three feature variants, scored on BOTH
  A = train-sim grouped CV (what CV sees)   and   B = test period (what the LB sees)
  abs : absolute levels (log scale, film national volume)
  rel : level relative to the film (pair per-cinema share) and film relative to the market
        (film volume minus median of films released in the prior 28 days, train+test windows)
  both
Decision rule: pick the variant that wins on B without collapsing on A.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from common import OUT, base_title, fig_dir, load, release_dates, style
from features import visible_windows

plt = style()
F = fig_dir("17_scale_shift")
A = pd.read_parquet(OUT / "cache" / "proxy_A.parquet")
B = pd.read_parquet(OUT / "cache" / "proxy_B.parquet")
d = load()
vis = pd.concat([visible_windows(d["train"], release_dates(d["train"])), d["hist"]])
v = vis.assign(d1=vis.groupby("movie_title").date_show.transform("min"))
v = v[v.date_show <= v.d1 + pd.Timedelta(days=1)]
ft = v.groupby("movie_title").agg(d1=("d1", "first"), T=("total_ticket", "sum"), nc=("cinema_ids", "nunique"))
ft["lpc"] = np.log1p(ft["T"] / 2 / ft.nc)
ft = ft.sort_values("d1")
mk = {m: ft[(ft.d1 < r.d1) & (ft.d1 >= r.d1 - pd.Timedelta(days=28))].lpc.median() for m, r in ft.iterrows()}


def add(X):
    X = X.copy()
    X["pair_vs_film"] = np.log(X.s) - (X.f_log - np.log(X.f_nc.clip(lower=1)))
    X["film_pc"] = X.f_log - np.log(X.f_nc.clip(lower=1))
    X["film_vs_mkt"] = X.film_pc - X.movie_title.map(mk)
    X["pair_vs_mkt"] = np.log(X.s) - X.movie_title.map(mk)
    return X


A, B = add(A), add(B)
shape = ["q1", "q2", "sh_tr", "f_tr", "cm", "d1_dow"]
V = {"abs": shape + ["log_s", "f_log", "f_nc", "sh1", "sh2", "occ2"],
     "rel": shape + ["pair_vs_film", "film_vs_mkt", "pair_vs_mkt", "f_nc"],
     "both": shape + ["log_s", "f_log", "f_nc", "sh1", "sh2", "occ2", "pair_vs_film", "film_vs_mkt", "pair_vs_mkt"]}
tgt = lambda f: f.y / f.s / f.cm
par = dict(objective="l1", n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, verbose=-1)
bins = [0, 5, 20, 50, 100, 200, 500, 1e9]
out, bucket = {}, {}
for n, cols in V.items():
    oof = np.zeros(len(A))
    for a, b in GroupKFold(5).split(A, groups=base_title(A.movie_title)):
        m = lgb.LGBMRegressor(**par, random_state=2026).fit(A.iloc[a][cols], tgt(A.iloc[a]), sample_weight=A.cm.iloc[a])
        oof[b] = np.clip(m.predict(A.iloc[b][cols]), 0, None) * A.cm.values[b] * A.s.values[b]
    m = lgb.LGBMRegressor(**par, random_state=2026).fit(A[cols], tgt(A), sample_weight=A.cm)
    pB = np.clip(m.predict(B[cols]), 0, None) * B.cm.values * B.s.values
    eA, eB = np.abs(A.y - oof) / A.s, np.abs(B.y - pB) / B.s
    out[n] = dict(A_cv=eA.mean(), B_test=eB.mean(), B_small=eB[B.s <= 50].mean(), B_big=eB[B.s > 200].mean(),
                  B_pred_zero=(pB / B.s < .05).mean(), B_true_zero=(B.y == 0).mean(), B_bias=np.median((pB - B.y) / B.s))
    bucket[n] = eB.groupby(pd.cut(B.s, bins), observed=False).mean()
res = pd.DataFrame(out).T
print("=== proxy D1,D2 -> D3: A = train-sim CV, B = test period ===")
print(res.round(4))
print("\nB (test period) MASE by scale bucket:\n", pd.DataFrame(bucket).round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
zt = pd.DataFrame({"train-sim": (A.y == 0).groupby(pd.cut(A.s, bins), observed=False).mean(),
                   "test period": (B.y == 0).groupby(pd.cut(B.s, bins), observed=False).mean()})
zt.plot.bar(ax=ax[0], rot=30); ax[0].set(title="D3 zero rate at equal absolute scale", xlabel="pair scale (mean D1-D2)")
x = np.arange(len(res))
ax[1].bar(x - .2, res.A_cv, .4, label="A: train-sim CV"); ax[1].bar(x + .2, res.B_test, .4, label="B: test period")
ax[1].set_xticks(x, res.index); ax[1].set(title="Proxy MASE by feature variant", ylim=(0.35, None)); ax[1].legend()
fig.savefig(F / "scale_shift.png")
print(f"figures -> {F}")
