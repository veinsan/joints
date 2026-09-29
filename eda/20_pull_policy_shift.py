"""20 - The constant ~0.07 CV->LB offset shared by v2 (TW 0.377 -> LB 0.4506) and v3 (0.388 -> 0.4573).

A constant offset across different models = a systematic difference every model inherits from train.
Candidate found in the test-period proxy (D1,D2 -> D3): at equal absolute scale, far fewer pairs stop
selling at D3 in the test period. Three explanations are tested:
  (a) season / quiet market   -> compare months with EQUAL market level (Nov-Dec 2025 vs train months)
  (b) data holes in train     -> 'sold, not sold, sold' hole rate, D1-D3 presence patterns
  (c) cinema pull policy changed in the test period -> what remains when (a) and (b) are ruled out
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, OUT, fig_dir, load, release_dates, simulate, style
from features import visible_windows

plt = style()
F = fig_dir("20_pull_policy")
d = load()
tr, th = d["train"], d["hist"]
A = pd.read_parquet(OUT / "cache" / "proxy_A.parquet")
B = pd.read_parquet(OUT / "cache" / "proxy_B.parquet")
bins = [0, 5, 20, 50, 100, 200, 1e9]
for X in (A, B):
    X["b"] = pd.cut(X.s, bins)
    X["z"] = X.y == 0

print("=== D3 zero rate (pair sold on D2) by scale bucket and D1 month ===")
zm = pd.concat([A.pivot_table(index="b", columns="month", values="z", aggfunc="mean", observed=False),
                B.pivot_table(index="b", columns="month", values="z", aggfunc="mean", observed=False)], axis=1)
print(zm.round(3))

print("\n=== (a) market level by D1 month: median tickets per cluster-day of films in their D1-D3 ===")
v = pd.concat([visible_windows(tr, release_dates(tr)), th])
v["d1"] = v.groupby("movie_title").date_show.transform("min")
f = v.groupby("movie_title").agg(m=("d1", "first"), T=("total_ticket", "sum"), n=("cinema_ids", "size"))
f["m"] = f.m.dt.to_period("M").astype(str)
lvl = (f["T"] / f.n).groupby(f.m).median().rename("tickets_per_cluster_day")
print(lvl.round(1).to_string())
print("-> Nov/Dec 2025 are as busy as train months, yet their D3 zero rates stay at test-period levels")

print("\n=== (b) data holes ===")
D1 = release_dates(tr)
hs, _ = simulate(tr, D1)


def pattern(h):
    d1 = h.groupby("movie_title").date_show.min()
    x = h.join(d1.rename("d1"), on="movie_title")
    x["d"] = (x.date_show - x.d1).dt.days + 1
    P = x.pivot_table(index=KEY, columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0) > 0
    return (P[1].astype(int).astype(str) + P[2].astype(int).astype(str) + P[3].astype(int).astype(str)).value_counts(normalize=True)


print(pd.DataFrame({"train_sim": pattern(hs), "test_history": pattern(th)}).round(4))

print("\n=== (c) same film-level market & same weekday, still different? tickets per show at D2 ===")
for n, X in [("train", A), ("test", B)]:
    X["tps2"] = X.s * 2 / (X.sh1 + X.sh2).clip(lower=1) if "sh1" in X else np.nan
q = pd.qcut(pd.concat([A.tps2, B.tps2]), 5)
qa, qb = q.iloc[: len(A)].values, q.iloc[len(A):].values
tt = pd.DataFrame({"train_zeroD3": A.z.groupby(qa, observed=False).mean(), "test_zeroD3": B.z.groupby(qb, observed=False).mean(),
                   "train_n": pd.Series(qa).value_counts(), "test_n": pd.Series(qb).value_counts()})
print("zero rate at D3 by tickets-per-show at D1-D2 (demand per screening):\n", tt.round(3))

fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
for c in zm.columns:
    col = "#2a78d6" if c < "2025-10" else "#e34948"
    ax[0].plot(range(len(zm)), zm[c].values, color=col, alpha=.8, marker="o", ms=3)
ax[0].set_xticks(range(len(zm)), [str(i) for i in zm.index], rotation=30)
ax[0].set(title="D3 zero rate by scale: train months (blue) vs test months (red)", ylabel="zero rate")
ax[1].bar(lvl.index, lvl.values, color=["#2a78d6" if m < "2025-10" else "#e34948" for m in lvl.index])
ax[1].set(title="Market level: tickets per cluster-day (D1-D3)"); ax[1].tick_params(axis="x", rotation=45)
x = np.arange(len(tt))
ax[2].bar(x - .2, tt.train_zeroD3, .4, label="train"); ax[2].bar(x + .2, tt.test_zeroD3, .4, label="test period")
ax[2].set_xticks(x, [f"q{i + 1}" for i in x]); ax[2].set(title="D3 zero rate by tickets/show quintile"); ax[2].legend()
fig.savefig(F / "pull_policy.png")
print(f"figures -> {F}")
