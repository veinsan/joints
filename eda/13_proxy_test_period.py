"""13 - Proxy task that HAS labels inside the test period: predict D3 from D1-D2.

test_history gives D1-D3 for every test film (Oct 2025 - Mar 2026). A model trained on train-sim
to map (D1, D2) -> D3 is scored on train-sim (CV) and on test_history. If the error / signed bias
is much worse on the test period, retention dynamics have shifted there (beyond composition),
and the shift can be localised by month / scale / segment. The same set is later used as a
test-period sanity check for design choices (weights, augmentation) that CV cannot see.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from common import KEY, OUT, base_title, fig_dir, load, release_dates, simulate, style
from features import calendar

plt = style()
F = fig_dir("13_proxy")
d = load()
tr, th = d["train"], d["hist"]
cal = calendar(d["hol"])
first = tr.groupby("movie_title").date_show.min()
D1 = release_dates(tr).drop(first[first == first.min()].index, errors="ignore")
hs, _ = simulate(tr, D1)


def proxy(h):
    """Pairs that sold on D2 (analogue of the D3 rule); target = D3 tickets / mean(D1, D2)."""
    d1 = h.groupby("movie_title").date_show.min().rename("d1")
    x = h.join(d1, on="movie_title")
    x["d"] = (x.date_show - x.d1).dt.days + 1
    P = x.pivot_table(index=KEY, columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    S = x.pivot_table(index=KEY, columns="d", values="total_show", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    O = x.pivot_table(index=KEY, columns="d", values="occupation_rate", fill_value=0).reindex(columns=[1, 2], fill_value=0)
    P = P[P[2] > 0]
    f = pd.DataFrame(index=P.index)
    f["s"] = ((P[1] + P[2]) / 2).clip(lower=1)
    f["y"] = P[3]
    f["q1"], f["q2"] = P[1] / f.s, P[2] / f.s
    f["log_s"] = np.log1p(f.s)
    f["sh1"], f["sh2"] = S.reindex(P.index)[1], S.reindex(P.index)[2]
    f["sh_tr"] = (f.sh2 + 1) / (f.sh1 + 1)
    f["occ2"] = O.reindex(P.index)[2]
    f = f.reset_index().join(d1, on="movie_title")
    g = x[x.d <= 2].groupby(["movie_title", "d"]).total_ticket.sum().unstack().reindex(columns=[1, 2], fill_value=0)
    f["f_tr"] = f.movie_title.map((g[2] + 1) / (g[1] + 1))
    f["f_log"] = f.movie_title.map(np.log1p(g.mean(axis=1)))
    f["f_nc"] = f.movie_title.map(x[x.d == 2].groupby("movie_title").cinema_ids.nunique())
    for k in range(3):
        f[f"c{k + 1}"] = cal.cal.reindex(f.d1 + pd.Timedelta(days=k)).values
    f["cm"] = f.c3 / ((f.c1 + f.c2) / 2)
    f["d1_dow"] = f.d1.dt.dayofweek
    f["month"] = f.d1.dt.to_period("M").astype(str)
    return f


A, B = proxy(hs), proxy(th)
cols = ["q1", "q2", "log_s", "sh1", "sh2", "sh_tr", "occ2", "f_tr", "f_log", "f_nc", "cm", "d1_dow"]
tgt = lambda f: f.y / f.s / f.cm
M = lambda f, p: np.abs(f.y - p) / f.s
par = dict(objective="l1", n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=100, verbose=-1)

oof = np.zeros(len(A))
for a, b in GroupKFold(5).split(A, groups=base_title(A.movie_title)):
    m = lgb.LGBMRegressor(**par, random_state=2026).fit(A.iloc[a][cols], tgt(A.iloc[a]), sample_weight=A.cm.iloc[a])
    oof[b] = np.clip(m.predict(A.iloc[b][cols]), 0, None) * A.cm.values[b] * A.s.values[b]
m = lgb.LGBMRegressor(**par, random_state=2026).fit(A[cols], tgt(A), sample_weight=A.cm)
pB = np.clip(m.predict(B[cols]), 0, None) * B.cm.values * B.s.values
A["err"], B["err"] = M(A, oof), M(B, pB)
A["bias"], B["bias"] = (oof - A.y) / A.s, (pB - B.y) / B.s
A["naive"], B["naive"] = M(A, A.cm * A.s * np.median(tgt(A))), M(B, B.cm * B.s * np.median(tgt(A)))

print("=== proxy (D1,D2 -> D3) ===")
print(f"train-sim CV : MASE {A.err.mean():.4f}  naive {A.naive.mean():.4f}  median bias {A.bias.median():+.4f}  rows {len(A)}")
print(f"test period  : MASE {B.err.mean():.4f}  naive {B.naive.mean():.4f}  median bias {B.bias.median():+.4f}  rows {len(B)}")
bins = [0, 5, 20, 50, 100, 200, 500, 1e9]
tab = pd.DataFrame({"sim_share": pd.cut(A.s, bins).value_counts(normalize=True), "test_share": pd.cut(B.s, bins).value_counts(normalize=True),
                    "sim_mase": A.groupby(pd.cut(A.s, bins), observed=False).err.mean(),
                    "test_mase": B.groupby(pd.cut(B.s, bins), observed=False).err.mean(),
                    "test_bias": B.groupby(pd.cut(B.s, bins), observed=False).bias.median()}).sort_index()
print(tab.round(3))
print(f"sim CV re-weighted to test scale mix: {(tab.test_share * tab.sim_mase).sum():.4f}  vs real test-period {B.err.mean():.4f}")
mon = B.groupby("month").agg(rows=("err", "size"), mase=("err", "mean"), bias=("bias", "median"),
                             y_over_s=("y", lambda s: (s / B.loc[s.index, "s"]).median()))
print("\nby test month:\n", mon.round(3))
print("train months:\n", A.groupby("month").agg(mase=("err", "mean"), bias=("bias", "median")).round(3))

fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
x = np.arange(len(tab))
ax[0].bar(x - .2, tab.sim_mase, .4, label="train-sim CV"); ax[0].bar(x + .2, tab.test_mase, .4, label="test period")
ax[0].set_xticks(x, [str(i) for i in tab.index], rotation=30); ax[0].set(title="Proxy MASE by scale"); ax[0].legend()
ax[1].bar(mon.index, mon.mase, color="#eb6834"); ax[1].axhline(A.err.mean(), color="#2a78d6", ls="--", label="train-sim CV")
ax[1].set(title="Proxy MASE by test month"); ax[1].legend(); ax[1].tick_params(axis="x", rotation=30)
ax[2].bar(mon.index, mon.bias, color="#4a3aa7"); ax[2].axhline(0, color="k", lw=.6)
ax[2].set(title="Median signed error (pred - y)/s by month"); ax[2].tick_params(axis="x", rotation=30)
fig.savefig(F / "proxy.png")
A.to_parquet(OUT / "cache" / "proxy_A.parquet"); B.to_parquet(OUT / "cache" / "proxy_B.parquet")
print(f"figures -> {F}")
