"""22 - Does a cluster's D3 drop propensity predict its D4-D10 pulls? (bridge from D3 labels to the target)

eda/21: in the test period the odds that a low-demand pair stops selling at D3 are ~0.28x train (a=-1.27),
stable across months. D4-D10 labels do not exist in the test period, so the D3 shift can only be carried
to the target if, in TRAIN, D3 drop behaviour and D4-D10 pulls move together. Test it cross-sectionally:
  unit = cluster x release month; x = logit D3-drop rate of low-demand visible pairs (demand-adjusted),
  y = logit D4-D10 zero rate of simulated targets (demand-adjusted). Slope ~ 1 -> the shift transfers.
Then build the feature that carries it: a trailing, past-only 'D3 drop propensity' per cluster computed
from visible D1-D3 windows (train windows for train, test_history for test), and check its distribution.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import KEY, OUT, fig_dir, load, release_dates, style
from features import dataset, visible_windows

plt = style()
F = fig_dir("22_propensity")
d = load()
tr, th = d["train"], d["hist"]
Xtr, Xte, _ = dataset()


def d3_events(vis):
    """Visible pairs that sold on D2: did they sell on D3? + demand per show at D1-D2."""
    x = vis.assign(d1=vis.groupby("movie_title").date_show.transform("min"))
    x["d"] = (x.date_show - x.d1).dt.days + 1
    P = x.pivot_table(index=KEY, columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    S = x.pivot_table(index=KEY, columns="d", values="total_show", fill_value=0).reindex(columns=[1, 2], fill_value=0)
    e = pd.DataFrame({"sold2": P[2] > 0, "drop3": P[3] == 0, "tps": (P[1] + P[2]) / (S[1] + S[2]).clip(lower=1)}).reset_index()
    e = e[e.sold2].join(x.groupby("movie_title").d1.first(), on="movie_title")
    return e


ev_tr = d3_events(visible_windows(tr, release_dates(tr)))
ev_te = d3_events(th)
# demand adjustment: expected drop from a train-fitted curve on log tickets-per-show
cur = lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=7, min_child_samples=200, verbose=-1,
                         random_state=2026).fit(np.log1p(ev_tr[["tps"]]), ev_tr.drop3)
for e in (ev_tr, ev_te):
    e["exp_drop"] = cur.predict_proba(np.log1p(e[["tps"]]))[:, 1]
    e["m"] = e.d1.dt.to_period("M").astype(str)

# D4-D10 zeros in train-sim, demand-adjusted by a classifier on pair features (no cluster info)
pf = ["log_s", "p1", "p2", "p3", "fp3", "share", "h", "dow", "cal_mult", "d1_dow", "n_hist", "sh_trend"]
zc = lgb.LGBMClassifier(n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=200, verbose=-1,
                        random_state=2026).fit(Xtr[pf], Xtr.total_ticket == 0)
T = Xtr.assign(z=(Xtr.total_ticket == 0).astype(float), exp_z=zc.predict_proba(Xtr[pf])[:, 1],
               m=Xtr.d1.dt.to_period("M").astype(str))
lg = lambda p: np.log(np.clip(p, .01, .99) / (1 - np.clip(p, .01, .99)))
u3 = ev_tr.groupby(["cinema_ids", "m"]).agg(o3=("drop3", "mean"), e3=("exp_drop", "mean"), n3=("drop3", "size"))
u10 = T.groupby(["cinema_ids", "m"]).agg(o10=("z", "mean"), e10=("exp_z", "mean"), n10=("z", "size"))
U = u3.join(u10, how="inner")
U = U[(U.n3 >= 15) & (U.n10 >= 100)]
U["x"] = lg(U.o3) - lg(U.e3)
U["y"] = lg(U.o10) - lg(U.e10)
b = np.polyfit(U.x, U.y, 1)
print(f"units (cluster x month): {len(U)} | corr(D3 excess logit, D4-10 excess logit) = {U[['x', 'y']].corr().iloc[0, 1]:.3f}"
      f" | slope {b[0]:.3f}, intercept {b[1]:.3f}")
um = pd.DataFrame({"x": ev_tr.groupby("m").apply(lambda e: lg(e.drop3.mean()) - lg(e.exp_drop.mean()), include_groups=False),
                   "y": T.groupby("m").apply(lambda t: lg(t.z.mean()) - lg(t.exp_z.mean()), include_groups=False)})
print("by train month (demand-adjusted excess logit):\n", um.round(3))
te_x = ev_te.groupby("m").apply(lambda e: lg(e.drop3.mean()) - lg(e.exp_drop.mean()), include_groups=False)
print("test months D3 excess logit (the shift to carry):\n", te_x.round(3))
print(f"implied D4-D10 excess logit in test via slope: {(b[0] * te_x + b[1]).round(3).to_dict()}")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].scatter(U.x, U.y, s=np.sqrt(U.n10), alpha=.4, color="#2a78d6")
xx = np.linspace(U.x.min(), U.x.max(), 10)
ax[0].plot(xx, np.polyval(b, xx), color="#e34948")
ax[0].set(title="Cluster x month: D3 drop vs D4-D10 pull (excess logit)", xlabel="D3 excess logit", ylabel="D4-D10 excess logit")
ax[1].bar(list(um.index) + list(te_x.index), list(um.x) + list(te_x.values),
          color=["#2a78d6"] * len(um) + ["#e34948"] * len(te_x))
ax[1].axhline(0, color="k", lw=.6); ax[1].tick_params(axis="x", rotation=45)
ax[1].set(title="D3 drop excess logit by month (blue train, red test)")
fig.savefig(F / "propensity.png")
U.to_parquet(OUT / "cache" / "propensity_units.parquet")
print(f"figures -> {F}")
