"""15 - Cinema-level competition visible on the TARGET date.

test_history holds D1-D3 of every wide release in the test period, per cluster. So for a target row
(film f, cluster c, date t), the rows of OTHER films in their D1-D3 window at (c, t) are official,
visible data: how many new titles opened at c, how many shows / tickets they took. At a Wed/Thu
programme change new titles take screens and old titles get pulled - exactly the zero pattern of
eda/14. The train-sim gets the same visibility: D1-D3 windows of all wide releases in train.

Checks: (1) the feature explains zero-rate in train-sim (monotone relation, figure),
        (2) its distribution in train-sim vs test, (3) TW-MASE with vs without.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import KEY, fig_dir, load, release_dates, style
from evaluate import folds, report, test_weights
from features import dataset, feature_cols
from model import LEVEL, NEVER, PARAMS

plt = style()
F = fig_dir("15_competition")
d = load()
tr, th = d["train"], d["hist"]
Xtr, Xte = dataset()


def visible_windows(tx, d1):
    x = tx.merge(d1.rename("d1"), left_on="movie_title", right_index=True)
    return x[(x.date_show >= x.d1) & (x.date_show <= x.d1 + pd.Timedelta(days=2))]


def add_comp(X, vis, cin_shows):
    """Per (cluster, date): new titles opening there, their shows/tickets; excludes the film's own base title."""
    from common import base_title
    v = vis.assign(base=base_title(vis.movie_title))
    agg = v.groupby(["cinema_ids", "date_show", "base"]).agg(sh=("total_show", "sum"), tx=("total_ticket", "sum")).reset_index()
    X = X.assign(base=base_title(X.movie_title))
    m = X[["cinema_ids", "date_show", "base"]].reset_index().merge(agg, on=["cinema_ids", "date_show"], how="left")
    m = m[m.base_x != m.base_y]
    s = m.groupby("index").agg(c_new_n=("base_y", "nunique"), c_new_sh=("sh", "sum"), c_new_tx=("tx", "sum"))
    X = X.join(s).fillna({"c_new_n": 0, "c_new_sh": 0, "c_new_tx": 0})
    X["c_new_sh_rel"] = X.c_new_sh / X.cinema_ids.map(cin_shows).fillna(X.c_new_sh.median() + 1)
    X["c_new_sh_vs_own"] = X.c_new_sh / X.sh3.clip(lower=1)
    X["c_new_tx_vs_own"] = np.log1p(X.c_new_tx) - np.log1p(X.y3)
    # cumulative: new titles opened at c between D4 and t (programme changes already happened)
    X["c_new_n_cum"] = X.groupby(KEY).c_new_n.cumsum() if X.index.is_monotonic_increasing else X.c_new_n
    return X.drop(columns="base")


first = tr.groupby("movie_title").date_show.min()
vis_tr = visible_windows(tr, release_dates(tr))
vis_te = th
cin_shows = tr.groupby(["cinema_ids", "date_show"]).total_show.sum().groupby("cinema_ids").median()
Xtr = add_comp(Xtr.sort_values(KEY + ["date_show"]).reset_index(drop=True), vis_tr, cin_shows)
Xte = add_comp(Xte.sort_values(KEY + ["date_show"]).reset_index(drop=True), vis_te, cin_shows)
new = ["c_new_n", "c_new_sh", "c_new_tx", "c_new_sh_rel", "c_new_sh_vs_own", "c_new_tx_vs_own", "c_new_n_cum"]
print("=== distribution train-sim vs test ===")
print(pd.DataFrame({"train_mean": Xtr[new].mean(), "test_mean": Xte[new].mean(),
                    "train_share>0": (Xtr[new] > 0).mean(), "test_share>0": (Xte[new] > 0).mean()}).round(3))

z = (Xtr.total_ticket == 0)
q = pd.qcut(Xtr.c_new_sh_vs_own.where(Xtr.c_new_sh > 0), 5, duplicates="drop")
print("\n=== zero rate vs competition (train-sim) ===")
print("no new title at c on t:", z[Xtr.c_new_sh == 0].mean().round(3), "| any new title:", z[Xtr.c_new_sh > 0].mean().round(3))
print(z.groupby(q, observed=True).mean().round(3).to_string())
print("zero rate by h, split by any new title at (c,t):\n",
      z.groupby([Xtr.h, Xtr.c_new_n > 0]).mean().unstack().round(3).rename(columns={False: "none", True: "new title"}).T)

cols = [c for c in feature_cols(Xtr) if c not in NEVER and (c not in LEVEL or c in ("log_s", "f_logT"))]
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)


def cv(cs):
    o = np.zeros(len(Xtr))
    for k in range(5):
        a, v = fold != k, fold == k
        Xa = Xtr[a]
        m = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": 500}, random_state=2026).fit(
            Xa[cs], Xa.total_ticket / Xa.scale / Xa.cal_mult, sample_weight=Xa.cal_mult)
        o[v] = np.clip(m.predict(Xtr[v][cs]), 0, None) * Xtr.cal_mult.values[v] * Xtr.scale.values[v]
    return o


print("\n=== TW-MASE ===")
base_cols = [c for c in cols if c not in new]
o0 = cv(base_cols); s0 = report(Xtr, o0, w_te, "without cinema competition")
o1 = cv(base_cols + new); s1 = report(Xtr, o1, w_te, "with cinema competition")
e0 = np.abs(Xtr.total_ticket - o0) / Xtr.scale
e1 = np.abs(Xtr.total_ticket - o1) / Xtr.scale
print("gain by horizon:", (e0 - e1).groupby(Xtr.h).mean().round(4).to_dict())
Xtr[new].to_parquet(F / "comp_train.parquet"); Xte[new + ["id"]].to_parquet(F / "comp_test.parquet")

fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
zz = z.groupby([Xtr.h, Xtr.c_new_n > 0]).mean().unstack()
ax[0].plot(zz.index, zz[False], marker="o", label="no new title at cluster")
ax[0].plot(zz.index, zz[True], marker="o", label="new title(s) opening at cluster")
ax[0].set(title="Zero rate by horizon (train-sim)", xlabel="D"); ax[0].legend()
zq = z.groupby(q, observed=True).mean()
ax[1].bar(range(len(zq)), zq.values, color="#e34948"); ax[1].set_xticks(range(len(zq)), [f"q{i + 1}" for i in range(len(zq))])
ax[1].set(title="Zero rate by quintile of new-title shows / own D3 shows")
ax[2].bar(["without", "with"], [s0, s1], color=["#2a78d6", "#1baf7a"]); ax[2].set(title="TW-MASE", ylim=(min(s0, s1) - .01, max(s0, s1) + .005))
fig.savefig(F / "competition.png")
print(f"figures -> {F}")
