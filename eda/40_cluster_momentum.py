"""40 - Time-local cluster momentum from OTHER films' visible D1-D3 windows.

Film-level error is not predictable from film features (eda/38). The other big oracle is cluster x date.
eda/29 used OTHER films' sales on the target date (no signal). Here: a cluster's recent DYNAMICS. For every
visible pair (film f, cluster c) the relative trend
    rt = log((y3 + 1) / (y1 + 1)) - log((F3 + 1) / (F1 + 1))       (F = film national total)
removes film appeal and calendar. A cluster that quickly drops screens/sales for every new film has rt < 0.
cmi(c, t) = mean rt over OTHER films with D1 in [t - 21, t] at cluster c (available in test from test_history,
computed the same way on train visible windows). Also cmi_zero = share of those pairs with y3 = 0 after y1 > 0.
Checks: correlation with the v6 OOF signed residual and with the zero indicator, by scale bucket; and
whether the test period differs (distribution shift of cmi between train and test).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import OUT, base_title, fig_dir, load, release_dates, style
from features import dataset, visible_windows

plt = style()
F = fig_dir("40_cluster_momentum")
d = load()
tr, th = d["train"], d["hist"]
Xtr, Xte, Xlim = dataset()
o = pd.read_csv(Path(__file__).resolve().parents[1] / "results/v7/oof.csv")
res = ((o.y - o.oof_final) / o.scale).values


def pair_trends(vis):
    v = vis.copy()
    d1 = v.groupby("movie_title").date_show.min()
    v["d"] = (v.date_show - v.movie_title.map(d1)).dt.days + 1
    P = v.pivot_table(index=["movie_title", "cinema_ids"], columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    Fn = P.groupby(level=0).sum()
    P = P[P[1] > 0]
    ft = np.log((Fn[3] + 1) / (Fn[1] + 1)).reindex(P.index.get_level_values(0)).values
    out = pd.DataFrame({"rt": np.log((P[3] + 1) / (P[1] + 1)).values - ft, "z3": (P[3] == 0).values.astype(float)}, index=P.index).reset_index()
    out["d1"] = out.movie_title.map(d1).values
    out["base"] = base_title(out.movie_title)
    return out


def cmi(X, pt, win=21):
    keys = X[["movie_title", "cinema_ids", "d1"]].drop_duplicates()
    keys["base"] = base_title(keys.movie_title)
    m = keys.merge(pt[["cinema_ids", "d1", "base", "rt", "z3"]], on="cinema_ids", suffixes=("", "_o"))
    m = m[(m.base != m.base_o) & (m.d1_o <= m.d1) & (m.d1_o >= m.d1 - pd.Timedelta(days=win))]
    a = m.groupby(["movie_title", "cinema_ids"]).agg(cmi=("rt", "mean"), cmi_zero=("z3", "mean"), cmi_n=("rt", "size")).reset_index()
    return X[["movie_title", "cinema_ids"]].merge(a, on=["movie_title", "cinema_ids"], how="left")


vis_tr = visible_windows(tr, release_dates(tr))
pt_tr, pt_te = pair_trends(vis_tr), pair_trends(th)
Ctr, Cte = cmi(Xtr, pt_tr), cmi(Xte, pt_te)
print(f"coverage: train {Ctr.cmi.notna().mean():.3f}, test {Cte.cmi.notna().mean():.3f}; median n films train {Ctr.cmi_n.median()}, test {Cte.cmi_n.median()}")
print("cmi distribution  train:", Ctr.cmi.describe()[["mean", "std"]].round(3).to_dict(), " test:", Cte.cmi.describe()[["mean", "std"]].round(3).to_dict())
print("cmi_zero          train:", Ctr.cmi_zero.describe()[["mean", "std"]].round(3).to_dict(), " test:", Cte.cmi_zero.describe()[["mean", "std"]].round(3).to_dict())

y = Xtr.total_ticket.values
b = pd.cut(Xtr.scale, [0, 20, 100, 1e9]).astype(str).values
rows = []
for f in ["cmi", "cmi_zero"]:
    v = Ctr[f].values
    ok = ~np.isnan(v)
    rows.append(dict(feature=f, bucket="all", rho_resid=spearmanr(v[ok], res[ok])[0], rho_zero=spearmanr(v[ok], (y == 0)[ok])[0],
                     rho_pairtrend=spearmanr(v[ok], (Xtr.p3_rel.values)[ok])[0]))
    for bb in np.unique(b):
        k = ok & (b == bb)
        rows.append(dict(feature=f, bucket=bb, rho_resid=spearmanr(v[k], res[k])[0], rho_zero=spearmanr(v[k], (y == 0)[k])[0],
                         rho_pairtrend=np.nan))
print("\n=== Spearman: cluster momentum vs OOF residual / zero target ===\n", pd.DataFrame(rows).round(3).to_string(index=False))

q = pd.qcut(Ctr.cmi, 5)
T = pd.DataFrame({"q": q, "res": res, "zero": y == 0}).groupby("q", observed=True).agg(mean_res=("res", "mean"), zero=("zero", "mean"), n=("res", "size"))
print("\nOOF residual by cmi quintile:\n", T.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].hist(Ctr.cmi.dropna(), bins=60, density=True, alpha=.6, label="train-sim"); ax[0].hist(Cte.cmi.dropna(), bins=60, density=True, alpha=.6, label="test")
ax[0].set(title="cluster momentum index (other films, prior 21 days)"); ax[0].legend()
ax[1].bar(range(5), T.mean_res, color="#2a78d6"); ax[1].set(title="OOF signed residual by cmi quintile", xlabel="cmi quintile")
fig.savefig(F / "cmi.png")
pd.concat([Ctr.assign(part="tr"), Cte.assign(part="te")]).to_parquet(OUT / "cache" / "cmi.parquet")
print(f"figures -> {F}")
