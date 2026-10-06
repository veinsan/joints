"""29 - Contemporaneous cluster-day demand from OTHER films on the target date.

Validation is now calibrated to the LB (TW-MASE in the kappa=0.5 world: v5 0.4066 vs LB 0.4087), so the
remaining gap to the top must come from validated model quality. One unused, legitimate signal:
test_history holds every wide release's D1-D3 rows, so on a target date t the SAME cluster c usually sells
other films that are in their D1-D3 window. Normalised by those films' national sales on t, that is a
cluster-day demand index (local event, quiet cluster, closure). The train-sim gets the identical signal from
train D1-D3 windows. Features:
  vis_n          other titles in window nationally on t
  c_vis_share    share of those titles' national tickets on t sold at cluster c
  c_dem_idx      log(c_vis_share / cluster's typical share)   - demand shock of c on t
  c_absent       visible titles shown in >= 50% of clusters on t, none at c -> closed / not reporting?
Checks: residual correlation vs the v5 OOF, zero rate vs c_absent, and hurdle TW-MASE with/without.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import OUT, base_title, fig_dir, load, release_dates, style
from evaluate import folds, test_weights
from features import dataset, visible_windows

plt = style()
F = fig_dir("29_cinema_day")
d = load()
tr, th = d["train"], d["hist"]
Xtr, Xte, Xlim = dataset()


def cinema_day(X, vis, ref_share=None):
    v = vis.assign(base=base_title(vis.movie_title))
    nat = v.groupby(["date_show", "base"]).agg(nat_tx=("total_ticket", "sum"), nat_nc=("cinema_ids", "nunique")).reset_index()
    cel = v.groupby(["cinema_ids", "date_show", "base"]).total_ticket.sum().rename("c_tx").reset_index()
    n_cl = vis.cinema_ids.nunique()
    t = X[["cinema_ids", "date_show"]].assign(base=base_title(X.movie_title)).reset_index()
    m = t.merge(nat, on="date_show", how="left", suffixes=("", "_o"))
    m = m[m.base != m.base_o]
    m = m.merge(cel.rename(columns={"base": "base_o"}), on=["cinema_ids", "date_show", "base_o"], how="left").fillna({"c_tx": 0})
    g = m.groupby("index").agg(vis_n=("base_o", "nunique"), nat_tx=("nat_tx", "sum"), c_tx=("c_tx", "sum"),
                               wide=("nat_nc", lambda s: int((s >= 0.5 * n_cl).sum())))
    has = m[m.c_tx > 0].groupby("index").base_o.nunique().rename("c_has")
    g = g.join(has).fillna({"c_has": 0})
    X = X.join(g).fillna({"vis_n": 0, "nat_tx": 0, "c_tx": 0, "wide": 0, "c_has": 0})
    X["c_vis_share"] = np.where(X.nat_tx > 0, X.c_tx / X.nat_tx.clip(lower=1), np.nan)
    if ref_share is None:
        ref_share = X.groupby("cinema_ids").c_vis_share.median()
    X["c_dem_idx"] = np.log((X.c_vis_share + 1e-4) / (X.cinema_ids.map(ref_share) + 1e-4))
    X["c_absent"] = ((X.wide > 0) & (X.c_has == 0)).astype(int)
    return X, ref_share


vis_tr = visible_windows(tr, release_dates(tr))
Xtr, ref = cinema_day(Xtr, vis_tr)
Xlim, _ = cinema_day(Xlim, vis_tr, ref)
Xte, _ = cinema_day(Xte, th, ref)
NEW = ["vis_n", "c_vis_share", "c_dem_idx", "c_absent"]
print("=== distribution train-sim vs test ===")
print(pd.DataFrame({"train_mean": Xtr[NEW].mean(), "test_mean": Xte[NEW].mean(),
                    "train_nonnull": Xtr[NEW].notna().mean(), "test_nonnull": Xte[NEW].notna().mean()}).round(3))
z = Xtr.total_ticket == 0
print("\nzero rate by c_absent (train-sim):", z.groupby(Xtr.c_absent).mean().round(3).to_dict(), "| rows", Xtr.c_absent.value_counts().to_dict())
print("test rows with c_absent=1:", int(Xte.c_absent.sum()))

o5 = pd.read_csv(OUT.parent / "results/v5/oof.csv")
res = (o5.y - o5.oof_final) / o5.scale
q = pd.qcut(Xtr.c_dem_idx, 5, duplicates="drop")
tab = pd.DataFrame({"mean_residual_(y-pred)/s": res.groupby(q, observed=True).mean(), "true_r": (o5.y / o5.scale).groupby(q, observed=True).mean(),
                    "pred_r": (o5.oof_final / o5.scale).groupby(q, observed=True).mean(), "zero_rate": z.groupby(q, observed=True).mean()})
print("\nv5 OOF residual by cluster-day demand quintile:\n", tab.round(3))
print("spearman(residual, c_dem_idx):", round(pd.Series(res).corr(Xtr.c_dem_idx, method="spearman"), 3))

BASE = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
        'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
        'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
        'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
        'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
COMP = ["c_new_n", "c_new_sh", "c_new_sh_rel", "c_new_sh_vs_own", "c_new_tx_vs_own"]
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=63, min_child_samples=100, feature_fraction=0.7,
         verbose=-1, deterministic=True, force_row_wise=True, random_state=2026)
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
rr = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def hurdle_oof(zcols, qcols):
    p0, Q = np.zeros(len(Xtr)), np.zeros((len(Xtr), len(QS)))
    for k in range(5):
        Xa, v = pd.concat([Xtr[fold != k], Xlim], ignore_index=True), fold == k
        p0[v] = lgb.LGBMClassifier(**P).fit(Xa[zcols], Xa.total_ticket == 0).predict_proba(Xtr[v][zcols])[:, 1]
        pos = Xa[Xa.total_ticket > 0]
        Q[v] = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **P).fit(pos[qcols], rr(pos)).predict(Xtr[v][qcols])
                                        for q in QS]), axis=1)
    return p0, Q


def med(p0, Q, lam):
    p = sig(lg(p0) + lam * -1.32)
    t = np.clip((0.5 - p) / (1 - p + 1e-9), QS[0], QS[-1])
    return np.where(p >= .5, 0, np.clip([np.interp(ti, QS, qi) for ti, qi in zip(t, Q)], 0, None)) * cm * s


def worlds(pred, p0_ref, Q_ref):
    out = {"real": np.average(np.abs(y - pred) / s, weights=w_te)}
    vals = []
    for seed in range(3):
        rng = np.random.default_rng(seed)
        ps = sig(lg(p0_ref) + 0.5 * -1.32)
        un = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(p0_ref, 1e-6, None))
        dr = np.array([np.interp(u, QS, q) for u, q in zip(rng.random(len(y)), Q_ref)])
        ys = np.where(un, np.clip(dr, 0, None) * cm * s, y)
        vals.append(np.average(np.abs(ys - pred) / s, weights=w_te))
    out["kappa0.5"] = float(np.mean(vals))
    return out


z0 = np.load(OUT / "cache" / "hurdle_oof.npz")
rows = {}
for name, zc, qc in [("v5 (comp in classifier)", BASE + COMP, BASE),
                     ("+ cluster-day in classifier", BASE + COMP + NEW, BASE),
                     ("+ cluster-day in classifier and quantiles", BASE + COMP + NEW, BASE + NEW)]:
    p0, Q = hurdle_oof(zc, qc)
    rows[name] = worlds(med(p0, Q, 0.5), z0["p0"], z0["Q"])
    print(name, {k: round(v, 4) for k, v in rows[name].items()})
R = pd.DataFrame(rows).T
print("\n", R.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].bar(range(len(tab)), tab.iloc[:, 0], color="#2a78d6"); ax[0].axhline(0, color="k", lw=.6)
ax[0].set_xticks(range(len(tab)), [f"q{i + 1}" for i in range(len(tab))]); ax[0].set(title="v5 OOF residual by cluster-day demand quintile")
ax[1].barh(R.index, R["kappa0.5"], color="#1baf7a"); ax[1].set(title="TW-MASE (kappa=0.5 world)", xlim=(R["kappa0.5"].min() - .005, R["kappa0.5"].max() + .002))
fig.savefig(F / "cinema_day.png")
pd.concat([Xtr[NEW]], axis=1).to_parquet(OUT / "cache" / "cday_train.parquet")
print(f"figures -> {F}")
