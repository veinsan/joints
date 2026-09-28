"""07 - Build features for train-sim and test, then VERIFY the preprocessing did what we expect.

Checks: scale matches the official metric code, no row loss, NaN audit, train-vs-test drift
(per-feature KS + adversarial validation), and a visual look at the key distributions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from common import KEY, OUT, fig_dir, load, release_dates, scale, simulate, style
from features import build, feature_cols, make_ctx

plt = style()
F = fig_dir("07_prepro")
d = load()
tr, th, te = d["train"], d["hist"], d["test"]

first = tr.groupby("movie_title").date_show.min()
D1 = release_dates(tr).drop(first[first == first.min()].index, errors="ignore")
hs, ts = simulate(tr, D1)
Xtr = build(hs, ts, make_ctx(hs, d["hol"], d["movies"], d["price"], tr))
Xte = build(th, te, make_ctx(th, d["hol"], d["movies"], d["price"], tr))

print("=== integrity ===")
print("train-sim rows:", len(ts), "->", len(Xtr), "| test rows:", len(te), "->", len(Xte))
assert len(Xte) == len(te) and (Xte.id.values == te.id.values).all(), "test row order changed"
off = Xte.join(scale(th), on=KEY, rsuffix="_ref")
print("scale == official hitung_skala:", np.allclose(off.scale, off.scale_ref))
assert np.allclose(off.scale, off.scale_ref)
print("h range test:", Xte.h.min(), Xte.h.max(), "| train:", Xtr.h.min(), Xtr.h.max())
cols = feature_cols(Xtr)
print(f"{len(cols)} features")
na = pd.DataFrame({"train_na": Xtr[cols].isna().mean(), "test_na": Xte[cols].isna().mean()})
print("features with NaN:\n", na[(na > 0).any(axis=1)].round(4))

print("\n=== drift: KS statistic train-sim vs test (top 15) ===")
ks = pd.Series({c: ks_2samp(Xtr[c].dropna(), Xte[c].dropna()).statistic for c in cols}).sort_values(ascending=False)
print(ks.head(15).round(3).to_string())

print("\n=== adversarial validation (can a model tell train-sim from test?) ===")
A = pd.concat([Xtr[cols].assign(t=0, g=Xtr.movie_title), Xte[cols].assign(t=1, g=Xte.movie_title)], ignore_index=True)
# grouped by film: film-level features are constant per film, a random split lets the model memorise films
cv = list(StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(A, A.t, A.g))
imp, oof = pd.Series(0.0, index=cols), np.zeros(len(A))
for a, b in cv:
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=2026, verbose=-1)
    m.fit(A.iloc[a][cols], A.t.iloc[a])
    oof[b] = m.predict_proba(A.iloc[b][cols])[:, 1]
    imp += pd.Series(m.booster_.feature_importance("gain"), index=cols)
print(f"AUC all features: {roc_auc_score(A.t, oof):.3f}")
print("top drift drivers:", (imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_dict())
calendar_like = ["d1_month", "cohort_logT", "comp_n", "comp_logT", "comp_maxT", "t_school", "t_ramadan", "cin_new",
                 "n_comp_open", "t_hol", "hol_in_hist", "cal_mult", "cin_nfilms"]
core = [c for c in cols if c not in calendar_like]
oof2 = np.zeros(len(A))
for a, b in cv:
    m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=2026, verbose=-1).fit(A.iloc[a][core], A.t.iloc[a])
    oof2[b] = m.predict_proba(A.iloc[b][core])[:, 1]
print(f"AUC without calendar/schedule features: {roc_auc_score(A.t, oof2):.3f}  (pair/film behaviour only)")
q = ["f_occ", "occ_mean", "f_tps3", "tps3", "log_s", "f_logT", "p3", "fp3", "f_per_cin"]
print("\nmedians train-sim vs test:\n", pd.DataFrame({"train": Xtr[q].median(), "test": Xte[q].median()}).round(3))

Xtr.to_parquet(OUT / "Xtr.parquet")
Xte.to_parquet(OUT / "Xte.parquet")

show = ["log_s", "p3", "fp3", "f_logT", "share", "cal_mult", "h", "cin_size", "comp_logT"]
fig, ax = plt.subplots(2, 5, figsize=(16, 6))
for a, c in zip(ax.flat, show):
    lo, hi = np.nanpercentile(pd.concat([Xtr[c], Xte[c]]), [1, 99])
    bins = np.linspace(lo, hi, 40)
    a.hist(Xtr[c].clip(lo, hi), bins=bins, alpha=.55, density=True, label="train-sim")
    a.hist(Xte[c].clip(lo, hi), bins=bins, alpha=.55, density=True, label="test")
    a.set_title(f"{c} (KS {ks[c]:.2f})")
ax.flat[0].legend()
ax.flat[-1].barh(ks.head(10).index[::-1], ks.head(10).values[::-1], color="#e34948")
ax.flat[-1].set_title("KS drift top-10")
fig.savefig(F / "drift.png")
print(f"figures -> {F}")
