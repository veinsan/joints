"""08 - Baselines and validation design: which target / feature set generalises?

Two validation schemes (both grouped by base film so IMAX/3D copies never leak across folds):
  * GroupKFold(5)         - many films, measures modelling quality
  * temporal (D1 >= Aug)  - trains on the past only, closest to the real train->test time gap
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from common import OUT, fig_dir, style
from features import feature_cols
from model import LEVEL, NEVER, fit_predict, groups

plt = style()
F = fig_dir("08_cv")
X = pd.read_parquet(OUT / "Xtr.parquet")
X["err0"] = 0.0
mase = lambda p, idx=slice(None): float(np.mean(np.abs(X.total_ticket.values[idx] - p[idx]) / X.scale.values[idx]))
r = X.total_ticket / X.scale
g = groups(X)
folds = list(GroupKFold(5).split(X, groups=g))
tmp = X.d1 >= "2025-08-01"
print(f"rows={len(X)} films={X.movie_title.nunique()}  temporal val films={X[tmp].movie_title.nunique()} rows={tmp.sum()}")

res = {}
print("\n=== baselines (no learning) ===")
res["zero"] = mase(np.zeros(len(X)))
res["naive s"] = mase(X.scale.values)
res["median r * s"] = mase(r.median() * X.scale.values)
# per-horizon median of r (learned OOF)
oof = np.zeros(len(X))
for a, b in folds:
    med = r.iloc[a].groupby(X.h.iloc[a]).median()
    oof[b] = X.h.iloc[b].map(med).values * X.scale.values[b]
res["median r by h"] = mase(oof)
oof = np.zeros(len(X))
rc = r / X.cal_mult
for a, b in folds:
    med = rc.iloc[a].groupby([X.h.iloc[a], X.d1_dow.iloc[a]]).median()
    k = pd.MultiIndex.from_arrays([X.h.iloc[b], X.d1_dow.iloc[b]])
    oof[b] = med.reindex(k).fillna(rc.iloc[a].median()).values * X.cal_mult.values[b] * X.scale.values[b]
res["median r/cal by (h, D1 dow) * cal"] = mase(oof)
# film-momentum heuristic: pair p3 * national decay
for k, v in res.items():
    print(f"  {k:38s} {v:.4f}")

cols_all = [c for c in feature_cols(X) if c not in NEVER + ["err0"]]
cols_rel = [c for c in cols_all if c not in LEVEL]
print(f"\nfeatures: all={len(cols_all)}  relative-only={len(cols_rel)}")


def run(cols, use_cal, name, n=600):
    o = np.zeros(len(X))
    for a, b in folds:
        o[b], _ = fit_predict(X.iloc[a], X.iloc[b], cols, use_cal, n_estimators=n)
    pt, _ = fit_predict(X[~tmp], X[tmp], cols, use_cal, n_estimators=n)
    s_cv, s_t = mase(o), float(np.mean(np.abs(X.total_ticket[tmp] - pt) / X.scale[tmp]))
    print(f"  {name:38s} GroupKFold={s_cv:.4f}  temporal={s_t:.4f}")
    return o, s_cv, s_t


print("\n=== LightGBM, L1 on MASE-normalised target ===")
runs = {}
runs["r, all feats"] = run(cols_all, False, "target r, all features")
runs["r/cal, all feats"] = run(cols_all, True, "target r/cal_mult, all features")
runs["r/cal, relative"] = run(cols_rel, True, "target r/cal_mult, relative features")
runs["r/cal, rel + log_s"] = run(cols_rel + ["log_s", "f_logT"], True, "r/cal, relative + log_s + f_logT")
tmp_base = X[tmp]
print("  (temporal baseline median r/cal by (h,dow)):",
      round(float(np.mean(np.abs(X.total_ticket[tmp] - oof[tmp]) / X.scale[tmp])), 4))

best = min(runs, key=lambda k: runs[k][1] + runs[k][2])
o = runs[best][0]
print(f"\nbest by CV+temporal: {best}")
X["pred"], X["ae"] = o, np.abs(X.total_ticket - o) / X.scale
print("\n=== error anatomy of best OOF ===")
print("by horizon:", X.groupby("h").ae.mean().round(3).to_dict())
print("by D1 weekday:", X.groupby("d1_dow").ae.mean().round(3).to_dict())
sb = pd.qcut(X.scale, 5)
print("by scale quintile:", X.groupby(sb, observed=True).ae.mean().round(3).to_dict())
print("zero-target rows: share of total error", round(X.ae[X.total_ticket == 0].sum() / X.ae.sum(), 3),
      "| mean err on zeros", round(X.ae[X.total_ticket == 0].mean(), 3))
fe = X.groupby("movie_title").ae.mean().sort_values()
print("worst films:", fe.tail(8).round(3).to_dict())
bias = ((X.pred - X.total_ticket) / X.scale).groupby(X.h).median()
print("median signed error by h (pred - y)/s:", bias.round(3).to_dict())

# correlation of residual with features: what the model still misses
res_s = (X.total_ticket - X.pred) / X.scale
cc = X[cols_all].corrwith(res_s, method="spearman").abs().sort_values(ascending=False)
print("\n|spearman(residual, feature)| top 10:", cc.head(10).round(3).to_dict())

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].bar(list(runs), [v[1] for v in runs.values()], color="#2a78d6", label="GroupKFold")
ax[0].bar(list(runs), [v[2] - v[1] for v in runs.values()], bottom=[v[1] for v in runs.values()], color="#eb6834", alpha=.5, label="temporal - cv")
ax[0].set(title="MASE by variant", ylim=(0.2, None)); ax[0].tick_params(axis="x", rotation=20); ax[0].legend()
ax[1].plot(X.groupby("h").ae.mean(), marker="o"); ax[1].set(title="OOF MASE by horizon", xlabel="D")
corr = X[cols_rel[:0] + ["p1", "p2", "p3", "fp1", "fp3", "cal_mult", "h", "share", "cin_size", "comp_n"]].assign(res=res_s).corr("spearman")
im = ax[2].imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax[2].set_xticks(range(len(corr)), corr.columns, rotation=90); ax[2].set_yticks(range(len(corr)), corr.columns)
ax[2].set_title("Spearman corr (incl. residual)"); fig.colorbar(im, ax=ax[2], shrink=.8)
fig.savefig(F / "cv.png")
X[["movie_title", "cinema_ids", "date_show", "h", "total_ticket", "scale", "pred", "ae"]].to_parquet(OUT / "oof_best.parquet")
print(f"figures -> {F}")
