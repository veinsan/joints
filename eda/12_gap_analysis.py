"""12 - Why is CV ~0.30-0.33 while public LB is ~0.45? Quantify the CV->LB gap.

Idea: MASE is a mean over rows, so if the test mixes easy/hard rows differently from the train-sim,
the plain CV is biased. Re-weight OOF errors to the test composition:
  (a) by scale bucket only,  (b) by an adversarial density ratio w = p/(1-p) on pair-level features.
If the re-weighted CV lands near the public score, the gap is composition (fixable by weighting /
targeted modelling); if not, it is a regime shift in retention itself.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from common import OUT, fig_dir, style
from features import dataset, feature_cols
from model import LEVEL, NEVER, fit_predict, groups

plt = style()
F = fig_dir("12_gap")
Xtr, Xte = dataset()
cols = [c for c in feature_cols(Xtr) if c not in NEVER and (c not in LEVEL or c in ("log_s", "f_logT"))]
g = groups(Xtr)
ae = lambda p: np.abs(Xtr.total_ticket.values - p) / Xtr.scale.values

cache = OUT / "cache" / "oof_lgb.npy"
if cache.exists():
    oof = np.load(cache)
else:
    oof = np.zeros(len(Xtr))
    for a, b in GroupKFold(5).split(Xtr, groups=g):
        oof[b] = fit_predict(Xtr.iloc[a], Xtr.iloc[b], cols, True, n_estimators=600)[0]
    np.save(cache, oof)
Xtr["err"] = ae(oof)
print(f"LightGBM OOF MASE (unweighted): {Xtr.err.mean():.4f}")

v2 = pd.read_csv(OUT.parent / "results/v2/oof.csv")
v2["err"] = (v2.y - v2.pred).abs() / v2.scale
print(f"v2 TabPFN-3.5 OOF MASE (unweighted): {v2.err.mean():.4f}   public LB 0.45061")

print("\n=== (a) scale composition ===")
bins = [0, 5, 20, 50, 100, 200, 500, 1e9]
cut = lambda s: pd.cut(s, bins)
t = pd.DataFrame({"test_share": cut(Xte.scale).value_counts(normalize=True),
                  "sim_share": cut(Xtr.scale).value_counts(normalize=True),
                  "lgb_mase": Xtr.groupby(cut(Xtr.scale), observed=False).err.mean(),
                  "v2_mase": v2.groupby(cut(v2.scale), observed=False).err.mean()}).sort_index()
t["lgb_contrib_test"] = t.test_share * t.lgb_mase
print(t.round(3))
print(f"scale-reweighted: LGB {(t.test_share * t.lgb_mase).sum():.4f} | v2 {(t.test_share * t.v2_mase).sum():.4f}")

print("\n=== (b) adversarial density ratio on pair-level features ===")
pf = ["log_s", "p1", "p2", "p3", "n_hist", "f_logT", "fnc1", "fnc3", "share", "d1_dow", "occ_mean", "tps3",
      "f_occ", "f_per_cin", "cin_size", "fmt", "rating", "g_Horror", "g_Drama", "sh3", "h", "dow"]
A = pd.concat([Xtr[pf].assign(t=0, g=g.values), Xte[pf].assign(t=1, g=Xte.movie_title.values)], ignore_index=True)
p = np.zeros(len(A))
for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=2026).split(A, A.t, A.g):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=200,
                           random_state=2026, verbose=-1).fit(A.iloc[a][pf], A.t.iloc[a])
    p[b] = m.predict_proba(A.iloc[b][pf])[:, 1]
ps = np.clip(p[: len(Xtr)], 0.02, 0.98)
w = ps / (1 - ps) * (len(Xtr) / len(Xte))
w = np.clip(w / w.mean(), 0, 20)
ess = w.sum() ** 2 / (w ** 2).sum()
print(f"weights: mean 1, p99 {np.percentile(w, 99):.2f}, max {w.max():.1f}, effective sample size {ess:.0f}/{len(w)}")
print(f"density-ratio weighted LGB MASE: {np.average(Xtr.err, weights=w):.4f}")
Xtr["w_adv"] = w
imp = pd.Series(m.booster_.feature_importance("gain"), index=pf).sort_values(ascending=False)
print("shift drivers (gain share):", (imp / imp.sum()).head(8).round(3).to_dict())

print("\n=== error mass by the dominant shift driver (weighted to test) ===")
for c in ["n_hist", "d1_dow"]:
    tt = pd.DataFrame({"test_share": Xte[c].value_counts(normalize=True), "sim_share": Xtr[c].value_counts(normalize=True),
                       "lgb_mase": Xtr.groupby(c).err.mean()}).round(3)
    print(f"-- {c} --\n{tt}")
Xtr[["err", "w_adv"]].to_parquet(OUT / "cache" / "oof_err_weights.parquet")

fig, ax = plt.subplots(1, 3, figsize=(16, 3.8))
x = np.arange(len(t))
ax[0].bar(x - .2, t.sim_share, .4, label="train-sim")
ax[0].bar(x + .2, t.test_share, .4, label="test")
ax[0].set_xticks(x, [str(i) for i in t.index], rotation=30); ax[0].set(title="Share of rows by pair scale"); ax[0].legend()
ax[1].bar(x, t.lgb_mase, color="#e34948"); ax[1].set_xticks(x, [str(i) for i in t.index], rotation=30)
ax[1].set(title="OOF MASE by pair scale (LightGBM)")
vals = {"OOF\nunweighted": Xtr.err.mean(), "scale\nreweighted": (t.test_share * t.lgb_mase).sum(),
        "density-ratio\nweighted": np.average(Xtr.err, weights=w), "public LB\n(v1 / v2)": 0.453}
ax[2].bar(list(vals), list(vals.values()), color=["#2a78d6", "#2a78d6", "#2a78d6", "#eb6834"])
ax[2].set(title="CV estimate vs public LB", ylim=(0.25, 0.5))
fig.savefig(F / "gap.png")
print(f"figures -> {F}")
