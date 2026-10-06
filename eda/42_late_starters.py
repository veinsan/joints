"""42 - Late-starting pairs: first sale on D3 (or D2) of the official window.

v6 OOF: pairs with first_day = 3 are ~4% of the test-weighted rows but ~0.11 of the 0.385 TW-MASE. They are
bimodal: 60-68% sell nothing on D4-D10 (one-off show), the rest sell at a full rate on a tiny scale
(s = y3 / 3), so r = y / s reaches 5-10 while the model predicts ~0.8. Questions:
  1. composition: is their test share the same as in the (bucket-weighted) train-sim? -> weights by bucket x fd
  2. can the two modes be told apart from D3 itself (shows, tickets, occupancy on D3)?
  3. does a specialist model on late starters (fd >= 2, plus limited-release rows) beat the v6 OOF on them?
  4. does a level-normalised target r' = y / max(s, y_last) help the generic LightGBM?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from evaluate import kappa_worlds, test_weights, test_weights_fd
from features import dataset, feature_cols

plt = style()
F = fig_dir("42_late_starters")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
o = pd.read_csv(Path(__file__).resolve().parents[1] / "results/v7/oof.csv")
p6, y, s, cm = o.oof_final.values, Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
fold = o.fold.values
w1, w2 = test_weights(Xtr, Xte), test_weights_fd(Xtr, Xte)
z0 = np.load(C / "hurdle_oof.npz")
W = kappa_worlds(y, s, cm, z0["p0"], z0["Q"])


def sc(p, w, m=None):
    m = np.ones(len(y), bool) if m is None else m
    return (float(np.average(np.abs(y[m] - p[m]) / s[m], weights=w[m])),
            float(np.mean([np.average(np.abs(t[m] - p[m]) / s[m], weights=w[m]) for t in W])))


print("=== 1. v6 OOF under bucket weights vs bucket x first_day weights (real, kappa) ===")
print(f"  bucket weights      {np.round(sc(p6, w1), 4)}")
print(f"  bucket x fd weights {np.round(sc(p6, w2), 4)}")
for fd in (1, 2, 3):
    m = Xtr.first_day.values == fd
    print(f"  fd={fd}: weight share {w1[m].sum() / w1.sum():.3f} -> {w2[m].sum() / w2.sum():.3f}, test row share {(Xte.first_day == fd).mean():.3f}")

late = Xtr.first_day.values >= 2
print("\n=== 2. late starters: D4-D10 zero rate by D3 shows and D3 tickets (fd = 3) ===")
L = Xtr[Xtr.first_day == 3].assign(zero=(y[Xtr.first_day.values == 3] == 0))
print(pd.crosstab(pd.cut(L.sh3, [0, 1, 2, 4, 100]), pd.cut(L.y3, [0, 5, 15, 40, 1e5]), values=L.zero, aggfunc="mean").round(2))
print(pd.crosstab(pd.cut(L.sh3, [0, 1, 2, 4, 100]), pd.cut(L.y3, [0, 5, 15, 40, 1e5])))

print("\n=== 3. specialist median model on late starters ===")
FE = feature_cols(Xtr)
XL = pd.concat([Xtr[late], Xlim[Xlim.first_day >= 2]], ignore_index=True)
gl = np.r_[fold[late], np.full((Xlim.first_day >= 2).sum(), -1)]
rL = (XL.total_ticket / XL.scale).values
P = dict(objective="quantile", alpha=0.5, n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=30,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=2026, deterministic=True, force_row_wise=True)
spec = np.zeros(late.sum())
for k in range(5):
    m = lgb.LGBMRegressor(**P).fit(XL[FE][gl != k], rL[gl != k])
    spec[fold[late] == k] = np.clip(m.predict(Xtr[late][FE][fold[late] == k]), 0, None)
p_spec = p6.copy(); p_spec[late] = spec * s[late]
rows = {"v6 OOF": p6, "specialist on fd>=2": p_spec}
for a in (0.25, 0.5, 0.75):
    q = p6.copy(); q[late] = (1 - a) * p6[late] + a * spec * s[late]; rows[f"blend specialist {a}"] = q
R = pd.DataFrame({k: [*sc(v, w2, late), *sc(v, w2)] for k, v in rows.items()},
                 index=["late real", "late kappa", "ALL real", "ALL kappa"]).T
print(R.round(4))

print("\n=== 4. level-normalised target for the generic LightGBM L1 ===")
lvl = np.maximum(s, Xtr[["y2", "y3"]].max(axis=1).values * np.where(Xtr.first_day.values == 3, 1.0, 0.0))
P1 = dict(objective="l1", n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8,
          subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=2026, deterministic=True, force_row_wise=True)
out = {}
for name, L_ in [("r = y/s", s), ("r' = y/max(s, y3 if fd=3)", lvl)]:
    pr = np.zeros(len(y))
    for k in range(5):
        tr_ = fold != k
        m = lgb.LGBMRegressor(**P1).fit(Xtr[FE][tr_], (y / L_ / cm)[tr_], sample_weight=(L_ * cm / s)[tr_])
        pr[~tr_] = np.clip(m.predict(Xtr[FE][~tr_]), 0, None) * L_[~tr_] * cm[~tr_]
    out[name] = [*sc(pr, w2, late), *sc(pr, w2)]
print(pd.DataFrame(out, index=["late real", "late kappa", "ALL real", "ALL kappa"]).T.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for fd, c in zip((1, 2, 3), ("#2a78d6", "#eb6834", "#1baf7a")):
    m = Xtr.first_day.values == fd
    ax[0].hist(np.clip(y[m] / s[m], 0, 12), bins=48, density=True, histtype="step", color=c, label=f"first day {fd}")
ax[0].set(title="target r = y / s by first sale day", yscale="log"); ax[0].legend()
ax[1].scatter(np.clip(y[late] / s[late], 0, 15), np.clip(spec, 0, 15), s=4, alpha=.3, color="#2a78d6")
ax[1].set(xlabel="true r", ylabel="specialist median", title="late starters: specialist")
fig.savefig(F / "late_starters.png")
np.save(C / "late_spec_oof.npy", spec)
print(f"figures -> {F}")
