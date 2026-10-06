"""36 - Three-model ensemble on the LB-calibrated metric: LightGBM mix + TabPFN-3.5 + TabICL v2.

All components produce a MASE-optimal point (mixture median with the same pull shift lambda = 0.5) and are
blended linearly; weights searched on a 0.1 simplex grid with TW-MASE in the kappa = 0.5 world. Also:
error correlation between components (diversity), and TabICL per-horizon vs pooled context on fold 0.
"""
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from evaluate import folds, test_weights
from features import dataset

plt = style()
F = fig_dir("36_three_models")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
TQS = np.round(np.arange(0.02, 1.0, 0.02), 2)
A_SHIFT, LAM, HW, EPS = -1.32, 0.5, 0.75, 0.02
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
z0 = np.load(C / "hurdle_oof.npz")
worlds = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    ps = sig(lg(z0["p0"]) + 0.5 * A_SHIFT)
    un = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(z0["p0"], 1e-6, None))
    dr = np.array([np.interp(u, QS, q) for u, q in zip(rng.random(len(y)), z0["Q"])])
    worlds.append(np.where(un, np.clip(dr, 0, None) * cm * s, y))


def sc(pred, idx=slice(None)):
    return (float(np.average(np.abs(y[idx] - pred[idx]) / s[idx], weights=w_te[idx])),
            float(np.mean([np.average(np.abs(w[idx] - pred[idx]) / s[idx], weights=w_te[idx]) for w in worlds])))


def mixmed(p, Qm, qs, lam):
    pp = sig(lg(p) + lam * A_SHIFT)
    t = np.clip((0.5 - pp) / (1 - pp + 1e-9), qs[0], qs[-1])
    return np.where(pp >= 0.5, 0.0, np.clip([np.interp(ti, qs, qi) for ti, qi in zip(t, Qm)], 0, None))


def tfm_median(TQ, lam):
    tp0 = np.array([np.interp(EPS, q, TQS, left=0.0, right=1.0) for q in TQ])
    u = tp0[:, None] + (1 - tp0[:, None]) * QS[None, :]
    tq = np.clip(np.array([np.interp(ui, TQS, qi) for ui, qi in zip(u, TQ)]), 0, None)
    return mixmed(tp0, tq, QS, lam)


z6 = np.load(C / "lgb_parts_v6.npz")
comp = {"lgb": ((1 - HW) * z6["l1"] + HW * mixmed(z6["p0"], z6["Q"], QS, LAM)) * cm * s,
        "tabpfn35": tfm_median(np.load(C / "tab35_oofQ.npy"), LAM) * cm * s,
        "tabicl": tfm_median(np.load(C / "tabicl_oofQ.npy"), LAM) * cm * s}
print("=== single components (TW real, TW kappa0.5) ===")
for k, v in comp.items():
    print(f"  {k:10s}", np.round(sc(v), 4))
err = pd.DataFrame({k: (y - v) / s for k, v in comp.items()})
print("signed-error correlation between components:\n", err.corr().round(3))

rows = []
grid = np.round(np.arange(0, 1.01, 0.1), 1)
for a, b in product(grid, grid):
    if a + b <= 1.0 + 1e-9:
        c = round(1 - a - b, 1)
        p = a * comp["lgb"] + b * comp["tabpfn35"] + c * comp["tabicl"]
        r, k = sc(p)
        rows.append(dict(w_lgb=a, w_tabpfn=b, w_tabicl=c, real=r, kappa=k))
G = pd.DataFrame(rows).sort_values("kappa")
print("\n=== best simplex weights (kappa world) ===\n", G.head(8).round(4).to_string(index=False))
two = G[G.w_tabicl == 0].iloc[0]
print(f"best without TabICL: {two.to_dict()}")

pooled = C / "tabicl_pooled_fold0Q.npy"
if pooled.exists():
    idx = np.where(fold == 0)[0]
    ph = comp["tabicl"]
    pp = np.zeros(len(y)); pp[idx] = tfm_median(np.load(pooled), LAM) * cm[idx] * s[idx]
    print(f"\nfold 0 TabICL: per-horizon {np.round(sc(ph, idx), 4)} vs pooled context {np.round(sc(pp, idx), 4)}")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
best = G.iloc[0]
ax[0].bar(list(comp) + ["best blend"], [sc(v)[1] for v in comp.values()] + [best.kappa], color=["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"])
ax[0].set(title="TW-MASE kappa=0.5", ylim=(min(best.kappa, min(sc(v)[1] for v in comp.values())) - .005, None))
piv = G.pivot_table(index="w_lgb", columns="w_tabpfn", values="kappa")
im = ax[1].imshow(piv.values, cmap="Blues_r", origin="lower"); ax[1].set_xticks(range(len(piv.columns)), piv.columns)
ax[1].set_yticks(range(len(piv.index)), piv.index); ax[1].set(xlabel="w TabPFN-3.5", ylabel="w LightGBM", title="TW-MASE kappa (rest = TabICL)")
fig.colorbar(im, ax=ax[1], shrink=.8)
fig.savefig(F / "three_models.png")
print(f"figures -> {F}")
