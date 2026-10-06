"""46 - Re-tune the decision knobs under composition-correct validation.

All v4-v7 choices (lambda, hurdle weight, LGB/TabPFN blend) were made with bucket-only test weights, which
over-weight late starters 2.3x (eda/42). Re-scored here with three validation lenses, each in kappa worlds:
  bfd  : scale bucket x first sale day weights
  adv  : adversarial density-ratio weights (eda/44)
  b    : old bucket weights (reference)
Grid: lambda x hurdle weight x TabPFN weight (+ integer rounding). Decision = min over the max regret across
lenses and kappa in {0.25, 0.5, 0.75}, so no single (possibly wrong) world drives the choice.
"""
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from evaluate import kappa_worlds, test_weights, test_weights_fd
from features import dataset

plt = style()
F = fig_dir("46_retune")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
TQS = np.round(np.arange(0.02, 1.0, 0.02), 2)
A_SHIFT, EPS = -1.32, 0.02
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
z0, z6 = np.load(C / "hurdle_oof.npz"), np.load(C / "lgb_parts_v6.npz")
TQ = np.load(C / "tab35_oofQ.npy")


def mixmed(p, Qm, qs, lam):
    pp = sig(lg(p) + lam * A_SHIFT)
    t = np.clip((0.5 - pp) / (1 - pp + 1e-9), qs[0], qs[-1])
    return np.where(pp >= 0.5, 0.0, np.clip([np.interp(ti, qs, qi) for ti, qi in zip(t, Qm)], 0, None))


tp0 = np.array([np.interp(EPS, q, TQS, left=0.0, right=1.0) for q in TQ])
u = tp0[:, None] + (1 - tp0[:, None]) * QS[None, :]
tq = np.clip(np.array([np.interp(ui, TQS, qi) for ui, qi in zip(u, TQ)]), 0, None)

lenses = {"b": test_weights(Xtr, Xte), "bfd": test_weights_fd(Xtr, Xte), "adv": np.load(C / "w_adv.npy")}
worlds = {k: kappa_worlds(y, s, cm, z0["p0"], z0["Q"], kappa=k) for k in (0.25, 0.5, 0.75)}
rows = []
cache_h, cache_t = {}, {}
for lam, hw, wt, rnd in product((0.0, 0.25, 0.5, 0.75, 1.0), (0.5, 0.75, 1.0), np.round(np.arange(0, 1.01, 0.1), 1), (False, True)):
    if lam not in cache_h:
        cache_h[lam], cache_t[lam] = mixmed(z6["p0"], z6["Q"], QS, lam), mixmed(tp0, tq, QS, lam)
    lgbp = ((1 - hw) * z6["l1"] + hw * cache_h[lam]) * cm * s
    p = (1 - wt) * lgbp + wt * cache_t[lam] * cm * s
    if rnd:
        p = np.round(p)
    r = dict(lam=lam, hw=hw, w_tab=wt, rnd=rnd)
    for ln, w in lenses.items():
        for k, W in worlds.items():
            r[f"{ln}_k{k}"] = np.mean([np.average(np.abs(t - p) / s, weights=w) for t in W])
    rows.append(r)
G = pd.DataFrame(rows)
crit = [c for c in G.columns if c.startswith(("bfd_", "adv_"))]
reg = G[crit] - G[crit].min()
G["max_regret"] = reg.max(axis=1)
G = G.sort_values("max_regret")
cur = G[(G.lam == 0.5) & (G.hw == 0.75) & (G.w_tab == 0.5) & (~G.rnd)].iloc[0]
print("current v6 setting (lam 0.5, hw 0.75, w_tab 0.5):\n", cur.drop("rnd").astype(float).round(4).to_string())
print("\n=== best by minimax regret over bfd/adv lenses x kappa 0.25/0.5/0.75 ===")
show = ["lam", "hw", "w_tab", "rnd", "b_k0.5", "bfd_k0.5", "adv_k0.5", "bfd_k0.25", "bfd_k0.75", "max_regret"]
print(G[show].head(12).round(4).to_string(index=False))
best = G.iloc[0]
print(f"\nbest vs current: bfd_k0.5 {best['bfd_k0.5'] - cur['bfd_k0.5']:+.4f} | adv_k0.5 {best['adv_k0.5'] - cur['adv_k0.5']:+.4f} | b_k0.5 {best['b_k0.5'] - cur['b_k0.5']:+.4f}")
for col in ("lam", "hw", "w_tab"):
    print(f"marginal best bfd_k0.5 by {col}:", G.groupby(col)["bfd_k0.5"].min().round(4).to_dict())
fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for lam, g in G[(G.hw == 0.75) & (~G.rnd)].groupby("lam"):
    g = g.sort_values("w_tab"); ax[0].plot(g.w_tab, g["bfd_k0.5"], marker="o", ms=3, label=f"lambda {lam}")
ax[0].set(xlabel="TabPFN weight", ylabel="TW (bucket x fd, kappa 0.5)", title="blend x pull shift, hurdle w 0.75"); ax[0].legend(fontsize=7)
for lam, g in G[(G.hw == 0.75) & (~G.rnd)].groupby("lam"):
    g = g.sort_values("w_tab"); ax[1].plot(g.w_tab, g["adv_k0.5"], marker="o", ms=3, label=f"lambda {lam}")
ax[1].set(xlabel="TabPFN weight", ylabel="TW (adversarial, kappa 0.5)", title="same, adversarial lens")
fig.savefig(F / "retune.png")
G.to_csv(OUT / "cache" / "retune_grid.csv", index=False)
print(f"figures -> {F}")
