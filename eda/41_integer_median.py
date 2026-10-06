"""41 - Tickets are integers: for small pairs the MASE-optimal point is an integer.

The expected absolute loss of an integer-valued y is piecewise linear between integers, so its minimum sits at
an integer (the median of the discrete distribution). Our medians are continuous (quantile interpolation),
which wastes error where s is small (s <= 5: 1.5% of test rows but MASE 2.56). Checks on the v6 OOF:
round / floor / ceil / threshold rounding (round up when frac > t) per scale bucket, real & kappa-world TW.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from evaluate import test_weights
from features import dataset

plt = style()
F = fig_dir("41_integer")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
o = pd.read_csv(Path(__file__).resolve().parents[1] / "results/v7/oof.csv")
p, y, s, cm = o.oof_final.values, Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
w = test_weights(Xtr, Xte)
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda q: np.log(np.clip(q, 1e-4, 1 - 1e-4) / (1 - np.clip(q, 1e-4, 1 - 1e-4)))
z0 = np.load(C / "hurdle_oof.npz")
worlds = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    ps = sig(lg(z0["p0"]) - 0.66)
    un = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(z0["p0"], 1e-6, None))
    dr = np.array([np.interp(u, QS, q) for u, q in zip(rng.random(len(y)), z0["Q"])])
    worlds.append(np.where(un, np.round(np.clip(dr, 0, None) * cm * s), y))
sc = lambda q, m: (float(np.average(np.abs(y[m] - q[m]) / s[m], weights=w[m])),
                   float(np.mean([np.average(np.abs(W[m] - q[m]) / s[m], weights=w[m]) for W in worlds])))
b = pd.cut(s, [0, 2, 5, 10, 20, 1e9]).astype(str)
rows = []
for bb in sorted(np.unique(b)):
    m = b == bb
    base = sc(p, m)
    r = dict(bucket=bb, w_share=w[m].sum() / w.sum(), base_real=base[0], base_kappa=base[1])
    for t in (0.3, 0.5, 0.7):
        q = np.floor(p) + (p - np.floor(p) > t)
        r[f"round_t{t}_kappa"] = sc(q, m)[1] - base[1]
    rows.append(r)
T = pd.DataFrame(rows)
print("=== change in TW-MASE (kappa world) within bucket when rounding to integers ===\n", T.round(4).to_string(index=False))
print("\ntotal effect on overall TW-MASE (kappa) of rounding with t=0.5 for s <= cutoff:")
for cut in (2, 5, 10, 20):
    q = np.where(s <= cut, np.floor(p) + (p - np.floor(p) > 0.5), p)
    print(f"  s <= {cut:>2}: real {sc(q, slice(None))[0] - sc(p, slice(None))[0]:+.4f} | kappa {sc(q, slice(None))[1] - sc(p, slice(None))[1]:+.4f}")
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(T.bucket, T["round_t0.5_kappa"], color="#2a78d6"); ax.axhline(0, color="k", lw=.6)
ax.set(title="TW-MASE change from integer rounding (kappa world)", xlabel="scale bucket")
fig.savefig(F / "integer.png")
print(f"figures -> {F}")
