"""37 - Where is the remaining error? Oracle decomposition of the v6-style blend (LightGBM mix + TabPFN-3.5, 50/50).

Each oracle replaces ONE piece of the prediction with the truth and measures how much TW-MASE (real and
kappa=0.5 world) would drop. The biggest drop = the biggest lever left:
  film oracle     : true film-level (base title x horizon) ratio sum(y)/sum(pred) applied to every pair
  film-day oracle : same at film x date (national daily trajectory, incl. calendar)
  zero oracle     : true zero/non-zero known (pred -> 0 on true zeros, rescaled median of positives otherwise)
  cluster oracle  : true cluster x date ratio (cinema-day demand)
Also: error share by scale bucket / horizon / true-zero rows.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, base_title, fig_dir, style
from evaluate import test_weights
from features import dataset

plt = style()
F = fig_dir("37_oracles")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
TQS = np.round(np.arange(0.02, 1.0, 0.02), 2)
A_SHIFT, LAM, HW, EPS = -1.32, 0.5, 0.75, 0.02
w_te = test_weights(Xtr, Xte)
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


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
pred = 0.5 * ((1 - HW) * z6["l1"] + HW * mixmed(z6["p0"], z6["Q"], QS, LAM)) * cm * s \
     + 0.5 * tfm_median(np.load(C / "tab35_oofQ.npy"), LAM) * cm * s
tw = lambda p, m=slice(None): float(np.average(np.abs(y[m] - p[m]) / s[m], weights=w_te[m]))
e = np.abs(y - pred) / s * w_te / w_te.sum()

print(f"blend TW-MASE (train truth): {tw(pred):.4f}")
print("\n=== error share ===")
b = pd.cut(s, [0, 5, 20, 50, 100, 200, 500, 1e9]).astype(str)
D = pd.DataFrame({"b": b, "h": Xtr.h.values, "zero": y == 0, "e": e, "w": w_te / w_te.sum()})
for k in ["b", "h", "zero"]:
    T = D.groupby(k).agg(err=("e", "sum"), wt=("w", "sum"))
    T["mase_in_group"] = T.err / T.wt
    print(T.round(4))

g = base_title(Xtr.movie_title).values
keys = {"film x h": pd.Series(g) + "|" + Xtr.h.astype(str).values,
        "film x date": pd.Series(g) + "|" + Xtr.date_show.astype(str).values,
        "cluster x date": Xtr.cinema_ids.astype(str).values + "|" + Xtr.date_show.astype(str).values}
res = {"blend": tw(pred)}
for name, k in keys.items():
    k = pd.Series(np.asarray(k))
    ratio = (pd.Series(y).groupby(k).transform("sum") / pd.Series(pred).groupby(k).transform("sum").clip(lower=1e-9)).values
    res[f"{name} oracle"] = tw(pred * ratio)
pz = np.where(y == 0, 0, np.where(pred == 0, np.median(y[y > 0] / s[y > 0]) * s, pred))
res["zero oracle"] = tw(pz)
pz2 = np.where(y == 0, 0, pred)
res["zero oracle (only zero out)"] = tw(pz2)
R = pd.Series(res)
print("\n=== oracle TW-MASE (lower = bigger lever) ===\n", R.round(4))

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(R.index, R.values, color="#2a78d6"); ax.set(title="TW-MASE if one piece were known", xlim=(0, R.max() * 1.05))
fig.savefig(F / "oracles.png")
print(f"figures -> {F}")
