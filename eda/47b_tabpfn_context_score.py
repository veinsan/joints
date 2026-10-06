"""47b - Score the eda/47 context-size A/B (run with .venv): TabPFN-3.5 per-horizon vs neighbour vs pooled."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT
from evaluate import folds, kappa_worlds, test_weights_fd
from features import dataset

C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
w = test_weights_fd(Xtr, Xte)
z0 = np.load(C / "hurdle_oof.npz")
W = kappa_worlds(y, s, cm, z0["p0"], z0["Q"])
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
TQS = np.round(np.arange(0.02, 1.0, 0.02), 2)
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def tfm_median(TQ, lam=0.5, eps=0.02):
    tp0 = np.array([np.interp(eps, q, TQS, left=0.0, right=1.0) for q in TQ])
    u = tp0[:, None] + (1 - tp0[:, None]) * QS[None, :]
    tq = np.clip(np.array([np.interp(ui, TQS, qi) for ui, qi in zip(u, TQ)]), 0, None)
    pp = sig(lg(tp0) - 1.32 * lam)
    t = np.clip((0.5 - pp) / (1 - pp + 1e-9), QS[0], QS[-1])
    return np.where(pp >= 0.5, 0.0, np.clip([np.interp(ti, QS, qi) for ti, qi in zip(t, tq)], 0, None))


fold = folds(Xtr)
rows = []
for hz in (4, 8):
    idx = np.where((fold == 0) & (Xtr.h.values == hz))[0]
    for name in ("per-horizon", "neighbour", "pooled"):
        p = tfm_median(np.load(C / f"tabctx_h{hz}_{name}.npy")) * cm[idx] * s[idx]
        real = np.average(np.abs(y[idx] - p) / s[idx], weights=w[idx])
        kap = np.mean([np.average(np.abs(t[idx] - p) / s[idx], weights=w[idx]) for t in W])
        rows.append(dict(h=hz, context=name, real=real, kappa=kap))
R = pd.DataFrame(rows)
print(R.round(4).to_string(index=False))
print("\nmean over h:\n", R.groupby("context")[["real", "kappa"]].mean().round(4))

print("\n=== average of per-horizon and pooled medians ===")
for hz in (4, 8):
    idx = np.where((fold == 0) & (Xtr.h.values == hz))[0]
    a = tfm_median(np.load(C / f"tabctx_h{hz}_per-horizon.npy")) * cm[idx] * s[idx]
    b = tfm_median(np.load(C / f"tabctx_h{hz}_pooled.npy")) * cm[idx] * s[idx]
    for wt in (0.0, 0.3, 0.5):
        p = (1 - wt) * a + wt * b
        print(f"h={hz} w_pooled {wt}: real {np.average(np.abs(y[idx] - p) / s[idx], weights=w[idx]):.4f} | "
              f"kappa {np.mean([np.average(np.abs(t[idx] - p) / s[idx], weights=w[idx]) for t in W]):.4f}")
