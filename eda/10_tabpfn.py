"""10 - Pretrained tabular foundation model: TabPFN v2 (Prior-Labs/TabPFN-v2-reg, Nature Jan 2025).

Why this one: every series has only 3 observed points, so time-series foundation models
(Chronos-Bolt, TimesFM, Moirai) have nothing to condition on. The signal lives in cross-sectional
features -> a tabular FM that outputs a full predictive distribution; its MEDIAN is the
MAE/MASE-optimal point forecast. Tested on the temporal split (train D1 < Aug, validate >= Aug).
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor
from common import OUT, fig_dir, load, release_dates, simulate, style
from features import build, feature_cols, make_ctx
from model import LEVEL, NEVER, fit_predict, target

plt = style()
F = fig_dir("10_tabpfn")
torch.manual_seed(2026)
d = load()
tr = d["train"]
first = tr.groupby("movie_title").date_show.min()
D1 = release_dates(tr).drop(first[first == first.min()].index, errors="ignore")
hs, ts = simulate(tr, D1)
X = build(hs, ts, make_ctx(hs, d["hol"], d["movies"], d["price"], tr))
cols = [c for c in feature_cols(X) if c not in NEVER and (c not in LEVEL or c in ("log_s", "f_logT"))]
va = (X.d1 >= "2025-08-01").values
Xa, Xb = X[~va], X[va]
# ponytail: CPU TabPFN is slow -> score on a fixed 5k-row subsample of the temporal fold (full fold on GPU in the notebook)
Xb = Xb.sample(5000, random_state=2026)
M = lambda p: float(np.mean(np.abs(Xb.total_ticket.values - p) / Xb.scale.values))

p_lgb, _ = fit_predict(Xa, Xb, cols, True, seeds=(2026, 2027, 2028), n_estimators=600)
print(f"LightGBM (3 seeds)            temporal MASE = {M(p_lgb):.4f}")

rng = np.random.RandomState(2026)
res = {}
for n_ctx in [5000]:  # 10k context runs on the T4 in the notebook
    idx = rng.choice(len(Xa), n_ctx, replace=False)
    t0 = time.time()
    m = TabPFNRegressor(device="cuda" if torch.cuda.is_available() else "cpu", n_estimators=4, random_state=2026,
                        ignore_pretraining_limits=True)
    m.fit(Xa.iloc[idx][cols].values.astype(np.float32), target(Xa.iloc[idx]).values)
    q = np.concatenate([m.predict(Xb[cols].values[i:i + 4000].astype(np.float32), output_type="median")
                        for i in range(0, len(Xb), 4000)])
    p = np.clip(q, 0, None) * Xb.scale.values * Xb.cal_mult.values
    res[n_ctx] = p
    print(f"TabPFN v2 median, ctx={n_ctx:5d}  temporal MASE = {M(p):.4f}   ({time.time() - t0:.0f}s)")

p_tab = res[5000]
for w in [0.3, 0.5, 0.6, 0.7, 0.8]:
    print(f"blend {1 - w:.1f}*LGB + {w:.1f}*TabPFN          temporal MASE = {M((1 - w) * p_lgb + w * p_tab):.4f}")
print("corr of normalised preds:", round(np.corrcoef(p_lgb / Xb.scale, p_tab / Xb.scale)[0, 1], 3))
pd.DataFrame({"lgb": p_lgb, "tabpfn": p_tab}).to_parquet(OUT / "temporal_tabpfn.parquet")

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.scatter(p_lgb / Xb.scale, p_tab / Xb.scale, s=2, alpha=.2, color="#2a78d6")
ax.plot([0, 3], [0, 3], color="#e34948", lw=1)
ax.set(xlim=(0, 3), ylim=(0, 3), title="LightGBM vs TabPFN (pred / scale)", xlabel="LightGBM", ylabel="TabPFN")
fig.savefig(F / "tabpfn_vs_lgb.png")
print(f"figures -> {F}")
