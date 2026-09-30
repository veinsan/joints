"""26 - What ELSE is shifted in the test period? Full calibration audit of the train-fitted D3 proxy model.

v4 confirmed on the LB that a shift measured on test-period labels (pull odds at D3) carries to D4-D10
(v3 0.4573 -> v4 0.4481, matching a world where ~half of the D3 odds shift applies). About 0.04 of the
offset is still unexplained, so the same instrument is pointed at the remaining parts of the distribution:
  1. PIT of the positive part: where do true test-period values fall in the train-fitted quantiles of r|r>0?
     (uniform = calibrated; mass above 0.5 = films hold better than train expects)
  2. the level ratio true / predicted median, by D3 weekday, month, scale, film size
  3. which features explain the shift (residual model on test-period rows)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from common import OUT, base_title, fig_dir, style

plt = style()
F = fig_dir("26_regime")
A = pd.read_parquet(OUT / "cache" / "proxy_A.parquet").reset_index(drop=True)
B = pd.read_parquet(OUT / "cache" / "proxy_B.parquet").reset_index(drop=True)
cols = ["q1", "q2", "log_s", "sh1", "sh2", "sh_tr", "occ2", "f_tr", "f_log", "f_nc", "cm", "d1_dow"]
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, verbose=-1,
         deterministic=True, force_row_wise=True, random_state=2026)
rr = lambda X: (X.y / X.s / X.cm).values
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def quantiles(Xa, Xb):
    pos = Xa[Xa.y > 0]
    return np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=q, **P).fit(pos[cols], rr(pos)).predict(Xb[cols])
                                    for q in QS]), axis=1)


QA = np.zeros((len(A), len(QS)))
for a, b in GroupKFold(5).split(A, groups=base_title(A.movie_title)):
    QA[b] = quantiles(A.iloc[a], A.iloc[b])
QB = quantiles(A, B)
np.savez(OUT / "cache" / "proxy_quantiles.npz", QA=QA, QB=QB)


def pit(X, Q):
    r = rr(X)
    return np.array([np.interp(ri, qi, QS, left=0.0, right=1.0) for ri, qi in zip(r, Q)])


for X, Q in ((A, QA), (B, QB)):
    X["pit"] = pit(X, Q)
    X["med"] = Q[:, len(QS) // 2]
    X["ratio"] = rr(X) / np.clip(X.med, 1e-3, None)
    X["d3_dow"] = (X.d1_dow + 2) % 7
pa, pb = A[A.y > 0], B[B.y > 0]
print("=== 1. PIT of true positive values in train-fitted quantiles (calibrated: mean 0.5) ===")
print(f"train-sim OOF: mean PIT {pa.pit.mean():.3f}, share above median {np.mean(pa.pit > .5):.3f}")
print(f"test period  : mean PIT {pb.pit.mean():.3f}, share above median {np.mean(pb.pit > .5):.3f}")
print(f"median ratio true/pred-median: train {pa.ratio.median():.3f} | test {pb.ratio.median():.3f}")

print("\n=== 2. where: median ratio true / predicted median (positive rows) ===")
t1 = pd.DataFrame({"train": pa.groupby("d3_dow").ratio.median(), "test": pb.groupby("d3_dow").ratio.median(),
                   "n_test": pb.groupby("d3_dow").size()}).rename(index=dict(enumerate(DOW)))
print("by D3 weekday:\n", t1.round(3))
print("by test month:", pb.groupby("month").ratio.median().round(3).to_dict())
print("by train month:", pa.groupby("month").ratio.median().round(3).to_dict())
bins = [0, 20, 50, 100, 200, 500, 1e9]
t2 = pd.DataFrame({"train": pa.groupby(pd.cut(pa.s, bins), observed=False).ratio.median(),
                   "test": pb.groupby(pd.cut(pb.s, bins), observed=False).ratio.median()})
print("by pair scale:\n", t2.round(3))
fq = pd.qcut(pd.concat([pa.f_log, pb.f_log]), 4)
t3 = pd.DataFrame({"train": pa.ratio.groupby(fq.iloc[: len(pa)].values, observed=False).median(),
                   "test": pb.ratio.groupby(fq.iloc[len(pa):].values, observed=False).median()})
print("by film national volume quartile:\n", t3.round(3))

print("\n=== 3. what explains the test-period residual? (LightGBM on log ratio, test rows only, importance) ===")
xb = pb.assign(lr=np.log(pb.ratio.clip(.05, 20)), mnum=pb.d1.dt.month + 12 * (pb.d1.dt.year - 2025))
fe = cols + ["mnum"]
m = lgb.LGBMRegressor(n_estimators=200, learning_rate=.05, num_leaves=15, min_child_samples=100, verbose=-1, random_state=2026).fit(xb[fe], xb.lr)
imp = pd.Series(m.booster_.feature_importance("gain"), index=fe).sort_values(ascending=False)
print((imp / imp.sum()).round(3).head(8).to_dict())

fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
ax[0].hist(pa.pit, bins=20, alpha=.55, density=True, label="train-sim OOF")
ax[0].hist(pb.pit, bins=20, alpha=.55, density=True, label="test period")
ax[0].axhline(1, color="k", lw=.6); ax[0].set(title="PIT of positive D3 values (flat = calibrated)"); ax[0].legend()
x = np.arange(len(t1))
ax[1].bar(x - .2, t1.train, .4, label="train"); ax[1].bar(x + .2, t1.test, .4, label="test period")
ax[1].axhline(1, color="k", lw=.6); ax[1].set_xticks(x, t1.index); ax[1].set(title="true / predicted median by D3 weekday"); ax[1].legend()
mm = pd.concat([pa.groupby("month").ratio.median(), pb.groupby("month").ratio.median()])
ax[2].bar(mm.index, mm.values, color=["#2a78d6" if i < "2025-10" else "#e34948" for i in mm.index])
ax[2].axhline(1, color="k", lw=.6); ax[2].tick_params(axis="x", rotation=45); ax[2].set(title="true / predicted median by month")
fig.savefig(F / "regime.png")
print(f"figures -> {F}")
