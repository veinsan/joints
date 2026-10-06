"""43 - Is the test period intrinsically harder, and where? (labelled test-period proxy D1,D2 -> D3)

Under composition-correct weights (scale bucket x first sale day, eda/42) every version's LB sits ~0.065 above
the kappa-world validation, independent of the model. A model-independent offset means part of the test is
harder than anything in train. The proxy (eda/13: model trained on train-period D1,D2 -> D3, applied to the
test period) has real labels in the test period, so per test month / calendar segment:
    excess = real proxy MASE - train-CV proxy MASE re-weighted to the same composition (bucket x started-D2)
A uniform excess = noisier period; an excess concentrated in Ramadan / Christmas = segment problem.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style

plt = style()
F = fig_dir("43_proxy_segments")
A = pd.read_parquet(OUT / "cache" / "proxy_A.parquet")
B = pd.read_parquet(OUT / "cache" / "proxy_B.parquet")
BINS = [0, 5, 20, 50, 100, 200, 500, 1e9]
for D in (A, B):
    D["key"] = pd.cut(D.s, BINS).astype(str) + "|" + (D.q1 == 0).astype(str)
    D["d3"] = D.d1 + pd.Timedelta(days=2)
    D["seg"] = "normal"
    D.loc[D.d3.between("2025-12-20", "2026-01-04"), "seg"] = "xmas"
    D.loc[D.d3.between("2026-02-19", "2026-03-20"), "seg"] = "ramadan"
    D["zero"] = D.y == 0
cellA = A.groupby("key").err.mean()
cellA_naive = A.groupby("key").naive.mean()
B["exp_err"] = B.key.map(cellA)
B["exp_naive"] = B.key.map(cellA_naive)
print(f"A (train period) proxy MASE {A.err.mean():.4f} | B real {B.err.mean():.4f} | B expected from A cells {B.exp_err.mean():.4f}")
print(f"naive (y3 = mean(y1,y2) x cal): A {A.naive.mean():.4f} | B real {B.naive.mean():.4f} | B expected {B.exp_naive.mean():.4f}")
for col in ("month", "seg"):
    T = B.groupby(col).agg(rows=("err", "size"), real=("err", "mean"), expected=("exp_err", "mean"), naive_real=("naive", "mean"),
                           naive_exp=("exp_naive", "mean"), zero=("zero", "mean"), bias=("bias", "mean"))
    T["excess"] = T.real - T.expected
    T["naive_excess"] = T.naive_real - T.naive_exp
    print(f"\n=== by {col} ===\n", T.round(4).to_string())
T2 = B.groupby(pd.cut(B.s, BINS)).agg(real=("err", "mean"), expected=("exp_err", "mean"), zero=("zero", "mean"), n=("err", "size"))
print("\n=== by scale bucket ===\n", T2.round(4))
fig, ax = plt.subplots(figsize=(9, 3.6))
M = B.groupby("month").agg(real=("err", "mean"), expected=("exp_err", "mean"))
ax.plot(M.index.astype(str), M.real, marker="o", label="real test-period proxy MASE")
ax.plot(M.index.astype(str), M.expected, marker="s", label="expected from train (same composition)")
ax.set(title="Test-period proxy (D1,D2 -> D3): real vs expected"); ax.legend()
fig.savefig(F / "proxy_month.png")
print(f"figures -> {F}")
