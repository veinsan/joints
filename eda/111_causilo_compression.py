"""Can Causilo (148.4 MB float32) sit next to TabPFN-3.5 int7 (169.1 MB) under 200 MB? Weights only.

eda/109: a 0.7 TabPFN-3.5 + 0.3 Causilo median blend improves cohort TW by 0.0014 (CI crosses 0).
Budget without LightGBM (its blend weight is 0 there): 200 - 169.1 - 1.5 margin = 29.4 MB.
Measured for every float tensor: symmetric per-row intN codes + zlib-9, relative L2 error, and size;
tensor dtypes and the share of non-matrix parameters (kept float32) are printed first.
"""
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import fig_dir, style

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/causilo_ckpt/regressor/model.safetensors")


def main():
    out = fig_dir("111_causilo_compression")
    dtypes, other, mats = {}, 0, []
    with safe_open(SRC, framework="np") as f:
        for k in f.keys():
            a = f.get_tensor(k)
            dtypes[str(a.dtype)] = dtypes.get(str(a.dtype), 0) + a.nbytes
            if a.dtype == np.float32 and a.ndim >= 2:
                mats.append(a.reshape(a.shape[0], -1))
            else:
                other += a.nbytes
    print(f"source {SRC.stat().st_size / 1e6:.1f} MB; bytes by dtype {({k: round(v / 1e6, 1) for k, v in dtypes.items()})}; "
          f"non-matrix kept float32 {other / 1e6:.2f} MB; matrices {len(mats)}")
    rows = {}
    for bits in (8, 7, 6, 5, 4):
        lev = 2 ** (bits - 1) - 1
        codes, sc_bytes, e2, r2 = [], 0, 0.0, 0.0
        for m in mats:
            s = (np.abs(m).max(axis=1, keepdims=True) / lev).astype(np.float16).astype(np.float32)
            s[s == 0] = 1.0
            q = np.clip(np.round(m / s), -lev, lev).astype(np.int8)
            e2 += float(np.square(q * s - m.astype(np.float64)).sum())
            r2 += float(np.square(m.astype(np.float64)).sum())
            codes.append(q.ravel())
            sc_bytes += s.size * 2
        zl = len(zlib.compress(np.concatenate(codes).tobytes(), 9))
        rows[f"int{bits}"] = dict(MB=(zl + sc_bytes + other) / 1e6, relative_L2_error=np.sqrt(e2 / r2))
    R = pd.DataFrame(rows).T
    R["total_with_tp35_int7_MB"] = R.MB + 169.14
    R["fits_without_lgb"] = R.total_with_tp35_int7_MB + 1.5 <= 200
    R["fits_with_lgb_20MB"] = R.total_with_tp35_int7_MB + 20.1 + 1.5 <= 200
    pd.set_option("display.width", 200)
    print(R.round(4).to_string())
    R.to_csv(out / "sizes.csv")
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].barh(R.index, R.total_with_tp35_int7_MB, color=["#2a78d6" if f else "#e34948" for f in R.fits_without_lgb])
    ax[0].axvline(198.5, color="k", ls="--", label="200 MB - margin"); ax[0].set(title="TabPFN-3.5 int7 + Causilo intN (MB)"); ax[0].legend()
    ax[1].barh(R.index, R.relative_L2_error); ax[1].set_xscale("log"); ax[1].set(title="Causilo relative L2 weight error")
    fig.tight_layout(); fig.savefig(out / "causilo_compression.png")


if __name__ == "__main__":
    main()
