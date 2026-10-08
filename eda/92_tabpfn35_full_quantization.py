"""Can the full 24-layer TabPFN-3.5 (best single model, v8 TW 0.3215) fit the 200 MB rule? Weights only.

The checkpoint is 876 MB float32. Budget for the foundation model: 200 - LightGBM (~25 MB) - TabM
(~6 MB) - margin (~2 MB) = ~167 MB. Schemes for every float32 tensor with ndim >= 2 (vectors stay fp32):
  fp16        : 2 bytes / weight
  int8 / row  : symmetric per output row, 1 byte + fp16 scale per row
  int4 / g64  : symmetric per group of 64 along the input axis, packed 2 per byte + fp16 scale per group
  int4 / g32  : same with groups of 32
Reported: stored size, relative L2 error of all matrix weights, worst tensors, and error vs the FP16
Fast conversion already validated on Kaggle (v13/v14). No inference is run; prediction impact must be
measured on Kaggle (fp32 vs quantized on the same fold) before the quantized file is trusted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import fig_dir, style

SRC = Path.home() / ".cache/tabpfn/tabpfn-v3.5-20260909.safetensors"


def q_int8_row(a):
    s = np.abs(a).max(axis=1, keepdims=True) / 127.0
    s[s == 0] = 1.0
    q = np.clip(np.round(a / s), -127, 127)
    return q * s.astype(np.float16).astype(np.float32), a.size * 1 + s.size * 2


def q_int4_group(a, g):
    rows, cols = a.shape
    pad = (-cols) % g
    b = np.pad(a, ((0, 0), (0, pad))).reshape(rows, -1, g)
    s = (np.abs(b).max(axis=2, keepdims=True) / 7.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1.0
    q = np.clip(np.round(b / s), -8, 7)
    deq = (q * s).reshape(rows, -1)[:, :cols]
    return deq, (rows * (cols + pad)) // 2 + s.size * 2


def main():
    out = fig_dir("92_tabpfn35_quant")
    rows, tot = [], {"fp32": 0, "vectors_fp32": 0}
    schemes = ["fp16", "int8_row", "int4_g64", "int4_g32"]
    err = {k: 0.0 for k in schemes}
    size = {k: 0 for k in schemes}
    ref = 0.0
    with safe_open(SRC, framework="np") as f:
        for name in f.keys():
            a = f.get_tensor(name)
            tot["fp32"] += a.nbytes
            if not (a.dtype == np.float32 and a.ndim >= 2):
                tot["vectors_fp32"] += a.nbytes
                continue
            m = a.reshape(a.shape[0], -1).astype(np.float32)
            r2 = float(np.square(m.astype(np.float64)).sum())
            ref += r2
            rec = {"fp16": (m.astype(np.float16).astype(np.float32), m.size * 2), "int8_row": q_int8_row(m),
                   "int4_g64": q_int4_group(m, 64), "int4_g32": q_int4_group(m, 32)}
            row = dict(name=name, shape=str(a.shape), MB=a.nbytes / 1e6)
            for k, (d, b) in rec.items():
                e = float(np.square(d.astype(np.float64) - m).sum())
                err[k] += e
                size[k] += b
                row[f"relerr_{k}"] = np.sqrt(e / max(r2, 1e-30))
            rows.append(row)
    T = pd.DataFrame(rows)
    T.to_csv(out / "tensors.csv", index=False)
    vec = tot["vectors_fp32"]
    S = pd.DataFrame({k: dict(stored_MB=(size[k] + vec) / 1e6, relative_L2_error=np.sqrt(err[k] / ref)) for k in schemes}).T
    S["fits_167MB"] = S.stored_MB <= 167
    pd.set_option("display.width", 200)
    print(f"source {tot['fp32'] / 1e6:.1f} MB float32; matrices {len(T)}, vectors/other {vec / 1e6:.1f} MB kept fp32")
    print(S.round(5).to_string())
    print("\nreference: Fast (8 layers) matrix-FP16, relative L2 error 0.000172, validated on Kaggle (v13 TW 0.3252 at lambda 0 in v14)")
    print("\nworst tensors under int4_g64:")
    print(T.sort_values("relerr_int4_g64", ascending=False).head(8)[["name", "shape", "MB", "relerr_int8_row", "relerr_int4_g64", "relerr_int4_g32"]].round(4).to_string(index=False))
    # mixed scheme: int8 for the most sensitive tensors (embeddings/encoders/decoders, small), int4_g64 for the rest
    sens = T.name.str.contains("encoder|decoder|embed|output|head", case=False) | (T.MB < 1.0)
    mixed_bytes = vec + sum((T.MB[sens] * 1e6 / 4 * 1).values) + sum((T.MB[~sens] * 1e6 / 4 * 0.5 * (1 + 4 / 64)).values)
    print(f"\nmixed int8 (sensitive/small, {int(sens.sum())} tensors, {T.MB[sens].sum():.1f} MB fp32) + int4_g64 (rest): ~{mixed_bytes / 1e6:.1f} MB")
    S.to_csv(out / "schemes.csv")
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].bar(S.index, S.stored_MB, color=["#2a78d6" if f else "#e34948" for f in S.fits_167MB])
    ax[0].axhline(167, color="k", ls="--", label="budget for the FM (~167 MB)"); ax[0].set(title="Stored size, TabPFN-3.5 full"); ax[0].legend()
    ax[1].bar(S.index, S.relative_L2_error); ax[1].axhline(0.000172, color="#e34948", ls="--", label="Fast FP16 (validated)")
    ax[1].set_yscale("log"); ax[1].set(title="Relative L2 weight error"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "quantization.png")


if __name__ == "__main__":
    main()
