"""Does entropy coding make int8/int7/int6 TabPFN-3.5 fit the budget? Weights only.

int8 per-row codes of Gaussian-like weights use far fewer than 8 bits of information. Measured:
Shannon entropy of the codes, and the real compressed size with lzma (preset 9e) and zlib (9), for
the whole checkpoint. Decompression is exact, so the stored file reproduces the int8 weights bit for
bit (relative L2 error 0.0102 from eda/92). Also: int8 with a per-row scale vs per-group-128 scale.
"""
import lzma
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import fig_dir, style

SRC = Path.home() / ".cache/tabpfn/tabpfn-v3.5-20260909.safetensors"


def per_row_codes(bits):
    """Symmetric per-row codes with 2**(bits-1)-1 levels each side; returns (codes, scale bytes, other bytes, rel err)."""
    lev = 2 ** (bits - 1) - 1
    codes, scales, other, e2, r2 = [], 0, 0, 0.0, 0.0
    with safe_open(SRC, framework="np") as f:
        for name in f.keys():
            a = f.get_tensor(name)
            if not (a.dtype == np.float32 and a.ndim >= 2):
                other += a.nbytes
                continue
            m = a.reshape(a.shape[0], -1)
            s = (np.abs(m).max(axis=1, keepdims=True) / lev).astype(np.float16).astype(np.float32)
            s[s == 0] = 1.0
            q = np.clip(np.round(m / s), -lev, lev).astype(np.int8)
            e2 += float(np.square(q * s - m.astype(np.float64)).sum())
            r2 += float(np.square(m.astype(np.float64)).sum())
            codes.append(q.ravel())
            scales += s.size * 2
    return np.concatenate(codes), scales, other, np.sqrt(e2 / r2)


def main_bits():
    out = fig_dir("93_int8_entropy")
    rows = {}
    for bits in (8, 7, 6, 5):
        c, sc, ot, err = per_row_codes(bits)
        zl = len(zlib.compress(c.tobytes(), 9))
        rows[f"int{bits} per-row + zlib"] = dict(MB=(zl + sc + ot) / 1e6, relative_L2_error=err)
        print(f"int{bits}: {rows[f'int{bits} per-row + zlib']}", flush=True)
    R = pd.DataFrame(rows).T
    R["fits_167_with_TabM"] = R.MB <= 167
    R["fits_178_without_TabM"] = R.MB <= 178
    print(R.round(4).to_string())
    R.to_csv(out / "bits.csv")
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    ax[0].barh(R.index, R.MB, color=["#2a78d6" if f else "#e34948" for f in R.fits_167_with_TabM]); ax[0].axvline(167, color="k", ls="--")
    ax[0].set(title="Stored size (MB)")
    ax[1].barh(R.index, R.relative_L2_error); ax[1].set_xscale("log"); ax[1].set(title="Relative L2 weight error")
    fig.tight_layout(); fig.savefig(out / "bits.png")


def main():
    out = fig_dir("93_int8_entropy")
    codes, scales, other = [], 0, 0
    ent_bits, n = 0.0, 0
    with safe_open(SRC, framework="np") as f:
        for name in f.keys():
            a = f.get_tensor(name)
            if not (a.dtype == np.float32 and a.ndim >= 2):
                other += a.nbytes
                continue
            m = a.reshape(a.shape[0], -1)
            s = np.abs(m).max(axis=1, keepdims=True) / 127.0
            s[s == 0] = 1.0
            q = np.clip(np.round(m / s), -127, 127).astype(np.int8)
            codes.append(q.ravel())
            scales += s.size * 2
            p = np.bincount((q.ravel().astype(np.int16) + 127), minlength=255) / q.size
            p = p[p > 0]
            ent_bits += -(p * np.log2(p)).sum() * q.size
            n += q.size
    allc = np.concatenate(codes).tobytes()
    raw = len(allc)
    zl = len(zlib.compress(allc, 9))
    lz = len(lzma.compress(allc, preset=9 | lzma.PRESET_EXTREME))
    R = pd.DataFrame({
        "int8 raw": dict(MB=(raw + scales + other) / 1e6),
        "int8 + zlib-9": dict(MB=(zl + scales + other) / 1e6),
        "int8 + lzma-9e": dict(MB=(lz + scales + other) / 1e6),
        "entropy bound": dict(MB=(ent_bits / 8 + scales + other) / 1e6)}).T
    R["fits_167"] = R.MB <= 167
    R["fits_175_without_TabM"] = R.MB <= 175
    print(f"matrix weights {n / 1e6:.1f} M, mean code entropy {ent_bits / n:.2f} bits (of 8)")
    print(R.round(1).to_string())
    R.to_csv(out / "sizes.csv")
    plt = style()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.barh(R.index, R.MB, color=["#2a78d6" if f else "#e34948" for f in R.fits_167])
    ax.axvline(167, color="k", ls="--", label="FM budget with TabM"); ax.axvline(175, color="#8a8984", ls=":", label="without TabM")
    ax.set(title="TabPFN-3.5 full, int8 per-row: stored size", xlabel="MB"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "int8_entropy.png")


if __name__ == "__main__":
    main_bits() if len(sys.argv) > 1 and sys.argv[1] == "bits" else main()
