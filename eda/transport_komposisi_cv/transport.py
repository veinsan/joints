"""EDA diagnostik: timbang ulang galat OOF lama menurut komposisi skala test, tanpa fitting model."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
pair = pd.read_csv(ROOT / "eda/pergeseran_skala_mase/output/pairs.csv")
test_scale = pair[pair.period == "test"].scale
bins = [0, 1, 3, 10, 20, 50, 100, 300, 1000, np.inf]
labels = ["1", "1-3", "3-10", "10-20", "20-50", "50-100", "100-300", "300-1000", ">1000"]
test_bin = pd.cut(test_scale, bins, labels=labels, include_lowest=True)
test_mix = test_bin.value_counts(normalize=True).reindex(labels, fill_value=0)

paths = {
    "xgb_locked": ROOT / "temp/exp_045_xgb_lock/output/final_legs_v1/oof.parquet",
    "tabpfn_e9": ROOT / "temp/exp_067_tabpfn_local/output/tabpfn35_median_e9surv/oof.parquet",
}
rows = []
details = []
rng = np.random.default_rng(2026)
for name, path in paths.items():
    o = pd.read_parquet(path)
    needed = {"y", "pred", "scale"}
    if not needed.issubset(o.columns):
        raise ValueError(f"{name}: kolom {sorted(o.columns)}")
    o["err"] = (o.y - o.pred).abs() / o.scale
    o["scale_bin"] = pd.cut(o.scale, bins, labels=labels, include_lowest=True)
    group = o.groupby("scale_bin", observed=True).err.agg(["size", "mean"]).reindex(labels)
    group["model"] = name
    group["test_share"] = test_mix
    group["train_share"] = group["size"] / len(o)
    details.append(group.reset_index())
    valid = group["mean"].notna()
    transported = float((group.loc[valid, "mean"] * test_mix.loc[valid]).sum() / test_mix.loc[valid].sum())
    o["base"] = o.movie_title.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True)
    film_bucket = o.groupby(["base", "scale_bin"], observed=True).err.agg(["size", "sum"]).reset_index()
    bases = pd.Index(film_bucket.base.unique())
    nmat = np.zeros((len(bases), len(labels)))
    emat = np.zeros_like(nmat)
    rr = bases.get_indexer(film_bucket.base)
    cc = pd.Index(labels).get_indexer(film_bucket.scale_bin.astype(str))
    nmat[rr, cc] = film_bucket["size"]
    emat[rr, cc] = film_bucket["sum"]
    deltas = []
    for _ in range(1000):
        draw = rng.integers(0, len(bases), len(bases))
        mult = np.bincount(draw, minlength=len(bases))
        n = mult @ nmat
        e = mult @ emat
        ok = n > 0
        mix = test_mix.to_numpy()[ok]
        trans = np.sum(mix * (e[ok] / n[ok])) / mix.sum()
        ordinary = e.sum() / n.sum()
        deltas.append(trans - ordinary)
    ci = np.quantile(deltas, [0.025, 0.975]).tolist()
    rows.append({"model": name, "oof_rows": len(o), "oof_mase": float(o.err.mean()),
        "scale_mix_reweighted_mase": transported, "difference": transported - float(o.err.mean()),
        "film_bootstrap_delta_ci_low": ci[0], "film_bootstrap_delta_ci_high": ci[1],
        "test_mix_coverage": float(test_mix.loc[valid].sum())})
summary = pd.DataFrame(rows)
detail = pd.concat(details, ignore_index=True)
summary.to_csv(OUT / "summary.csv", index=False)
detail.to_csv(OUT / "by_scale.csv", index=False)
print("summary\n", summary.round(4).to_string(index=False))
print("\nby scale\n", detail.round(4).to_string(index=False))
