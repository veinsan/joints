"""EDA kontinuitas kapasitas tersirat D1-D3 vs kontrol klaster."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "temp/exp_020_data_fix"))
from build_windows_v2 import build_windows, test_windows  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
tw, _, _ = build_windows(tr)
vw = test_windows(th)

summary = {}
rng = np.random.default_rng(20261005)
for period, wide in [("train", tw), ("test", vw)]:
    x = wide.copy()
    valid = np.ones(len(x), dtype=bool)
    for d in (1, 3):
        valid &= x[f"total_ticket_d{d}"].ge(20).to_numpy()
        valid &= x[f"total_show_d{d}"].ge(2).to_numpy()
        valid &= x[f"occupation_rate_d{d}"].between(5, 99.99).to_numpy()
        x[f"cap_d{d}"] = 100 * x[f"total_ticket_d{d}"] / (
            x[f"total_show_d{d}"] * x[f"occupation_rate_d{d}"])
        valid &= x[f"cap_d{d}"].between(50, 400).to_numpy()
    x = x.loc[valid].copy()
    x["abs_log_cap_ratio"] = np.abs(np.log(x.cap_d3 / x.cap_d1))
    observed = float(x.abs_log_cap_ratio.median())
    null = []
    null_show = []
    indices = x.groupby("cinema_ids").indices
    x["d3_show_bucket"] = pd.cut(x.total_show_d3, [1, 3, 7, 15, np.inf], labels=False)
    indices_show = x.groupby(["cinema_ids", "d3_show_bucket"]).indices
    d1 = x.cap_d1.to_numpy()
    d3 = x.cap_d3.to_numpy()
    for _ in range(100):
        shuffled = d3.copy()
        for idx in indices.values():
            shuffled[idx] = rng.permutation(shuffled[idx])
        null.append(float(np.median(np.abs(np.log(shuffled / d1)))))
        shuffled_show = d3.copy()
        for idx in indices_show.values():
            shuffled_show[idx] = rng.permutation(shuffled_show[idx])
        null_show.append(float(np.median(np.abs(np.log(shuffled_show / d1)))))
    x["cap_change_pct"] = 100 * (x.cap_d3 / x.cap_d1 - 1)
    x["show_change_pct"] = 100 * (x.total_show_d3 / x.total_show_d1 - 1)
    bins = pd.cut(x.show_change_pct, [-101, -50, -10, 10, 50, np.inf],
                  labels=["cut_ge50", "cut_10_50", "stable_10", "up_10_50", "up_gt50"])
    by_show = x.groupby(bins, observed=True).agg(
        pairs=("cap_change_pct", "size"), median_cap_change_pct=("cap_change_pct", "median"),
        median_abs_log_cap_ratio=("abs_log_cap_ratio", "median")
    ).reset_index()
    by_show.to_csv(OUT / f"by_show_{period}.csv", index=False)
    summary[period] = {
        "n_valid_pairs": len(x),
        "share_all_pairs_valid": float(len(x) / len(wide)),
        "median_abs_log_cap_ratio_observed": observed,
        "median_abs_log_cap_ratio_shuffled_mean": float(np.mean(null)),
        "median_abs_log_cap_ratio_shuffled_q025_q975": [float(v) for v in np.quantile(null, [.025, .975])],
        "median_abs_log_cap_ratio_shuffled_same_cinema_show_mean": float(np.mean(null_show)),
        "median_abs_log_cap_ratio_shuffled_same_cinema_show_q025_q975": [
            float(v) for v in np.quantile(null_show, [.025, .975])],
        "median_cap_change_pct": float(x.cap_change_pct.median()),
        "share_cap_change_abs_le10pct": float(x.cap_change_pct.abs().le(10).mean()),
    }

(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
