"""EDA murni: uji konsistensi mekanis tiket, okupansi, dan show lintas periode."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
tr = pd.read_csv(ROOT / "data/train.csv")
te = pd.read_csv(ROOT / "data/test_history.csv")
tr["period"] = "train"
te["period"] = "test_history"
df = pd.concat([tr, te], ignore_index=True)
df["seats_implied"] = 100 * df.total_ticket / (df.occupation_rate * df.total_show)
df["tps"] = df.total_ticket / df.total_show
df["capacity_days"] = 100 * df.total_ticket / df.occupation_rate

basic = df.groupby("period").agg(rows=("total_ticket", "size"),
    pct_occ_over100=("occupation_rate", lambda x: (x > 100).mean()),
    pct_occ_zero=("occupation_rate", lambda x: (x == 0).mean()),
    pct_tickets_over_implied_seats=("seats_implied", lambda x: (x < 1).mean()),
    median_implied_seats_per_show=("seats_implied", "median"),
    p01_implied_seats_per_show=("seats_implied", lambda x: x.quantile(.01)),
    p99_implied_seats_per_show=("seats_implied", lambda x: x.quantile(.99)))

# Median kapasitas per show tidak harus konstan: film bisa pindah auditorium.
# Uji konsistensi per klaster dan per periode hanya untuk hari berokupansi memadai.
filtered = df[(df.total_show >= 3) & (df.occupation_rate >= 5)].copy()
cin = filtered.groupby(["period", "cinema_ids"]).seats_implied.agg(["size", "median", "std"])
cin["cv"] = cin["std"] / cin["median"]
overlap = cin.loc["train", ["median", "size"]].join(cin.loc["test_history", ["median", "size"]],
    lsuffix="_train", rsuffix="_test", how="inner")
overlap = overlap[(overlap.size_train >= 30) & (overlap.size_test >= 30)].copy()
overlap["relative_diff"] = overlap.median_test / overlap.median_train - 1
summary = {
    "overlap_clusters": len(overlap),
    "spearman_capacity_train_test": float(overlap.median_train.corr(overlap.median_test, method="spearman")),
    "median_abs_relative_diff": float(overlap.relative_diff.abs().median()),
    "p90_abs_relative_diff": float(overlap.relative_diff.abs().quantile(.9)),
    "within_cluster_median_cv_train": float(cin.loc["train"].cv.median()),
    "within_cluster_median_cv_test": float(cin.loc["test_history"].cv.median()),
    "occ_decimal_places_two_train": float((np.abs(tr.occupation_rate * 100 - np.round(tr.occupation_rate * 100)) < 1e-8).mean()),
    "occ_decimal_places_two_test": float((np.abs(te.occupation_rate * 100 - np.round(te.occupation_rate * 100)) < 1e-8).mean()),
}

basic.to_csv(OUT / "basic.csv")
cin.reset_index().to_csv(OUT / "cinema_capacity.csv", index=False)
overlap.to_csv(OUT / "overlap_capacity.csv")
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("basic\n", basic.round(3).to_string())
print("capacity consistency\n", json.dumps(summary, indent=2))
