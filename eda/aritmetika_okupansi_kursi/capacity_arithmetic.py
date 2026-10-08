"""Audit aritmetika okupansi, kursi integer, dan kapasitas satu show."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

results = {}
single_parts = []
for period, filename in [("train", "train.csv"), ("test", "test_history.csv")]:
    raw = pd.read_csv(ROOT / "data" / filename)
    x = raw[(raw.total_ticket > 0) & raw.occupation_rate.between(1, 100) & raw.total_show.gt(0)].copy()
    c = 100 * x.total_ticket / x.occupation_rate
    nearest = np.rint(c)
    lower = np.floor(c).clip(lower=1)
    upper = np.ceil(c).clip(lower=1)
    reconstructed_lower = np.round(100 * x.total_ticket / lower, 2)
    reconstructed_upper = np.round(100 * x.total_ticket / upper, 2)
    exact = (np.isclose(reconstructed_lower, x.occupation_rate, atol=1e-9, rtol=0)
             | np.isclose(reconstructed_upper, x.occupation_rate, atol=1e-9, rtol=0))
    x["implied_total_seats"] = c
    x["nearest_seats"] = nearest
    x["integer_reconstructs_occ"] = exact
    x["seat_fraction_distance"] = np.abs(c - nearest)
    x["implied_per_show"] = c / x.total_show
    single = x[x.total_show.eq(1)].copy()
    single["period"] = period
    single_parts.append(single)
    nonfull = single[single.occupation_rate.between(10, 99.995, inclusive="left")]
    results[period] = {
        "all_valid_rows": len(x),
        "share_floor_or_ceil_integer_reconstructs_occupancy": float(exact.mean()),
        "share_seat_fraction_within_0_1": float(x.seat_fraction_distance.le(.1).mean()),
        "single_show_rows": len(single),
        "single_share_reconstructs": float(single.integer_reconstructs_occ.mean()),
        "single_median_implied_seats": float(single.implied_total_seats.median()),
        "single_share_implied_seats_50_to_400": float(single.implied_total_seats.between(50, 400).mean()),
        "single_nonfull_occ_ge10_share_fraction_within_0_1": float(nonfull.seat_fraction_distance.le(.1).mean()),
        "single_nonfull_occ_ge10_median_fraction_distance": float(nonfull.seat_fraction_distance.median()),
        "single_exactly_full_rows": int(single.occupation_rate.eq(100).sum()),
    }
    for threshold in (10, 20, 80):
        subset = single[single.occupation_rate.ge(threshold) & single.occupation_rate.lt(99.995)]
        results[period][f"single_occ_ge{threshold}_rows"] = len(subset)
        results[period][f"single_occ_ge{threshold}_share_reconstructs"] = float(subset.integer_reconstructs_occ.mean())
    single["month"] = pd.to_datetime(single.date_show).dt.to_period("M").astype(str)
    monthly = single[single.occupation_rate.between(10, 99.995, inclusive="left")].groupby("month").agg(
        rows=("total_ticket", "size"), share_reconstructs=("integer_reconstructs_occ", "mean"),
        median_occ=("occupation_rate", "median"), median_tickets=("total_ticket", "median")
    ).reset_index()
    monthly.insert(0, "period", period)
    monthly.to_csv(OUT / f"single_occupancy_month_{period}.csv", index=False)

s = pd.concat(single_parts, ignore_index=True)
s = s[s.integer_reconstructs_occ & s.nearest_seats.between(50, 400)].copy()
cap = s.groupby(["period", "nearest_seats"]).agg(rows=("total_ticket", "size"),
                                                  films=("movie_title", "nunique"),
                                                  clusters=("cinema_ids", "nunique")).reset_index()
cap.sort_values(["period", "rows"], ascending=[True, False]).to_csv(OUT / "single_show_capacity_counts.csv", index=False)
for period in ("train", "test"):
    q = cap[cap.period.eq(period)]
    results[period]["single_top10_capacities"] = q.nlargest(10, "rows")[
        ["nearest_seats", "rows", "films", "clusters"]].to_dict(orient="records")

(OUT / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
