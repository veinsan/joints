"""Audit hipotesis tiket berbayar vs admissions yang dipakai okupansi."""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)


def possible_count(numerator, occ):
    implied = 100 * numerator / occ
    lower = np.floor(implied).clip(50, 400)
    upper = np.ceil(implied).clip(50, 400)
    for seats in (lower, upper):
        rebuilt = np.round(100 * numerator / seats, 2)
        valid = np.isclose(rebuilt, occ, atol=1e-9, rtol=0)
        valid &= implied >= 49.5
        valid &= implied <= 400.5
        yield valid


def scan(ticket, occ):
    plus = np.full(len(ticket), -1, dtype=int)
    minus = np.full(len(ticket), -1, dtype=int)
    for k in range(11):
        for sign, target in [(1, plus), (-1, minus)]:
            if sign == -1 and k == 0:
                continue
            numerator = ticket + sign * k
            good = numerator > 0
            a, b = possible_count(numerator, occ)
            found = good & (a | b) & (target == -1)
            target[found] = k
    return plus, minus


summary = {}
for period, filename in [("train", "train.csv"), ("test", "test_history.csv")]:
    x = pd.read_csv(ROOT / "data" / filename)
    x = x[x.total_show.eq(1) & x.total_ticket.between(1, 400)
          & x.occupation_rate.between(10, 99.995, inclusive="left")].copy()
    ticket = x.total_ticket.to_numpy()
    occ = x.occupation_rate.to_numpy()
    m = len(x)
    plus, minus = scan(ticket, occ)
    same = plus == 0
    plus_only = (plus > 0) & (minus == -1)
    minus_only = (minus > 0) & (plus == -1)
    both = (plus > 0) & (minus > 0)
    neither = (plus == -1) & (minus == -1)
    summary[period] = {
        "n": m,
        "exact_without_gap": int(same.sum()),
        "positive_gap_only_le10": int(plus_only.sum()),
        "negative_gap_only_le10": int(minus_only.sum()),
        "both_signs_le10": int(both.sum()),
        "neither_sign_le10": int(neither.sum()),
        "median_min_positive_gap_where_possible": float(np.median(plus[plus > 0])) if (plus > 0).any() else None,
        "median_min_negative_gap_where_possible": float(np.median(minus[minus > 0])) if (minus > 0).any() else None,
    }
    observed_bias = (plus_only.sum() - minus_only.sum()) / m
    rng = np.random.default_rng(20261005)
    buckets = pd.qcut(ticket, 10, labels=False, duplicates="drop")
    shuffled_bias = []
    for _ in range(100):
        shuffled = occ.copy()
        for bucket in np.unique(buckets):
            idx = np.flatnonzero(buckets == bucket)
            shuffled[idx] = rng.permutation(shuffled[idx])
        p_null, n_null = scan(ticket, shuffled)
        shuffled_bias.append((((p_null > 0) & (n_null == -1)).sum()
                              - ((n_null > 0) & (p_null == -1)).sum()) / m)
    summary[period]["observed_positive_only_minus_negative_only_share"] = float(observed_bias)
    summary[period]["shuffled_bias_mean"] = float(np.mean(shuffled_bias))
    summary[period]["shuffled_bias_q025_q975"] = [float(v) for v in np.quantile(shuffled_bias, [.025, .975])]
    out = x[["date_show", "cinema_ids", "movie_title", "total_ticket", "occupation_rate"]].copy()
    out["min_added_admissions"] = plus
    out["min_removed_admissions"] = minus
    out.to_csv(OUT / f"one_show_gap_{period}.csv", index=False)

(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
