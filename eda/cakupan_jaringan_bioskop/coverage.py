"""EDA murni: bandingkan agregat tiket internal dengan tonggak penonton publik bertanggal."""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
titles = tr.groupby("movie_title").total_ticket.sum().sort_values(ascending=False)
monthly = tr.groupby(tr.date_show.dt.to_period("M")).total_ticket.sum()
titles.head(30).to_csv(OUT / "top_titles.csv", header=["train_tickets"])
monthly.to_csv(OUT / "monthly_tickets.csv", header=["tickets"])
coverage = {
    "dataset_april_tickets": int(monthly.loc[pd.Period("2025-04")]),
    "dataset_q2_tickets": int(monthly.loc[pd.Period("2025-04")] + monthly.loc[pd.Period("2025-05")] + monthly.loc[pd.Period("2025-06")]),
    "cinema_xxi_april_public_claim": "lebih dari 14 juta admissions, rilis perusahaan 2 Mei 2025",
    "cinema_xxi_h1_public_claim": "42,5 juta admissions, Q2 lebih dari 2 kali Q1, rilis perusahaan 28 Juli 2025",
}
(OUT / "benchmark_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
queries = [
    ("JUMBO", "2025-06-02", 10073332, "Katadata 2 Jun 2025, 10.073.332 penonton"),
    ("SORE ISTRI DARI MASA DEPAN", "2025-07-24", 1700000, "Kemenekraf 24 Jul 2025, lebih dari 1,7 juta"),
    ("PABRIK GULA", "2025-04-07", 2600000, "Cinepoint dikutip Databoks 7 Apr 2025, hampir 2,6 juta"),
]
rows = []
for name, date, public, source in queries:
    match = tr.movie_title.str.contains(name, case=False, regex=False)
    sub = tr[match & tr.date_show.le(pd.Timestamp(date))]
    rows.append({"query": name, "matched_titles": "; ".join(sorted(sub.movie_title.unique())),
        "date": date, "dataset_tickets_from_apr1": int(sub.total_ticket.sum()),
        "public_admissions_approx": public, "dataset_to_public_ratio": sub.total_ticket.sum() / public,
        "source_note": source})
result = pd.DataFrame(rows)
result.to_csv(OUT / "milestone_comparison.csv", index=False)
print("top titles\n", titles.head(25).to_string())
print("\nmonthly tickets\n", monthly.to_string())
print("\npublic milestones\n", result.to_string(index=False))
