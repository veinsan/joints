"""Audit exact row transitions around the early-June 2025 show collapse."""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])

def compare(day0, day1):
    before = df[df.date_show.eq(day0)]
    after = df[df.date_show.eq(day1)]
    paired = before.merge(after, on=["cinema_ids", "movie_title"], suffixes=("_before", "_after"))
    paired["show_ratio"] = paired.total_show_after / paired.total_show_before
    paired["ticket_ratio"] = paired.total_ticket_after / paired.total_ticket_before
    paired["show_delta"] = paired.total_show_after - paired.total_show_before
    ratio_counts = paired.show_ratio.round(6).value_counts().head(15).rename_axis("show_ratio").reset_index(name="pairs")
    name = f"{day0[5:]}_to_{day1[5:]}".replace("-", "")
    paired[["cinema_ids", "movie_title", "city_name_before", "total_show_before", "total_show_after",
            "show_ratio", "total_ticket_before", "total_ticket_after", "ticket_ratio",
            "occupation_rate_before", "occupation_rate_after"]].to_csv(OUT / f"paired_{name}.csv", index=False)
    ratio_counts.to_csv(OUT / f"ratios_{name}.csv", index=False)
    by_id_before = before.groupby("cinema_ids", as_index=False).total_show.sum().rename(columns={"total_show": "shows_before"})
    by_id_after = after.groupby("cinema_ids", as_index=False).total_show.sum().rename(columns={"total_show": "shows_after"})
    id_paired = by_id_before.merge(by_id_after, on="cinema_ids")
    id_paired["show_ratio"] = id_paired.shows_after / id_paired.shows_before
    id_paired.to_csv(OUT / f"ids_{name}.csv", index=False)
    return {
        "transition": f"{day0}->{day1}", "before_rows": len(before), "after_rows": len(after),
        "paired_rows": len(paired), "before_shows_all": int(before.total_show.sum()),
        "after_shows_all": int(after.total_show.sum()),
        "paired_show_ratio_sum": round(paired.total_show_after.sum()/paired.total_show_before.sum(), 4),
        "paired_ticket_ratio_sum": round(paired.total_ticket_after.sum()/paired.total_ticket_before.sum(), 4),
        "pct_paired_show_same": round(paired.total_show_before.eq(paired.total_show_after).mean(), 4),
        "pct_paired_show_one_third_exact": round(paired.total_show_after.mul(3).eq(paired.total_show_before).mean(), 4),
        "pct_paired_show_one_third_rounded": round(paired.total_show_after.eq((paired.total_show_before/3).round()).mean(), 4),
        "median_row_show_ratio": round(float(paired.show_ratio.median()), 4),
        "paired_ids": len(id_paired),
        "median_id_show_ratio": round(float(id_paired.show_ratio.median()), 4),
        "pct_ids_show_ratio_le_half": round(float(id_paired.show_ratio.le(.5).mean()), 4),
        "top_show_ratios": ratio_counts.to_dict("records"),
    }

dates = [("2025-05-24", "2025-05-25"), ("2025-05-30", "2025-05-31"),
         ("2025-05-31", "2025-06-01"),
         ("2025-06-01", "2025-06-02"), ("2025-06-05", "2025-06-06"),
         ("2025-06-16", "2025-06-17"), ("2025-06-21", "2025-06-22")]
results = [compare(a, b) for a, b in dates]
(OUT / "transition_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
for result in results:
    print({k:v for k,v in result.items() if k != "top_show_ratios"})
    print("top ratios:", result["top_show_ratios"][:8])
