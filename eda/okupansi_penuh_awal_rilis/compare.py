"""Bandingkan baris satu show D1-D3 train/test dan konsentrasi okupansi penuh."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("windows_v2", ROOT / "temp/exp_020_data_fix/build_windows_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def early_rows(file, period):
    data = pd.read_csv(ROOT / file, parse_dates=["date_show"])
    if period == "train":
        wide, _, _ = v2.build_windows(data)
    else:
        wide = v2.test_windows(data)
    frames = []
    for d in (1, 2, 3):
        f = wide[["movie_title", "cinema_ids", f"total_ticket_d{d}",
                  f"total_show_d{d}", f"occupation_rate_d{d}"]].copy()
        f.columns = ["movie_title", "cinema_ids", "total_ticket", "total_show", "occupation_rate"]
        f["d"] = d
        frames.append(f)
    x = pd.concat(frames, ignore_index=True)
    x = x[x.total_show > 0].copy()
    x["period"] = period
    return x


rows = pd.concat([early_rows("data/train.csv", "train"),
                  early_rows("data/test_history.csv", "test")], ignore_index=True)
rows["one_show"] = rows.total_show.eq(1)
rows["exact_100"] = rows.occupation_rate.eq(100)
rows["near_100"] = rows.occupation_rate.ge(95)
summary = rows.groupby(["period", "d"]).agg(
    rows=("movie_title", "size"), one_show=("one_show", "sum"),
    exact_100=("exact_100", "sum"), near_100=("near_100", "sum"),
    med_tickets=("total_ticket", "median"),
).reset_index()
summary["one_show_share"] = summary.one_show / summary.rows
summary.to_csv(OUT / "summary_by_day.csv", index=False)

one = rows[rows.one_show].copy()
one_summary = one.groupby(["period", "d"]).agg(
    rows=("movie_title", "size"), exact_100=("exact_100", "sum"),
    near_100=("near_100", "sum"), med_tickets=("total_ticket", "median"),
    low_ticket_share=("total_ticket", lambda s: s.le(8).mean()),
).reset_index()
one_summary["share_exact_100"] = one_summary.exact_100 / one_summary.rows
one_summary.to_csv(OUT / "one_show_by_day.csv", index=False)
print("all rows:\n", summary.to_string(index=False))
print("one show:\n", one_summary.to_string(index=False))

films = one.groupby(["period", "movie_title"]).agg(
    one_show=("movie_title", "size"), exact_100=("exact_100", "sum")
).reset_index().sort_values(["period", "exact_100"], ascending=[True, False])
films.to_csv(OUT / "by_film.csv", index=False)
for period, g in films.groupby("period"):
    print(period, "top exact100 films:\n", g.head(12).to_string(index=False))
