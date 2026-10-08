"""Deskripsi intensitas layar sebelum dan sesudah rilis luas Tabayyun."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
x = tr[tr.movie_title.str.fullmatch("TABAYYUN", case=False)].copy()
x = x[x.date_show <= "2025-05-14"]
daily = x.groupby("date_show").agg(
    cinema_clusters=("cinema_ids", "nunique"),
    shows=("total_show", "sum"),
    tickets=("total_ticket", "sum"),
).reset_index()
daily["tickets_per_show"] = daily.tickets / daily.shows
daily.to_csv(OUT / "tabayyun_daily.csv", index=False)
print(daily.to_string(index=False))
preview = x[x.date_show < "2025-05-08"]
release = x[x.date_show == "2025-05-08"]
print("preview days", preview.date_show.nunique(), "clusters", preview.cinema_ids.nunique(),
      "shows", preview.total_show.sum(), "tickets", preview.total_ticket.sum())
print("release clusters", release.cinema_ids.nunique(), "shows", release.total_show.sum(),
      "tickets", release.total_ticket.sum())
