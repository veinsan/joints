"""EDA deskriptif pola satu show pada preview sebelum D1 rilis luas."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("windows_v2", ROOT / "temp/exp_020_data_fix/build_windows_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
d1 = v2.film_d1(tr)
x = tr.merge(d1.reset_index(), on="movie_title", how="inner")
x["rel_day"] = (x.date_show - x.D1).dt.days
x["phase"] = np.select([x.rel_day < 0, x.rel_day <= 2], ["preview", "D1_D3"], default="later")
x["one_show"] = x.total_show.eq(1)
x["band_130_140"] = x.total_ticket.between(130, 140)
x["one_show_band"] = x.one_show & x.band_130_140

summary = x.groupby("phase").agg(
    rows=("movie_title", "size"), films=("movie_title", "nunique"),
    one_show=("one_show", "sum"), band=("band_130_140", "sum"),
    one_show_band=("one_show_band", "sum"),
).reset_index()
summary.to_csv(OUT / "phase_summary.csv", index=False)
print(summary.to_string(index=False))

preview = x[x.phase == "preview"].copy()
film = preview.groupby("movie_title").agg(
    rows=("movie_title", "size"), dates=("date_show", "nunique"),
    clusters=("cinema_ids", "nunique"), one_show_band=("one_show_band", "sum"),
    tickets=("total_ticket", "sum"),
).reset_index().sort_values(["one_show_band", "rows"], ascending=False)
film.to_csv(OUT / "preview_by_film.csv", index=False)
print("top preview films:\n", film.head(25).to_string(index=False))

single = x[x.one_show].copy()
dist = single.groupby(["phase", "total_ticket"]).size().rename("rows").reset_index()
dist.to_csv(OUT / "one_show_ticket_distribution.csv", index=False)
print("one-show ticket peaks:\n", dist[dist.phase == "preview"].nlargest(25, "rows").to_string(index=False))

concentration = preview[preview.one_show_band].groupby(["cinema_ids", "city_name"]).agg(
    rows=("movie_title", "size"), films=("movie_title", "nunique"),
    dates=("date_show", "nunique"),
).reset_index().sort_values("rows", ascending=False)
concentration.to_csv(OUT / "preview_band_by_cluster.csv", index=False)
print("top clusters:\n", concentration.head(20).to_string(index=False))

preview[["date_show", "movie_title", "cinema_ids", "city_name", "rel_day", "total_ticket", "total_show",
         "occupation_rate", "one_show_band"]].to_csv(OUT / "preview_rows.csv", index=False)

one = x[x.one_show].copy()
phase_stats = one.groupby("phase").agg(
    rows=("total_ticket", "size"),
    median_tickets=("total_ticket", "median"),
    q90_tickets=("total_ticket", lambda s: s.quantile(.9)),
    median_occupancy=("occupation_rate", "median"),
    high_occupancy=("occupation_rate", lambda s: (s >= 80).mean()),
    tickets_100_200=("total_ticket", lambda s: s.between(100, 200).mean()),
).reset_index()
phase_stats["exact_100_occupancy"] = one.groupby("phase").occupation_rate.apply(lambda s: s.eq(100).mean()).to_numpy()
phase_stats.to_csv(OUT / "one_show_phase_stats.csv", index=False)
print("one-show phase stats:\n", phase_stats.to_string(index=False))

repeat = one[one.phase == "preview"].groupby(["movie_title", "cinema_ids", "city_name"]).agg(
    dates=("date_show", "nunique"), min_tickets=("total_ticket", "min"),
    max_tickets=("total_ticket", "max"), median_tickets=("total_ticket", "median"),
    tickets_std=("total_ticket", "std"),
).reset_index()
repeat = repeat[repeat.dates >= 3].sort_values(["dates", "tickets_std"], ascending=[False, True])
repeat.to_csv(OUT / "repeated_one_show_preview.csv", index=False)
print("repeat groups >=3:", len(repeat))
print("range <=5 and median >=100:",
      int(((repeat.max_tickets - repeat.min_tickets <= 5) & (repeat.median_tickets >= 100)).sum()))
print(repeat.head(25).to_string(index=False))

preview_one = one[one.phase == "preview"]
film_one = preview_one.groupby("movie_title").agg(
    rows=("movie_title", "size"), exact_100=("occupation_rate", lambda s: s.eq(100).sum()),
    med_tickets=("total_ticket", "median"),
).reset_index().sort_values("exact_100", ascending=False)
film_one.to_csv(OUT / "preview_one_show_occupancy_by_film.csv", index=False)
print("preview one-show exact 100 by film:\n", film_one.head(25).to_string(index=False))
without_believe = preview_one[preview_one.movie_title != "BELIEVE - TAKDIR, MIMPI, KEBERANIAN"]
print("without BELIEVE preview one-show rows", len(without_believe),
      "exact 100", int(without_believe.occupation_rate.eq(100).sum()),
      "fraction", without_believe.occupation_rate.eq(100).mean())
believe = preview[preview.movie_title == "BELIEVE - TAKDIR, MIMPI, KEBERANIAN"]
print("BELIEVE preview rows", len(believe), "dates", believe.date_show.nunique(),
      "clusters", believe.cinema_ids.nunique(), "shows", believe.total_show.sum(),
      "tickets", believe.total_ticket.sum())
