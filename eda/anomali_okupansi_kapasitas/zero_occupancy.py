"""EDA murni: cari pola occupation_rate nol walau ada tiket positif."""

from pathlib import Path
import importlib.util
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
frames = []
for name, file in [("train", "train.csv"), ("test_history", "test_history.csv")]:
    frame = pd.read_csv(ROOT / "data" / file, parse_dates=["date_show"])
    frame["period"] = name
    frames.append(frame)
df = pd.concat(frames, ignore_index=True)
z = df[(df.total_ticket > 0) & (df.occupation_rate == 0)].copy()
z["month"] = z.date_show.dt.to_period("M").astype(str)
z["tps"] = z.total_ticket / z.total_show
z["base"] = z.movie_title.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True)

report = {
    "count": z.groupby("period").size().to_dict(),
    "ticket_median": z.groupby("period").total_ticket.median().to_dict(),
    "ticket_max": z.groupby("period").total_ticket.max().to_dict(),
    "show_median": z.groupby("period").total_show.median().to_dict(),
    "n_cinema": z.groupby("period").cinema_ids.nunique().to_dict(),
    "n_films": z.groupby("period").movie_title.nunique().to_dict(),
    "n_dates": z.groupby("period").date_show.nunique().to_dict(),
}
sept17 = df[(df.period == "train") & (df.date_show == pd.Timestamp("2025-09-17"))]
report["sep17_rows"] = len(sept17)
report["sep17_zero_occ"] = int(sept17.occupation_rate.eq(0).sum())
report["sep17_zero_occ_tickets_ge20"] = int(((sept17.occupation_rate == 0) & (sept17.total_ticket >= 20)).sum())
report["zero_occ_tickets_ge20_by_period"] = z.groupby("period").total_ticket.apply(lambda x: int((x >= 20).sum())).to_dict()
spec = importlib.util.spec_from_file_location("windows_v2", ROOT / "temp/exp_020_data_fix/build_windows_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
wide, _, _ = v2.build_windows(frames[0])
bad_d = []
for d in [1, 2, 3]:
    bad_d.append((wide[f"total_ticket_d{d}"] > 0) & (wide[f"occupation_rate_d{d}"] == 0))
bad = bad_d[0] | bad_d[1] | bad_d[2]
report["train_window_pairs_with_zero_occ_positive_tickets"] = int(bad.sum())
report["train_window_pairs_with_zero_occ_positive_tickets_pct"] = float(bad.mean())
report["train_window_pairs_sep17"] = int((bad & (((wide.D1 <= pd.Timestamp("2025-09-17")) &
    (wide.D1 + pd.Timedelta(days=2) >= pd.Timestamp("2025-09-17"))))).sum())
test_wide = v2.test_windows(frames[1])
test_pairs = pd.read_csv(ROOT / "data/test.csv")[["movie_title", "cinema_ids"]].drop_duplicates()
test_wide = test_wide.merge(test_pairs, on=["movie_title", "cinema_ids"])
test_bad = pd.Series(False, index=test_wide.index)
for d in [1, 2, 3]:
    test_bad |= (test_wide[f"total_ticket_d{d}"] > 0) & (test_wide[f"occupation_rate_d{d}"] == 0)
report["test_window_pairs_with_zero_occ_positive_tickets"] = int(test_bad.sum())
report["test_window_pairs_with_zero_occ_positive_tickets_pct"] = float(test_bad.mean())
by_month = df.assign(month=df.date_show.dt.to_period("M").astype(str), zero_occ=df.occupation_rate.eq(0)).groupby(["period", "month"]).zero_occ.agg(["size", "sum", "mean"])
by_movie = z.groupby(["period", "movie_title"]).agg(n=("total_ticket", "size"),
    min_date=("date_show", "min"), max_date=("date_show", "max"),
    median_tix=("total_ticket", "median"), max_tix=("total_ticket", "max")).sort_values("n", ascending=False)
by_cinema = z.groupby(["period", "cinema_ids", "city_name"]).agg(n=("total_ticket", "size"),
    median_tix=("total_ticket", "median")).sort_values("n", ascending=False)
by_date = z.groupby(["period", "date_show"]).size().sort_values(ascending=False).rename("n")
date_film = df.groupby(["period", "date_show", "movie_title"]).agg(
    rows=("total_ticket", "size"), zero_occ=("occupation_rate", lambda x: x.eq(0).sum()),
    median_tix=("total_ticket", "median"), max_tix=("total_ticket", "max"))
date_film = date_film[date_film.zero_occ > 0].sort_values("zero_occ", ascending=False)
large = z[z.total_ticket >= 20].sort_values("total_ticket", ascending=False)

for name, table in [("by_month", by_month), ("by_movie", by_movie),
                    ("by_cinema", by_cinema), ("by_date", by_date)]:
    table.to_csv(OUT / f"{name}.csv")
z.to_csv(OUT / "zero_occ_rows.csv", index=False)
date_film.to_csv(OUT / "date_film.csv")
large.to_csv(OUT / "large_ticket_zero_occ.csv", index=False)
(OUT / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print("summary", report)
print("\nmonth\n", by_month.round(4).to_string())
print("\nmovies\n", by_movie.head(15).to_string())
print("\ncinemas\n", by_cinema.head(15).to_string())
print("\ndates\n", by_date.head(15).to_string())
print("\ndate x film\n", date_film.head(20).to_string())
print("\nhigh-ticket zero occupancy count", len(large), "\n", large.head(15).to_string(index=False))
