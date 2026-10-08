"""EDA sumber: bandingkan jejak geografis dan besaran show dengan laporan operator publik."""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])

summary = {}
for name, df in [("train", tr), ("test_history", th)]:
    mapping = df.groupby("cinema_ids").city_name.nunique()
    summary[name] = {
        "n_cinema_ids": int(df.cinema_ids.nunique()),
        "n_city_names": int(df.city_name.nunique()),
        "cinema_ids_multiple_cities": int(mapping.gt(1).sum()),
        "ticket_sum": int(df.total_ticket.sum()),
        "show_sum": int(df.total_show.sum()),
        "max_show_film_cluster_day": int(df.total_show.max()),
        "film_cluster_days_over_100_shows": int(df.total_show.gt(100).sum()),
    }
    cluster_day = df.groupby(["cinema_ids", "city_name", "date_show"], as_index=False).agg(
        show_sum=("total_show", "sum"), ticket_sum=("total_ticket", "sum"), n_films=("movie_title", "nunique"))
    cluster_day.sort_values("show_sum", ascending=False).head(30).to_csv(
        OUT / f"{name}_top_cluster_days.csv", index=False)
    summary[name]["max_show_cluster_day_all_films"] = int(cluster_day.show_sum.max())
    summary[name]["median_show_cluster_day_all_films"] = float(cluster_day.show_sum.median())
    by_month = df.assign(month=df.date_show.dt.to_period("M").astype(str)).groupby("month").agg(
        days=("date_show", "nunique"), city_names=("city_name", "nunique"),
        cinema_ids=("cinema_ids", "nunique"), ticket_sum=("total_ticket", "sum"),
        show_sum=("total_show", "sum"))
    by_month["tickets_per_show"] = by_month.ticket_sum / by_month.show_sum
    by_month.to_csv(OUT / f"{name}_monthly.csv")

sept = tr[tr.date_show.dt.month.eq(9)].copy()
sept_daily = sept.groupby("date_show").agg(city_names=("city_name", "nunique"),
                                            cinema_ids=("cinema_ids", "nunique"),
                                            show_sum=("total_show", "sum"),
                                            ticket_sum=("total_ticket", "sum"))
sept_daily.to_csv(OUT / "september_daily.csv")
summary["september_train"] = {
    "city_names_monthly": int(sept.city_name.nunique()),
    "cinema_ids_monthly": int(sept.cinema_ids.nunique()),
    "median_city_names_daily": float(sept_daily.city_names.median()),
    "median_cinema_ids_daily": float(sept_daily.cinema_ids.median()),
    "median_shows_daily": float(sept_daily.show_sum.median()),
    "max_shows_daily": int(sept_daily.show_sum.max()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
