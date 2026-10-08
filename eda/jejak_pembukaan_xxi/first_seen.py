"""Audit tanggal pertama kota dan cinema_ids dalam transaksi train/test_history."""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
test = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
all_rows = pd.concat([train.assign(period="train"), test.assign(period="test")], ignore_index=True)

city = all_rows.groupby("city_name").agg(
    first_seen=("date_show", "min"), last_seen=("date_show", "max"),
    ids=("cinema_ids", "nunique"), rows=("total_ticket", "size")
).reset_index().sort_values("first_seen")
city.to_csv(OUT / "city_first_seen.csv", index=False)

ids = all_rows.groupby(["cinema_ids", "city_name"]).agg(
    first_seen=("date_show", "min"), last_seen=("date_show", "max"),
    rows=("total_ticket", "size")
).reset_index().sort_values("first_seen")
ids.to_csv(OUT / "id_first_seen.csv", index=False)

official = pd.read_csv(Path(__file__).resolve().parent / "official_openings.csv", parse_dates=["opening_date"])
official["data_city_first_seen"] = official.city_name.map(city.set_index("city_name").first_seen)
official["data_ids_first_seen_on_opening"] = official.apply(
    lambda r: int(((ids.city_name == r.city_name) & (ids.first_seen == r.opening_date)).sum()), axis=1)
official["data_cumulative_city_count"] = official.opening_date.map(
    lambda date: int(city.first_seen.le(max(date, train.date_show.min())).sum()))
official["city_count_matches"] = official.data_cumulative_city_count.eq(official.official_city_count)
official.to_csv(OUT / "official_opening_check.csv", index=False)

first_day = ids.merge(all_rows, left_on=["cinema_ids", "first_seen"],
                      right_on=["cinema_ids", "date_show"])
first_day["implied_scheduled_seats"] = first_day.total_ticket * 100 / first_day.occupation_rate.where(
    first_day.occupation_rate.gt(0))
first_day = first_day.groupby(["cinema_ids", "city_name_x", "first_seen"]).agg(
    first_day_films=("movie_title", "nunique"),
    first_day_shows=("total_show", "sum"),
    first_day_tickets=("total_ticket", "sum"),
    first_day_implied_scheduled_seats=("implied_scheduled_seats", "sum")
).reset_index().rename(columns={"city_name_x": "city_name"})
first_day.to_csv(OUT / "id_first_day_activity.csv", index=False)

later = city[city.first_seen.gt(train.date_show.min())]
print("cities first seen after train start:\n", later.to_string(index=False))
print("\nIDs first seen after train start:\n", ids[ids.first_seen.gt(train.date_show.min())].to_string(index=False))
print("\nPematang first day:\n", first_day[first_day.city_name.eq("PEMATANG SIANTAR")].to_string(index=False))
print("\nOfficial opening check:\n", official[["opening_date", "location", "data_city_first_seen",
                                            "data_ids_first_seen_on_opening", "data_cumulative_city_count",
                                            "official_city_count"]].to_string(index=False))

summary = {
    "train_city_count": int(train.city_name.nunique()),
    "train_id_count": int(train.cinema_ids.nunique()),
    "all_city_count": int(all_rows.city_name.nunique()),
    "all_id_count": int(all_rows.cinema_ids.nunique()),
    "cities_first_seen_after_apr1": int(len(later)),
    "ids_first_seen_after_apr1": int(ids.first_seen.gt(train.date_show.min()).sum()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
