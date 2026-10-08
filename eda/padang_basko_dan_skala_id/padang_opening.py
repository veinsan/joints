"""Descriptive audit of the Basko City Mall XXI opening and cinema_ids scale."""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
daily_id = df.groupby(["date_show", "city_name", "cinema_ids"], as_index=False).agg(
    shows=("total_show", "sum"), tickets=("total_ticket", "sum"),
    films=("movie_title", "nunique"),
)

opening = pd.Timestamp("2025-07-09")
# 21-day windows keep weekdays balanced and avoid the June 7-16 train outage.
before = (daily_id.date_show >= opening - pd.Timedelta(days=21)) & (daily_id.date_show < opening)
after = (daily_id.date_show >= opening) & (daily_id.date_show < opening + pd.Timedelta(days=21))
window = daily_id[before | after].copy()
window["phase"] = window.date_show.ge(opening).map({False: "before", True: "after"})
id_comparison = window.groupby(["city_name", "cinema_ids", "phase"], as_index=False).agg(
    days=("date_show", "nunique"), mean_shows=("shows", "mean"),
    mean_tickets=("tickets", "mean"), mean_films=("films", "mean"),
)
id_comparison[id_comparison.city_name.eq("PADANG")].to_csv(OUT / "padang_id_comparison.csv", index=False)

city_daily = daily_id.groupby(["date_show", "city_name"], as_index=False).agg(
    shows=("shows", "sum"), tickets=("tickets", "sum"), ids=("cinema_ids", "nunique")
)
city_daily = city_daily[(city_daily.date_show >= opening - pd.Timedelta(days=21)) &
                        (city_daily.date_show < opening + pd.Timedelta(days=21))].copy()
city_daily["phase"] = city_daily.date_show.ge(opening).map({False: "before", True: "after"})
city_comparison = city_daily.groupby(["city_name", "phase"], as_index=False).agg(
    days=("date_show", "nunique"), mean_shows=("shows", "mean"),
    mean_tickets=("tickets", "mean"), mean_ids=("ids", "mean"),
)
wide = city_comparison.pivot(index="city_name", columns="phase")
wide.columns = [f"{metric}_{phase}" for metric, phase in wide.columns]
wide = wide.reset_index()
wide["show_ratio"] = wide.mean_shows_after / wide.mean_shows_before
wide["ticket_ratio"] = wide.mean_tickets_after / wide.mean_tickets_before
wide["show_delta"] = wide.mean_shows_after - wide.mean_shows_before
wide = wide.sort_values("show_ratio", ascending=False)
wide.to_csv(OUT / "city_comparison.csv", index=False)

all_profile = daily_id.groupby(["cinema_ids", "city_name"], as_index=False).agg(
    days=("date_show", "nunique"), median_daily_shows=("shows", "median"),
    p95_daily_shows=("shows", lambda s: s.quantile(.95)),
    max_daily_shows=("shows", "max"), mean_daily_films=("films", "mean"),
)
all_profile.to_csv(OUT / "id_scale.csv", index=False)

padang_daily = daily_id[daily_id.city_name.eq("PADANG")].copy()
padang_daily.to_csv(OUT / "padang_full_daily.csv", index=False)

padang = wide[wide.city_name.eq("PADANG")].iloc[0]
summary = {
    "padang_ids": int(daily_id[daily_id.city_name.eq("PADANG")].cinema_ids.nunique()),
    "padang_days_before": int(padang.days_before),
    "padang_days_after": int(padang.days_after),
    "padang_mean_shows_before": round(float(padang.mean_shows_before), 2),
    "padang_mean_shows_after": round(float(padang.mean_shows_after), 2),
    "padang_show_ratio": round(float(padang.show_ratio), 3),
    "padang_rank_show_ratio": int(wide.reset_index(drop=True).index[wide.city_name.eq("PADANG")][0] + 1),
    "comparison_cities": int(len(wide)),
    "id_count_train": int(len(all_profile)),
    "id_median_daily_shows_quantiles": {str(q): float(all_profile.median_daily_shows.quantile(q))
                                       for q in [0, .25, .5, .75, .9, 1]},
    "id_over_40_median_daily_shows": int(all_profile.median_daily_shows.gt(40).sum()),
    "id_over_100_median_daily_shows": int(all_profile.median_daily_shows.gt(100).sum()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("\nPadang IDs:\n", id_comparison[id_comparison.city_name.eq("PADANG")].to_string(index=False))
print("\nTop city changes:\n", wide[["city_name", "show_ratio", "show_delta", "ticket_ratio"]].head(12).to_string(index=False))
