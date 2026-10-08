"""Descriptive audit of the 1 May 2025 Pluit Junction XXI closure."""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
daily = df.groupby(["date_show", "city_name", "cinema_ids"], as_index=False).agg(
    shows=("total_show", "sum"), tickets=("total_ticket", "sum"),
    films=("movie_title", "nunique")
)
start, end = pd.Timestamp("2025-04-17"), pd.Timestamp("2025-05-14")
window = daily[daily.date_show.between(start, end)].copy()
window["phase"] = window.date_show.lt(pd.Timestamp("2025-05-01")).map({True: "before", False: "after"})
summary = window.groupby(["city_name", "cinema_ids", "phase"], as_index=False).agg(
    days=("date_show", "nunique"), mean_shows=("shows", "mean"),
    mean_tickets=("tickets", "mean"), mean_films=("films", "mean"),
)
wide = summary.pivot(index=["city_name", "cinema_ids"], columns="phase")
wide.columns = [f"{x}_{y}" for x, y in wide.columns]
wide = wide.reset_index()
wide["show_delta"] = wide.mean_shows_after - wide.mean_shows_before
wide["show_ratio"] = wide.mean_shows_after / wide.mean_shows_before
wide.sort_values("show_delta").to_csv(OUT / "all_id_comparison.csv", index=False)
day0 = daily[daily.date_show.eq("2025-04-30")][["city_name", "cinema_ids", "shows"]]
day1 = daily[daily.date_show.eq("2025-05-01")][["city_name", "cinema_ids", "shows"]]
step = day0.merge(day1, on=["city_name", "cinema_ids"], suffixes=("_apr30", "_may1"))
step["show_delta"] = step.shows_may1 - step.shows_apr30
step = step.sort_values("show_delta")
step.to_csv(OUT / "apr30_may1_id_change.csv", index=False)
daily[daily.city_name.eq("JAKARTA") & daily.date_show.between("2025-04-24", "2025-05-08")].to_csv(
    OUT / "jakarta_daily.csv", index=False)
summary[summary.city_name.eq("JAKARTA")].to_csv(OUT / "jakarta_comparison.csv", index=False)
target_id = "4ed104ac0de10ba7eef50158603e0b32"
target_step = step[step.cinema_ids.eq(target_id)].iloc[0]
target_wide = wide[wide.cinema_ids.eq(target_id)].iloc[0]
result = {
    "target_id": target_id,
    "shows_apr30": int(target_step.shows_apr30),
    "shows_may1": int(target_step.shows_may1),
    "immediate_delta": int(target_step.show_delta),
    "rank_negative_daily_delta": int(step.reset_index(drop=True).index[step.cinema_ids.eq(target_id)][0] + 1),
    "ids_with_both_days": int(len(step)),
    "target_mean_shows_before_14d": round(float(target_wide.mean_shows_before), 2),
    "target_mean_shows_after_14d": round(float(target_wide.mean_shows_after), 2),
    "target_mean_delta_14d": round(float(target_wide.show_delta), 2),
    "rank_negative_14d_delta": int(wide.sort_values("show_delta").reset_index(drop=True).index[
        wide.sort_values("show_delta").cinema_ids.eq(target_id)][0] + 1),
    "ids_with_both_14d_windows": int(len(wide)),
}
(OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
print("Jakarta:\n", wide[wide.city_name.eq("JAKARTA")].to_string(index=False))
print("\nLargest decreases:\n", wide.sort_values("show_delta")[["city_name", "cinema_ids", "show_delta", "show_ratio"]].head(12).to_string(index=False))
print("\nJakarta daily:\n", daily[daily.city_name.eq("JAKARTA") & daily.date_show.between("2025-04-28", "2025-05-04")].to_string(index=False))
