"""EDA deskriptif: apakah cuti bersama 18 Agustus tampak berbeda dari Senin tetangga?"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
dates = pd.to_datetime(["2025-08-11", "2025-08-18", "2025-08-25"])
x = df[df.date_show.isin(dates)].copy()
x["tps"] = x.total_ticket / x.total_show.replace(0, np.nan)

daily = x.groupby("date_show").agg(tickets=("total_ticket", "sum"),
                                     shows=("total_show", "sum"),
                                     rows=("total_ticket", "size"),
                                     films=("movie_title", "nunique"))
daily["tps"] = daily.tickets / daily.shows
daily.to_csv(OUT / "daily.csv")

p = x.pivot(index=["movie_title", "cinema_ids"], columns="date_show",
            values=["total_ticket", "total_show", "tps"]).dropna()
p.columns = [f"{metric}_{dt.date()}" for metric, dt in p.columns]
p = p.reset_index()
p["base"] = p.movie_title.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True).str.strip()
for metric in ("total_ticket", "total_show", "tps"):
    before = p[f"{metric}_2025-08-11"]
    event = p[f"{metric}_2025-08-18"]
    after = p[f"{metric}_2025-08-25"]
    p[f"{metric}_ratio"] = event / np.sqrt(before * after)

by_film = p.groupby("base").agg(n_pairs=("cinema_ids", "size"),
                                 ticket_median_ratio=("total_ticket_ratio", "median"),
                                 tps_median_ratio=("tps_ratio", "median"),
                                 show_median_ratio=("total_show_ratio", "median"))
by_film.to_csv(OUT / "by_film.csv")

# Kontrol deskriptif tambahan: bandingkan rasio Senin/Selasa dalam tiga minggu.
six_dates = pd.to_datetime(["2025-08-11", "2025-08-12", "2025-08-18",
                            "2025-08-19", "2025-08-25", "2025-08-26"])
z = df[df.date_show.isin(six_dates)].copy()
z["tps"] = z.total_ticket / z.total_show.replace(0, np.nan)
six = z.pivot(index=["movie_title", "cinema_ids"], columns="date_show", values="tps").dropna()
week_ratios = [six[pd.Timestamp(m)] / six[pd.Timestamp(t)] for m, t in
               [("2025-08-11", "2025-08-12"), ("2025-08-18", "2025-08-19"),
                ("2025-08-25", "2025-08-26")]]
six["event_monday_tuesday_ratio_relative"] = week_ratios[1] / np.sqrt(week_ratios[0] * week_ratios[2])
six.reset_index()[["movie_title", "cinema_ids", "event_monday_tuesday_ratio_relative"]].to_csv(
    OUT / "balanced_six_day_ratios.csv", index=False)

summary = {
    "n_balanced_pairs": len(p),
    "n_base_films": int(p.base.nunique()),
    "daily_ticket_event_vs_neighbor_geomean": float(daily.loc[dates[1], "tickets"] /
                                                    np.sqrt(daily.loc[dates[0], "tickets"] * daily.loc[dates[2], "tickets"])),
    "daily_show_event_vs_neighbor_geomean": float(daily.loc[dates[1], "shows"] /
                                                  np.sqrt(daily.loc[dates[0], "shows"] * daily.loc[dates[2], "shows"])),
    "balanced_pair_ticket_median_ratio": float(p.total_ticket_ratio.median()),
    "balanced_pair_tps_median_ratio": float(p.tps_ratio.median()),
    "balanced_pair_show_median_ratio": float(p.total_show_ratio.median()),
    "films_tps_median_ratio_above_1": int(by_film.tps_median_ratio.gt(1).sum()),
    "films_tps_median_ratio_below_1": int(by_film.tps_median_ratio.lt(1).sum()),
    "n_six_day_balanced_pairs": len(six),
    "six_day_relative_monday_tuesday_tps_median_ratio": float(six.event_monday_tuesday_ratio_relative.median()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(daily.to_string())
print(json.dumps(summary, indent=2))
