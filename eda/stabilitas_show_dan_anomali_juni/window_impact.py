"""Count v2 train windows touching the newly found June 1-5 anomaly."""

from pathlib import Path
import importlib.util
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
BUILDER = ROOT / "temp/exp_020_data_fix/build_windows_v2.py"
spec = importlib.util.spec_from_file_location("window_builder_v2", BUILDER)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

train = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
wide, targets, _ = builder.build_windows(train)
anomaly_start, anomaly_end = pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-05")
touch = wide[(wide.D1.le(anomaly_end)) &
             ((wide.D1 + pd.Timedelta(days=9)).ge(anomaly_start))].copy()
affected_tgt = targets[targets.date_show.between(anomaly_start, anomaly_end)].copy()
affected_tgt["base_title"] = builder.base_title(affected_tgt.movie_title)
per_title = touch.groupby(["movie_title", "D1"], as_index=False).agg(pairs=("cinema_ids", "size"))
per_title["base_title"] = builder.base_title(per_title.movie_title)
per_title.to_csv(OUT / "affected_titles.csv", index=False)
per_day = affected_tgt.groupby("date_show", as_index=False).agg(
    target_rows=("total_ticket", "size"), positive_rows=("total_ticket", lambda s: s.gt(0).sum()),
    tickets=("total_ticket", "sum"), shows=("total_show", "sum"),
    films=("movie_title", "nunique"),
)
per_day.to_csv(OUT / "affected_target_days.csv", index=False)
film_days = targets[targets.movie_title.isin(touch.movie_title)].groupby(
    ["movie_title", "date_show", "d"], as_index=False).agg(
        target_rows=("total_ticket", "size"), positive_rows=("total_ticket", lambda s: s.gt(0).sum()),
        tickets=("total_ticket", "sum"), shows=("total_show", "sum"),
    )
film_days.to_csv(OUT / "affected_film_daily.csv", index=False)
summary = {
    "v2_pairs_total": len(wide), "affected_pairs": len(touch),
    "affected_fraction": round(len(touch)/len(wide), 4),
    "affected_movie_titles": int(touch.movie_title.nunique()),
    "affected_base_titles": int(builder.base_title(touch.movie_title).nunique()),
    "affected_target_rows": len(affected_tgt),
    "affected_target_positive_rows": int(affected_tgt.total_ticket.gt(0).sum()),
    "target_rows_total": len(targets),
    "affected_target_fraction": round(len(affected_tgt)/len(targets), 4),
}
(OUT / "window_impact.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("\nTitles:\n", per_title.to_string(index=False))
print("\nTarget days:\n", per_day.to_string(index=False))
print("\nFilm days:\n", film_days.to_string(index=False))
