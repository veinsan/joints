"""Susun tanggal D1 test_history per judul dasar untuk audit sumber luar."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
th["base"] = th.movie_title.str.replace(
    r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True).str.strip()
dates = th.groupby("base").agg(
    D1=("date_show", "min"), rows=("movie_title", "size"),
    clusters=("cinema_ids", "nunique"),
).reset_index()
dates["month"] = dates.D1.dt.to_period("M").astype(str)
dates = dates.sort_values(["month", "rows"], ascending=[True, False])
dates.to_csv(OUT / "test_titles.csv", index=False)
for month, g in dates.groupby("month"):
    print(month)
    print(g[["base", "D1", "rows", "clusters"]].head(15).to_string(index=False))
