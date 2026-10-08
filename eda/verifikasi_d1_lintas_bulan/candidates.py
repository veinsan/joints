"""EDA audit D1: susun kandidat judul dasar train lintas bulan dan preview."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

x = pd.read_csv(ROOT / "eda/verifikasi_tanggal_rilis/output/release_audit.csv",
                parse_dates=["D1", "first_transaction"])
x["base"] = x.movie_title.str.replace(
    r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True).str.strip()
x = x.sort_values(["base", "lead_days"], ascending=[True, False]).drop_duplicates("base")
x["month"] = x.D1.dt.to_period("M").astype(str)
x["lead_group"] = pd.cut(x.lead_days, [-1, 0, 7, 365], labels=["same_day", "1_to_7", "over_7"])
x = x.sort_values(["month", "lead_group", "lead_days", "base"])
x[["base", "D1", "first_transaction", "lead_days", "month", "lead_group"]].to_csv(
    OUT / "all_candidate_titles.csv", index=False)
print(x.groupby(["month", "lead_group"], observed=True).size().to_string())
for month, g in x.groupby("month"):
    print("\n", month)
    print(g[["base", "D1", "lead_days"]].to_string(index=False))
