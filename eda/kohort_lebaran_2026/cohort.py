"""EDA kohort film rilis 18 Maret 2026 dan cakupan target libur."""

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

th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
wide = v2.test_windows(th)
wide["base"] = v2.base_title(wide.movie_title)
wide["d3_tps"] = wide.total_ticket_d3 / wide.total_show_d3
wide["scale"] = np.maximum((wide.total_ticket_d1 + wide.total_ticket_d2 + wide.total_ticket_d3) / 3, 1)
wide["low_d3"] = wide.d3_tps.le(8)
wide["low_scale"] = wide.scale.le(20)

cohort = wide[wide.D1 == "2026-03-18"].copy()
others = wide[(wide.D1 >= "2026-02-01") & (wide.D1 != "2026-03-18")].copy()
other_march = wide[(wide.D1 >= "2026-03-01") & (wide.D1 < "2026-04-01") &
                   (wide.D1 != "2026-03-18")].copy()
print("cohort pairs", len(cohort), "base films", cohort.base.nunique(),
      "test pair share", len(cohort) / len(wide),
      "target row count", len(cohort) * 7)

def summarize(g):
    return pd.Series({
        "pairs": len(g), "films": g["base"].nunique() if "base" in g else 1,
        "median_scale": g.scale.median(), "low_scale_share": g.low_scale.mean(),
        "median_d3_shows": g.total_show_d3.median(),
        "median_d3_tps": g.d3_tps.median(),
        "low_d3_share": g.low_d3.mean(),
        "median_d3_occ": g.occupation_rate_d3.median(),
    })

periods = pd.DataFrame([summarize(cohort), summarize(other_march), summarize(others)],
                       index=["March18", "other_March", "other_FebMar"])
periods.to_csv(OUT / "cohort_summary.csv")
print(periods.to_string())

films = cohort.groupby("base").apply(summarize, include_groups=False).reset_index().sort_values("pairs", ascending=False)
films.to_csv(OUT / "cohort_by_film.csv", index=False)
print("films:\n", films.to_string(index=False))

target = pd.DataFrame({"date_show": pd.date_range("2026-03-21", "2026-03-27")})
target["horizon"] = range(4, 11)
target["target_rows"] = len(cohort)
calendar = pd.read_csv(ROOT / "data/holidays.csv", parse_dates=["date"])
target = target.merge(calendar.rename(columns={"date": "date_show"}), on="date_show", how="left",
                      validate="one_to_one")
official_cuti = pd.read_csv(ROOT / "eda/cuti_bersama_tidak_terkode/output/official_cuti_vs_package.csv",
                            parse_dates=["date"])
target["official_cuti"] = target.date_show.isin(official_cuti.date)
target.to_csv(OUT / "target_dates.csv", index=False)
print("target dates:\n", target.to_string(index=False))
cohort_cuti_rows = int(target.loc[target.official_cuti, "target_rows"].sum())
all_cuti_rows = int(official_cuti.test_target_rows.sum())
march_cuti_rows = int(official_cuti.loc[official_cuti.date.dt.month == 3, "test_target_rows"].sum())
print("cohort cuti target rows", cohort_cuti_rows,
      "share of all test cuti rows", cohort_cuti_rows / all_cuti_rows,
      "share of March cuti rows", cohort_cuti_rows / march_cuti_rows)
