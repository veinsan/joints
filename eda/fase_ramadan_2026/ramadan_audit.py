"""EDA waktu D3 dan paparan Ramadan 1447 H tanpa target test."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
x = pd.read_csv(ROOT / "eda/persaingan_film_baru/output/d3_selected_opening_context.csv",
                parse_dates=["date_show", "D1"])
x = x[x.period == "test"].copy()
x["weak_tps"] = x.tps.le(8)
x["base"] = x.movie_title.str.replace(
    r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True).str.strip()

def tag(date):
    if date < pd.Timestamp("2026-02-01"):
        return "Oct-Jan"
    if date < pd.Timestamp("2026-02-19"):
        return "Feb1-18_pre"
    if date < pd.Timestamp("2026-03-18"):
        return "Feb19-Mar17_Ramadan"
    if date <= pd.Timestamp("2026-03-20"):
        return "Mar18-20_preEid_cuti"
    return "after"

x["phase"] = x.date_show.map(tag)
phase = x.groupby("phase").agg(
    pairs=("movie_title", "size"), films=("base", "nunique"),
    median_tps=("tps", "median"), weak_tps_share=("weak_tps", "mean"),
    median_occ=("occupation_rate", "median"), median_shows=("total_show", "median"),
).reset_index()
phase.to_csv(OUT / "phase_summary.csv", index=False)
print("phase:\n", phase.to_string(index=False))

x["week"] = x.date_show.dt.to_period("W-SUN").astype(str)
weekly = x.groupby("week").agg(
    first_d3=("date_show", "min"), last_d3=("date_show", "max"),
    pairs=("movie_title", "size"), films=("base", "nunique"),
    median_tps=("tps", "median"), weak_tps_share=("weak_tps", "mean"),
    median_occ=("occupation_rate", "median"),
).reset_index()
weekly.to_csv(OUT / "weekly_summary.csv", index=False)
print("Feb-Mar weeks:\n", weekly[weekly.first_d3 >= "2026-02-01"].to_string(index=False))

film = x[x.phase.isin(["Feb1-18_pre", "Feb19-Mar17_Ramadan", "Mar18-20_preEid_cuti"])].groupby(
    ["phase", "base"]).agg(
        pairs=("movie_title", "size"), median_tps=("tps", "median"),
        weak_tps_share=("weak_tps", "mean"), median_occ=("occupation_rate", "median"),
    ).reset_index().sort_values(["phase", "pairs"], ascending=[True, False])
film.to_csv(OUT / "film_by_phase.csv", index=False)

city = x.groupby(["phase", "city_name"]).agg(
    pairs=("weak_tps", "size"), weak_share=("weak_tps", "mean")
).reset_index()
wide = city.pivot(index="city_name", columns="phase", values=["pairs", "weak_share"])
for a, b in [("Oct-Jan", "Feb1-18_pre"), ("Feb1-18_pre", "Feb19-Mar17_Ramadan")]:
    ok = wide[("pairs", a)].ge(10) & wide[("pairs", b)].ge(10)
    delta = wide.loc[ok, ("weak_share", b)] - wide.loc[ok, ("weak_share", a)]
    print(f"city weak change {a} -> {b}: {int(ok.sum())} shared >=10, "
          f"{int(delta.gt(0).sum())} up, {int(delta.lt(0).sum())} down, "
          f"median change {delta.median():.4f}")
city.to_csv(OUT / "city_by_phase.csv", index=False)
