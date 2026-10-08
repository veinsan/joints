"""Audit total-show stability and persistent jumps for each train cinema_ids."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
daily = df.groupby(["cinema_ids", "city_name", "date_show"], as_index=False).agg(
    shows=("total_show", "sum"), tickets=("total_ticket", "sum")
)
network = df.groupby("date_show", as_index=False).agg(
    shows=("total_show", "sum"), tickets=("total_ticket", "sum"),
    rows=("total_ticket", "size"), ids=("cinema_ids", "nunique"),
    films=("movie_title", "nunique"),
)
network.to_csv(OUT / "daily_network.csv", index=False)
dates = pd.date_range(df.date_show.min(), df.date_show.max(), freq="D")
bad = pd.date_range("2025-06-01", "2025-06-16", freq="D")
profiles, candidates = [], []
for (cid, city), group in daily.groupby(["cinema_ids", "city_name"]):
    s = group.set_index("date_show").shows.reindex(dates).astype(float)
    s.loc[bad] = np.nan
    valid = s.dropna()
    consecutive = s.notna() & s.shift(1).notna()
    daily_delta = s.diff()[consecutive]
    profiles.append({
        "cinema_ids": cid, "city_name": city, "observed_days": len(valid),
        "mean_shows": valid.mean(), "median_shows": valid.median(),
        "cv_shows": valid.std() / valid.mean() if valid.mean() else np.nan,
        "median_abs_daily_change": daily_delta.abs().median(),
        "pct_exact_same_consecutive": daily_delta.eq(0).mean(),
        "pct_change_le_2_consecutive": daily_delta.abs().le(2).mean(),
    })
    for i in range(7, len(dates) - 6):
        before = s.iloc[i-7:i]
        after = s.iloc[i:i+7]
        if before.notna().all() and after.notna().all():
            delta = after.mean() - before.mean()
            candidates.append({
                "date": dates[i], "cinema_ids": cid, "city_name": city,
                "before_mean_7d": before.mean(), "after_mean_7d": after.mean(),
                "delta_7d": delta, "ratio_7d": after.mean()/before.mean() if before.mean() else np.nan,
            })

profiles = pd.DataFrame(profiles).sort_values("cv_shows")
profiles.to_csv(OUT / "id_profiles.csv", index=False)
candidates = pd.DataFrame(candidates)
candidates.to_csv(OUT / "all_7d_changes.csv", index=False)

# Per-ID non-overlapping peaks remove adjacent dates that describe the same transition.
peaks = []
for cid, group in candidates.groupby("cinema_ids"):
    chosen = []
    for row in group.reindex(group.delta_7d.abs().sort_values(ascending=False).index).itertuples():
        if all(abs((row.date - other.date).days) > 14 for other in chosen):
            chosen.append(row)
    peaks.extend(chosen)
peaks = pd.DataFrame([row._asdict() for row in peaks]).drop(columns="Index")
peaks["abs_delta"] = peaks.delta_7d.abs()
peaks = peaks.sort_values("abs_delta", ascending=False)
peaks.to_csv(OUT / "nonoverlapping_peaks.csv", index=False)

known = candidates[((candidates.date.eq("2025-05-01")) &
                    candidates.cinema_ids.eq("4ed104ac0de10ba7eef50158603e0b32")) |
                   ((candidates.date.eq("2025-07-09")) &
                    candidates.cinema_ids.eq("53b77c875c67b68561415f23065e6e8d"))]
known.to_csv(OUT / "known_events.csv", index=False)
eligible = profiles[profiles.observed_days.ge(150)]
summary = {
    "ids": len(profiles), "ids_at_least_150_days": len(eligible),
    "median_id_cv": round(float(eligible.cv_shows.median()), 4),
    "median_id_pct_exact_same_consecutive": round(float(eligible.pct_exact_same_consecutive.median()), 4),
    "median_id_pct_change_le_2_consecutive": round(float(eligible.pct_change_le_2_consecutive.median()), 4),
    "known_events": known[["date", "city_name", "delta_7d", "ratio_7d"]].assign(
        date=lambda x: x.date.dt.strftime("%Y-%m-%d")).to_dict("records"),
    "number_nonoverlap_peaks_abs_delta_ge_20": int(peaks.abs_delta.ge(20).sum()),
    "unique_ids_with_peak_ge_20": int(peaks[peaks.abs_delta.ge(20)].cinema_ids.nunique()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("\nTop peaks:\n", peaks[["date", "city_name", "cinema_ids", "delta_7d", "ratio_7d"]].head(20).to_string(index=False))
print("\nLate May network:\n", network[network.date_show.between("2025-05-25", "2025-06-06")].to_string(index=False))
