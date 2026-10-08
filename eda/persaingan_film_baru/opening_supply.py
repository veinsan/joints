"""EDA pasokan film baru D1-D3 pada klaster-tanggal yang sama."""

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("windows_v2", ROOT / "temp/exp_020_data_fix/build_windows_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


def early(file, period):
    data = pd.read_csv(ROOT / file, parse_dates=["date_show"])
    data["base"] = v2.base_title(data.movie_title)
    if period == "train":
        d1 = v2.film_d1(data)
        start, end = data.date_show.min(), data.date_show.max()
        first = data.groupby("base").date_show.min()
        mapped_first = first.reindex(v2.base_title(pd.Series(d1.index))).to_numpy()
        d1 = d1[(d1 > start) & (d1 + pd.Timedelta(days=9) <= end) & (mapped_first > start)]
        d1 = d1[~((d1 <= v2.OUTAGE[1]) & (d1 + pd.Timedelta(days=9) >= v2.OUTAGE[0]))]
    else:
        d1_base = data.groupby("base").date_show.min()
        titles = data.drop_duplicates("movie_title").set_index("movie_title").base
        d1 = titles.map(d1_base).rename("D1")
    x = data.merge(d1.reset_index(), on="movie_title", how="inner")
    x["d"] = (x.date_show - x.D1).dt.days + 1
    x = x[x.d.between(1, 3)].copy()
    x["period"] = period
    return x


early_rows = pd.concat([early("data/train.csv", "train"),
                        early("data/test_history.csv", "test")], ignore_index=True)
assert not early_rows.duplicated(["period", "movie_title", "cinema_ids", "date_show"]).any()
opening = early_rows.groupby(["period", "cinema_ids", "date_show"]).agg(
    opening_films=("base", "nunique"), opening_titles=("movie_title", "nunique"),
    opening_shows=("total_show", "sum"), opening_tickets=("total_ticket", "sum"),
).reset_index()
opening.to_csv(OUT / "cluster_day_opening_supply.csv", index=False)

selected = early_rows[early_rows.d == 3].merge(
    opening, on=["period", "cinema_ids", "date_show"], validate="many_to_one")
selected["other_opening_films"] = selected.opening_films - 1
selected["other_opening_shows"] = selected.opening_shows - selected.total_show
selected["tps"] = selected.total_ticket / selected.total_show
selected["one_show"] = selected.total_show.eq(1)
selected["weak_tps"] = selected.tps.le(8)
selected.to_csv(OUT / "d3_selected_opening_context.csv", index=False)

summary = selected.groupby("period").agg(
    pairs=("movie_title", "size"),
    median_opening_films=("opening_films", "median"),
    median_other_opening_shows=("other_opening_shows", "median"),
    median_own_shows=("total_show", "median"),
    one_show_share=("one_show", "mean"),
    weak_tps_share=("weak_tps", "mean"),
).reset_index()
summary.to_csv(OUT / "d3_period_summary.csv", index=False)
print("D3 selected summary:\n", summary.to_string(index=False))

selected["competitor_bucket"] = pd.cut(selected.other_opening_films, [-1, 0, 1, 2, 3, 99],
                                        labels=["0", "1", "2", "3", "4plus"])
bucket = selected.groupby(["period", "competitor_bucket"], observed=True).agg(
    pairs=("movie_title", "size"), own_shows_median=("total_show", "median"),
    one_show_share=("one_show", "mean"), weak_tps_share=("weak_tps", "mean"),
).reset_index()
bucket.to_csv(OUT / "d3_by_competitor_bucket.csv", index=False)
print("by new-film competitors:\n", bucket.to_string(index=False))
train_rates = bucket[bucket.period == "train"].set_index("competitor_bucket").weak_tps_share
test_bucket = bucket[bucket.period == "test"].set_index("competitor_bucket")
counterfactual = (test_bucket.pairs * train_rates).sum() / test_bucket.pairs.sum()
print("test mix x train weak_tps rates:", round(counterfactual, 6))

strata = selected.groupby(["period", "cinema_ids", "competitor_bucket"], observed=True).agg(
    n=("weak_tps", "size"), weak_rate=("weak_tps", "mean")
).reset_index()
tr_s = strata[strata.period == "train"].drop(columns="period").rename(
    columns={"n": "train_n", "weak_rate": "train_rate"})
te_s = strata[strata.period == "test"].drop(columns="period").rename(
    columns={"n": "test_n", "weak_rate": "test_rate"})
common = tr_s.merge(te_s, on=["cinema_ids", "competitor_bucket"], validate="one_to_one")
common = common[(common.train_n >= 5) & (common.test_n >= 5)].copy()
common.to_csv(OUT / "shared_cluster_competitor_strata.csv", index=False)
weight = common.test_n / common.test_n.sum()
print("shared cluster x competition strata", len(common), "test pair coverage",
      round(common.test_n.sum() / len(selected[selected.period == "test"]), 6),
      "test-weighted train/test weak rates",
      round((weight * common.train_rate).sum(), 6),
      round((weight * common.test_rate).sum(), 6))

selected["month"] = selected.date_show.dt.to_period("M").astype(str)
monthly = selected.groupby(["period", "month"]).agg(
    pairs=("movie_title", "size"), median_opening_films=("opening_films", "median"),
    median_other_opening_shows=("other_opening_shows", "median"),
    one_show_share=("one_show", "mean"), weak_tps_share=("weak_tps", "mean"),
).reset_index()
monthly.to_csv(OUT / "d3_monthly.csv", index=False)
print("monthly:\n", monthly.to_string(index=False))
