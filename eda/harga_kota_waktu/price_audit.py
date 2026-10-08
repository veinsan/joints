"""Audit kelengkapan tabel harga statis dan hubungannya dengan ekor D3."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
x = pd.read_csv(ROOT / "eda/persaingan_film_baru/output/d3_selected_opening_context.csv",
                parse_dates=["date_show"])
prices = pd.read_csv(ROOT / "data/ticket_prices.csv")
assert not prices.duplicated(["city_name", "price_day"]).any()

weekday = x.date_show.dt.dayofweek
x["price_day"] = "Weekday"
x.loc[weekday.eq(4), "price_day"] = "Friday"
x.loc[weekday.ge(5), "price_day"] = "Weekend"
x = x.merge(prices, on=["city_name", "price_day"], how="left", validate="many_to_one")
x["weak_tps"] = x.tps.le(8)
print("D3 pairs", x.groupby("period").size().to_dict(),
      "missing price", x.ceil.isna().groupby(x.period).sum().to_dict())
print("prices city count", prices.city_name.nunique(),
      "price_day each", prices.groupby("city_name").price_day.nunique().value_counts().to_dict())

x["price_quartile"] = pd.qcut(x.ceil, 4, duplicates="drop")
summary = x.groupby(["period", "price_quartile"], observed=True).agg(
    pairs=("movie_title", "size"), weak_tps_share=("weak_tps", "mean"),
    median_tps=("tps", "median"), median_price=("ceil", "median"),
).reset_index()
summary.to_csv(OUT / "by_price_quartile.csv", index=False)
print(summary.to_string(index=False))
tr = summary[summary.period == "train"].set_index("price_quartile")
te = summary[summary.period == "test"].set_index("price_quartile")
print("test price mix x train weak rate",
      (te.pairs * tr.weak_tps_share).sum() / te.pairs.sum())

city = x.groupby(["period", "city_name"]).agg(
    n=("weak_tps", "size"), weak_rate=("weak_tps", "mean"),
    median_tps=("tps", "median"), median_price=("ceil", "median"),
).reset_index()
tr_c = city[city.period == "train"].drop(columns="period").rename(
    columns={"n": "train_n", "weak_rate": "train_rate", "median_tps": "train_med_tps"})
te_c = city[city.period == "test"].drop(columns="period").rename(
    columns={"n": "test_n", "weak_rate": "test_rate", "median_tps": "test_med_tps"})
common = tr_c.merge(te_c, on="city_name", suffixes=("_train", "_test"))
common = common[(common.train_n >= 30) & (common.test_n >= 30)].copy()
common.to_csv(OUT / "shared_cities.csv", index=False)
print("shared cities >=30 pairs", len(common),
      "higher weak rate test", int((common.test_rate > common.train_rate).sum()),
      "test coverage", common.test_n.sum() / len(x[x.period == "test"]))
