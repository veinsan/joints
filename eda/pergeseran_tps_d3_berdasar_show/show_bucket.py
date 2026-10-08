"""Audit pergeseran tiket per show D3 pada strata jumlah show."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
x = pd.read_csv(ROOT / "eda/persaingan_film_baru/output/d3_selected_opening_context.csv")
assert len(x[x.period == "train"]) == 7988
assert len(x[x.period == "test"]) == 10373

x["show_bucket"] = pd.cut(x.total_show, [0, 1, 2, 4, 8, 16, float("inf")],
                          labels=["1", "2", "3-4", "5-8", "9-16", "17plus"])
x["weak_tps"] = x.tps.le(8)
summary = x.groupby(["period", "show_bucket"], observed=True).agg(
    pairs=("movie_title", "size"), median_tickets=("total_ticket", "median"),
    median_tps=("tps", "median"), weak_tps_share=("weak_tps", "mean"),
).reset_index()
summary.to_csv(OUT / "by_show_bucket.csv", index=False)
print(summary.to_string(index=False))

tr = summary[summary.period == "train"].set_index("show_bucket")
te = summary[summary.period == "test"].set_index("show_bucket")
cf = (te.pairs * tr.weak_tps_share).sum() / te.pairs.sum()
print("test show mix x train weak rate", round(cf, 6),
      "observed test", round(x[x.period == "test"].weak_tps.mean(), 6))

strata = x.groupby(["period", "cinema_ids", "show_bucket"], observed=True).agg(
    n=("weak_tps", "size"), weak_rate=("weak_tps", "mean")
).reset_index()
tr_s = strata[strata.period == "train"].drop(columns="period").rename(
    columns={"n": "train_n", "weak_rate": "train_rate"})
te_s = strata[strata.period == "test"].drop(columns="period").rename(
    columns={"n": "test_n", "weak_rate": "test_rate"})
common = tr_s.merge(te_s, on=["cinema_ids", "show_bucket"], validate="one_to_one")
common = common[(common.train_n >= 5) & (common.test_n >= 5)].copy()
common.to_csv(OUT / "shared_cluster_show_strata.csv", index=False)
weights = common.test_n / common.test_n.sum()
print("shared cluster x show strata", len(common),
      "test coverage", round(common.test_n.sum() / len(x[x.period == "test"]), 6),
      "train/test weak rate", round((weights * common.train_rate).sum(), 6),
      round((weights * common.test_rate).sum(), 6))
