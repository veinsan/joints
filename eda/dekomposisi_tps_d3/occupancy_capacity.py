"""Dekomposisi deskriptif tiket/show D3 menjadi okupansi dan kapasitas tersirat."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(parents=True, exist_ok=True)
x = pd.read_csv(ROOT / "eda/persaingan_film_baru/output/d3_selected_opening_context.csv")
x["show_bucket"] = pd.cut(x.total_show, [0, 1, 2, 4, 8, 16, float("inf")],
                          labels=["1", "2", "3-4", "5-8", "9-16", "17plus"])
x["valid_occ"] = x.occupation_rate.between(0.01, 100)
x["capacity_per_show"] = np.where(x.valid_occ,
                                  x.total_ticket * 100 / x.occupation_rate / x.total_show,
                                  np.nan)
x["low_occ"] = x.occupation_rate.le(10)

summary = x.groupby(["period", "show_bucket"], observed=True).agg(
    pairs=("movie_title", "size"), valid_occ=("valid_occ", "sum"),
    median_tps=("tps", "median"),
    median_occ=("occupation_rate", "median"),
    median_capacity=("capacity_per_show", "median"),
    low_occ_share=("low_occ", "mean"),
).reset_index()
summary.to_csv(OUT / "by_show_bucket.csv", index=False)
print(summary.to_string(index=False))

overall = x.groupby("period").agg(
    pairs=("movie_title", "size"), invalid_occ=("valid_occ", lambda s: (~s).sum()),
    zero_occ=("occupation_rate", lambda s: s.eq(0).sum()),
    over_100_occ=("occupation_rate", lambda s: s.gt(100).sum()),
    below_1_occ=("occupation_rate", lambda s: ((s > 0) & (s < 1)).sum()),
    median_tps=("tps", "median"), median_occ=("occupation_rate", "median"),
    median_capacity=("capacity_per_show", "median"),
    low_occ_share=("low_occ", "mean"),
).reset_index()
overall.to_csv(OUT / "overall.csv", index=False)
print("overall:\n", overall.to_string(index=False))

positive = x[x.valid_occ].copy()
positive["log_tps"] = np.log(positive.tps)
positive["log_occ"] = np.log(positive.occupation_rate / 100)
positive["log_capacity"] = np.log(positive.capacity_per_show)
means = positive.groupby("period")[["log_tps", "log_occ", "log_capacity"]].mean()
diff = means.loc["test"] - means.loc["train"]
means.to_csv(OUT / "log_means.csv")
print("test minus train log means:\n", diff.to_string())
print("identity residual", diff.log_tps - diff.log_occ - diff.log_capacity)

sensible = positive[(positive.occupation_rate >= 1) & positive.capacity_per_show.between(50, 400)]
sensible_means = sensible.groupby("period")[["log_tps", "log_occ", "log_capacity"]].mean()
print("sensitivity 1<=occ<=100 and 50<=capacity<=400 counts",
      sensible.groupby("period").size().to_dict())
print("sensitivity log mean differences:\n",
      (sensible_means.loc["test"] - sensible_means.loc["train"]).to_string())

strata = positive.groupby(["period", "cinema_ids", "show_bucket"], observed=True).agg(
    n=("movie_title", "size"), mean_log_tps=("log_tps", "mean"),
    mean_log_occ=("log_occ", "mean"), mean_log_capacity=("log_capacity", "mean"),
).reset_index()
tr = strata[strata.period == "train"].drop(columns="period")
te = strata[strata.period == "test"].drop(columns="period")
common = tr.merge(te, on=["cinema_ids", "show_bucket"], suffixes=("_train", "_test"),
                  validate="one_to_one")
common = common[(common.n_train >= 5) & (common.n_test >= 5)].copy()
common.to_csv(OUT / "shared_cluster_show_strata.csv", index=False)
w = common.n_test / common.n_test.sum()
print("common strata", len(common), "test coverage", round(common.n_test.sum() / len(x[x.period == "test"]), 6))
for col in ["mean_log_tps", "mean_log_occ", "mean_log_capacity"]:
    delta = (w * (common[f"{col}_test"] - common[f"{col}_train"])).sum()
    print(col, "test-train weighted log difference", round(delta, 6),
          "ratio", round(np.exp(delta), 6))
