"""EDA murni: komposisi skala MASE dan hubungan deskriptif dengan nol target."""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("windows_v2", ROOT / "temp/exp_020_data_fix/build_windows_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
train, target, _ = v2.build_windows(tr)
test = v2.test_windows(th)
te_pairs = pd.read_csv(ROOT / "data/test.csv")[["movie_title", "cinema_ids"]].drop_duplicates()
test = test.merge(te_pairs, on=["movie_title", "cinema_ids"])
assert len(test) == len(te_pairs)

bins = [0, 1, 3, 10, 20, 50, 100, 300, 1000, np.inf]
labels = ["1", "1-3", "3-10", "10-20", "20-50", "50-100", "100-300", "300-1000", ">1000"]
all_pairs = pd.concat([train.assign(period="train"), test.assign(period="test")], ignore_index=True)
all_pairs["scale_bin"] = pd.cut(all_pairs.scale, bins=bins, labels=labels, include_lowest=True)
all_pairs["base"] = v2.base_title(all_pairs.movie_title)
all_pairs["film_median_scale"] = all_pairs.groupby(["period", "base"]).scale.transform("median")
all_pairs["scale_vs_film"] = all_pairs.scale / all_pairs.film_median_scale
all_pairs["d3_tps"] = all_pairs.total_ticket_d3 / all_pairs.total_show_d3.replace(0, np.nan)
all_pairs["month"] = all_pairs.D1.dt.to_period("M").astype(str)

summary = all_pairs.groupby("period").agg(n=("scale", "size"), films=("base", "nunique"),
    median_scale=("scale", "median"), p10_scale=("scale", lambda x: x.quantile(.1)),
    pct_scale_le20=("scale", lambda x: (x <= 20).mean()),
    median_d3_tps=("d3_tps", "median"), median_d3_occ=("occupation_rate_d3", "median"),
    median_rel_scale=("scale_vs_film", "median"))
composition = all_pairs.groupby(["period", "scale_bin"], observed=True).agg(
    n=("scale", "size"), median_scale=("scale", "median"),
    median_d3_tps=("d3_tps", "median"), median_d3_occ=("occupation_rate_d3", "median"))
composition["share"] = composition.n / composition.groupby(level=0).n.transform("sum")
by_month = all_pairs.groupby(["period", "month"]).agg(n=("scale", "size"),
    median_scale=("scale", "median"), pct_scale_le20=("scale", lambda x: (x <= 20).mean()),
    median_d3_occ=("occupation_rate_d3", "median"))

target = target.merge(train[["movie_title", "cinema_ids", "scale"]], on=["movie_title", "cinema_ids"])
target["scale_bin"] = pd.cut(target.scale, bins=bins, labels=labels, include_lowest=True)
zero_by_scale = target.groupby("scale_bin", observed=True).agg(n=("total_ticket", "size"),
    zero_rate=("total_ticket", lambda x: x.eq(0).mean()),
    median_positive_tix=("total_ticket", lambda x: x[x > 0].median()),
    median_y_over_scale=("total_ticket", lambda x: np.median(x / target.loc[x.index, "scale"])))
zero_by_scale["train_pair_share"] = composition.loc["train"].share.reindex(zero_by_scale.index)
zero_by_scale["test_pair_share"] = composition.loc["test"].share.reindex(zero_by_scale.index)

summary.to_csv(OUT / "summary.csv")
composition.to_csv(OUT / "composition.csv")
by_month.to_csv(OUT / "by_month.csv")
zero_by_scale.to_csv(OUT / "train_zero_by_scale.csv")
all_pairs[["period", "movie_title", "cinema_ids", "scale", "scale_vs_film", "d3_tps", "occupation_rate_d3"]].to_csv(OUT / "pairs.csv", index=False)
print("summary\n", summary.round(3).to_string())
print("\ncomposition\n", composition.round(3).to_string())
print("\nby month\n", by_month.round(3).to_string())
print("\ntrain target zero by scale\n", zero_by_scale.round(3).to_string())

plot = composition.reset_index().pivot(index="scale_bin", columns="period", values="share")
ax = plot[["train", "test"]].plot(kind="bar", figsize=(9, 4), color=["#24527a", "#de7a45"])
ax.set_ylabel("Proporsi pasangan")
ax.set_xlabel("Bucket skala MASE")
ax.set_title("Evaluasi test memuat lebih banyak pasangan berskala kecil")
ax.legend(title="Periode")
plt.tight_layout()
plt.savefig(OUT / "scale_composition.png", dpi=160)
plt.close()
