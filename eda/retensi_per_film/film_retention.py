"""EDA: konsentrasi dan sebaran retensi D3 pasangan D2 lemah per film."""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("windows_v2", ROOT / "temp/exp_020_data_fix/build_windows_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])

d1_train = v2.film_d1(tr)
first = tr.groupby(v2.base_title(tr.movie_title)).date_show.min()
first_for_title = first.reindex(v2.base_title(pd.Series(d1_train.index))).to_numpy()
d1_train = d1_train[(d1_train > tr.date_show.min()) &
                    (d1_train + pd.Timedelta(days=9) <= tr.date_show.max()) &
                    (first_for_title > tr.date_show.min())]
d1_train = d1_train[~((d1_train <= v2.OUTAGE[1]) &
                      (d1_train + pd.Timedelta(days=9) >= v2.OUTAGE[0]))]
d1_test_base = th.groupby(v2.base_title(th.movie_title)).date_show.min()
title_to_base = v2.base_title(pd.Series(th.movie_title.unique()))
d1_test = pd.Series(d1_test_base.reindex(title_to_base).to_numpy(),
                    index=pd.Index(th.movie_title.unique(), name="movie_title"), name="D1")


def low_d2(df, d1, period):
    x = df.merge(d1.reset_index(), on="movie_title", how="inner")
    x["d"] = (x.date_show - x.D1).dt.days + 1
    d2 = x[x.d.eq(2)].copy()
    d2["tps_d2"] = d2.total_ticket / d2.total_show.replace(0, np.nan)
    d2 = d2[d2.tps_d2.le(8)].copy()
    d3 = x[x.d.eq(3)][["movie_title", "cinema_ids"]].drop_duplicates()
    d2 = d2.merge(d3.assign(active_d3=True), on=["movie_title", "cinema_ids"], how="left")
    d2["active_d3"] = d2.active_d3.fillna(False).astype(bool)
    d2["base"] = v2.base_title(d2.movie_title)
    d2["period"] = period
    d2["month"] = d2.D1.dt.to_period("M").astype(str)
    return d2


p = pd.concat([low_d2(tr, d1_train, "train"), low_d2(th, d1_test, "test")], ignore_index=True)
film = p.groupby(["period", "base", "month"]).active_d3.agg(n="size", selected="sum", rate="mean").reset_index()
film.to_csv(OUT / "by_film.csv", index=False)
adequate = film[film.n.ge(10)]
by_month = adequate.groupby(["period", "month"]).agg(n_films=("base", "nunique"),
                                                       median_film_rate=("rate", "median"),
                                                       share_films_rate_ge70=("rate", lambda s: s.ge(.7).mean()))
by_month.to_csv(OUT / "by_month.csv")

summary = {}
for period in ("train", "test"):
    f = film[film.period.eq(period)]
    a = adequate[adequate.period.eq(period)]
    selected = f.selected.sort_values(ascending=False)
    summary[period] = {
        "all_films_with_low_d2": len(f),
        "films_with_at_least_10_low_d2": len(a),
        "median_retention_across_adequate_films": float(a.rate.median()),
        "q25_retention_across_adequate_films": float(a.rate.quantile(.25)),
        "q75_retention_across_adequate_films": float(a.rate.quantile(.75)),
        "share_adequate_films_retention_ge70": float(a.rate.ge(.7).mean()),
        "share_adequate_films_retention_le50": float(a.rate.le(.5).mean()),
        "top5_film_share_of_selected_low_d2": float(selected.head(5).sum() / selected.sum()),
        "top10_film_share_of_selected_low_d2": float(selected.head(10).sum() / selected.sum()),
        "n_selected_low_d2": int(selected.sum()),
    }
summary["test_adequate_films_above_train_q75"] = float(
    adequate.loc[adequate.period.eq("test"), "rate"].gt(summary["train"]["q75_retention_across_adequate_films"]).mean())
rng = np.random.default_rng(2026)
train_rates = adequate.loc[adequate.period.eq("train"), "rate"].to_numpy()
test_rates = adequate.loc[adequate.period.eq("test"), "rate"].to_numpy()
delta = np.empty(3000)
for i in range(len(delta)):
    rt = rng.choice(train_rates, size=len(train_rates), replace=True)
    re = rng.choice(test_rates, size=len(test_rates), replace=True)
    delta[i] = np.median(re) - np.median(rt)
summary["median_film_retention_gap_test_minus_train"] = float(np.median(test_rates) - np.median(train_rates))
summary["film_bootstrap_median_gap_ci95"] = np.quantile(delta, [.025, .975]).tolist()
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("\nBy month:\n", by_month.round(3).to_string())
