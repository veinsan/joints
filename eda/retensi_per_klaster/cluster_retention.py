"""EDA: apakah pergeseran retensi D2 ke D3 terjadi di klaster bioskop yang sama?"""

from pathlib import Path
import importlib.util
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

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


def panel(df, d1, period):
    x = df.merge(d1.reset_index(), on="movie_title", how="inner")
    x["d"] = (x.date_show - x.D1).dt.days + 1
    d2 = x.loc[x.d.eq(2), ["movie_title", "cinema_ids", "city_name", "D1", "total_ticket", "total_show"]].copy()
    d3 = x.loc[x.d.eq(3), ["movie_title", "cinema_ids"]].drop_duplicates()
    d2 = d2.merge(d3.assign(active_d3=True), on=["movie_title", "cinema_ids"], how="left")
    d2["active_d3"] = d2.active_d3.fillna(False).astype(bool)
    d2["tps"] = d2.total_ticket / d2.total_show.replace(0, np.nan)
    d2["period"] = period
    d2["base"] = v2.base_title(d2.movie_title)
    return d2


p = pd.concat([panel(tr, d1_train, "train"), panel(th, d1_test, "test")], ignore_index=True)
p["tps_bucket"] = pd.cut(p.tps, [0, 8, 15, 25, 40, np.inf],
                         labels=["<=8", "8-15", "15-25", "25-40", ">40"], include_lowest=True)
shared = np.intersect1d(p.loc[p.period.eq("train"), "cinema_ids"].unique(),
                        p.loc[p.period.eq("test"), "cinema_ids"].unique())
p_shared = p[p.cinema_ids.isin(shared)].copy()

low = p_shared[p_shared.tps.le(8)].copy()
by_cinema = low.groupby(["cinema_ids", "period"]).active_d3.agg(["size", "mean"]).reset_index()
wide = by_cinema.pivot(index="cinema_ids", columns="period", values=["size", "mean"]).dropna()
wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
wide["rate_gap"] = wide.mean_test - wide.mean_train
wide["n_min"] = wide[["size_train", "size_test"]].min(axis=1)
wide.to_csv(OUT / "by_cinema_low_tps.csv")

by_city = low.groupby(["city_name", "period"]).active_d3.agg(["size", "mean"]).reset_index()
city = by_city.pivot(index="city_name", columns="period", values=["size", "mean"]).dropna()
city.columns = [f"{metric}_{period}" for metric, period in city.columns]
city["rate_gap"] = city.mean_test - city.mean_train
city.to_csv(OUT / "by_city_low_tps.csv")

well = wide[(wide.size_train >= 5) & (wide.size_test >= 5)]
n_positive = int(well.rate_gap.gt(0).sum())
n_negative = int(well.rate_gap.lt(0).sum())
sign_p = float(binomtest(n_positive, n_positive + n_negative, .5, alternative="greater").pvalue)

# Standardisasi langsung di strata bioskop dan demand D2. Hanya overlap yang cukup.
st = p_shared.groupby(["cinema_ids", "tps_bucket", "period"], observed=True).active_d3.agg(["size", "mean"]).reset_index()
all_sw = st.pivot(index=["cinema_ids", "tps_bucket"], columns="period", values=["size", "mean"]).dropna()
sw = all_sw[(all_sw[("size", "train")] >= 3) & (all_sw[("size", "test")] >= 3)]
sw.to_csv(OUT / "matched_cinema_tps.csv")
w = sw[("size", "test")] / sw[("size", "test")].sum()
matched_gap = float((w * (sw[("mean", "test")] - sw[("mean", "train")])).sum())

thresholds = []
for minimum in (3, 5, 10):
    tab = all_sw[(all_sw[("size", "train")] >= minimum) & (all_sw[("size", "test")] >= minimum)]
    tw = tab[("size", "test")] / tab[("size", "test")].sum()
    thresholds.append({"minimum_per_period": minimum, "n_strata": len(tab),
                       "test_coverage": float(tab[("size", "test")].sum() / p_shared.period.eq("test").sum()),
                       "gap": float((tw * (tab[("mean", "test")] - tab[("mean", "train")])).sum())})
pd.DataFrame(thresholds).to_csv(OUT / "threshold_sensitivity.csv", index=False)

# Bootstrap film: pasangan bioskop dalam satu judul tidak independen.
strata = sw.index
matrices = {}
for period in ("train", "test"):
    part = p_shared[p_shared.period.eq(period)].copy()
    part["stratum"] = list(zip(part.cinema_ids, part.tps_bucket))
    part = part[part.stratum.isin(strata)]
    g = part.groupby(["base", "stratum"], observed=True).active_d3.agg(["size", "sum"]).reset_index()
    bases = pd.Index(g.base.unique())
    counts = np.zeros((len(bases), len(strata)), dtype=float)
    successes = np.zeros_like(counts)
    row = bases.get_indexer(g.base)
    col = strata.get_indexer(g.stratum)
    counts[row, col] = g["size"]
    successes[row, col] = g["sum"]
    matrices[period] = (counts, successes)
rng = np.random.default_rng(2026)
boot = []
for _ in range(1000):
    samples = {}
    for period in ("train", "test"):
        counts, successes = matrices[period]
        draw = rng.integers(0, len(counts), len(counts))
        multiplicity = np.bincount(draw, minlength=len(counts))
        samples[period] = multiplicity @ counts, multiplicity @ successes
    tn, ty = samples["train"]
    en, ey = samples["test"]
    common = (tn > 0) & (en > 0)
    weights = en[common] / en[common].sum()
    boot.append(float(np.sum(weights * (ey[common] / en[common] - ty[common] / tn[common]))))

summary = {
    "n_shared_cinemas": len(shared),
    "train_cinema_coverage": float(p_shared.period.eq("train").sum() / p.period.eq("train").sum()),
    "test_cinema_coverage": float(p_shared.period.eq("test").sum() / p.period.eq("test").sum()),
    "low_tps_shared_train_n": int(low.period.eq("train").sum()),
    "low_tps_shared_test_n": int(low.period.eq("test").sum()),
    "low_tps_shared_train_d3_rate": float(low.loc[low.period.eq("train"), "active_d3"].mean()),
    "low_tps_shared_test_d3_rate": float(low.loc[low.period.eq("test"), "active_d3"].mean()),
    "well_observed_cinemas": len(well),
    "positive_cinema_gap": n_positive,
    "negative_cinema_gap": n_negative,
    "zero_cinema_gap": int(well.rate_gap.eq(0).sum()),
    "median_cinema_gap": float(well.rate_gap.median()),
    "sign_test_one_sided_p": sign_p,
    "matched_cinema_tps_strata": len(sw),
    "matched_test_coverage": float(sw[("size", "test")].sum() / p_shared.period.eq("test").sum()),
    "matched_gap": matched_gap,
    "matched_film_bootstrap_ci_95": np.quantile(boot, [.025, .975]).tolist(),
    "minimum_count_sensitivity": thresholds,
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
