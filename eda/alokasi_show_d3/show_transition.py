"""EDA: alokasi show D2->D3 pada pasangan lemah yang masih aktif."""

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


def panel(df, d1, period):
    x = df.merge(d1.reset_index(), on="movie_title", how="inner")
    x["d"] = (x.date_show - x.D1).dt.days + 1
    x = x[x.d.isin([2, 3])]
    p = x.pivot_table(index=["movie_title", "cinema_ids"], columns="d",
                      values=["total_ticket", "total_show", "occupation_rate"],
                      aggfunc="first", fill_value=0)
    p.columns = [f"{a}_d{b}" for a, b in p.columns]
    p = p.reset_index()
    for a in ("total_ticket", "total_show", "occupation_rate"):
        for d in (2, 3):
            col = f"{a}_d{d}"
            if col not in p:
                p[col] = 0
    p["period"] = period
    p["tps_d2"] = p.total_ticket_d2 / p.total_show_d2.replace(0, np.nan)
    p["tps_d3"] = p.total_ticket_d3 / p.total_show_d3.replace(0, np.nan)
    p["show_ratio_32"] = p.total_show_d3 / p.total_show_d2.replace(0, np.nan)
    return p


p = pd.concat([panel(tr, d1_train, "train"), panel(th, d1_test, "test")], ignore_index=True)
all_d2 = p[p.total_ticket_d2.gt(0)].copy()
all_d2["d2_weak"] = all_d2.tps_d2.le(8)
unconditional = all_d2.groupby(["period", "d2_weak"]).agg(
    n=("tps_d2", "size"), d3_active=("total_ticket_d3", lambda s: s.gt(0).mean()),
    mean_d3_show_to_d2_ratio=("show_ratio_32", "mean"),
    median_d3_show_to_d2_ratio=("show_ratio_32", "median"),
    mean_d3_shows=("total_show_d3", "mean"))
unconditional.to_csv(OUT / "unconditional_d2_transition.csv")
p = p[p.total_ticket_d2.gt(0) & p.total_ticket_d3.gt(0)].copy()
p["d2_weak"] = p.tps_d2.le(8)
summary_table = p.groupby(["period", "d2_weak"]).agg(
    n=("tps_d2", "size"), n_films=("movie_title", "nunique"),
    median_d2_tps=("tps_d2", "median"), median_d3_tps=("tps_d3", "median"),
    median_d2_shows=("total_show_d2", "median"), median_d3_shows=("total_show_d3", "median"),
    median_show_ratio=("show_ratio_32", "median"),
    share_d3_one_show=("total_show_d3", lambda s: s.eq(1).mean()),
    share_show_cut=("show_ratio_32", lambda s: s.lt(1).mean()),
    share_show_same=("show_ratio_32", lambda s: s.eq(1).mean()),
    share_d3_ticket_le8=("total_ticket_d3", lambda s: s.le(8).mean()),
)
summary_table.to_csv(OUT / "transition_summary.csv")

weak = p[p.d2_weak].copy()
weak["show_d2_bucket"] = pd.cut(weak.total_show_d2, [0, 1, 2, 5, np.inf],
                                labels=["1", "2", "3-5", ">5"])
show_bins = weak.groupby(["period", "show_d2_bucket"], observed=True).agg(
    n=("tps_d2", "size"), median_d3_shows=("total_show_d3", "median"),
    median_show_ratio=("show_ratio_32", "median"),
    share_show_cut=("show_ratio_32", lambda s: s.lt(1).mean()),
    share_d3_one_show=("total_show_d3", lambda s: s.eq(1).mean()))
show_bins.to_csv(OUT / "weak_by_d2_show_bucket.csv")

rates = show_bins.share_show_cut.unstack("period")
weights = show_bins.n.xs("test") / show_bins.n.xs("test").sum()
adjusted_cut = {
    "train_cut_rate_at_test_d2_show_mix": float((weights * rates.train).sum()),
    "test_cut_rate_at_test_d2_show_mix": float((weights * rates.test).sum()),
}
adjusted_cut["test_minus_train_gap"] = (adjusted_cut["test_cut_rate_at_test_d2_show_mix"] -
                                         adjusted_cut["train_cut_rate_at_test_d2_show_mix"])
(OUT / "adjusted_cut.json").write_text(json.dumps(adjusted_cut, indent=2), encoding="utf-8")

print(summary_table.round(4).to_string())
print("\nUnconditional D2 panel:\n", unconditional.round(4).to_string())
print("\nBy show bucket:\n", show_bins.round(4).to_string())
print("\nAdjusted cut:", json.dumps(adjusted_cut, indent=2))
