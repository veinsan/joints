"""EDA: uraikan kenaikan pasangan lemah terpilih D3 menjadi campuran D2 dan retensi."""

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
    x = x[x.d.between(1, 3)]
    p = x.pivot_table(index=["movie_title", "cinema_ids"], columns="d",
                      values=["total_ticket", "total_show"], aggfunc="first", fill_value=0)
    p.columns = [f"{a}_d{b}" for a, b in p.columns]
    p = p.reset_index()
    for a in ("total_ticket", "total_show"):
        for d in (1, 2, 3):
            col = f"{a}_d{d}"
            if col not in p:
                p[col] = 0
    p["period"] = period
    p["tps_d2"] = p.total_ticket_d2 / p.total_show_d2.replace(0, np.nan)
    p["active_d2"] = p.total_ticket_d2.gt(0)
    p["active_d3"] = p.total_ticket_d3.gt(0)
    p["scale"] = np.maximum(p[["total_ticket_d1", "total_ticket_d2", "total_ticket_d3"]].sum(axis=1) / 3, 1)
    return p


panel_all = pd.concat([panel(tr, d1_train, "train"), panel(th, d1_test, "test")], ignore_index=True)
eligible = panel_all[panel_all.active_d2].copy()
eligible["bucket"] = pd.cut(eligible.tps_d2, [0, 8, 15, 25, 40, np.inf],
                            labels=["<=8", "8-15", "15-25", "25-40", ">40"], include_lowest=True)
tab = eligible.groupby(["period", "bucket"], observed=True).agg(
    eligible=("active_d3", "size"), selected=("active_d3", "sum"))
tab["continuation"] = tab.selected / tab.eligible
tab["eligible_share"] = tab.eligible / tab.groupby(level=0).eligible.transform("sum")
tab["selected_share"] = tab.selected / tab.groupby(level=0).selected.transform("sum")
tab.to_csv(OUT / "d2_mix_retention.csv")

train_rate = tab.xs("train").continuation
test_eligible = tab.xs("test").eligible
counterfactual_selected = test_eligible * train_rate
counterfactual_low_share = float(counterfactual_selected.loc["<=8"] / counterfactual_selected.sum())

selected = panel_all[panel_all.active_d3].copy()
selected["low_d2"] = selected.tps_d2.le(8)
selected["small_scale"] = selected.scale.le(20)
cross = selected.groupby(["period", "low_d2", "small_scale"]).size().rename("n").reset_index()
cross.to_csv(OUT / "selected_cross_tab.csv", index=False)

summary = {
    "train_eligible_low_d2_share": float(tab.loc[("train", "<=8"), "eligible_share"]),
    "test_eligible_low_d2_share": float(tab.loc[("test", "<=8"), "eligible_share"]),
    "train_selected_low_d2_share_among_d2_active": float(tab.loc[("train", "<=8"), "selected_share"]),
    "test_selected_low_d2_share_among_d2_active": float(tab.loc[("test", "<=8"), "selected_share"]),
    "counterfactual_test_d2_mix_with_train_retention_low_share": counterfactual_low_share,
    "train_all_selected_low_d2_share": float(selected.loc[selected.period.eq("train"), "low_d2"].mean()),
    "test_all_selected_low_d2_share": float(selected.loc[selected.period.eq("test"), "low_d2"].mean()),
    "train_all_selected_small_scale_share": float(selected.loc[selected.period.eq("train"), "small_scale"].mean()),
    "test_all_selected_small_scale_share": float(selected.loc[selected.period.eq("test"), "small_scale"].mean()),
    "train_small_scale_that_are_low_d2_share": float(selected.loc[selected.period.eq("train") & selected.small_scale, "low_d2"].mean()),
    "test_small_scale_that_are_low_d2_share": float(selected.loc[selected.period.eq("test") & selected.small_scale, "low_d2"].mean()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(tab.round(4).to_string())
print(json.dumps(summary, indent=2))
