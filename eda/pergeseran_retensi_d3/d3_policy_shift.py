"""EDA murni: audit kelanjutan penayangan D1/D2 ke D3 pada train vs test_history."""

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


def make_panel(df, d1, period):
    x = df.merge(d1.rename("D1").reset_index(), on="movie_title", how="inner")
    x["d"] = (x.date_show - x.D1).dt.days + 1
    x = x[x.d.between(1, 3)].copy()
    p = x.pivot_table(
        index=["movie_title", "cinema_ids", "city_name", "D1"], columns="d",
        values=["total_ticket", "total_show", "occupation_rate"], aggfunc="first", fill_value=0,
    )
    p.columns = [f"{a}_d{b}" for a, b in p.columns]
    p = p.reset_index()
    for a in ["total_ticket", "total_show", "occupation_rate"]:
        for d in [1, 2, 3]:
            col = f"{a}_d{d}"
            if col not in p:
                p[col] = 0
    p["period"] = period
    p["active_d1"] = p.total_ticket_d1.gt(0)
    p["active_d2"] = p.total_ticket_d2.gt(0)
    p["active_d3"] = p.total_ticket_d3.gt(0)
    p["tps_d2"] = p.total_ticket_d2 / p.total_show_d2.replace(0, np.nan)
    p["month"] = p.D1.dt.to_period("M").astype(str)
    p["dow"] = p.D1.dt.day_name()
    p["n_pair_film"] = p.groupby("movie_title").cinema_ids.transform("size")
    return p


tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
te = pd.read_csv(ROOT / "data/test.csv")
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
train = make_panel(tr, d1_train, "train")
test = make_panel(th, d1_test, "test")
panel = pd.concat([train, test], ignore_index=True)

test_pairs = te[["movie_title", "cinema_ids"]].drop_duplicates()
matched = test.merge(test_pairs.assign(in_test=True), on=["movie_title", "cinema_ids"], how="left")
assert matched.in_test.fillna(False).eq(matched.active_d3).all(), "Aturan seleksi D3 tidak tepat"

def summary(g):
    return pd.Series({"n": len(g), "n_films": g.movie_title.nunique(),
                      "d3_rate": g.active_d3.mean(), "d3_dropout": 1 - g.active_d3.mean(),
                      "median_tps_d2": g.tps_d2.median(), "median_occ_d2": g.occupation_rate_d2.median()})

eligible = panel[panel.active_d2].copy()
eligible["tps_bucket"] = pd.cut(eligible.tps_d2, [0, 8, 15, 25, 40, np.inf],
                                labels=["<=8", "8-15", "15-25", "25-40", ">40"], include_lowest=True)
eligible["occ_bucket"] = pd.cut(eligible.occupation_rate_d2, [0, 5, 10, 20, 40, 100],
                                labels=["<=5", "5-10", "10-20", "20-40", ">40"], include_lowest=True)
eligible["film_size"] = pd.cut(eligible.n_pair_film, [0, 40, 80, 120, np.inf],
                               labels=["<=40", "40-80", "80-120", ">120"])
eligible["show_bucket"] = pd.cut(eligible.total_show_d2, [0, 1, 2, 5, np.inf],
                                  labels=["1", "2", "3-5", ">5"])

overall = eligible.groupby("period").apply(summary, include_groups=False)
by_tps = eligible.groupby(["period", "tps_bucket"], observed=True).apply(summary, include_groups=False)
by_occ = eligible.groupby(["period", "occ_bucket"], observed=True).apply(summary, include_groups=False)
by_size = eligible.groupby(["period", "film_size"], observed=True).apply(summary, include_groups=False)
by_month = eligible.groupby(["period", "month"]).apply(summary, include_groups=False)
by_month_tps = eligible.groupby(["period", "month", "tps_bucket"], observed=True).active_d3.agg(["size", "mean"])
by_month_tps["dropout"] = 1 - by_month_tps["mean"]
by_dow = eligible.groupby(["period", "dow"]).apply(summary, include_groups=False)
by_show = eligible.groupby(["period", "show_bucket"], observed=True).apply(summary, include_groups=False)

first_day = panel[panel.active_d1].copy()
first_day["tps_d1"] = first_day.total_ticket_d1 / first_day.total_show_d1.replace(0, np.nan)
first_day["tps_bucket"] = pd.cut(first_day.tps_d1, [0, 8, 15, 25, 40, np.inf],
    labels=["<=8", "8-15", "15-25", "25-40", ">40"], include_lowest=True)
d2_rate = first_day.groupby(["period", "tps_bucket"], observed=True).agg(
    n=("active_d2", "size"), d2_rate=("active_d2", "mean"))

# Standardisasi langsung pada komposisi test, hanya strata dengan data kedua periode.
eligible["stratum"] = eligible.tps_bucket.astype(str) + "/" + eligible.film_size.astype(str) + "/" + eligible.dow
st = eligible.groupby(["stratum", "period"]).active_d3.agg(["size", "mean"]).reset_index()
piv = st.pivot(index="stratum", columns="period", values=["size", "mean"]).dropna()
piv = piv[(piv[("size", "train")] >= 20) & (piv[("size", "test")] >= 20)]
weights = piv[("size", "test")] / piv[("size", "test")].sum()
standardized = {"n_strata": len(piv), "test_coverage": float(piv[("size", "test")].sum() / len(test[test.active_d2])),
                "train_d3_rate_at_test_mix": float((weights * piv[("mean", "train")]).sum()),
                "test_d3_rate_at_test_mix": float((weights * piv[("mean", "test")]).sum())}

# Bootstrap per judul dasar, karena pasangan dalam film saling bergantung.
bootstrap = []
rng = np.random.default_rng(2026)
valid_strata = piv.index
matrices = {}
for period in ("train", "test"):
    part = eligible[(eligible.period == period) & eligible.stratum.isin(valid_strata)].copy()
    part["base"] = v2.base_title(part.movie_title)
    groups = part.groupby(["base", "stratum"]).active_d3.agg(["size", "sum"]).reset_index()
    bases = pd.Index(groups.base.unique())
    n = np.zeros((len(bases), len(valid_strata)), dtype=float)
    y = np.zeros_like(n)
    row = bases.get_indexer(groups.base)
    col = valid_strata.get_indexer(groups.stratum)
    n[row, col] = groups["size"]
    y[row, col] = groups["sum"]
    matrices[period] = (n, y)
for _ in range(1000):
    counts = {}
    for period in ("train", "test"):
        n, y = matrices[period]
        draw = rng.integers(0, len(n), len(n))
        multiplicity = np.bincount(draw, minlength=len(n))
        counts[period] = (multiplicity @ n, multiplicity @ y)
    ntr, ytr = counts["train"]
    nte, yte = counts["test"]
    common = (ntr > 0) & (nte > 0)
    w = nte[common] / nte[common].sum()
    bootstrap.append(float(np.sum(w * (yte[common] / nte[common] - ytr[common] / ntr[common]))))
standardized["d3_rate_gap_test_minus_train"] = standardized["test_d3_rate_at_test_mix"] - standardized["train_d3_rate_at_test_mix"]
standardized["film_bootstrap_95ci"] = np.quantile(bootstrap, [0.025, 0.975]).tolist()

# Uji ketahanan kedua: samakan tiket per show, jumlah show, dan ukuran film.
eligible["stratum_show"] = eligible.tps_bucket.astype(str) + "/" + eligible.show_bucket.astype(str) + "/" + eligible.film_size.astype(str)
ss = eligible.groupby(["stratum_show", "period"]).active_d3.agg(["size", "mean"]).reset_index()
sp = ss.pivot(index="stratum_show", columns="period", values=["size", "mean"]).dropna()
sp = sp[(sp[("size", "train")] >= 20) & (sp[("size", "test")] >= 20)]
sw = sp[("size", "test")] / sp[("size", "test")].sum()
standardized["show_adjusted_strata"] = len(sp)
standardized["show_adjusted_test_coverage"] = float(sp[("size", "test")].sum() / len(test[test.active_d2]))
standardized["show_adjusted_gap"] = float((sw * (sp[("mean", "test")] - sp[("mean", "train")])).sum())

for name, obj in [("overall", overall), ("by_tps", by_tps), ("by_occ", by_occ),
                  ("by_film_size", by_size), ("by_month", by_month), ("by_month_tps", by_month_tps),
                  ("by_dow", by_dow),
                  ("by_show", by_show),
                  ("d2_by_d1_tps", d2_rate)]:
    obj.to_csv(OUT / f"{name}.csv")
    print(f"\n{name}\n{obj.round(3).to_string()}")
print("\nstandardized", standardized)
print("selection_exact", bool(matched.in_test.fillna(False).eq(matched.active_d3).all()))
(OUT / "summary.json").write_text(json.dumps({"overall": overall.reset_index().to_dict(orient="records"),
    "standardized": standardized, "selection_exact": True}, indent=2), encoding="utf-8")

plot = by_tps.reset_index().pivot(index="tps_bucket", columns="period", values="d3_dropout")
ax = plot[["train", "test"]].plot(kind="bar", figsize=(8, 4), color=["#24527a", "#de7a45"])
ax.set_ylabel("Proporsi berhenti sebelum D3")
ax.set_xlabel("Tiket per show pada D2")
ax.set_title("Pasangan lemah lebih sering bertahan di periode test")
ax.legend(title="Periode")
plt.tight_layout()
plt.savefig(OUT / "dropout_by_tps.png", dpi=160)
plt.close()
