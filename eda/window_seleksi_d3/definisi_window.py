"""EDA 003: bagaimana host memilih D1 dan pasangan (film, klaster) di test.

python eda/window_seleksi_d3/definisi_window.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
D = Path("data")
tr = pd.read_csv(D / "train.csv", parse_dates=["date_show"])
th = pd.read_csv(D / "test_history.csv", parse_dates=["date_show"])
te = pd.read_csv(D / "test.csv", parse_dates=["date_show"])
res = {}


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


# 1. film uji yang juga ada di train
win = th.groupby("movie_title").date_show.min().rename("D1")
ov = sorted(set(te.movie_title) & set(tr.movie_title))
rows = []
for m in ov:
    t = tr[tr.movie_title == m]
    d = t.groupby("date_show").agg(n_cin=("cinema_ids", "nunique"), tix=("total_ticket", "sum"))
    rows.append({"film": m, "D1": win[m].date(), "train_first": d.index.min().date(), "train_last": d.index.max().date(),
                 "train_days": len(d), "train_tix": int(d.tix.sum()), "train_max_cin": int(d.n_cin.max()),
                 "hist_D1_cin": int(th[(th.movie_title == m) & (th.date_show == win[m])].cinema_ids.nunique()),
                 "hist_D1_tix": int(th[(th.movie_title == m) & (th.date_show == win[m])].total_ticket.sum())})
ovdf = pd.DataFrame(rows)
print(ovdf.to_string())
ovdf.to_csv(OUT / "overlap_films.csv", index=False)

# 2. film uji dengan history di klaster sebelum D1? (tidak mungkin di test_history, cek D1 per klaster)
first_pair = th.groupby(["movie_title", "cinema_ids"]).date_show.min().reset_index()
first_pair["off"] = (first_pair.date_show - first_pair.movie_title.map(win)).dt.days
p("test_pair_first_day_offset", first_pair.off.value_counts().sort_index().to_dict())

# 3. film train: kurva nasional dan tanggal rilis
nat = tr.groupby(["movie_title", "date_show"]).agg(n_cin=("cinema_ids", "nunique"), tix=("total_ticket", "sum"),
                                                   shows=("total_show", "sum")).reset_index()
rows = []
for m, g in nat.groupby("movie_title"):
    g = g.sort_values("date_show")
    mx = g.n_cin.max()
    wide = g[g.n_cin >= 0.5 * mx].date_show.min()
    # tanggal pertama dengan tiket >= 20% puncak 7 hari pertama setelah wide
    rows.append({"film": m, "first": g.date_show.min(), "wide": wide, "last": g.date_show.max(),
                 "n_days": len(g), "max_cin": mx, "tix": g.tix.sum(),
                 "pre_wide_days": int((g.date_show < wide).sum()),
                 "pre_wide_tix_share": g[g.date_show < wide].tix.sum() / g.tix.sum()})
fl = pd.DataFrame(rows)
fl["first_dow"] = fl["first"].dt.day_name()
fl["wide_dow"] = fl["wide"].dt.day_name()
fl["censored_left"] = fl["first"] == tr.date_show.min()
p("train_film_first_dow(not left-censored)", fl[~fl.censored_left].first_dow.value_counts().to_dict())
p("train_film_wide_dow(not left-censored)", fl[~fl.censored_left].wide_dow.value_counts().to_dict())
p("train_film_left_censored", int(fl.censored_left.sum()))
p("train_film_pre_wide_days", fl[~fl.censored_left].pre_wide_days.value_counts().sort_index().head(15).to_dict())
p("test_film_D1_dow", win.dt.day_name().value_counts().to_dict())
fl.to_csv(OUT / "train_film_release.csv", index=False)

# contoh film dengan pre-wide days (preview)
ex = fl[(~fl.censored_left) & (fl.pre_wide_days > 0)].sort_values("tix", ascending=False).head(8)
for m in ex.film:
    g = nat[nat.movie_title == m].sort_values("date_show").head(8)
    print(m, g[["date_show", "n_cin", "tix", "shows"]].assign(dow=g.date_show.dt.day_name().str[:3]).to_string(index=False))

# 4. pasangan history yang tidak masuk test
ph = th.groupby(["movie_title", "cinema_ids"]).agg(days=("date_show", "nunique"), tix=("total_ticket", "sum"),
                                                  last=("date_show", "max"), occ=("occupation_rate", "mean"),
                                                  shows=("total_show", "sum")).reset_index()
ph["last_off"] = (ph["last"] - ph.movie_title.map(win)).dt.days
tp = te[["movie_title", "cinema_ids"]].drop_duplicates().assign(in_test=1)
ph = ph.merge(tp, how="left").fillna({"in_test": 0})
p("excluded_pairs", int((ph.in_test == 0).sum()))
p("excluded_films_all_pairs", sorted(set(ph.movie_title) - set(te.movie_title)))
summ = ph.groupby("in_test").agg(n=("tix", "size"), med_tix=("tix", "median"), mean_days=("days", "mean"),
                                 frac_last_D3=("last_off", lambda s: (s == 2).mean()), med_occ=("occ", "median"))
print(summ.to_string())
p("in_test_rate_by_last_day_offset", ph.groupby("last_off").in_test.mean().round(3).to_dict())
p("in_test_rate_by_days", ph.groupby("days").in_test.mean().round(3).to_dict())
ph["tix_bin"] = pd.qcut(ph.tix, 10, duplicates="drop")
p("in_test_rate_by_tix_decile", {str(k): round(v, 3) for k, v in ph.groupby("tix_bin", observed=True).in_test.mean().items()})
# per film
pf = ph.groupby("movie_title").in_test.mean()
p("in_test_rate_per_film_desc", pf.describe().round(3).to_dict())
p("films_partial_exclusion", int(((pf > 0) & (pf < 1)).sum()))
ph.to_csv(OUT / "history_pairs.csv", index=False)

(OUT / "definisi_window.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
