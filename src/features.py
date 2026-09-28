"""Feature builder shared by local CV and the Kaggle notebook.

build(hist, tgt, ctx) -> one row per target (film, cinema, date) with features + scale + cal.
Works identically for simulated train samples and the real test, as long as `hist`/`tgt`
follow the test_history/test layout.
"""
import re

import numpy as np
import pandas as pd

from common import KEY, base_title, fmt

DOW_PROF = np.array([0.811, 0.789, 0.811, 0.783, 0.848, 1.290, 1.282])  # eda/04
HOL_LEVEL = 1.29          # a public holiday on a weekday behaves ~like a Saturday (eda/04)
SCHOOL_WEEKDAY = 1.15     # school-break Mon-Thu excess (eda/04)
RAMADAN_F = 1.0           # Ramadan evening depression; 1.0 = neutral, tuned by LB probe (see eda/10)
# External calendar facts, all public before 30 Sep 2025 (sources in the notebook):
#  - DKI Jakarta school calendar 2024/25 (break 28 Jun-12 Jul 2025) and 2025/26 (Kepdis 89/2025: 22-31 Dec 2025)
#  - SKB 3 Menteri cuti bersama 2025 (signed Oct 2024) and 2026 (signed 19 Sep 2025)
#  - Ramadan 1447 H = 30 days before 1 Syawal (21 Mar 2026 per SKB 2026) -> 19 Feb-20 Mar 2026
SCHOOL_BREAKS = [("2025-06-28", "2025-07-12"), ("2025-12-22", "2025-12-31")]
CUTI_BERSAMA = pd.to_datetime(["2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07", "2025-05-13", "2025-05-30",
                               "2025-06-09", "2025-08-18", "2025-12-26", "2026-02-16", "2026-03-18", "2026-03-20",
                               "2026-03-23", "2026-03-24"])
RAMADAN = ("2026-02-19", "2026-03-20")
GENRES = ["Horror", "Drama", "Action", "Comedy", "Romance", "Animation", "Family", "Thriller", "Mystery"]
RATING = {"Semua Umur": 0, "Remaja": 1, "Dewasa": 2, "Dewasa 21": 3}


def calendar(hol: pd.DataFrame, ramadan_f=RAMADAN_F) -> pd.DataFrame:
    c = hol.rename(columns={"date": "date_show"})[["date_show", "holiday_tipe"]].copy()
    c["dow"] = c.date_show.dt.dayofweek
    c["is_hol"] = ((c.holiday_tipe == "holiday") | c.date_show.isin(CUTI_BERSAMA)).astype(int)
    c["school"] = 0
    for a, b in SCHOOL_BREAKS:
        c.loc[c.date_show.between(a, b), "school"] = 1
    c["ramadan"] = c.date_show.between(*RAMADAN).astype(int)
    base = DOW_PROF[c.dow]
    base = np.where((c.school == 1) & (c.dow < 4), base * SCHOOL_WEEKDAY, base)
    # holiday uplift is learned outside Ramadan only; a cuti bersama in the last Ramadan days is mudik time
    base = np.where((c.is_hol == 1) & (c.ramadan == 0), np.maximum(base, HOL_LEVEL), base)
    c["cal"] = base * np.where(c.ramadan == 1, ramadan_f, 1.0)
    return c.drop(columns="holiday_tipe").set_index("date_show")


def build(hist, tgt, ctx):
    """ctx: dict(cal, movies, price, cin_size, releases) - see make_ctx."""
    cal, mv = ctx["cal"], ctx["movies"].set_index("original_title")
    d1 = hist.groupby("movie_title").date_show.min().rename("d1")
    h = hist.join(d1, on="movie_title")
    h["d"] = (h.date_show - h.d1).dt.days + 1

    # pair-level D1..D3 (zero-filled)
    piv = lambda v, p: h.pivot_table(index=KEY, columns="d", values=v, fill_value=0) \
        .reindex(columns=[1, 2, 3], fill_value=0).add_prefix(p)
    P = pd.concat([piv("total_ticket", "y"), piv("total_show", "sh"), piv("occupation_rate", "occ")], axis=1)
    P["mean3"] = P[["y1", "y2", "y3"]].mean(axis=1)
    P["scale"] = P.mean3.clip(lower=1)
    P["log_s"] = np.log1p(P.mean3)
    for i in (1, 2, 3):
        P[f"p{i}"] = P[f"y{i}"] / P.scale
    P["n_hist"] = (P[["y1", "y2", "y3"]] > 0).sum(axis=1)
    P["tps3"] = P.y3 / P.sh3.clip(lower=1)
    P["sh_trend"] = P.sh3 / P[["sh1", "sh2"]].max(axis=1).clip(lower=1)
    P["occ_mean"] = P[["occ1", "occ2", "occ3"]].mean(axis=1)
    P = P.reset_index()

    # film-level aggregates
    Fd = h.groupby(["movie_title", "d"]).agg(T=("total_ticket", "sum"), nc=("cinema_ids", "nunique")).unstack().fillna(0)
    Fd.columns = [f"f{a}{b}" for a, b in Fd.columns]
    Fd = Fd.reindex(columns=[f"f{a}{b}" for a in ["T", "nc"] for b in (1, 2, 3)], fill_value=0)
    Fm = Fd[["fT1", "fT2", "fT3"]].mean(axis=1).clip(lower=1)
    for i in (1, 2, 3):
        Fd[f"fp{i}"] = Fd[f"fT{i}"] / Fm
    Fd["f_logT"] = np.log1p(Fm)
    Fd["f_per_cin"] = Fm / Fd[["fnc1", "fnc2", "fnc3"]].max(axis=1).clip(lower=1)
    Fd["f_nc_trend"] = Fd.fnc3 / Fd.fnc1.clip(lower=1)
    Fd["f_occ"] = h.groupby("movie_title").occupation_rate.mean()
    Fd["f_tps3"] = h[h.d == 3].groupby("movie_title").total_ticket.sum() / h[h.d == 3].groupby("movie_title").total_show.sum()
    Fd = Fd.join(d1)
    Fd["base"] = base_title(pd.Series(Fd.index, index=Fd.index))
    Fd["fmt"] = fmt(pd.Series(Fd.index, index=Fd.index)).map({"2D": 0, "3D": 1, "IMAX 2D": 2, "IMAX 3D": 3})
    bT = h.assign(base=base_title(h.movie_title)).groupby("base").total_ticket.sum() / 3
    Fd["base_logT"] = np.log1p(Fd.base.map(bT))
    Fd["fmt_share"] = np.log1p(Fm) - Fd.base_logT
    Fd["d1_dow"] = Fd.d1.dt.dayofweek
    Fd["d1_month"] = Fd.d1.dt.month
    # metadata
    m = mv.reindex(Fd.base)
    Fd["rating"] = m.age_rating.map(RATING).values
    for g in GENRES:
        Fd[f"g_{g}"] = m.genre.fillna("").str.contains(g).astype(int).values
    Fd["n_genre"] = m.genre.fillna("").str.count(",").values + 1
    Fd["n_cast"] = m.casts.fillna("").str.count(",").values + 1
    # competition from the (known) release schedule: films opening after D1 within +9 days
    rel = ctx["releases"]
    comp = []
    for f, r in Fd.iterrows():
        o = rel[(rel.base != r.base) & (rel.d1 > r.d1) & (rel.d1 <= r.d1 + pd.Timedelta(days=9))]
        comp.append((len(o), np.log1p(o.tix.sum()), np.log1p(o.tix.max()) if len(o) else 0))
    Fd[["comp_n", "comp_logT", "comp_maxT"]] = comp
    same = rel[rel.d1.isin(Fd.d1.unique())].groupby("d1").tix.sum()
    Fd["cohort_logT"] = np.log1p(Fd.d1.map(same))
    Fd["f_share_cohort"] = Fd.base_logT - Fd.cohort_logT

    X = tgt.merge(P, on=KEY, how="left").merge(Fd.drop(columns=["base"]), left_on="movie_title", right_index=True, how="left")
    X["h"] = (X.date_show - X.d1).dt.days + 1
    cT = cal.reindex(X.date_show)
    X["dow"] = cT.dow.values
    X["t_hol"], X["t_school"], X["t_ramadan"] = cT.is_hol.values, cT.school.values, cT.ramadan.values
    X["cal_t"] = cT.cal.values
    ch = np.stack([cal.cal.reindex(X.d1 + pd.Timedelta(days=k)).values for k in range(3)], 1)
    X["cal_hist"] = ch.mean(1)
    X["cal_mult"] = X.cal_t / X.cal_hist
    X["hol_in_hist"] = np.stack([cal.is_hol.reindex(X.d1 + pd.Timedelta(days=k)).values for k in range(3)], 1).sum(1)
    X["n_comp_open"] = _open_count(X, ctx["releases"])
    # relative pair vs film
    X["share"] = X.log_s - np.log1p(X[["fT1", "fT2", "fT3"]].mean(axis=1) / X.fnc3.clip(lower=1))
    X["p3_rel"] = X.p3 - X.fp3
    X["p1_rel"] = X.p1 - X.fp1
    # cinema
    X["cin_size"] = X.cinema_ids.map(ctx["cin_size"])
    X["cin_nfilms"] = X.cinema_ids.map(ctx["cin_nfilms"])
    X["cin_new"] = X.cin_size.isna().astype(int)
    pr = ctx["price"].pivot(index="city_name", columns="price_day", values="ceil")
    X["price_wkd"] = X.city_name.map(pr.Weekday)
    X["price_prem"] = X.city_name.map(pr.Weekend / pr.Weekday)
    return X


def _open_count(X, rel):
    """Number of other titles opening in (D1, target date] - known release schedule."""
    d = np.sort(rel.d1.values.astype("datetime64[D]"))
    a = np.searchsorted(d, X.d1.values.astype("datetime64[D]"), side="right")
    b = np.searchsorted(d, X.date_show.values.astype("datetime64[D]"), side="right")
    return b - a


def make_ctx(hist, hol, movies, price, size_tx, ramadan_f=RAMADAN_F):
    """size_tx: transactions used for cinema size (train for both CV and test = past-only info)."""
    d1 = hist.groupby("movie_title").date_show.min()
    rel = hist.assign(base=base_title(hist.movie_title)).groupby("base").agg(
        d1=("date_show", "min"), tix=("total_ticket", "sum")).reset_index()
    rel["tix"] /= 3
    cs = size_tx.groupby("cinema_ids").total_ticket.sum() / size_tx.groupby("cinema_ids").date_show.nunique()
    nf = size_tx.groupby(["cinema_ids", "date_show"]).movie_title.nunique().groupby("cinema_ids").mean()
    return dict(cal=calendar(hol, ramadan_f), movies=movies, price=price, releases=rel,
                cin_size=np.log10(cs), cin_nfilms=nf)


def feature_cols(X):
    drop = {"id", "movie_title", "cinema_ids", "city_name", "date_show", "d1", "total_ticket", "scale", "mean3",
            "cal_t", "cal_hist", "r", "w"}
    return [c for c in X.columns if c not in drop and X[c].dtype != object]
