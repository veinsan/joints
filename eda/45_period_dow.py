"""45 - Is the weekly amplitude different in the test period? (model-independent calendar error)

Every model shares cal_mult = DOW profile estimated on Apr-Sep 2025. If Oct 2025-Mar 2026 has a steeper
weekend/weekday contrast, all versions are too flat on D4-D10 (weekend under, weekdays over): a
model-independent error, like the ~0.04-0.065 LB offset left after composition weights (eda/42, eda/44).
Same estimator in both periods, only on the D1-D3 windows (what the test exposes):
    log T(film, day) = film effect + decay(d) + dow(date) [+ holiday]          (film-national totals)
fitted on train-period visible windows and on test_history, plus the same model at PAIR level (film x cluster
effect) for the large-pair half. dow is relative to Thursday.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import fig_dir, load, release_dates, style
from features import CUTI_BERSAMA, visible_windows

plt = style()
F = fig_dir("45_period_dow")
d = load()
tr, th, hol = d["train"], d["hist"], d["hol"]
holi = set(hol[hol.holiday_tipe == "holiday"].date) | set(CUTI_BERSAMA)
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def fit(v, level="film"):
    v = v.copy()
    v["d1"] = v.movie_title.map(v.groupby("movie_title").date_show.min())
    v["d"] = (v.date_show - v.d1).dt.days + 1
    keys = ["movie_title"] if level == "film" else ["movie_title", "cinema_ids"]
    T = v.groupby(keys + ["date_show", "d"]).total_ticket.sum().reset_index()
    if level == "pair":
        T = T[T.groupby(keys).total_ticket.transform("min") >= 30]          # large, steady pairs only
        T = T[T.groupby(keys).d.transform("size") == 3]
    T["unit"] = T[keys].astype(str).agg("|".join, axis=1)
    T["dow"] = T.date_show.dt.dayofweek
    T["hol"] = T.date_show.isin(holi).astype(int)
    T = T[T.hol == 0]
    yv = np.log(T.total_ticket.values + 1)
    # within-unit demeaning (film / pair fixed effects), then OLS on decay + dow dummies
    Xd = pd.get_dummies(T.d.astype(str), prefix="d").drop(columns="d_1").astype(float)
    Xw = pd.get_dummies(T.dow.map(dict(enumerate(DOW))), prefix="w").astype(float).drop(columns="w_Thu", errors="ignore")
    X = pd.concat([Xd, Xw], axis=1)
    g = T.unit.values
    Xm = X - X.groupby(g).transform("mean")
    ym = yv - pd.Series(yv).groupby(g).transform("mean").values
    beta, *_ = np.linalg.lstsq(Xm.values, ym, rcond=None)
    b = pd.Series(beta, index=X.columns)
    n = T.dow.value_counts().reindex(range(7), fill_value=0)
    return b, n, len(T.unit.unique())


vis_tr = visible_windows(tr, release_dates(tr))
out = {}
for lvl in ("film", "pair"):
    for name, v in (("train", vis_tr), ("test", th)):
        b, n, u = fit(v, lvl)
        out[(lvl, name)] = b
        print(f"\n[{lvl}] {name}: {u} units | obs per dow {dict(zip(DOW, n.values))}")
        print("   dow vs Thu (multiplier):", {k[2:]: round(float(np.exp(x)), 3) for k, x in b.items() if k.startswith("w_")})
        print("   decay vs D1:", {k: round(float(np.exp(x)), 3) for k, x in b.items() if k.startswith("d_")})

# full-data train profile (what cal_mult uses) for reference
x = tr[~tr.date_show.isin(holi)].groupby("date_show").total_ticket.sum()
prof = x.groupby(x.index.dayofweek).mean()
print("\ncal_mult reference (all train days, national): Sat/Thu", round(prof[5] / prof[3], 3), "Sun/Thu", round(prof[6] / prof[3], 3),
      "Fri/Thu", round(prof[4] / prof[3], 3))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a_, lvl in zip(ax, ("film", "pair")):
    for name, c in (("train", "#2a78d6"), ("test", "#eb6834")):
        b = out[(lvl, name)]
        ks = [k for k in ["w_Wed", "w_Fri", "w_Sat", "w_Sun"] if k in b]
        a_.plot([k[2:] for k in ks], np.exp(b[ks].values), marker="o", color=c, label=name)
    a_.axhline(1, color="k", lw=.6); a_.set(title=f"DOW multiplier vs Thursday ({lvl} level, D1-D3 windows)"); a_.legend()
fig.savefig(F / "dow.png")
print(f"figures -> {F}")
