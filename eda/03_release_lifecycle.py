"""03 - Reconstruct D1 in train so simulated samples look like test; film lifecycle curves."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, fig_dir, load, release_dates, scale, simulate, style

plt = style()
F = fig_dir("03_lifecycle")
d = load()
tr, th = d["train"], d["hist"]


def profile(h):
    """Film-level D1-D3 signature used to compare simulated vs real test history."""
    d1 = h.groupby("movie_title").date_show.min()
    x = h.join(d1.rename("d1"), on="movie_title")
    x["d"] = (x.date_show - x.d1).dt.days + 1
    nc = x.groupby(["movie_title", "d"]).cinema_ids.nunique().unstack().reindex(columns=[1, 2, 3]).fillna(0)
    t = x.groupby(["movie_title", "d"]).total_ticket.sum().unstack().reindex(columns=[1, 2, 3]).fillna(0)
    return pd.DataFrame({"dow": d1.dt.dayofweek, "nc1_nc3": nc[1] / nc[3].clip(lower=1),
                         "t3_t1": t[3] / t[1].clip(lower=1), "nc3": nc[3]})


first_sale = tr.groupby("movie_title").date_show.min()
running = first_sale[first_sale == tr.date_show.min()].index  # already in cinemas before 1 Apr
pt = profile(th)
print("films already running on 2025-04-01 (excluded from simulation):", len(running))
print("\n=== choose D1 rule: first day with coverage >= frac * peak coverage ===")
print("test reference:", {k: round(v, 3) for k, v in {
    "Wed/Thu share": pt.dow.isin([2, 3]).mean(), "median nc1/nc3": pt.nc1_nc3.median(),
    "median T3/T1": pt.t3_t1.median(), "median nc3": pt.nc3.median()}.items()})
rows = []
for frac, mn in [(f, m) for m in [0, 25] for f in [0.3, 0.5, 0.7]]:
    r = release_dates(tr, frac, mn).drop(running, errors="ignore")
    hs, _ = simulate(tr, r)
    p = profile(hs)
    rows.append(dict(frac=frac, min_nc=mn, films=len(p), wed_thu=p.dow.isin([2, 3]).mean(), nc1_nc3=p.nc1_nc3.median(),
                     t3_t1=p.t3_t1.median(), nc3=p.nc3.median()))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# Alternative rule: largest single-day jump in coverage (release day = wide opening)
nc = tr.groupby(["movie_title", "date_show"]).cinema_ids.nunique().rename("nc").reset_index()
full = nc.set_index("date_show").groupby("movie_title").nc.apply(lambda s: s.asfreq("D", fill_value=0)).reset_index()
full["jump"] = full.groupby("movie_title").nc.diff().fillna(full.nc)
rj = full.loc[full.groupby("movie_title").jump.idxmax()].set_index("movie_title").date_show.rename("d1").drop(running, errors="ignore")
p = profile(simulate(tr, rj)[0])
print(f"max-jump rule: films={len(p)} wed_thu={p.dow.isin([2, 3]).mean():.3f} nc1/nc3={p.nc1_nc3.median():.3f} "
      f"T3/T1={p.t3_t1.median():.3f}")
r5 = release_dates(tr, 0.5).drop(running, errors="ignore")
print("agreement jump-rule vs frac=0.5:", (rj.reindex(r5.index) == r5).mean().round(3))

# ---------------------------------------------------------------- lifecycle with chosen rule
D1 = release_dates(tr, 0.5, 25).drop(running, errors="ignore")
x = tr.join(D1, on="movie_title", how="inner")
x["d"] = (x.date_show - x.d1).dt.days + 1
film = x.groupby(["movie_title", "d"]).total_ticket.sum().unstack().fillna(0)
film = film.loc[:, (film.columns >= -6) & (film.columns <= 42)]
norm = film.div(film[[1, 2, 3]].mean(axis=1).clip(lower=1), axis=0)
print("\n=== national film curve, normalised by mean(D1..D3) ===")
print("median by day:", norm.loc[:, 1:14].median().round(3).to_dict())
print("pre-release share of tickets (d<1):", (film.loc[:, film.columns < 1].sum().sum() / film.sum().sum()).round(4))

dow = D1.dt.dayofweek
fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
for i, (k, g) in enumerate(norm.groupby(dow.reindex(norm.index))):
    if len(g) >= 10:
        ax[0].plot(g.columns[(g.columns >= 1) & (g.columns <= 21)], g.loc[:, 1:21].median(),
                   label=f"D1={['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][int(k)]} (n={len(g)})")
ax[0].axvspan(3.5, 10.5, color="#eda100", alpha=.12)
ax[0].set(title="Median national curve / mean(D1-D3)", xlabel="day since D1", ylabel="ratio"); ax[0].legend()

hs, ts = simulate(tr, D1)
ts = ts.join(scale(hs), on=KEY)
ts["h"] = (ts.date_show - ts.movie_title.map(D1)).dt.days + 1
ts["r"] = ts.total_ticket / ts.scale
m = ts.groupby(["h", ts.movie_title.map(dow)]).r.median().unstack()
m.columns = [["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][c] for c in m.columns]
print("\n=== pair-level median r by horizon x D1 weekday ===\n", m.round(3))
m[[c for c in ["Wed", "Thu", "Fri"] if c in m]].plot(ax=ax[1], marker="o")
ax[1].set(title="Pair-level median r by horizon", xlabel="D (4..10)", ylabel="median y/s")

# share of pairs still showing by horizon: survival
surv = ts.assign(on=ts.total_ticket > 0).groupby("h").on.mean()
ax[2].plot(surv.index, surv.values, marker="o")
ax[2].set(title="Share of pairs still selling (survival)", xlabel="D", ylim=(0, 1))
fig.savefig(F / "lifecycle.png")

# heterogeneity: distribution of film-level D4-10 / D1-3 ratio
fr = film.loc[:, 4:10].mean(axis=1) / film.loc[:, 1:3].mean(axis=1).clip(lower=1)
print("\nfilm-level mean(D4..10)/mean(D1..3) quantiles:", fr.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
f3 = film[3] / film[1].clip(lower=1)
print("corr(film D3/D1, film D4-10 ratio) spearman:", round(pd.concat([f3, fr], axis=1).corr("spearman").iloc[0, 1], 3))
fig, ax = plt.subplots(figsize=(5, 3.6))
ax.scatter(f3.clip(upper=4), fr.clip(upper=3), s=10, alpha=.5, color="#2a78d6")
ax.set(title="Film momentum: D3/D1 vs D4-10 ratio", xlabel="D3/D1 (national)", ylabel="mean(D4-10)/mean(D1-3)")
fig.savefig(F / "momentum.png")
print(f"figures -> {F}")
