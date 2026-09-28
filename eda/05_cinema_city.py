"""05 - Cinema clusters & cities: size, stickiness of cinema behaviour, new clusters, ticket price."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, fig_dir, load, release_dates, scale, simulate, style

plt = style()
F = fig_dir("05_cinema")
d = load()
tr, th, te, price = d["train"], d["hist"], d["test"], d["price"]

cin = tr.groupby("cinema_ids").agg(city=("city_name", "first"), tix_day=("total_ticket", "sum"),
                                   days=("date_show", "nunique"), films=("movie_title", "nunique"),
                                   shows=("total_show", "sum"), occ=("occupation_rate", "mean"))
cin["tix_day"] /= cin.days
print("=== cinema cluster size (train, tickets per active day) ===")
print(cin.tix_day.describe(percentiles=[.1, .5, .9]).round(0).to_string())
print("top-5 clusters share of all train tickets:", (cin.tix_day * cin.days).nlargest(5).sum() / tr.total_ticket.sum())
print("clusters per city (top):", cin.city.value_counts().head(8).to_dict())
new = sorted(set(te.cinema_ids) - set(tr.cinema_ids))
print("\nnew clusters in test:", len(new), "| their test rows:", te.cinema_ids.isin(new).sum(),
      "| cities:", th[th.cinema_ids.isin(new)].city_name.unique().tolist())
print("new clusters D1-D3 tickets/day vs old median:",
      th[th.cinema_ids.isin(new)].groupby("cinema_ids").total_ticket.mean().round(0).to_dict(), cin.tix_day.median())

# Is a cinema's "legginess" persistent? split train films into 2 halves and correlate cinema mean residual r
hs, ts = simulate(tr, release_dates(tr))
ts = ts.join(scale(hs), on=KEY)
ts["r"] = ts.total_ticket / ts.scale
ts["film_r"] = ts.groupby(["movie_title", "date_show"]).r.transform("median")
ts["res"] = ts.r - ts.film_r
half = ts.movie_title.map(dict(zip(sorted(ts.movie_title.unique()), np.arange(ts.movie_title.nunique()) % 2)))
a = ts.groupby([half, "cinema_ids"]).res.mean().unstack(0).dropna()
print(f"\ncinema residual (r - film median r) split-half correlation: {a.corr().iloc[0, 1]:.3f}  (n={len(a)})")
print("=> cinema-level target encoding carries signal only if this is clearly > 0")

# Pair share of film vs cinema size: big clusters keep films longer?
ts["cin_size"] = np.log10(ts.cinema_ids.map(cin.tix_day))
m = ts.groupby(pd.qcut(ts.cin_size, 5)).agg(r=("r", "median"), zero=("total_ticket", lambda s: (s == 0).mean()))
print("\nmedian r / zero-share by cinema size quintile:\n", m.round(3))

# ticket price
pv = price.pivot(index="city_name", columns="price_day", values="ceil")
print("\nticket price (Rp) summary:\n", pv.describe().round(0))
print("cities in data without price:", sorted(set(te.city_name) - set(pv.index)))
cc = cin.join(pv, on="city")
print("corr(log cinema size, weekend price) spearman:", round(cc[["tix_day", "Weekend"]].corr("spearman").iloc[0, 1], 3))
pv["wk_prem"] = pv.Weekend / pv.Weekday

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].hist(np.log10(cin.tix_day), bins=30, color="#2a78d6")
ax[0].set(title="Cluster size (log10 tickets/day)", xlabel="log10")
ax[1].scatter(a[0], a[1], s=12, alpha=.6, color="#2a78d6")
ax[1].set(title=f"Cinema residual r, split-half (rho={a.corr().iloc[0, 1]:.2f})", xlabel="films half A", ylabel="films half B")
ax[2].scatter(cc.Weekend / 1000, np.log10(cc.tix_day), s=12, alpha=.6, color="#eb6834")
ax[2].set(title="Weekend price vs cluster size", xlabel="price (Rp 000)", ylabel="log10 tickets/day")
fig.savefig(F / "cinema.png")
print(f"figures -> {F}")
