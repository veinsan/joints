"""01 - Data audit: integrity, overlaps, and HOW the organiser built the test set."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, fig_dir, load, style

plt = style()
F = fig_dir("01_audit")
d = load()
tr, th, te, mv = d["train"], d["hist"], d["test"], d["movies"]

print("=== shapes / date ranges ===")
for n, x in [("train", tr), ("test_history", th), ("test", te)]:
    print(f"{n:13s} {x.shape}  {x.date_show.min().date()} -> {x.date_show.max().date()}  "
          f"films={x.movie_title.nunique()} cinemas={x.cinema_ids.nunique()} cities={x.city_name.nunique()}")

print("\n=== missing / duplicates ===")
print("NA train:", tr.isna().sum().sum(), " NA hist:", th.isna().sum().sum(), " NA test:", te.isna().sum().sum())
for n, x in [("train", tr), ("hist", th)]:
    print(f"{n}: dup (date,cinema,film)={x.duplicated(['date_show'] + KEY).sum()}  full dup={x.duplicated().sum()}")
print("cinema -> multiple cities?", (pd.concat([tr, th]).groupby("cinema_ids").city_name.nunique() > 1).sum())
print("test ids unique & contiguous:", te.id.is_unique, te.id.min(), te.id.max(), len(te))

print("\n=== value sanity ===")
print(tr[["total_ticket", "occupation_rate", "total_show"]].describe(percentiles=[.01, .5, .99]).T.round(2))
print("rows with ticket<=0:", (tr.total_ticket <= 0).sum(), " occupancy>100:", (tr.occupation_rate > 100).sum())
# implied seats per show = tickets / (occ% * shows)
seats = tr.total_ticket / (tr.occupation_rate / 100 * tr.total_show)
print("implied seats/show quantiles:", seats.replace(np.inf, np.nan).quantile([.01, .25, .5, .75, .99]).round(0).to_dict())

print("\n=== overlaps ===")
ft, fh = set(tr.movie_title), set(te.movie_title)
print(f"test films also in train: {len(ft & fh)}/{len(fh)}  (these are pre-release previews, see below)")
print(f"test cinemas unseen in train: {len(set(te.cinema_ids) - set(tr.cinema_ids))}")
print(f"films in movies.csv: train {len(ft & set(mv.original_title))}/{len(ft)}  test {len(fh & set(mv.original_title))}/{len(fh)}")
for m in sorted(ft & fh):
    a = tr[tr.movie_title == m]
    print(f"   {m:30s} train {a.date_show.min().date()}..{a.date_show.max().date()} cinemas={a.cinema_ids.nunique():2d} "
          f"tickets={a.total_ticket.sum():5d}  | D1={th[th.movie_title == m].date_show.min().date()}")

print("\n=== test construction (the key finding) ===")
d1 = th.groupby("movie_title").date_show.min().rename("d1")
h = th.join(d1, on="movie_title")
h["d"] = (h.date_show - h.d1).dt.days + 1
ph = h.groupby(KEY).agg(ndays=("d", "size"), last=("d", "max"), s=("total_ticket", "sum")).reset_index()
ph = ph.merge(te[KEY].drop_duplicates().assign(in_test=1), how="left").fillna({"in_test": 0})
print(pd.crosstab(ph["last"], ph.in_test, margins=True).rename_axis("last day with sale"))
print("=> a (film,cinema) pair is in test IFF it sold on D3. Pairs that dropped the film before D3 are excluded.")
print("test rows per pair:", te.groupby(KEY).size().value_counts().to_dict())
gap = te.groupby("movie_title").date_show.min() - th.groupby("movie_title").date_show.max()
print("gap D3 -> first target day:", gap.dt.days.value_counts().to_dict())
print("D1 weekday (test):", d1.dt.day_name().value_counts().to_dict())
print("history titles without test rows:", sorted(set(th.movie_title) - fh))
print("pairs with <3 history days (zeros inside D1-D3):", (ph.query("in_test==1").ndays < 3).mean().round(3))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ph.groupby(["last", "in_test"]).size().unstack(fill_value=0).plot.bar(ax=ax[0], width=0.8)
ax[0].set(title="Pairs by last history day with a sale", xlabel="last day with sale (D1..D3)", ylabel="pairs")
ax[0].legend(["not in test", "in test"])
wk = pd.concat([d1.dt.dayofweek.value_counts().sort_index().rename("test D1")], axis=1)
wk.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(wk)] if len(wk) == 7 else [
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][i] for i in wk.index]
wk.plot.bar(ax=ax[1], legend=False)
ax[1].set(title="Weekday of D1 (test films)", ylabel="films")
dd = pd.concat([tr.groupby("date_show").total_ticket.sum().rename("train (all films)"),
                th.groupby("date_show").total_ticket.sum().rename("test_history (D1-D3 only)")], axis=1)
dd.plot(ax=ax[2], lw=1)
ax[2].set(title="Daily tickets: train vs visible test history", ylabel="tickets", xlabel="")
fig.savefig(F / "audit.png")
print(f"\nfigure -> {F/'audit.png'}")
