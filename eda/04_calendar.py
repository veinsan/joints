"""04 - Calendar effects: weekday, holidays, school break, and test-only periods (Xmas, Ramadan, Lebaran)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, fig_dir, load, style

plt = style()
F = fig_dir("04_calendar")
d = load()
tr, th, te, hol = d["train"], d["hist"], d["test"], d["hol"]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Detrended daily multiplier: each pair-day relative to the pair's centred 7-day mean.
# Removes film age + cinema size, leaves pure calendar effect.
x = tr.set_index("date_show").groupby(KEY).total_ticket.apply(lambda s: s.asfreq("D", fill_value=0)).reset_index()
x["m7"] = x.groupby(KEY).total_ticket.transform(lambda s: s.rolling(7, center=True).mean())
x = x[x.m7 >= 20]
x["mult"] = x.total_ticket / x.m7
x = x.merge(hol, left_on="date_show", right_on="date", how="left")
x["dow"] = x.date_show.dt.dayofweek
day = x.groupby("date_show").agg(mult=("mult", "median"), dow=("dow", "first"),
                                 hol=("holiday_tipe", "first"), name=("holiday_name", "first"))
prof = day[day.hol == "normal"].groupby("dow").mult.median()
print("=== weekday multiplier (non-holiday days, median of pair-day / centred 7d mean) ===")
print(prof.rename(index=dict(enumerate(DOW))).round(3).to_string())
day["excess"] = day.mult / day.dow.map(prof)
print("\n=== holidays: multiplier vs normal same weekday ===")
print(day[day.hol == "holiday"][["name", "dow", "mult", "excess"]].assign(dow=lambda a: a.dow.map(dict(enumerate(DOW)))).round(3).to_string())

# school holiday window (DKI Jakarta academic calendar 2024/25: 28 Jun - 13 Jul 2025)
sch = (day.index >= "2025-06-28") & (day.index <= "2025-07-13")
wk = day.dow < 4
print(f"\nschool-break weekday (Mon-Thu) excess: {day[sch & wk & (day.hol == 'normal')].excess.median():.3f}"
      f"   vs term-time Mon-Thu: {day[~sch & wk & (day.hol == 'normal')].excess.median():.3f}")
post_leb = (day.index <= "2025-04-07")
print(f"post-Lebaran week 2025 (1-7 Apr) excess: {day[post_leb].excess.median():.3f}")

# ---------------------------------------------------------------- test-period calendar map
tt = te.merge(hol, left_on="date_show", right_on="date", how="left")
ramadan = (tt.date_show >= "2026-02-19") & (tt.date_show <= "2026-03-19")
lebaran = (tt.date_show >= "2026-03-20") & (tt.date_show <= "2026-03-29")
xmas = (tt.date_show >= "2025-12-20") & (tt.date_show <= "2026-01-04")
print("\n=== share of TEST rows in periods never seen in train ===")
for n, m in [("Ramadan 1447H (19 Feb-19 Mar 2026)", ramadan), ("Lebaran window (20-29 Mar 2026)", lebaran),
             ("Christmas/New-Year school break", xmas), ("any national holiday", tt.holiday_tipe == "holiday")]:
    print(f"  {n:40s} rows={m.sum():6d}  share={m.mean():.3%}  films={tt[m].movie_title.nunique()}")
d1 = th.groupby("movie_title").date_show.min()
leb_films = d1[(d1 >= "2026-03-12")].sort_values()
print("films whose D4-D10 touch Lebaran:", leb_films.dt.date.to_dict())

# Ramadan depression visible in test_history: mean tickets per cinema-day on D1-D3, by release month
h = th.join(d1.rename("d1"), on="movie_title")
per = h.groupby("movie_title").agg(t=("total_ticket", "mean"), d1=("d1", "first"))
per["period"] = np.where((per.d1 >= "2026-02-18") & (per.d1 <= "2026-03-19"), "Ramadan", "other")
print("\nmedian D1-D3 tickets per cinema-day, by release period (confounded by film quality):")
print(per.groupby("period").t.describe()[["count", "25%", "50%", "75%"]].round(1))

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].bar(DOW, prof.values, color="#2a78d6")
ax[0].set(title="Weekday multiplier (train, non-holiday)", ylabel="x of centred 7-day mean")
ax[1].plot(day.index, day.excess, lw=1)
for dt in day[day.hol == "holiday"].index:
    ax[1].axvline(dt, color="#e34948", lw=.6, alpha=.6)
ax[1].axvspan(pd.Timestamp("2025-06-28"), pd.Timestamp("2025-07-13"), color="#eda100", alpha=.2, label="school break")
ax[1].set(title="Daily excess over weekday profile (red = holiday)", ylabel="excess"); ax[1].legend()
cnt = te.groupby("date_show").size()
ax[2].bar(cnt.index, cnt.values, width=1, color="#2a78d6")
for a, b, c, n in [("2025-12-20", "2026-01-04", "#eda100", "Xmas/NY break"), ("2026-02-19", "2026-03-19", "#4a3aa7", "Ramadan"),
                   ("2026-03-20", "2026-03-29", "#e34948", "Lebaran")]:
    ax[2].axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=.18, label=n)
ax[2].set(title="Test target rows per date", ylabel="rows"); ax[2].legend(fontsize=7)
fig.autofmt_xdate()
fig.savefig(F / "calendar.png")
print(f"figures -> {F}")
