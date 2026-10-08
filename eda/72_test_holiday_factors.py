"""Holiday factors inside the test period, measured from test_history (official data), not assumed.

eda/66 found the year-end days stronger than the structural calendar (Dec 24->25 implied log ratio
0.61 vs assumed 0.32). Here every test-period holiday that falls inside some film's D1-D3 window is
measured: for each film-day transition (d -> d+1) touching a holiday date, the implied calendar
log-ratio = observed log ratio - train age effect, compared with the assumed c(t). Pairs selling on
all three days only (fixed panel). Also counts the target rows (D4-D10) that sit on each date, i.e.
the stakes of getting that date wrong.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eda"))
from common import KEY, fig_dir, load, release_dates, simulate, style
from features import calendar

m66 = __import__("66_calendar_residual_test_period")

DATES = {"2025-12-24": "Christmas Eve (school hol)", "2025-12-25": "Christmas", "2025-12-26": "cuti bersama",
         "2025-12-31": "New Year Eve", "2026-01-01": "New Year", "2026-01-02": "Fri after NY",
         "2026-01-16": "Isra Mikraj (Fri)", "2026-02-16": "cuti Imlek (Mon)", "2026-02-17": "Imlek (Tue)",
         "2026-03-19": "Nyepi (Thu)"}


def main():
    out = fig_dir("72_test_holidays")
    D = load()
    c = calendar(D["hol"])
    cal = c.cal
    hist, _ = simulate(D["train"], release_dates(D["train"]))
    tr = m66.panel(hist)
    tr["g"] = tr.lr - np.log(cal.reindex(tr.t1).values / cal.reindex(tr.t0).values)
    age = tr.groupby("d").g.median()
    te = m66.panel(D["hist"])
    te["assumed"] = np.log(cal.reindex(te.t1).values / cal.reindex(te.t0).values)
    te["implied"] = te.lr - te.d.map(age)
    te["gap"] = te.implied - te.assumed
    rows = []
    for ds, name in DATES.items():
        t = pd.Timestamp(ds)
        q = te[(te.t0 == t) | (te.t1 == t)]
        for _, r in q.iterrows():
            rows.append(dict(date=ds, name=name, film=r.movie_title, step=f"{r.t0:%m-%d}>{r.t1:%m-%d}",
                             side="into" if r.t1 == t else "out of", implied=r.implied, assumed=r.assumed, gap=r.gap, T=r["T"]))
    h = pd.DataFrame(rows)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    print("Film-level transitions touching a test-period holiday (implied vs assumed log calendar ratio):")
    print(h.round(3).to_string(index=False))
    h.to_csv(out / "holiday_transitions.csv", index=False)
    s = h.groupby(["date", "name", "side"]).agg(n=("gap", "size"), gap_median=("gap", "median"), gap_mean=("gap", "mean"))
    print("\nSummary (into = ratio holiday/previous day, out of = next day/holiday):")
    print(s.round(3).to_string())

    # stakes: target rows by date
    test = D["test"]
    cnt = test.date_show.value_counts()
    print("\nTarget rows (D4-D10) on each holiday date:")
    for ds, name in DATES.items():
        print(f"  {ds} {name:28s} rows {cnt.get(pd.Timestamp(ds), 0):5d}  assumed c={cal[pd.Timestamp(ds)]:.3f}")
    yr = test.date_show.between("2025-12-20", "2026-01-04")
    print(f"  all target rows 20 Dec - 4 Jan: {yr.sum()} ({yr.mean():.1%})")
    bali = test.city_name.str.contains("DENPASAR|BADUNG|BALI|GIANYAR|TABANAN|BULELENG|KLUNGKUNG", case=False)
    print(f"  Bali-cluster target rows on Nyepi 2026-03-19: {(bali & (test.date_show == '2026-03-19')).sum()}"
          f" (Bali clusters: {sorted(test.city_name[bali].unique())})")
    hb = D["hist"]
    hbali = hb.city_name.str.contains("DENPASAR|BADUNG|BALI|GIANYAR|TABANAN|BULELENG|KLUNGKUNG", case=False)
    print("  Bali history rows on 2026-03-19 (Nyepi):", int((hbali & (hb.date_show == "2026-03-19")).sum()),
          "| on 2026-03-18:", int((hbali & (hb.date_show == "2026-03-18")).sum()),
          "| on 2026-03-20:", int((hbali & (hb.date_show == "2026-03-20")).sum()))

    plt = style()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    k = h.groupby(["date", "side"]).gap.median().unstack()
    k.plot.bar(ax=ax, color=["#2a78d6", "#eb6834"])
    ax.axhline(0, color="#8a8984", lw=.8)
    ax.set_ylabel("implied - assumed log ratio"); ax.set_title("Holiday calendar error in the test period (median over films)")
    fig.tight_layout(); fig.savefig(out / "holiday_gaps.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
