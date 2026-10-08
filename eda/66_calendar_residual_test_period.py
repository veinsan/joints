"""Does the structural calendar c(t) make D1->D2->D3 film dynamics period-invariant?

The only labelled information inside the test period is test_history (D1-D3 of every test film).
For every film we take the pairs that sold on all three days (fixed panel, so coverage changes do
not leak into the ratio) and compute day-to-day log ratios of film totals:

    g_fd = log(T_{d+1} / T_d) - log(c(t_{d+1}) / c(t_d))       (calendar-adjusted)

If c(t) is right, g_fd is the pure age decay and should have the same distribution in train-sim and
in every test-period segment (normal, year-end, Ramadan). A shift in a segment = the calendar is
mis-specified there, and every model inherits it (cal_mult is shared). No model is fitted.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, fig_dir, load, release_dates, simulate, style
from features import calendar


def panel(hist):
    h = hist.copy()
    d1 = h.groupby("movie_title").date_show.min()
    h["d"] = (h.date_show - h.movie_title.map(d1)).dt.days + 1
    p = h.pivot_table(index=KEY, columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    full = p[(p > 0).all(axis=1)]
    T = full.groupby(level=0).sum()
    T = T[(T >= 50).all(axis=1)]          # avoid tiny-count films dominating the log ratio
    out = []
    for m, r in T.iterrows():
        for d in (1, 2):
            t0 = d1[m] + pd.Timedelta(days=d - 1)
            out.append(dict(movie_title=m, d=d, t0=t0, t1=t0 + pd.Timedelta(days=1),
                            lr=np.log(r[d + 1] / r[d]), npairs=len(full.loc[m]), T=r.sum()))
    return pd.DataFrame(out)


def segment(t):
    t = pd.Timestamp(t)
    if t < pd.Timestamp("2025-10-01"):
        return "train"
    if pd.Timestamp("2025-12-20") <= t <= pd.Timestamp("2026-01-04"):
        return "year-end"
    if pd.Timestamp("2026-02-19") <= t <= pd.Timestamp("2026-03-20"):
        return "ramadan"
    return "test-normal"


def main():
    out = fig_dir("66_calendar_residual")
    D = load()
    cal = calendar(D["hol"]).cal
    d1 = release_dates(D["train"])
    hist_tr, _ = simulate(D["train"], d1)
    rows = pd.concat([panel(hist_tr).assign(src="train"), panel(D["hist"]).assign(src="test")], ignore_index=True)
    rows["c0"], rows["c1"] = cal.reindex(rows.t0).values, cal.reindex(rows.t1).values
    rows["g"] = rows.lr - np.log(rows.c1 / rows.c0)
    rows["seg"] = rows.t1.map(segment)
    rows["trans"] = rows.t0.dt.day_name().str[:3] + ">" + rows.t1.dt.day_name().str[:3]
    rows.to_csv(out / "film_day_ratios.csv", index=False)

    pd.set_option("display.width", 200)
    print("Calendar-adjusted log ratio g (median, by step d and segment); n films in brackets")
    tab = rows.groupby(["d", "seg"]).g.agg(["median", "mean", "count"]).round(3)
    print(tab.to_string())

    # transition-specific: does the same weekday transition behave differently in the test period?
    print("\nRaw log ratio by weekday transition (step d), train vs test segments:")
    t2 = rows.pivot_table(index=["d", "trans"], columns="seg", values="lr", aggfunc="median").round(3)
    cnt = rows.pivot_table(index=["d", "trans"], columns="seg", values="lr", aggfunc="count")
    t2 = t2[cnt.fillna(0).sum(axis=1) >= 6]
    print(t2.to_string())
    print("\ncounts:\n", cnt.loc[t2.index].fillna(0).astype(int).to_string())

    # implied calendar ratio per transition (train age effect removed): what c1/c0 would make test match train?
    age = rows[rows.seg == "train"].groupby("d").g.median()
    rows["implied_logc"] = rows.lr - rows.d.map(age)
    imp = rows.groupby(["seg", "trans"]).agg(implied=("implied_logc", "median"), assumed=("c1", "first"), n=("g", "size"))
    imp["assumed"] = rows.groupby(["seg", "trans"]).apply(lambda q: np.median(np.log(q.c1 / q.c0)), include_groups=False)
    imp = imp[imp.n >= 4].round(3)
    imp["gap"] = (imp.implied - imp.assumed).round(3)
    print("\nImplied vs assumed calendar log-ratio per transition (age effect from train removed):")
    print(imp.to_string())

    # bootstrap test of segment shift in g relative to train, per step d
    rng = np.random.default_rng(2026)
    print("\nShift of mean g vs train (bootstrap 95% CI over films):")
    res = []
    for d in (1, 2):
        base = rows[(rows.seg == "train") & (rows.d == d)].g.values
        for s in ["test-normal", "year-end", "ramadan"]:
            x = rows[(rows.seg == s) & (rows.d == d)].g.values
            if len(x) < 4:
                continue
            bs = [rng.choice(x, len(x)).mean() - rng.choice(base, len(base)).mean() for _ in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res.append(dict(d=d, seg=s, n=len(x), shift=x.mean() - base.mean(), lo=lo, hi=hi))
            print(f"  d{d}->{d+1} {s:12s} n={len(x):3d} shift {x.mean()-base.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    pd.DataFrame(res).to_csv(out / "segment_shift.csv", index=False)

    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, d in enumerate((1, 2)):
        q = rows[rows.d == d]
        for j, s in enumerate(["train", "test-normal", "year-end", "ramadan"]):
            v = q[q.seg == s].g
            ax[i].scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-.15, .15, len(v)), v, s=10, alpha=.6)
            ax[i].hlines(v.median(), j - .3, j + .3, color="k")
        ax[i].set_xticks(range(4), ["train", "test-normal", "year-end", "ramadan"])
        ax[i].set_title(f"calendar-adjusted log(T{d+1}/T{d}) per film")
        ax[i].axhline(0, color="#8a8984", lw=.8)
    fig.tight_layout(); fig.savefig(out / "g_by_segment.png")

    fig, ax = plt.subplots(figsize=(12, 3.6))
    q = rows[rows.src == "test"].sort_values("t1")
    ax.scatter(q.t1, q.g, c=np.where(q.d == 1, "#2a78d6", "#eb6834"), s=12)
    ax.axhline(age[1], color="#2a78d6", lw=1, ls="--", label="train median d1>2")
    ax.axhline(age[2], color="#eb6834", lw=1, ls="--", label="train median d2>3")
    for a, b, lab in [("2025-12-20", "2026-01-04", "year-end"), ("2026-02-19", "2026-03-20", "Ramadan")]:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#e6e5e0", alpha=.6)
    ax.set_title("Test period: calendar-adjusted daily log ratio per film (blue d1>2, orange d2>3)")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "g_timeline_test.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
