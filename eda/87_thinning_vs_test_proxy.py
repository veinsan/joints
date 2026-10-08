"""Binomial thinning re-checked against the RIGHT reference: test-period small pairs.

History: thinning (y' ~ Binomial(y, pi)) was rejected in eda/14 because thinned small pairs had a D4-D10
zero rate 0.52 vs 0.76 for real small train pairs. But real small train pairs are dying pairs, while
test small pairs are normal pairs in a quiet market (eda/68-69). The test period's own labels (D3 given
D1, D2) are the right reference. For raw train-sim, thinned train-sim (pi = 0.5, 0.35) and test:
  by bucket of m12 = mean(y1, y2): P(y3 = 0), median q3 = y3 / m12, share of pairs.
If thinned train matches the test period where raw train does not, thinning is the augmentation that
creates training rows like the test's small pairs (the 'color jitter' that matches the shift).
Also a lookup transfer check: tables built on raw vs raw+thinned train, scored on test-period pairs.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, fig_dir, load, release_dates, simulate, style
from features import calendar

BK = [0, 3, 5, 10, 20, 50, 200, 1e9]


def thin(hist, pi, seed):
    h = hist.copy()
    h["total_ticket"] = np.random.default_rng(seed).binomial(h.total_ticket.values.astype(int), pi)
    return h[h.total_ticket > 0]


def proxy(hist, cal):
    d1 = hist.groupby("movie_title").date_show.min().rename("d1")
    h = hist.join(d1, on="movie_title")
    h["d"] = (h.date_show - h.d1).dt.days + 1
    p = h.pivot_table(index=KEY, columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    p = p[(p[1] > 0) & (p[2] > 0)].reset_index().join(d1, on="movie_title")
    c = [cal.reindex(p.d1 + pd.Timedelta(days=k)).values for k in range(3)]
    p["m12"] = (p[1] + p[2]) / 2
    p["q3"] = p[3] / p.m12 / (c[2] / ((c[0] + c[1]) / 2))
    p["z"] = (p[3] == 0).astype(int)
    p["p1"] = p[1] / p.m12
    p["bk"] = pd.cut(p.m12, BK)
    return p


def main():
    out = fig_dir("87_thinning")
    D = load()
    cal = calendar(D["hol"]).cal
    hist, _ = simulate(D["train"], release_dates(D["train"]))
    sets = {"train raw": proxy(hist, cal), "test period": proxy(D["hist"], cal)}
    for pi in (0.5, 0.35):
        sets[f"train thinned pi={pi}"] = pd.concat([proxy(thin(hist, pi, s), cal) for s in range(3)], ignore_index=True)
    pd.set_option("display.width", 220)
    tab = {}
    for n, p in sets.items():
        g = p.groupby("bk", observed=False)
        tab[n] = pd.DataFrame({"share": g.size() / len(p), "P(y3=0)": g.z.mean(), "median q3": g.q3.median()})
    T = pd.concat(tab, axis=1)
    print("By bucket of mean(y1, y2):")
    print(T.round(3).to_string())
    T.to_csv(out / "by_bucket.csv")

    # lookup transfer: (bucket of m12 x bucket of p1) median q3 and zero rate; scored on the test period
    te = sets["test period"]
    def score(train):
        k = lambda d: d.bk.astype(str) + "|" + pd.cut(d.p1, [0, .7, .9, 1.1, 1.3, 2.1]).astype(str)
        med, zr = train.q3.groupby(k(train)).median(), train.z.groupby(k(train)).mean()
        pq = k(te).map(med).fillna(train.q3.median()).values
        sm = (te.m12 <= 20).values
        return dict(mae=np.mean(np.abs(te.q3 - pq)), mae_small=np.mean(np.abs(te.q3 - pq)[sm]),
                    bias_small=np.median((te.q3 - pq)[sm]), zero_pred_small=k(te)[sm].map(zr).mean(), zero_obs_small=te.z[sm].mean())
    S = pd.DataFrame({n: score(p) for n, p in sets.items() if n != "test period"})
    S["raw + thinned 0.5"] = pd.Series(score(pd.concat([sets["train raw"], sets["train thinned pi=0.5"]], ignore_index=True)))
    print("\nLookup trained on each set, scored on test-period pairs (small = m12 <= 20):")
    print(S.T.round(4).to_string())
    S.T.to_csv(out / "transfer.csv")
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for n, col in zip(sets, ["#2a78d6", "#e34948", "#1baf7a", "#eda100"]):
        g = sets[n].groupby("bk", observed=False)
        xs = np.arange(len(BK) - 1)
        ax[0].plot(xs, g.z.mean().values, marker="o", color=col, label=n)
        ax[1].plot(xs, g.q3.median().values, marker="o", color=col, label=n)
    for a_, tt in zip(ax, ["P(no sale on D3) given D1, D2 sales", "median calendar-adjusted y3 / mean(y1, y2)"]):
        a_.set_xticks(np.arange(len(BK) - 1), [str(b) for b in pd.IntervalIndex.from_breaks(BK)], rotation=30)
        a_.set(title=tt, xlabel="mean(y1, y2) bucket")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "thinning_vs_test.png")


if __name__ == "__main__":
    main()
