"""Thinned 'quiet-market' world at the real horizon D4-D10 (lookup tables, no model).

Every train transaction is thinned, y' ~ Binomial(y, pi), for ALL films and days (the whole market
gets quieter, so relative positions stay), D1 is kept from the raw data, and the organiser's rule
(sell on D3, zero-fill, official scale) is re-applied on the thinned data. Questions:
 1. Does the thinned world look like the test in composition (scale buckets, first-sale day)?
 2. Lookup predictor (median r by h x D1-dow group x p3 bin x scale bucket), grouped 5-fold by film:
      raw -> raw, raw -> thinned(0.4), thinned-mix -> thinned(0.4), raw+thinned-mix -> raw / thinned(0.4)
    Plain (unweighted) MASE on the thinned world is compared with the public level (~0.40).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, release_dates, simulate, style
from features import calendar

SB = [0, 5, 20, 50, 100, 200, 500, 1e9]


def world(tr, d1, cal, pi=None, seed=0):
    tx = tr.copy()
    if pi is not None:
        tx["total_ticket"] = np.random.default_rng(seed).binomial(tx.total_ticket.values.astype(int), pi)
        tx = tx[tx.total_ticket > 0]
    hist, tgt = simulate(tx, d1)
    h = hist.join(d1.rename("d1"), on="movie_title")
    h["d"] = (h.date_show - h.d1).dt.days + 1
    p = h.pivot_table(index=KEY, columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    p.columns = ["y1", "y2", "y3"]
    X = tgt.join(p, on=KEY).join(d1.rename("d1"), on="movie_title")
    X["scale"] = X[["y1", "y2", "y3"]].mean(axis=1).clip(lower=1)
    X["p3"] = X.y3 / X.scale
    X["h"] = (X.date_show - X.d1).dt.days + 1
    ch = np.stack([cal.reindex(X.d1 + pd.Timedelta(days=k)).values for k in range(3)], 1).mean(1)
    X["cm"] = cal.reindex(X.date_show).values / ch
    X["fd"] = np.where(X.y1 > 0, 1, np.where(X.y2 > 0, 2, 3))
    X["dw"] = np.minimum(X.d1.dt.dayofweek, 4)
    X["base"] = base_title(X.movie_title)
    X["r"] = X.total_ticket / X.scale / X.cm
    X["cell"] = (X.h.astype(str) + "|" + X.dw.astype(str) + "|" + pd.cut(X.p3, [-1, .3, .7, 1, 1.3, 9]).astype(str) + "|"
                 + pd.cut(X.scale, SB).astype(str) + "|" + X.fd.clip(1, 2).astype(str))
    X["cell2"] = X.h.astype(str) + "|" + X.dw.astype(str) + "|" + pd.cut(X.p3, [-1, .3, .7, 1, 1.3, 9]).astype(str)
    return X.reset_index(drop=True)


def cv_lookup(train_sets, test_set, folds_of):
    """Grouped by base film: train cells from train_sets rows of other folds, score test_set rows of the fold."""
    err = np.zeros(len(test_set))
    for k in range(5):
        tr = pd.concat([d[d.base.map(folds_of) != k] for d in train_sets], ignore_index=True)
        va = test_set.base.map(folds_of) == k
        m1, m2 = tr.r.groupby(tr.cell).median(), tr.r.groupby(tr.cell2).median()
        n1 = tr.cell.value_counts()
        q = test_set[va]
        pr = np.where(q.cell.map(n1).fillna(0) >= 20, q.cell.map(m1), q.cell2.map(m2).fillna(tr.r.median()))
        err[va.values] = np.abs(q.total_ticket - pr * q.cm * q.scale) / q.scale
    return err


def main():
    out = fig_dir("88_thinned_world")
    D = load()
    cal = calendar(D["hol"]).cal
    tr = D["train"]
    first = tr.groupby("movie_title").date_show.min()
    d1 = release_dates(tr).drop(first[first == first.min()].index, errors="ignore")
    raw = world(tr, d1, cal)
    val40 = world(tr, d1, cal, 0.4, seed=101)
    mix = [world(tr, d1, cal, pi, seed=s) for pi, s in [(0.35, 1), (0.5, 2), (0.7, 3)]]
    te = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    te["fd"] = np.where(te.y1 > 0, 1, np.where(te.y2 > 0, 2, 3))
    comp = pd.DataFrame({n: pd.cut(d.scale, SB).value_counts(normalize=True).sort_index() for n, d in
                         [("train raw", raw), ("thinned 0.4", val40), ("test", te)]})
    comp.loc["first sale D2/D3"] = [np.mean(d.fd > 1) for d in (raw, val40, te)]
    comp.loc["rows"] = [len(raw), len(val40), len(te)]
    pd.set_option("display.width", 200)
    print("Composition:"); print(comp.round(3).to_string())
    films = np.array(sorted(raw.base.unique()))
    folds_of = dict(zip(films, np.random.RandomState(2026).permutation(len(films)) % 5))
    res = {}
    res["raw -> raw"] = cv_lookup([raw], raw, folds_of)
    res["raw -> thinned 0.4"] = cv_lookup([raw], val40, folds_of)
    res["thinned mix -> thinned 0.4"] = cv_lookup(mix, val40, folds_of)
    res["raw + thinned mix -> thinned 0.4"] = cv_lookup([raw] + mix, val40, folds_of)
    res["raw + thinned mix -> raw"] = cv_lookup([raw] + mix, raw, folds_of)
    rows = []
    for n, e in res.items():
        d = raw if n.endswith("-> raw") else val40
        sb = pd.cut(d.scale, SB)
        rows.append(dict(setup=n, MASE=e.mean(), small_s20=e[(d.scale <= 20).values].mean(),
                         s20_50=e[d.scale.between(20, 50).values].mean(), zero_rows=e[(d.total_ticket == 0).values].mean(),
                         positive_rows=e[(d.total_ticket > 0).values].mean(), zero_share=(d.total_ticket == 0).mean()))
    R = pd.DataFrame(rows).set_index("setup")
    print("\nGrouped 5-fold lookup MASE (plain mean, no weights):"); print(R.round(4).to_string())
    R.to_csv(out / "lookup_cv.csv"); comp.to_csv(out / "composition.csv")
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    comp.iloc[:7].plot.bar(ax=ax[0], rot=30); ax[0].set(title="Scale composition: raw train vs thinned 0.4 vs test")
    R.MASE.plot.barh(ax=ax[1]); ax[1].axvline(0.40, color="#e34948", ls="--", label="public level ~0.40"); ax[1].legend()
    ax[1].set(title="Lookup MASE by train -> validation world")
    fig.tight_layout(); fig.savefig(out / "thinned_world.png")


if __name__ == "__main__":
    main()
