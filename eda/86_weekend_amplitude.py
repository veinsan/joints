"""Is the weekend amplitude heterogeneous across clusters / films, and does it explain D4-D5 error?

Wednesday releases never show a weekend inside D1-D3, yet D4-D5 are their first Saturday-Sunday,
and the structural calendar uses one weekend multiplier for everyone. Checks:
 1. cluster weekend amplitude from train transactions (Sat+Sun vs Mon-Thu of the same film-week,
    so film age and film level cancel): spread across clusters and split-half reliability;
 2. LOO bound: cluster x weekday-type shock from OTHER films (does a weekend-specific cluster effect
    exist that the all-day cluster bound in eda/84 hid by cancelling?);
 3. residual of v13 on weekend target rows vs the cluster amplitude (fold-local by film);
 4. film-level amplitude by genre / rating (metadata) for the weekend rows of Wednesday releases.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, OUTAGE, base_title, fig_dir, load, style
from evaluate import test_weights_fd


def amplitude(tx):
    """log(mean weekend day / mean Mon-Thu day) per (film, cluster, ISO week), then median per cluster."""
    t = tx[~tx.date_show.isin(OUTAGE)].copy()
    t["wk"] = t.date_show.dt.isocalendar().week.astype(int)
    t["typ"] = np.where(t.date_show.dt.dayofweek >= 5, "we", np.where(t.date_show.dt.dayofweek <= 3, "wd", "fri"))
    g = t[t.typ != "fri"].groupby(KEY + ["wk", "typ"]).total_ticket.mean().unstack()
    g = g.dropna()
    g = g[(g.wd >= 3) & (g.we >= 3)]
    g["amp"] = np.log(g.we / g.wd)
    return g.reset_index()


def main():
    out = fig_dir("86_weekend_amplitude")
    D = load()
    A = amplitude(D["train"])
    A["base"] = base_title(A.movie_title)
    films = np.array(sorted(A.base.unique()))
    half = A.base.isin(films[::2])
    c1, c2 = A[half].groupby("cinema_ids").amp.median(), A[~half].groupby("cinema_ids").amp.median()
    both = pd.concat([c1, c2], axis=1, keys=["h1", "h2"]).dropna()
    rel = both.h1.corr(both.h2)
    camp = A.groupby("cinema_ids").amp.median()
    print(f"film-cluster-weeks {len(A)}; cluster weekend amplitude: median {camp.median():.3f} "
          f"(ratio {np.exp(camp.median()):.2f}), sd across clusters {camp.std():.3f}, p10-p90 {np.exp(camp.quantile(.1)):.2f}-{np.exp(camp.quantile(.9)):.2f}x;"
          f" split-half reliability (by film) {rel:.3f}")
    city = A.merge(D["train"].drop_duplicates("cinema_ids")[["cinema_ids", "city_name"]]).groupby("city_name").amp.median()
    print(f"city amplitude spread sd {city.std():.3f}; top {city.nlargest(3).round(2).to_dict()} bottom {city.nsmallest(3).round(2).to_dict()}")

    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    T = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(X, T)
    o = X[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    X["lr"] = np.log((X.total_ticket + 1) / (o.oof_final.values + 1))
    X["e"] = np.abs(X.total_ticket - o.oof_final.values) / X.scale
    X["w"], X["base"] = w, base_title(X.movie_title)
    X["we"] = X.date_show.dt.dayofweek >= 5
    # fold-local cluster amplitude: exclude the row's own film
    amp_ex = {}
    for b in X.base.unique():
        amp_ex[b] = A[A.base != b].groupby("cinema_ids").amp.median()
    X["camp"] = [amp_ex[b].get(c, np.nan) for b, c in zip(X.base, X.cinema_ids)]
    X["camp_dev"] = X.camp - camp.median()
    for d1dow, lab in [(2, "Wed release"), (3, "Thu release")]:
        q = X[(X.d1_dow == d1dow) & X.we]
        print(f"\n{lab}, weekend target rows {len(q)}: spearman(residual, cluster amplitude) {q.lr.corr(q.camp_dev, method='spearman'):+.3f};"
              f" TW-MASE {np.average(q.e, weights=q.w):.3f}")
        k = pd.qcut(q.camp_dev.rank(method="first"), 5, labels=False)
        print("  mean residual by cluster-amplitude quintile:", q.lr.groupby(k).mean().round(3).tolist())
        qw = X[(X.d1_dow == d1dow) & ~X.we]
        k2 = pd.qcut(qw.camp_dev.rank(method="first"), 5, labels=False)
        print("  weekday rows, same quintiles:", qw.lr.groupby(k2).mean().round(3).tolist())
    # oracle-style correction on weekend rows of Wed/Thu releases: pred x exp(beta * camp_dev), beta=1 (structural)
    p = o.oof_final.values.copy()
    for beta in (0.5, 1.0):
        adj = np.where(X.we.values & X.camp_dev.notna().values, np.exp(beta * X.camp_dev.fillna(0).values), 1.0)
        adj = adj * np.where(~X.we.values & X.camp_dev.notna().values, np.exp(-beta * X.camp_dev.fillna(0).values * 2 / 5), 1.0)
        e = np.abs(X.total_ticket.values - p * adj) / X.scale.values
        print(f"structural cluster-amplitude correction beta={beta}: TW {np.average(e, weights=w):.4f} vs {np.average(X.e, weights=w):.4f}")
    # genre amplitude among Wed-release weekend rows
    q = X[(X.d1_dow == 2) & X.we]
    g = pd.DataFrame({gn: q[q[f"g_{gn}"] == 1].lr.mean() for gn in ["Horror", "Drama", "Action", "Comedy", "Animation", "Family", "Romance"]}, index=["mean residual"]).T
    print("\nWed-release weekend residual by genre:"); print(g.round(3).to_string())
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].hist(np.exp(camp), bins=30); ax[0].set(title="Cluster weekend amplitude (weekend / Mon-Thu)")
    ax[1].scatter(both.h1, both.h2, s=10); ax[1].set(title=f"Split-half by film, r = {rel:.2f}", xlabel="half 1", ylabel="half 2")
    fig.tight_layout(); fig.savefig(out / "weekend_amplitude.png")


if __name__ == "__main__":
    main()
