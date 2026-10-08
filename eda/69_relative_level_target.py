"""Does the D4-D10 target depend on the pair's absolute level or on its level relative to the cluster?

eda/68: the D3 pull follows a cluster-relative yardstick (test offset -1.35 -> -0.21). Here the same
question for the target itself. For every target pair:
    log_s     = log1p(mean y1..y3)                         (absolute; a v12 feature)
    rel_clu   = log(scale) - log(median scale of the OTHER visible films' pairs in the same cluster
                                 whose D1 is within +-14 days)  (visible = simulated films in train,
                                 test films in test: the same information rule on both sides)
Lookup tables (h x decile) only. Three checks per yardstick:
  1. resolution inside train (zero log-loss, MAE of r around the bin median),
  2. drift: table from Apr-Jun films applied to Jul-Sep films (does it travel across market states?),
  3. what each yardstick implies for test rows (zero rate, median r).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, fig_dir, load, release_dates, simulate, style


def pair_scales(hist):
    d1 = hist.groupby("movie_title").date_show.min().rename("d1")
    s = (hist.groupby(KEY).total_ticket.sum() / 3).rename("sc").reset_index()
    return s.join(d1, on="movie_title")


def rel_clu(X, P, win=14):
    """log scale minus log median scale of other films' pairs in the same cluster within +-win days of D1."""
    out = np.full(len(X), np.nan)
    Xp = X.drop_duplicates(KEY)[KEY + ["d1", "scale"]]
    ref = {}
    for c, q in P.groupby("cinema_ids"):
        dd = q.d1.values.astype("datetime64[D]").astype(int)
        for m, di in Xp[Xp.cinema_ids == c][["movie_title", "d1"]].itertuples(index=False):
            di = np.datetime64(di, "D").astype(int)
            k = (np.abs(dd - di) <= win) & (q.movie_title.values != m)
            ref[(m, c)] = np.median(q.sc.values[k]) if k.sum() >= 2 else np.nan
    r = pd.Series(ref)
    v = np.array([r.get((m, c), np.nan) for m, c in zip(X.movie_title, X.cinema_ids)])
    return np.log(X.scale.values) - np.log(np.clip(v, 1, None)), v


def bins(x, edges):
    return np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)


def main():
    out = fig_dir("69_relative_level")
    D = load()
    d1 = release_dates(D["train"])
    hist, _ = simulate(D["train"], d1)
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    T = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    X["r"] = X.total_ticket / X.scale / X.cal_mult
    X["z"] = (X.total_ticket == 0).astype(int)
    X["rel_clu"], X["ref"] = rel_clu(X, pair_scales(hist))
    T["rel_clu"], T["ref"] = rel_clu(T, pair_scales(D["hist"]))
    print(f"rel_clu coverage: train {np.isfinite(X.rel_clu).mean():.3f}, test {np.isfinite(T.rel_clu).mean():.3f}")
    print("\nDistribution (train | test) quantiles 10/50/90:")
    for c in ["log_s", "rel_clu"]:
        print(f"  {c:8s} train {np.nanquantile(X[c], [.1, .5, .9]).round(2)} | test {np.nanquantile(T[c], [.1, .5, .9]).round(2)}")
    print(f"  cluster reference scale median train {np.nanmedian(X.ref):.1f} | test {np.nanmedian(T.ref):.1f}")

    X = X[np.isfinite(X.rel_clu)].copy()
    T = T[np.isfinite(T.rel_clu)].copy()
    early = (X.d1.dt.month <= 6).values
    res = []
    for name in ["log_s", "rel_clu"]:
        e = np.unique(np.quantile(X[name], np.linspace(0, 1, 11)))
        X["b"], T["b"] = X.h.astype(str) + "|" + bins(X[name].values, e).astype(str), T.h.astype(str) + "|" + bins(T[name].values, e).astype(str)
        z_l, r_l = X.groupby("b").z.mean(), X.groupby("b").r.median()
        ez = X.b.map(z_l).clip(1e-3, 1 - 1e-3)
        ll = -np.mean(X.z * np.log(ez) + (1 - X.z) * np.log(1 - ez))
        mae = np.mean(np.abs(X.r - X.b.map(r_l)))
        ze, re_ = X[early].groupby("b").z.mean(), X[early].groupby("b").r.median()
        late = X[~early]
        drift_z = late.z.mean() - late.b.map(ze).mean()
        drift_r = (late.r - late.b.map(re_)).median()
        mae_late = np.mean(np.abs(late.r - late.b.map(re_).fillna(X.r.median())))
        res.append(dict(yardstick=name, zero_logloss=ll, r_mae=mae, oot_zero_bias=drift_z, oot_r_median_bias=drift_r,
                        oot_r_mae=mae_late, test_zero_rate=T.b.map(z_l).mean(), test_median_r=T.b.map(r_l).mean()))
    # conditional value: does absolute level still matter once rel_clu is known (and vice versa)?
    ea, er = np.quantile(X.log_s, np.linspace(0, 1, 6)), np.quantile(X.rel_clu, np.linspace(0, 1, 6))
    X["ba"], X["br"] = bins(X.log_s.values, ea), bins(X.rel_clu.values, er)
    piv_z = X.pivot_table(index="br", columns="ba", values="z", aggfunc="mean").round(2)
    piv_r = X.pivot_table(index="br", columns="ba", values="r", aggfunc="median").round(2)
    piv_n = X.pivot_table(index="br", columns="ba", values="z", aggfunc="size")
    r = pd.DataFrame(res).set_index("yardstick").round(4)
    pd.set_option("display.width", 200)
    print(f"\ntrain rows with reference {len(X)}, test rows {len(T)}; actual train zero {X.z.mean():.3f}, median r {X.r.median():.3f}")
    print(r.to_string())
    r.to_csv(out / "yardsticks.csv")
    print("\nZero rate: rows = rel_clu quintile, cols = log_s quintile (train)")
    print(piv_z.to_string())
    print("\nMedian r (calendar-adjusted):")
    print(piv_r.to_string())
    print("\nrow counts:\n", piv_n.to_string())
    T["ba"], T["br"] = bins(T.log_s.values, ea), bins(T.rel_clu.values, er)
    print("\nTest row share per cell:")
    print((T.pivot_table(index="br", columns="ba", values="h", aggfunc="size") / len(T)).round(3).to_string())

    # the 2D table decomposed: zero rate by one dimension holding the other fixed (within-cell weights of test)
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for name, col in [("log_s", "#2a78d6"), ("rel_clu", "#eb6834")]:
        e = np.quantile(X[name], np.linspace(0, 1, 11))
        g = X.groupby(bins(X[name].values, e)).agg(z=("z", "mean"), r=("r", "median"))
        ax[0].plot(g.index, g.z, marker="o", color=col, label=name)
        ax[1].plot(g.index, g.r, marker="o", color=col, label=name)
        ax[2].hist(X[name], bins=40, alpha=.35, density=True, color=col, label=f"{name} train")
        ax[2].hist(T[name], bins=40, alpha=.35, density=True, color=col, histtype="step", lw=2, label=f"{name} test")
    ax[0].set_title("D4-D10 zero rate by decile"); ax[0].set_xlabel("decile"); ax[0].legend()
    ax[1].set_title("median r (y / s / cal) by decile"); ax[1].set_xlabel("decile")
    ax[2].set_title("train vs test distribution"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "yardsticks.png")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, piv, lab in [(ax[0], piv_z, "zero rate"), (ax[1], piv_r, "median r")]:
        im = a.imshow(piv.values, cmap="Blues", origin="lower")
        for (i, j), v in np.ndenumerate(piv.values):
            a.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8)
        a.set_xlabel("log_s quintile (absolute)"); a.set_ylabel("rel_clu quintile (vs cluster)"); a.set_title(lab)
        a.grid(False)
    fig.tight_layout(); fig.savefig(out / "abs_vs_rel_grid.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
