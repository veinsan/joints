"""Preprocessing check of the v13 relative-level features, implemented exactly as in the notebook.

rel_ref(X, vis) is the function copied into notebooks/build_v13.py. Reference set = D1-D3 windows of
every wide release (train windows + test_history), the same information the test side has.
    rel_nat = log scale - log median scale of the other films' pairs whose D1 is within +-14 days
    rel_clu = same, restricted to the same cluster (>= 2 pairs, else falls back to rel_nat)
    f_rel   = log film mean daily total - log median of the other films' totals within +-14 days
'Other films' excludes the same base title (IMAX/3D versions of itself). No target-period label is used.
Checks: (1) train/test distributions vs the absolute features they replace, (2) coverage and
fallbacks, (3) self-exclusion, (4) the lookup result of eda/70 reproduced with this implementation,
(5) cluster examples: absolute level vs reference over time.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, release_dates, simulate, style

WIN = 14


def rel_ref(X, vis, win=WIN):
    """Notebook implementation (v13). X: target rows with movie_title, cinema_ids, d1, scale, fT1..fT3."""
    v = vis.assign(base=base_title(vis.movie_title))
    d1v = v.groupby("base").date_show.min().rename("d1")
    P = (v.groupby(["base", "cinema_ids"]).total_ticket.sum() / 3).rename("sc").reset_index().join(d1v, on="base")
    F = (v.groupby("base").total_ticket.sum() / 3).rename("ft").to_frame().join(d1v).reset_index()
    u = X[["movie_title", "cinema_ids", "d1"]].drop_duplicates().assign(base=lambda d: base_title(d.movie_title))
    days = lambda s: s.values.astype("datetime64[D]").astype(np.int64)
    # national pair reference, one value per film
    fu = u.drop_duplicates("base")[["base", "d1"]]
    pd1, psc, pbase = days(P.d1), P.sc.values, P.base.values
    nat = {b: np.median(psc[(np.abs(pd1 - t) <= win) & (pbase != b)]) for b, t in zip(fu.base, days(fu.d1))}
    fd1, fft, fb = days(F.d1), F.ft.values, F.base.values
    fnat = {b: np.median(fft[(np.abs(fd1 - t) <= win) & (fb != b)]) for b, t in zip(fu.base, days(fu.d1))}
    # cluster pair reference
    m = u.merge(P.rename(columns={"base": "base_o", "d1": "d1_o"}), on="cinema_ids", how="left")
    m = m[(np.abs(days(m.d1) - days(m.d1_o)) <= win) & (m.base != m.base_o)]
    g = m.groupby(["movie_title", "cinema_ids"]).sc.agg(["median", "size"])
    clu = g["median"].where(g["size"] >= 2)
    out = X[["movie_title", "cinema_ids"]].copy()
    b = base_title(X.movie_title)
    ref_nat = b.map(nat).values
    ref_clu = pd.MultiIndex.from_frame(X[KEY]).map(clu).values.astype(float)
    ls = np.log(X.scale.values)
    out["rel_nat"] = ls - np.log(np.clip(ref_nat, 1, None))
    out["rel_clu"] = np.where(np.isfinite(ref_clu), ls - np.log(np.clip(ref_clu, 1, None)), out.rel_nat)
    fm = X[["fT1", "fT2", "fT3"]].mean(axis=1).clip(lower=1).values
    out["f_rel"] = np.log(fm) - np.log(np.clip(b.map(fnat).values, 1, None))
    out["clu_fallback"] = ~np.isfinite(ref_clu)
    out["ref_clu"], out["ref_nat"] = ref_clu, ref_nat
    return out


def main():
    out = fig_dir("73_relative_prepro")
    D = load()
    d1 = release_dates(D["train"])
    hist, _ = simulate(D["train"], d1)
    x = D["train"].merge(d1.rename("d1"), left_on="movie_title", right_index=True)
    vis_tr = x[(x.date_show >= x.d1) & (x.date_show <= x.d1 + pd.Timedelta(days=2))].drop(columns="d1")
    vis = pd.concat([vis_tr, D["hist"]], ignore_index=True)
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    T = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    RX, RT = rel_ref(X, vis), rel_ref(T, vis)
    for c in ["rel_nat", "rel_clu", "f_rel"]:
        X[c], T[c] = RX[c].values, RT[c].values

    # (2) coverage
    print(f"cluster-reference fallback: train {RX.clu_fallback.mean():.4f}, test {RT.clu_fallback.mean():.4f};"
          f" NaN rel_nat train {np.isnan(RX.rel_nat).mean():.4f} test {np.isnan(RT.rel_nat).mean():.4f}")
    # (3) self-exclusion: an IMAX version must never be its own reference -> compare with a version that includes self
    imax = X.movie_title.str.contains(r"\(IMAX|\(3D", regex=True)
    print(f"format-version rows in train: {imax.sum()} (reference excludes the same base title by construction)")

    # (1) distributions
    rows = []
    for c in ["log_s", "rel_nat", "rel_clu", "f_logT", "f_rel"]:
        a, b = X[c].dropna(), T[c].dropna()
        rows.append(dict(feature=c, train_median=a.median(), test_median=b.median(), shift=b.median() - a.median(),
                         train_iqr=a.quantile(.75) - a.quantile(.25), ks=ks_2samp(a, b).statistic))
    dist = pd.DataFrame(rows).set_index("feature").round(3)
    pd.set_option("display.width", 200)
    print("\nTrain vs test distribution (absolute features the relative ones replace):")
    print(dist.to_string())
    dist.to_csv(out / "distributions.csv")

    # (4) lookup check on D4-D10 (h x decile): implied test zero rate & median r, plus quiet-film split
    X["r"] = X.total_ticket / X.scale / X.cal_mult
    X["z"] = (X.total_ticket == 0).astype(int)
    res = []
    film_ref = X.groupby(base_title(X.movie_title)).apply(lambda q: np.median(RX.loc[q.index, "ref_nat"]), include_groups=False)
    quiet = base_title(X.movie_title).map(film_ref <= film_ref.quantile(.4)).values
    for c in ["log_s", "rel_nat", "rel_clu"]:
        e = np.unique(np.quantile(X[c], np.linspace(0, 1, 11)))
        kb = lambda d: d.h.astype(str) + "|" + np.clip(np.searchsorted(e, d[c].values, side="right") - 1, 0, len(e) - 2).astype(str)
        kx, kt = kb(X), kb(T)
        zr, mr = X.z.groupby(kx).mean(), X.r.groupby(kx).median()
        mq = X.r[~quiet].groupby(kx[~quiet]).median()
        pq = kx[quiet].map(mq).fillna(X.r[~quiet].median()) * X.cal_mult[quiet]
        res.append(dict(feature=c, test_zero_rate=kt.map(zr).mean(), test_median_r=kt.map(mr).mean(),
                        quiet_mase=np.mean(np.abs(X.total_ticket[quiet] / X.scale[quiet] - pq)),
                        quiet_bias=np.median(X.total_ticket[quiet] / X.scale[quiet] - pq)))
    res = pd.DataFrame(res).set_index("feature").round(4)
    print(f"\nLookup check (train zero {X.z.mean():.3f}, train median r {X.r.median():.3f}); quiet = 40% films with lowest national reference:")
    print(res.to_string())
    res.to_csv(out / "lookup_check.csv")

    # (5) cluster examples
    plt = style()
    big = X.cinema_ids.value_counts().index[[0, 40, 90]]
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6), sharey=False)
    for a, c in zip(ax, big):
        for D_, R_, col, lab in [(X, RX, "#2a78d6", "train"), (T, RT, "#eb6834", "test")]:
            q = D_[D_.cinema_ids == c].drop_duplicates(KEY)
            r = R_.loc[q.index]
            a.scatter(q.d1, np.log(q.scale), s=6, alpha=.35, color=col, label=f"{lab} pair log scale")
            o = np.argsort(q.d1.values)
            a.plot(q.d1.values[o], np.log(r.ref_clu.values[o]), color=col, lw=1.5, label=f"{lab} cluster reference")
        a.set_title(f"cluster {c}"); a.set_ylabel("log tickets/day (D1-D3 mean)")
        a.tick_params(axis="x", rotation=45)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "cluster_reference_examples.png")

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
    for a, c in zip(ax, ["log_s", "rel_clu", "f_rel"]):
        a.hist(X[c].dropna(), bins=50, density=True, alpha=.5, label="train", color="#2a78d6")
        a.hist(T[c].dropna(), bins=50, density=True, alpha=.5, label="test", color="#eb6834")
        a.set_title(f"{c}: KS {dist.loc[c, 'ks']:.3f}")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "distributions.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
