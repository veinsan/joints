"""Two out-of-distribution checks for absolute vs cluster-relative level (lookup tables, no models).

A. Test-period proxy (the only labels inside the test period): pairs selling on D1 and D2, target
   q3 = y3 / mean(y1, y2) / (c3 / mean(c1, c2)). Table built on train-sim pairs, scored on test pairs.
   Which yardstick predicts the test-period D3 level and zero with less bias / lower MAE?
B. Quiet-market split inside train: films ranked by their cluster-reference level (how busy the
   market around their release was). Tables from the busiest 60% of films, scored on the quietest
   40% - the direction of the test shift. Scored with MASE of the bin-median r.
Yardsticks: absolute log level, cluster-relative level, and both (2D).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, release_dates, simulate, style
from features import calendar

sys.path.insert(0, str(ROOT / "eda"))
rel_mod = __import__("69_relative_level_target")


def proxy_table(hist, cal):
    h = hist.copy()
    d1 = h.groupby("movie_title").date_show.min().rename("d1")
    h = h.join(d1, on="movie_title")
    h["d"] = (h.date_show - h.d1).dt.days + 1
    p = h.pivot_table(index=KEY, columns="d", values="total_ticket", fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)
    p = p[(p[1] > 0) & (p[2] > 0)].reset_index().join(d1, on="movie_title")
    c = [cal.reindex(p.d1 + pd.Timedelta(days=k)).values for k in range(3)]
    p["m12"] = (p[1] + p[2]) / 2
    p["q3"] = p[3] / p.m12 / (c[2] / ((c[0] + c[1]) / 2))
    p["z"] = (p[3] == 0).astype(int)
    p["abs"] = np.log1p(p.m12)
    # cluster reference: median D1-D2 level of the other films' pairs in the cluster within +-14 days
    p["rel"] = np.nan
    for cid, q in p.groupby("cinema_ids"):
        dd = q.d1.values.astype("datetime64[D]").astype(int)
        v = q.m12.values
        ref = np.array([np.median(v[(np.abs(dd - x) <= 14) & (np.arange(len(q)) != i)]) if ((np.abs(dd - x) <= 14).sum() > 2) else np.nan
                        for i, x in enumerate(dd)])
        p.loc[q.index, "rel"] = np.log(v) - np.log(ref)
    return p.dropna(subset=["rel"])


def keyer(train, cols, nb):
    edges = {c: np.unique(np.quantile(train[c], np.linspace(0, 1, nb + 1))) for c in cols}
    def f(d):
        k = pd.Series("", index=d.index)
        for c in cols:
            k = k + np.clip(np.searchsorted(edges[c], d[c].values, side="right") - 1, 0, len(edges[c]) - 2).astype(str) + "|"
        return k
    return f


YARD = {"absolute": (["abs"], 10), "relative": (["rel"], 10), "both": (["abs", "rel"], 5)}


def main():
    out = fig_dir("70_relative_validation")
    D = load()
    cal = calendar(D["hol"]).cal
    d1 = release_dates(D["train"])
    hist, _ = simulate(D["train"], d1)
    A, T = proxy_table(hist, cal), proxy_table(D["hist"], cal)
    print(f"A. proxy pairs train {len(A)}, test {len(T)}; median q3 train {A.q3.median():.3f} test {T.q3.median():.3f};"
          f" zero {A.z.mean():.3f} vs {T.z.mean():.3f}")
    resA = []
    for name, (cols, nb) in YARD.items():
        k = keyer(A, cols, nb)
        ka, kt = k(A), k(T)
        med, zr = A.q3.groupby(ka).median(), A.z.groupby(ka).mean()
        pt = kt.map(med).fillna(A.q3.median())
        resA.append(dict(yardstick=name, test_q3_bias=float(np.median(T.q3 - pt)), test_mae=float(np.mean(np.abs(T.q3 - pt))),
                         train_mae=float(np.mean(np.abs(A.q3 - ka.map(med)))),
                         test_zero_pred=float(kt.map(zr).fillna(A.z.mean()).mean()), test_zero_obs=float(T.z.mean())))
    ra = pd.DataFrame(resA).set_index("yardstick").round(4)
    pd.set_option("display.width", 200)
    print(ra.to_string())
    ra.to_csv(out / "A_proxy.csv")

    # ---- B. quiet-market split on the real D4-D10 target
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    X["rel"], X["ref"] = rel_mod.rel_clu(X, rel_mod.pair_scales(hist))
    X = X[np.isfinite(X.rel)].copy()
    X["abs"] = X.log_s
    X["r"] = X.total_ticket / X.scale / X.cal_mult
    film_ref = X.groupby(base_title(X.movie_title)).ref.median()
    quiet_films = film_ref[film_ref <= film_ref.quantile(.4)].index
    q = base_title(X.movie_title).isin(quiet_films).values
    print(f"\nB. quiet split: busy films {(~q).sum()} rows (ref median {X.ref[~q].median():.0f}),"
          f" quiet films {q.sum()} rows (ref median {X.ref[q].median():.0f})")
    resB = []
    for name, (cols, nb) in YARD.items():
        k = keyer(X[~q], cols, nb)
        kb = X.h.astype(str) + "#" + k(X)
        med = X.r[~q].groupby(kb[~q]).median()
        pred = kb[q].map(med).fillna(X.r[~q].median()) * X.cal_mult[q]
        e = np.abs(X.total_ticket[q] / X.scale[q] - pred)
        # same table scored on held-out busy films (5-fold over films) as the in-distribution reference
        g = base_title(X.movie_title[~q]).values
        ug = np.unique(g); f = dict(zip(ug, np.random.RandomState(2026).permutation(len(ug)) % 5))
        fold = np.array([f[x] for x in g])
        Xb, kbb = X[~q], kb[~q]
        ein = []
        for i in range(5):
            m = Xb.r[fold != i].groupby(kbb[fold != i]).median()
            p_ = kbb[fold == i].map(m).fillna(Xb.r[fold != i].median()) * Xb.cal_mult[fold == i]
            ein.append(np.abs(Xb.total_ticket[fold == i] / Xb.scale[fold == i] - p_))
        ein = pd.concat(ein)
        resB.append(dict(yardstick=name, mase_busy_cv=ein.mean(), mase_quiet=e.mean(),
                         gap=e.mean() - ein.mean(), quiet_bias=float(np.median(X.total_ticket[q] / X.scale[q] - pred))))
    rb = pd.DataFrame(resB).set_index("yardstick").round(4)
    print(rb.to_string())
    rb.to_csv(out / "B_quiet_split.csv")

    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, col in [("absolute", "#2a78d6"), ("relative", "#eb6834"), ("both", "#1baf7a")]:
        cols, nb = YARD[name]
        k = keyer(A, cols, nb)
        med = A.q3.groupby(k(A)).median()
        pt = k(T).map(med).fillna(A.q3.median())
        dec = pd.qcut(T["abs"], 10, labels=False)
        g = (T.q3 - pt).groupby(dec).median()
        ax[0].plot(g.index, g.values, marker="o", color=col, label=name)
    ax[0].axhline(0, color="#8a8984", lw=.8)
    ax[0].set_xlabel("test pair absolute level decile"); ax[0].set_ylabel("median(actual - predicted q3)")
    ax[0].set_title("A. test-period D3 level: bias by absolute size"); ax[0].legend()
    ax[1].bar(rb.index, rb.gap, color=["#2a78d6", "#eb6834", "#1baf7a"])
    ax[1].set_title("B. MASE gap quiet-market minus in-distribution"); ax[1].set_ylabel("gap")
    fig.tight_layout(); fig.savefig(out / "relative_validation.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def national_check():
    """Add-on (run: python eda/70_... national): national-market relative vs cluster relative on the test proxy."""
    D = load()
    cal = calendar(D["hol"]).cal
    hist, _ = simulate(D["train"], release_dates(D["train"]))
    A, T = proxy_table(hist, cal), proxy_table(D["hist"], cal)
    for P in (A, T):
        # national reference: median D1-D2 level of every other film's pairs whose D1 is within +-14 days
        dd = P.d1.values.astype("datetime64[D]").astype(int)
        films = P.movie_title.values
        ref = {}
        for m in np.unique(films):
            x = dd[films == m][0]
            k = (np.abs(dd - x) <= 14) & (films != m)
            ref[m] = np.median(P.m12.values[k]) if k.sum() > 10 else np.nan
        P["nat"] = np.log(P.m12) - np.log(P.movie_title.map(ref))
    A, T = A.dropna(subset=["nat"]), T.dropna(subset=["nat"])
    rows = []
    for name, (cols, nb) in {"absolute": (["abs"], 10), "national-relative": (["nat"], 10), "cluster-relative": (["rel"], 10),
                             "national x cluster": (["nat", "rel"], 5)}.items():
        k = keyer(A, cols, nb)
        med, zr = A.q3.groupby(k(A)).median(), A.z.groupby(k(A)).mean()
        pt = k(T).map(med).fillna(A.q3.median())
        sm = T.m12.between(5, 50).values
        rows.append(dict(yardstick=name, test_mae=np.mean(np.abs(T.q3 - pt)), test_bias=np.median(T.q3 - pt),
                         bias_scale_5_50=np.median((T.q3 - pt)[sm]), train_mae=np.mean(np.abs(A.q3 - k(A).map(med))),
                         test_zero_pred=k(T).map(zr).fillna(A.z.mean()).mean(), test_zero_obs=T.z.mean()))
    r = pd.DataFrame(rows).set_index("yardstick").round(4)
    print(r.to_string())
    r.to_csv(fig_dir("70_relative_validation") / "A_proxy_national.csv")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "national":
    national_check()
