"""Is the D3 pull decision absolute (tickets/show) or relative (vs the film / vs the market)?

eda/20-21 measured the test-period pull shift with an *absolute* tickets/show lookup: odds x0.27.
The test period is a quiet market (occupancy ~half of train). If cinemas pull films that do badly
*relative* to what is normal at the time, the 'shift' is an artefact of the absolute yardstick and
should vanish under a relative one. Then the same holds for D4-D10 zeros and the model's absolute
level features (log_s, tps) mislead it in the test period.

Lookup tables only (deciles from train-sim), no model fitting.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, fig_dir, load, release_dates, simulate, style

logit = lambda p: np.log(p / (1 - p))
sig = lambda t: 1 / (1 + np.exp(-t))


def offset(e, z):
    e = np.clip(e, 1e-3, 1 - 1e-3)
    return brentq(lambda b: sig(logit(e) + b).sum() - z.sum(), -8, 8)


def stops(tx, d1):
    x = tx.merge(d1.rename("d1"), left_on="movie_title", right_index=True)
    x["d"] = (x.date_show - x.d1).dt.days + 1
    x = x[x.d.between(1, 3)]
    piv = lambda v, pre: x.pivot_table(index=KEY, columns="d", values=v, fill_value=0).reindex(columns=[1, 2, 3], fill_value=0).add_prefix(pre)
    p = pd.concat([piv("total_ticket", "tot"), piv("total_show", "sh"), piv("occupation_rate", "occ")], axis=1)
    p = p[(p.tot1 > 0) & (p.tot2 > 0)].reset_index()
    p["tps"] = (p.tot1 + p.tot2) / (p.sh1 + p.sh2).clip(lower=1)
    p["occ12"] = (p.occ1 + p.occ2) / 2
    p["z"] = (p.tot3 == 0).astype(int)
    p = p.merge(d1.rename("d1"), left_on="movie_title", right_index=True)
    # film-relative: this cluster vs the film's national tickets/show
    ft = p.groupby("movie_title").apply(lambda q: (q.tot1 + q.tot2).sum() / (q.sh1 + q.sh2).sum(), include_groups=False)
    p["tps_film_rel"] = p.tps / p.movie_title.map(ft)
    # cluster-relative: vs the median tps of every *other* film's D1-D2 in this cluster within +-14 days
    p["tps_clu_rel"] = np.nan
    for c, q in p.groupby("cinema_ids"):
        dd = q.d1.values.astype("datetime64[D]").astype(int)
        t = q.tps.values
        ref = [np.median(t[(np.abs(dd - di) <= 14) & (np.arange(len(q)) != i)]) if ((np.abs(dd - di) <= 14).sum() > 1) else np.nan
               for i, di in enumerate(dd)]
        p.loc[q.index, "tps_clu_rel"] = t / np.array(ref)
    return p


def main():
    out = fig_dir("68_pull_relative")
    D = load()
    d1 = release_dates(D["train"])
    hist, _ = simulate(D["train"], d1)
    A = stops(D["train"], d1[d1.index.isin(hist.movie_title.unique())]).assign(src="train")
    T = stops(D["hist"], D["hist"].groupby("movie_title").date_show.min()).assign(src="test")
    print(f"pairs train {len(A)}, test {len(T)}; D3-stop train {A.z.mean():.3f}, test {T.z.mean():.3f}")
    print(f"median tps train {A.tps.median():.2f} test {T.tps.median():.2f} | occ12 {A.occ12.median():.1f} vs {T.occ12.median():.1f}")

    res = []
    defs = {"absolute tps": ["tps"], "occupancy": ["occ12"], "film-relative tps": ["tps_film_rel"],
            "cluster-relative tps": ["tps_clu_rel"], "abs x film-rel": ["tps", "tps_film_rel"],
            "cluster-rel x film-rel": ["tps_clu_rel", "tps_film_rel"]}
    for name, cols in defs.items():
        a, t = A.dropna(subset=cols).copy(), T.dropna(subset=cols).copy()
        nb = 10 if len(cols) == 1 else 5
        key_a, key_t = "", ""
        for c in cols:
            e = np.unique(np.quantile(np.log(a[c].clip(lower=1e-3)), np.linspace(0, 1, nb + 1)))
            ba = np.clip(np.searchsorted(e, np.log(a[c].clip(lower=1e-3)), side="right") - 1, 0, len(e) - 2)
            bt = np.clip(np.searchsorted(e, np.log(t[c].clip(lower=1e-3)), side="right") - 1, 0, len(e) - 2)
            key_a, key_t = key_a + ba.astype(str) + "|", key_t + bt.astype(str) + "|"
        look = a.z.groupby(key_a).mean()
        ea, et = key_a.map(look) if hasattr(key_a, "map") else pd.Series(key_a).map(look).values, pd.Series(key_t).map(look).fillna(a.z.mean()).values
        ea = pd.Series(key_a).map(look).values
        # in-sample log loss on train (resolution of the yardstick)
        ll = -np.mean(a.z * np.log(np.clip(ea, 1e-3, 1)) + (1 - a.z) * np.log(np.clip(1 - ea, 1e-3, 1)))
        # out-of-time: month 4-6 lookup scored on months 7-9 (does the yardstick travel across market states?)
        early = a.d1.dt.month <= 6
        lk = a.z[early.values].groupby(key_a[early.values]).mean()
        el = pd.Series(key_a[~early.values]).map(lk).fillna(a.z[early.values].mean()).values
        off_late = offset(el, a.z.values[~early.values])
        res.append(dict(yardstick=name, n_test=len(t), test_expected=et.mean(), test_observed=t.z.mean(),
                        test_offset=offset(et, t.z.values), train_logloss=ll, offset_jul_sep=off_late))
    r = pd.DataFrame(res).set_index("yardstick").round(3)
    pd.set_option("display.width", 200)
    print("\nTest-period D3-stop offset under each yardstick (0 = no policy shift needed):")
    print(r.to_string())
    r.to_csv(out / "offsets.csv")

    # monthly: does the absolute yardstick also drift inside train when the market level changes?
    A["month"] = A.d1.dt.month
    T["month"] = T.d1.dt.to_period("M").astype(str)
    mk = pd.concat([A.groupby("month").agg(occ=("occ12", "median"), tps=("tps", "median"), stop=("z", "mean"), n=("z", "size")),
                    T.groupby("month").agg(occ=("occ12", "median"), tps=("tps", "median"), stop=("z", "mean"), n=("z", "size"))])
    print("\nMarket state vs D3-stop rate by D1 month:")
    print(mk.round(3).to_string())

    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, (c, lab) in enumerate([("tps", "tickets/show D1-D2 (absolute)"), ("tps_film_rel", "tps / film national tps"),
                                  ("tps_clu_rel", "tps / other films in cluster +-14d")]):
        for q, col in [(A, "#2a78d6"), (T, "#eb6834")]:
            q = q.dropna(subset=[c])
            e = np.quantile(np.log(A[c].dropna().clip(lower=1e-3)), np.linspace(0, 1, 11))
            b = np.clip(np.searchsorted(e, np.log(q[c].clip(lower=1e-3)), side="right") - 1, 0, 9)
            g = q.z.groupby(b).mean()
            ax[i].plot(g.index, g.values, marker="o", color=col, label=q.src.iloc[0])
        ax[i].set_xlabel(f"decile of {lab}"); ax[i].set_ylabel("P(no sale on D3)")
        ax[i].set_title(lab)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "stop_curves.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
