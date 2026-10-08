"""Does a unit's 'stickiness' at D3 transfer to its D4-D10 zero rate?

Test-period evidence (eda/20-21): at equal tickets/show, test-period cinemas stop a film at D3 with
odds x0.27 (logit a = -1.32). Whether that shift carries to D4-D10 decides how much zero mass the
test really has, i.e. lambda. The old pair-level correlation (0.13, eda/22) is dominated by noise.

Here: units = cluster, cluster x month, month. For each unit
  a_u = logit offset of observed D3-stop vs a pooled lookup expectation (bins of log tps12 x D1 dow)
  b_u = logit offset of observed D4-D10 zeros vs a pooled lookup expectation (bins of h x p3 x scale)
Slope of b on a, corrected for the split-half reliability of a, is the transfer coefficient.
Lookup tables only, no model fitting.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, release_dates, simulate, style

logit = lambda p: np.log(p / (1 - p))
sig = lambda t: 1 / (1 + np.exp(-t))


def offset(e, z):
    """Logit shift b such that sum sigmoid(logit(e)+b) = sum z (unit-level calibration offset)."""
    e = np.clip(e, 1e-3, 1 - 1e-3)
    k = z.sum()
    if k == 0 or k == len(z):
        k = np.clip(k, .5, len(z) - .5)   # continuity correction
    return brentq(lambda b: sig(logit(e) + b).sum() - k, -8, 8)


def d3_stop_table(tx, d1):
    """Pairs that sold on D1 and D2 of a simulated window: z = no sale on D3."""
    x = tx.merge(d1.rename("d1"), left_on="movie_title", right_index=True)
    x["d"] = (x.date_show - x.d1).dt.days + 1
    x = x[x.d.between(1, 3)]
    p = x.pivot_table(index=KEY, columns="d", values=["total_ticket", "total_show"], fill_value=0)
    p = p[(p[("total_ticket", 1)] > 0) & (p[("total_ticket", 2)] > 0)]
    out = pd.DataFrame(index=p.index)
    out["tps"] = (p[("total_ticket", 1)] + p[("total_ticket", 2)]) / (p[("total_show", 1)] + p[("total_show", 2)]).clip(lower=1)
    out["z"] = (p.get(("total_ticket", 3), 0) == 0).astype(int)
    out = out.reset_index().merge(d1.rename("d1"), left_on="movie_title", right_index=True)
    out["dow"] = out.d1.dt.dayofweek
    return out


def main():
    out = fig_dir("67_pull_transfer")
    D = load()
    d1 = release_dates(D["train"])
    hist, tgt = simulate(D["train"], d1)
    d1s = d1[d1.index.isin(hist.movie_title.unique())]

    # --- D3 stop (train-sim films) + the same quantity in the test period
    A = d3_stop_table(D["train"], d1s)
    hd1 = D["hist"].groupby("movie_title").date_show.min()
    AT = d3_stop_table(D["hist"], hd1)
    edges = np.quantile(np.log1p(A.tps), np.linspace(0, 1, 11))
    for q in (A, AT):
        q["bin"] = np.clip(np.searchsorted(edges, np.log1p(q.tps), side="right") - 1, 0, 9).astype(str) + "|" + np.minimum(q.dow, 4).astype(str)
    look = A.groupby("bin").z.mean()
    A["e"], AT["e"] = A.bin.map(look).values, AT.bin.map(look).fillna(A.z.mean()).values
    a_test = offset(AT.e.values, AT.z.values)
    print(f"D3-stop rate train {A.z.mean():.3f} | test {AT.z.mean():.3f} | test expected {AT.e.mean():.3f} | test logit offset a = {a_test:+.3f}")

    # --- D4-D10 zeros
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    X["z"] = (X.total_ticket == 0).astype(int)
    X["bin"] = (X.h.astype(str) + "|" + pd.cut(X.p3, [-1, 0, .3, .6, .9, 1.2, 1.6, 9]).astype(str) + "|"
                + pd.cut(X.scale, [0, 5, 20, 50, 200, 1e9]).astype(str))
    X["e"] = X.bin.map(X.groupby("bin").z.mean())
    X["month"] = X.d1.dt.month
    A["month"] = A.d1.dt.month

    def unit_offsets(by):
        ka = A.groupby(by)
        kb = X.groupby(by)
        rows = []
        for u in set(ka.groups) & set(kb.groups):
            qa, qb = ka.get_group(u), kb.get_group(u)
            if len(qa) < 15 or len(qb) < 70:
                continue
            fa = base_title(qa.movie_title)
            ua = sorted(fa.unique())
            half = fa.isin(ua[::2]).values
            rows.append(dict(u=u, na=len(qa), nb=len(qb), a=offset(qa.e.values, qa.z.values),
                             a1=offset(qa.e.values[half], qa.z.values[half]) if half.sum() >= 5 else np.nan,
                             a2=offset(qa.e.values[~half], qa.z.values[~half]) if (~half).sum() >= 5 else np.nan,
                             b=offset(qb.e.values, qb.z.values)))
        return pd.DataFrame(rows)

    res = {}
    for name, by in [("cluster", "cinema_ids"), ("cluster_month", ["cinema_ids", "month"]), ("month", "month")]:
        u = unit_offsets(by)
        w = np.sqrt(u.na)
        slope = np.polyfit(u.a, u.b, 1, w=w)[0]
        ok = u[["a1", "a2"]].dropna()
        rel_half = ok.a1.corr(ok.a2) if len(ok) > 5 else np.nan
        rel = 2 * rel_half / (1 + rel_half) if rel_half == rel_half else np.nan   # Spearman-Brown
        corr = u.a.corr(u.b)
        res[name] = dict(units=len(u), corr_ab=corr, slope=slope, reliability_a=rel,
                         slope_corrected=slope / rel if rel and rel > 0.05 else np.nan,
                         sd_a=u.a.std(), sd_b=u.b.std())
        u.to_csv(out / f"offsets_{name}.csv", index=False)
        print(f"\n[{name}] units={len(u)} corr(a,b)={corr:+.3f} slope={slope:+.3f} "
              f"reliability(a)={rel:.3f} -> attenuation-corrected slope {res[name]['slope_corrected']:+.3f}"
              f" | sd a {u.a.std():.3f}, sd b {u.b.std():.3f}")
        if name == "month":
            print(u.sort_values("u").round(3).to_string(index=False))
    pd.DataFrame(res).T.to_csv(out / "transfer.csv")

    # implied test-period D4-D10 shift and its effect on the lookup zero rate of test rows
    te = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    te["bin"] = (te.h.astype(str) + "|" + pd.cut(te.p3, [-1, 0, .3, .6, .9, 1.2, 1.6, 9]).astype(str) + "|"
                 + pd.cut(te.scale, [0, 5, 20, 50, 200, 1e9]).astype(str))
    e_te = te.bin.map(X.groupby("bin").z.mean()).fillna(X.z.mean()).values
    print(f"\nLookup zero rate of test rows (train behaviour): {e_te.mean():.3f}")
    for lam in (0, .25, .5, .75, 1.0):
        print(f"  lambda {lam:4.2f}: test zero rate {sig(logit(np.clip(e_te, 1e-3, 1-1e-3)) + lam * a_test).mean():.3f}")

    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for i, name in enumerate(["cluster", "cluster_month", "month"]):
        u = pd.read_csv(out / f"offsets_{name}.csv")
        ax[i].scatter(u.a, u.b, s=np.clip(u.na / 5, 5, 60), alpha=.6)
        lim = np.array([u.a.min(), u.a.max()])
        ax[i].plot(lim, np.polyval(np.polyfit(u.a, u.b, 1, w=np.sqrt(u.na)), lim), color="#eb6834")
        ax[i].plot(lim, lim, color="#8a8984", ls="--", lw=1, label="slope 1")
        ax[i].set_xlabel("D3-stop logit offset a_u"); ax[i].set_ylabel("D4-D10 zero logit offset b_u")
        ax[i].set_title(f"{name}: r={res[name]['corr_ab']:+.2f}, slope*={res[name]['slope_corrected']:+.2f}")
    ax[0].legend()
    if True:
        ax[0].axvline(a_test, color="#e34948", lw=1, ls=":")
    fig.tight_layout(); fig.savefig(out / "transfer_scatter.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
