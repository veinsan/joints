"""Late starters (first sale on D2 or D3): is the deflated official scale the real problem?

For a pair that only started showing on D3, s = y3/3, so even a perfectly 'normal' D4 gives r ~ 3.
Questions:
 1. What fraction survive (y>0) per horizon, and what does y_t / y_last look like (level in units of
    the last observed day instead of the official scale)?
 2. Which D1-D3 signals separate survivors from vanishers: shows on the first day, tickets per show,
    occupancy, whether the whole film expanded coverage on D3 (wide late expansion), cluster size.
 3. Do the same signals look the same in the test period?
 4. v13 OOF error on these rows vs simple rules (y_last x calendar ratio, survival-gated).
Lookups and saved predictions only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, fig_dir, style
from evaluate import test_weights_fd


def main():
    out = fig_dir("80_late_starters")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    x["pred"], x["w"] = o.oof_final.values, w
    for d in (x, t):
        d["fd"] = np.where(d.y1 > 0, 1, np.where(d.y2 > 0, 2, 3))
        d["ylast"] = np.where(d.fd == 3, d.y3, (d.y2 + d.y3) / 2)
        d["shfirst"] = np.where(d.fd == 3, d.sh3, d.sh2)
        d["tps_last"] = d.y3 / d.sh3.clip(lower=1)
        d["f_exp"] = d.fnc3 / d.fnc1.clip(lower=1)
    x["y"] = x.total_ticket
    x["ratio_last"] = x.y / x.ylast / x.cal_mult            # level in units of the last observed day
    x["e"] = np.abs(x.y - x.pred) / x.scale
    pd.set_option("display.width", 220)
    for fd in (3, 2):
        L = x[x.fd == fd]
        print(f"\n===== first sale on D{fd}: train rows {len(L)} ({L.drop_duplicates(KEY).shape[0]} pairs), "
              f"test rows {(t.fd == fd).sum()} ({t[t.fd == fd].drop_duplicates(KEY).shape[0]} pairs)")
        print("survival (y>0) by h:", L.groupby("h").y.apply(lambda v: (v > 0).mean()).round(3).to_dict())
        print("median y/ylast/cal by h among survivors:", L[L.y > 0].groupby("h").ratio_last.median().round(2).to_dict())
        print(f"v13 TW-MASE {np.average(L.e, weights=L.w):.3f}; median r actual {np.median(L.y / L.scale):.2f} vs pred {np.median(L.pred / L.scale):.2f}")
        # univariate survival screens (pair-level survival = any sale D4-D10, and D4 survival)
        P = L.groupby(KEY).agg(surv=("y", lambda v: (v > 0).mean()), y_mean=("y", "mean"), ylast=("ylast", "first"),
                               shfirst=("shfirst", "first"), tps=("tps_last", "first"), occ3=("occ3", "first"),
                               f_exp=("f_exp", "first"), fnc3=("fnc3", "first"), cin=("cin_size", "first"),
                               d1_dow=("d1_dow", "first")).reset_index()
        T = t[t.fd == fd].groupby(KEY).agg(ylast=("ylast", "first"), shfirst=("shfirst", "first"), tps=("tps_last", "first"),
                                           occ3=("occ3", "first"), f_exp=("f_exp", "first"), fnc3=("fnc3", "first"),
                                           cin=("cin_size", "first"), d1_dow=("d1_dow", "first")).reset_index()
        for c in ["ylast", "shfirst", "tps", "occ3", "f_exp", "cin"]:
            q = pd.qcut(P[c].rank(method="first"), 4, labels=False)
            print(f"  {c:8s} quartile survival {P.surv.groupby(q).mean().round(2).tolist()} | spearman {P[c].corr(P.surv, method='spearman'):+.2f}"
                  f" | train median {P[c].median():.2f} test median {T[c].median():.2f}")
        P.to_csv(out / f"pairs_fd{fd}.csv", index=False)
        T.to_csv(out / f"test_pairs_fd{fd}.csv", index=False)
    # how much error would go away if late starters were perfect / were predicted with simple rules
    L = x[x.fd == 3].copy()
    tot = np.average(x.e, weights=x.w)
    for name, pr in {"v13": L.pred, "zero": 0 * L.pred,
                     "ylast x cal x 0.5": L.ylast * L.cal_mult * 0.5, "ylast x cal x 1.0": L.ylast * L.cal_mult}.items():
        e = np.abs(L.y - pr) / L.scale
        print(f"  late starter rule {name:18s} TW-MASE {np.average(e, weights=L.w):.3f} -> total {tot + ((e - L.e) * L.w).sum() / x.w.sum():.4f}")
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for fd, col in [(3, "#e34948"), (2, "#eda100"), (1, "#2a78d6")]:
        L = x[x.fd == fd]
        ax[0].plot(range(4, 11), L.groupby("h").y.apply(lambda v: (v > 0).mean()).values, marker="o", color=col, label=f"first sale D{fd}")
        ax[1].plot(range(4, 11), L.groupby("h").e.mean().values, marker="o", color=col, label=f"D{fd}")
    ax[0].set(title="Survival by horizon", xlabel="h"); ax[0].legend()
    ax[1].set(title="v13 OOF MASE by horizon", xlabel="h")
    L = x[(x.fd == 3) & (x.y > 0)]
    ax[2].hist(np.log2(L.ratio_last.clip(1/64, 64)), bins=40, color="#e34948")
    ax[2].set(title="late starters: log2(y / y3 / cal), survivors", xlabel="log2 ratio")
    fig.tight_layout(); fig.savefig(out / "late_starters.png")


if __name__ == "__main__":
    main()
