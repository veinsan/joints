"""Preprocessing checks for v15: (A) Causilo target transform, (B) release-week cohort folds.

(A) Causilo standardises the regression target with a plain StandardScaler (causilo/data/dataset.py) and
predicts 999 native quantiles in that space. Our target r = y / (s * cal_mult) has 30% exact zeros and
a tail above 100 for tiny scales, so the standardised target is a spike plus a few huge values. Checks
per horizon: skew, share of standardised values inside +-0.25 sd (resolution left for the bulk), and the
same for log1p(r). Quantiles commute with a monotone transform, so expm1(Q_log1p) = Q_r exactly; this
is verified empirically on the data (np.quantile on both scales).
(B) Film-grouped folds put 23 of 24 release weeks in more than one fold (eda/75), so validation films
share their week, calendar and competitors with training films. Cohort folds assign whole release
weeks to folds. Checks: fold sizes, weeks per fold, share of validation rows whose release week is in
training (must be 0), and the TW composition of every fold.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import base_title, fig_dir, style
from evaluate import test_weights_fd


def main():
    out = fig_dir("94_causilo_cohort")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    r = (x.total_ticket / x.scale / x.cal_mult).values
    rows = []
    for h in range(4, 11):
        v = r[x.h.values == h]
        for name, z in [("raw r", v), ("log1p r", np.log1p(v))]:
            zs = (z - z.mean()) / z.std()
            rows.append(dict(h=h, target=name, skew=skew(z), max_sd=zs.max(), share_within_0_25sd=np.mean(np.abs(zs) < .25),
                             iqr_in_sd=np.subtract(*np.percentile(zs, [75, 25]))))
    A = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("(A) target scale seen by Causilo after its StandardScaler:")
    print(A.groupby("target")[["skew", "max_sd", "share_within_0_25sd", "iqr_in_sd"]].mean().round(3).to_string())
    qs = np.round(np.arange(0.02, 1.0, 0.02), 2)
    back = np.expm1(np.quantile(np.log1p(r), qs, method="inverted_cdf"))
    direct = np.quantile(r, qs, method="inverted_cdf")
    print(f"    back-transform check: max |expm1(Q(log1p r)) - Q(r)| = {np.max(np.abs(back - direct)):.2e} over {len(qs)} levels")
    A.to_csv(out / "target_scale.csv", index=False)

    base = base_title(x.movie_title)
    wk = ((x.d1 - pd.Timestamp("2025-03-31")).dt.days // 7).values
    weeks = np.array(sorted(set(wk)))
    fw = dict(zip(weeks, np.random.RandomState(2026).permutation(len(weeks)) % 5))
    fold_c = np.array([fw[w_] for w_ in wk])
    ug = np.array(sorted(set(base)))
    ff = dict(zip(ug, np.random.RandomState(2026).permutation(len(ug)) % 5))
    fold_f = base.map(ff).values
    w = test_weights_fd(x, t)
    res = []
    for name, fold in [("film-grouped (v8-v14)", fold_f), ("release-week cohort (v15)", fold_c)]:
        for k in range(5):
            v = fold == k
            tr_weeks = set(wk[~v])
            res.append(dict(scheme=name, fold=k, rows=v.sum(), films=base[v].nunique(), weeks=len(set(wk[v])),
                            val_rows_week_in_train=np.mean(np.isin(wk[v], list(tr_weeks))),
                            tw_weight_share=w[v].sum() / w.sum(), small_share=np.mean(x.scale.values[v] <= 20)))
    B = pd.DataFrame(res)
    print("\n(B) fold geometry:")
    print(B.round(3).to_string(index=False))
    print(B.groupby("scheme")[["val_rows_week_in_train", "tw_weight_share"]].agg(["mean", "std"]).round(3).to_string())
    B.to_csv(out / "folds.csv", index=False)
    pd.DataFrame({"week": weeks, "fold": [fw[w_] for w_ in weeks]}).to_csv(out / "week_folds.csv", index=False)

    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    v = r[x.h.values == 6]
    for a_, z, ttl in [(ax[0], (v - v.mean()) / v.std(), "raw r, standardised (h=6)"), (ax[1], (np.log1p(v) - np.log1p(v).mean()) / np.log1p(v).std(), "log1p r, standardised (h=6)")]:
        a_.hist(np.clip(z, -3, 10), bins=80, color="#2a78d6"); a_.set(title=ttl, xlabel="sd units (clipped to [-3, 10])")
    for name, col in [("film-grouped (v8-v14)", "#e34948"), ("release-week cohort (v15)", "#1baf7a")]:
        q = B[B.scheme == name]
        ax[2].plot(q.fold, q.val_rows_week_in_train, marker="o", color=col, label=name)
    ax[2].set(title="Validation rows whose release week is in training", xlabel="fold"); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "causilo_cohort.png")


if __name__ == "__main__":
    main()
