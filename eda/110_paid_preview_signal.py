"""Do films with paid previews before the official D1 grow more after D3 than the model expects?

Paid previews (sneak shows days before the official release) are a distributor's bet on word of
mouth, i.e. exactly the post-D3 trajectory information the models miss (eda/84). Official data only:
  train : transactions of the film in the 14 days before its reconstructed D1 (outside D1-D3)
  test  : the same is visible ONLY when those days fall inside train.csv (<= 30 Sep 2025), so for test
          films released after mid October previews are unobservable -> availability check.
Measures: how many train films had previews, their v15 OOF residual by horizon vs films without,
film-level bootstrap, and test coverage of the signal.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, release_dates, style
from evaluate import test_weights_fd


def main():
    out = fig_dir("110_paid_preview")
    D = load()
    tr = D["train"].assign(base=base_title(D["train"].movie_title))
    first = tr.groupby("movie_title").date_show.min()
    d1 = release_dates(D["train"]).drop(first[first == first.min()].index, errors="ignore")
    d1b = pd.Series(d1.values, index=base_title(pd.Series(d1.index)).values).groupby(level=0).min()
    x = tr.join(d1b.rename("d1"), on="base").dropna(subset=["d1"])
    pre = x[(x.date_show < x.d1) & (x.date_show >= x.d1 - pd.Timedelta(days=14))]
    P = pre.groupby("base").agg(pre_tx=("total_ticket", "sum"), pre_days=("date_show", "nunique"), pre_nc=("cinema_ids", "nunique"))
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    T = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(X, T)
    o = X[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v15/oof.csv"), on=KEY + ["h"], how="left")
    X["base"] = base_title(X.movie_title)
    X["lr"] = np.log((X.total_ticket + 1) / (o.oof_final.values + 1))
    X["e"] = np.abs(X.total_ticket - o.oof_final.values) / X.scale
    X["w"] = w
    X = X.join(P, on="base")
    X["preview"] = X.pre_tx.fillna(0) > 0
    F = X.groupby("base").agg(preview=("preview", "first"), pre_tx=("pre_tx", "first"), pre_nc=("pre_nc", "first"),
                              d1T=("fT1", "first"), d1=("d1", "first"))
    F["pre_share"] = F.pre_tx / F.d1T
    print(f"train films with previews in the 14 days before D1: {int(F.preview.sum())}/{len(F)}")
    print(F[F.preview].sort_values("pre_share", ascending=False)[["pre_tx", "pre_nc", "d1T", "pre_share"]].head(20).round(3).to_string())
    tab = X.groupby(["preview", "h"]).apply(lambda q: np.average(q.lr, weights=q.w), include_groups=False).unstack()
    pd.set_option("display.width", 200)
    print("\nweighted mean residual log((y+1)/(pred+1)) by horizon (v15):"); print(tab.round(3).to_string())
    G = X.groupby(["base", "preview"]).apply(lambda q: np.average(q.lr, weights=q.w), include_groups=False).reset_index(name="res")
    a, b = G[G.preview].res.values, G[~G.preview].res.values
    rng = np.random.default_rng(2026)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
    print(f"\nfilm-level mean residual: preview {a.mean():+.3f} (n={len(a)}) vs none {b.mean():+.3f} (n={len(b)}); "
          f"difference 95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    print("TW-MASE by group:", {k: round(np.average(g.e, weights=g.w), 4) for k, g in X.groupby("preview")})
    # test availability: preview window inside train.csv only for early test films
    hd1 = D["hist"].groupby("movie_title").date_show.min()
    tb = pd.DataFrame({"d1": hd1}).assign(base=lambda d: base_title(pd.Series(d.index, index=d.index)))
    tb["window_observable"] = (tb.d1 - pd.Timedelta(days=14)) <= pd.Timestamp("2025-09-30")
    seen = set(tr[tr.date_show >= "2025-09-16"].base)
    tb["preview_seen"] = tb.base.isin(seen) & tb.window_observable
    rows = T.assign(base=base_title(T.movie_title)).base.map(tb.drop_duplicates("base").set_index("base").window_observable)
    print(f"\ntest films whose 14-day pre-D1 window is observable: {int(tb.window_observable.sum())}/{len(tb)} "
          f"({rows.mean():.1%} of test rows); of those with a visible preview: {int(tb.preview_seen.sum())}")
    F.to_csv(out / "train_films.csv"); tb.to_csv(out / "test_films.csv")
    plt = style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, col in [(True, "#e34948"), (False, "#2a78d6")]:
        ax.plot(tab.columns, tab.loc[k].values, marker="o", color=col, label="paid preview" if k else "no preview")
    ax.axhline(0, color="k", lw=.8); ax.set(title="v15 residual by horizon: films with vs without paid previews", xlabel="h"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "preview_residual.png")


if __name__ == "__main__":
    main()
