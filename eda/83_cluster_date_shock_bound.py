"""Honest upper bound of the cluster x date lever: leave-one-film-out common shocks.

eda/37 put the 'cluster x date multiplier known' oracle at TW 0.202 (from 0.325), but a per-cell
oracle computed with the row's own label is inflated when few films share a cell. Here the shock
for row (film A, cluster c, date t) uses ONLY other base films' rows at (c, t):
    shock = shrunk mean of log((y + 1) / (pred + 1)) over other films at (c, t)
Then pred_A x exp(shock) is scored. Variants: same date only; same cluster any date within +-3 days
(slow-moving cluster state); and the national date shock (all clusters, other films). Labels of other
films are NOT available at test time, so this is a bound, not a feature. A second part measures what
the test side can actually see: how many other films are inside their D1-D3 window at (c, t).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, style
from evaluate import test_weights_fd


def loo_shock(df, keys, k=3.0):
    """Mean log-ratio of OTHER base films in the same key cell, shrunk by n/(n+k)."""
    g = df.groupby(keys + ["base"]).lr.agg(["sum", "count"]).reset_index()
    tot = g.groupby(keys)[["sum", "count"]].sum().rename(columns={"sum": "S", "count": "N"}).reset_index()
    g = g.merge(tot, on=keys)
    g["n_o"], g["s_o"] = g.N - g["count"], g.S - g["sum"]
    g["shock"] = np.where(g.n_o > 0, g.s_o / (g.n_o + k), 0.0)
    g["n_other_films_rows"] = g.n_o
    return df.merge(g[keys + ["base", "shock", "n_other_films_rows"]], on=keys + ["base"], how="left")


def main():
    out = fig_dir("83_cluster_date_bound")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    df = x[KEY + ["date_show", "scale", "total_ticket", "h"]].copy()
    df["pred"], df["w"], df["base"] = o.oof_final.values, w, base_title(x.movie_title).values
    df["lr"] = np.log((df.total_ticket + 1) / (df.pred + 1))
    df["row"] = np.arange(len(df))
    base_tw = np.average(np.abs(df.total_ticket - df.pred) / df.scale, weights=w)
    res = [dict(variant="v13 OOF", TW=base_tw, mean_other_rows=np.nan)]
    for name, keys in [("cluster x date (other films)", ["cinema_ids", "date_show"]),
                       ("national date (other films)", ["date_show"]),
                       ("cluster x film-age h (other films)", ["cinema_ids", "h"])]:
        d = loo_shock(df, keys).sort_values("row")
        for k in (1.0, 3.0):
            d2 = loo_shock(df, keys, k).sort_values("row")
            p = d2.pred.values * np.exp(d2.shock.values)
            res.append(dict(variant=f"{name}, shrink k={k:g}", TW=np.average(np.abs(d2.total_ticket.values - p) / d2.scale.values, weights=w),
                            mean_other_rows=d2.n_other_films_rows.mean()))
        if name.startswith("cluster x date"):
            cd = d
    # cluster x week (slow state): other films within +-3 days at the same cluster
    df["wk"] = (df.date_show - pd.Timestamp("2025-03-31")).dt.days // 7
    d2 = loo_shock(df, ["cinema_ids", "wk"]).sort_values("row")
    p = d2.pred.values * np.exp(d2.shock.values)
    res.append(dict(variant="cluster x week (other films), k=3", TW=np.average(np.abs(d2.total_ticket.values - p) / d2.scale.values, weights=w),
                    mean_other_rows=d2.n_other_films_rows.mean()))
    R = pd.DataFrame(res).set_index("variant")
    R["gain"] = R.TW - base_tw
    pd.set_option("display.width", 200)
    print("Leave-one-film-out shock bounds (labels of OTHER films; not available at test time):")
    print(R.round(4).to_string())
    R.to_csv(out / "bounds.csv")
    # how strongly does a film's residual co-move with other films at (c, t)? correlation of own lr with shock
    print(f"\ncorr(own log-ratio, other-film cluster-date shock): {np.corrcoef(cd.lr, cd.shock)[0, 1]:+.3f}"
          f" | rows with >=1 other film at (c,t): {(cd.n_other_films_rows > 0).mean():.3f}")

    # what the test side can see: other films inside their D1-D3 window at (c, t)
    D = load()
    h = D["hist"].assign(base=base_title(D["hist"].movie_title))
    vis = h.groupby(["cinema_ids", "date_show"]).base.nunique().rename("n_vis")
    tt = t[["movie_title", "cinema_ids", "date_show"]].assign(base=base_title(t.movie_title))
    tt = tt.join(vis, on=["cinema_ids", "date_show"]).fillna({"n_vis": 0})
    print(f"test target rows with >=1 other film in its D1-D3 window at the same (cluster, date): {(tt.n_vis > 0).mean():.3f};"
          f" mean count {tt.n_vis.mean():.2f}")
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].barh(R.index, R.gain); ax[0].set(title="TW-MASE change (negative = better)")
    q = pd.qcut(cd.shock.rank(method="first"), 10, labels=False)
    ax[1].plot(cd.groupby(q).shock.mean(), cd.groupby(q).lr.mean(), marker="o")
    ax[1].set(xlabel="other-film shock (decile mean)", ylabel="own log-ratio (mean)", title="Co-movement at (cluster, date)")
    fig.tight_layout(); fig.savefig(out / "bounds.png")


if __name__ == "__main__":
    main()
