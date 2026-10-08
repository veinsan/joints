"""Where does v13's residual live? Honest leave-one-out shocks at every level.

For each level, the shock of a row comes only from OTHER rows that share the level key but not the
excluded unit (so the row's own label never enters):
  film x date      : same base film and date, other clusters        (film trajectory after D3)
  film             : same base film, other clusters, any horizon
  pair             : same (film, cluster), other horizons             (persistent pair level)
  cluster          : same cluster, other films                         (cluster stickiness)
Each bound = TW-MASE of pred x exp(shrunk mean log-ratio). Bounds use labels that do not exist at
test time; they only say where predictive information would have to come from.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, style
from evaluate import test_weights_fd


def loo(df, keys, unit, k):
    g = df.groupby(keys + [unit]).lr.agg(["sum", "count"]).reset_index()
    tot = g.groupby(keys)[["sum", "count"]].sum().rename(columns={"sum": "S", "count": "N"}).reset_index()
    g = g.merge(tot, on=keys)
    g["shock"] = np.where(g.N > g["count"], (g.S - g["sum"]) / (g.N - g["count"] + k), 0.0)
    return df.merge(g[keys + [unit, "shock"]], on=keys + [unit], how="left").sort_values("row").shock.values


def main():
    out = fig_dir("84_residual_levels")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    df = x[KEY + ["date_show", "scale", "total_ticket", "h"]].copy()
    df["pred"], df["base"], df["row"] = o.oof_final.values, base_title(x.movie_title).values, np.arange(len(x))
    df["pair"] = df.movie_title + "|" + df.cinema_ids
    df["lr"] = np.log((df.total_ticket + 1) / (df.pred + 1))
    y, s, p = df.total_ticket.values, df.scale.values, df.pred.values
    tw = lambda q: np.average(np.abs(y - q) / s, weights=w)
    base = tw(p)
    rows = [dict(level="v13 OOF", TW=base)]
    spec = {"film x date (other clusters)": (["base", "date_show"], "cinema_ids"),
            "film x horizon (other clusters)": (["base", "h"], "cinema_ids"),
            "film (other clusters)": (["base"], "cinema_ids"),
            "pair (other horizons)": (["pair"], "h"),
            "cluster (other films)": (["cinema_ids"], "base")}
    shocks = {}
    for name, (keys, unit) in spec.items():
        for k in (1.0, 5.0):
            sh = loo(df, keys, unit, k)
            rows.append(dict(level=f"{name}, k={k:g}", TW=tw(p * np.exp(sh)), corr=np.corrcoef(df.lr, sh)[0, 1]))
        shocks[name] = sh
    R = pd.DataFrame(rows).set_index("level")
    R["gain"] = R.TW - base
    pd.set_option("display.width", 200)
    print(R.round(4).to_string())
    R.to_csv(out / "bounds.csv")
    # is the film trajectory shock visible in D1-D3? correlate film-level shock with film D1-D3 shape stats
    F = x.assign(sh=shocks["film (other clusters)"], base=df.base).groupby("base").agg(
        sh=("sh", "mean"), fp1=("fp1", "first"), fp3=("fp3", "first"), f_nc_trend=("f_nc_trend", "first"),
        f_logT=("f_logT", "first"), f_occ=("f_occ", "first"), d1_dow=("d1_dow", "first"))
    print("\nfilm-level shock vs D1-D3 film statistics (spearman):")
    print(F.drop(columns="sh").corrwith(F.sh, method="spearman").round(3).to_string())
    plt = style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(R.index[1:], R.gain[1:], color=["#2a78d6" if g < 0 else "#e34948" for g in R.gain[1:]])
    ax.axvline(0, color="k", lw=.8); ax.set(title="LOO shock bounds: TW-MASE change (negative = information exists)")
    fig.tight_layout(); fig.savefig(out / "bounds.png")


if __name__ == "__main__":
    main()
