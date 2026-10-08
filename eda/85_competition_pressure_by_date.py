"""Does date-varying competition from newly opening films explain the film x date residual?

eda/84: the missing information is the film's trajectory per horizon (film x date residual, LOO corr
0.57, bound -0.046). A legitimate, date-varying signal exists on the test side: films that open
during A's D4-D10 are in test_history with their D1-D3 national tickets and shows. For film A and
target date t (visible windows only, same rule for train-sim and test):
  press_tx   = log(1 + sum of national tickets on t of other films inside their D1-D3 window) - log(A's D1-D3 mean)
  press_n    = number of such films on t
  cum_open   = log(1 + sum of D1-D3 mean tickets of films that opened after A's D3 and up to t) - log(A mean)
  big_open   = max over those films of (their D1-D3 mean / A's D1-D3 mean)
Target: film x date residual of v13 = mean over clusters of log((y + 1) / (pred + 1)).
Within-film: also correlate after removing each film's mean residual (pure trajectory shape).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, release_dates, style
from evaluate import test_weights_fd


def pressure(T, vis):
    v = vis.assign(base=base_title(vis.movie_title))
    d1 = v.groupby("base").date_show.min()
    daily = v.groupby(["date_show", "base"]).total_ticket.sum().rename("tx").reset_index()
    mean3 = v.groupby("base").total_ticket.sum() / 3
    F = T.assign(base=base_title(T.movie_title))[["base", "date_show", "d1"]].drop_duplicates()
    F["own"] = F.base.map(mean3)
    m = F.merge(daily.rename(columns={"base": "b_o"}), on="date_show", how="left")
    m = m[m.b_o != m.base]
    g = m.groupby(["base", "date_show"]).agg(press_tx=("tx", "sum"), press_n=("b_o", "nunique"))
    F = F.join(g, on=["base", "date_show"]).fillna({"press_tx": 0, "press_n": 0})
    F["press_tx"] = np.log1p(F.press_tx) - np.log1p(F.own)
    op = pd.DataFrame({"b_o": d1.index, "d1_o": d1.values, "m_o": mean3.reindex(d1.index).values})
    m = F.merge(op, how="cross")
    m = m[(m.b_o != m.base) & (m.d1_o > m.d1 + pd.Timedelta(days=2)) & (m.d1_o <= m.date_show)]
    g = m.groupby(["base", "date_show"]).agg(cum=("m_o", "sum"), big=("m_o", "max"))
    F = F.join(g, on=["base", "date_show"]).fillna({"cum": 0, "big": 0})
    F["cum_open"] = np.log1p(F.cum) - np.log1p(F.own)
    F["big_open"] = np.log1p(F.big) - np.log1p(F.own)
    return F[["base", "date_show", "press_tx", "press_n", "cum_open", "big_open"]]


def main():
    out = fig_dir("85_competition_pressure")
    D = load()
    d1 = release_dates(D["train"])
    x = D["train"].merge(d1.rename("d1"), left_on="movie_title", right_index=True)
    vis_tr = x[(x.date_show >= x.d1) & (x.date_show <= x.d1 + pd.Timedelta(days=2))].drop(columns="d1")
    vis = pd.concat([vis_tr, D["hist"]], ignore_index=True)
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    Te = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(X, Te)
    o = X[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    X["lr"] = np.log((X.total_ticket + 1) / (o.oof_final.values + 1))
    X["base"], X["w"] = base_title(X.movie_title), w
    fdres = X.groupby(["base", "date_show", "h"]).apply(lambda q: np.average(q.lr, weights=q.w), include_groups=False).rename("res").reset_index()
    P = pressure(X, vis)
    M = fdres.merge(P, on=["base", "date_show"], how="left")
    M["res_within"] = M.res - M.groupby("base").res.transform("mean")
    feats = ["press_tx", "press_n", "cum_open", "big_open"]
    for f in feats:
        M[f + "_within"] = M[f] - M.groupby("base")[f].transform("mean")
    pd.set_option("display.width", 200)
    print(f"film x date cells {len(M)}; residual sd {M.res.std():.3f}, within-film sd {M.res_within.std():.3f}")
    rows = []
    for f in feats:
        rows.append(dict(feature=f, spearman_res=M[f].corr(M.res, method="spearman"),
                         spearman_within=M[f + "_within"].corr(M.res_within, method="spearman"),
                         train_median=M[f].median()))
    PT = pressure(Te, D["hist"].pipe(lambda h: pd.concat([vis_tr, h])))
    R = pd.DataFrame(rows).set_index("feature")
    R["test_median"] = [PT[f].median() for f in feats]
    print(R.round(3).to_string())
    # by horizon: does pressure matter more late in the window?
    print("\nwithin-film spearman(cum_open, residual) by h:",
          {h: round(g.cum_open_within.corr(g.res_within, method="spearman"), 3) for h, g in M.groupby("h")})
    q = pd.qcut(M.cum_open_within.rank(method="first"), 5, labels=False)
    print("within-film residual by cum_open quintile:", M.res_within.groupby(q).mean().round(3).tolist())
    R.to_csv(out / "signal.csv"); M.to_csv(out / "film_date_residuals.csv", index=False)
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(M.res_within.groupby(q).mean().values, marker="o"); ax[0].set(title="within-film residual by cum_open quintile", xlabel="quintile")
    ax[1].hist(M.cum_open, bins=40, alpha=.5, density=True, label="train"); ax[1].hist(PT.cum_open, bins=40, alpha=.5, density=True, label="test")
    ax[1].set(title="cum_open distribution"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "pressure.png")


if __name__ == "__main__":
    main()
