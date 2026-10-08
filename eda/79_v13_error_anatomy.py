"""Where does v13's remaining test-weighted error live, and how much is reachable?

Contribution of each cell to TW-MASE (v13 OOF, bfd weights) and the gain if that cell's rows were
predicted by an in-sample per-cell oracle median of r (upper bound of what better conditioning on
that cell could buy). Saved predictions only.
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
    out = fig_dir("79_error_anatomy")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    y, s, p = x.total_ticket.values, x.scale.values, o.oof_final.values
    e = np.abs(y - p) / s
    tot = np.average(e, weights=w)
    x["n_hist_pat"] = (x.y1 > 0).astype(int).astype(str) + (x.y2 > 0).astype(int).astype(str) + "1"
    x["sb"] = pd.cut(x.scale, [0, 5, 20, 50, 200, 1e9]).astype(str)
    x["sh3b"] = pd.cut(x.sh3, [-1, 1, 2, 4, 8, 1e9]).astype(str)
    x["grow"] = np.where(y == 0, "zero", np.where(y > x.y3, "grow>y3", "pos<=y3"))
    x["e"], x["w"] = e, w
    print(f"v13 OOF TW-MASE {tot:.4f}")
    pd.set_option("display.width", 200)
    for col in ["n_hist_pat", "sb", "h", "sh3b", "grow", "d1_dow"]:
        g = x.groupby(col).apply(lambda q: pd.Series(dict(
            w_share=q.w.sum() / w.sum(), tw_mase=np.average(q.e, weights=q.w),
            contribution=(q.e * q.w).sum() / w.sum(),
            mean_signed=np.average((p[q.index] - y[q.index]) / s[q.index], weights=q.w))), include_groups=False)
        g["contrib_share"] = g.contribution / tot
        print(f"\n--- by {col}"); print(g.round(4).to_string())
        g.to_csv(out / f"by_{col}.csv")
    # heavy tail: share of total weighted error from the top 1% / 5% rows
    c = np.sort(e * w)[::-1]
    print(f"\nTop 1% rows carry {c[:len(c)//100].sum()/c.sum():.1%}, top 5% {c[:len(c)//20].sum()/c.sum():.1%} of weighted error")
    # error by test-share of late starters x scale
    k = x.groupby(["n_hist_pat", "sb"]).apply(lambda q: pd.Series(dict(w_share=q.w.sum()/w.sum(), contribution=(q.e*q.w).sum()/w.sum(),
                                                                    tw=np.average(q.e, weights=q.w), n=len(q))), include_groups=False)
    print("\n--- history pattern x scale bucket"); print(k.sort_values("contribution", ascending=False).head(12).round(4).to_string())
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    g = pd.read_csv(out / "by_sb.csv", index_col=0)
    ax[0].bar(g.index, g.contribution); ax[0].set(title=f"Contribution to TW-MASE {tot:.3f} by scale", ylabel="contribution")
    ax[0].tick_params(axis="x", rotation=30)
    g = pd.read_csv(out / "by_h.csv", index_col=0)
    ax[1].bar(g.index.astype(str), g.tw_mase); ax[1].set(title="TW-MASE by horizon", xlabel="h")
    fig.tight_layout(); fig.savefig(out / "anatomy.png")


if __name__ == "__main__":
    main()
