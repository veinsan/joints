"""Do the submitted models extrapolate in absolute size on test rows?

Compare each version's prediction on test rows with its OOF prediction on train rows *in the same cell*
of (h x cluster-relative level x pair shape p3). If the model reasons in relative terms, the
calendar-adjusted prediction r_hat = pred / (scale * cal_mult) should be similar for test and train
rows of the same cell regardless of absolute size. If it reasons in absolute terms, test rows with
small absolute scale (quiet market) get lower r_hat than train rows of the same relative level.
Uses saved predictions only.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eda"))
from common import fig_dir, load, release_dates, simulate, style

rel_mod = __import__("69_relative_level_target")


def main():
    out = fig_dir("71_model_abs_extrapolation")
    D = load()
    hist, _ = simulate(D["train"], release_dates(D["train"]))
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    T = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    X["rel"], _ = rel_mod.rel_clu(X, rel_mod.pair_scales(hist))
    T["rel"], _ = rel_mod.rel_clu(T, rel_mod.pair_scales(D["hist"]))
    leb = T.date_show.between("2026-03-21", "2026-03-27")
    T = T[~leb & np.isfinite(T.rel)].copy()
    X = X[np.isfinite(X.rel)].copy()
    er = np.quantile(X.rel, np.linspace(0, 1, 6))
    ep = [-1, .3, .7, 1.0, 1.3, 9]
    for d in (X, T):
        d["cell"] = d.h.astype(str) + "|" + np.clip(np.searchsorted(er, d.rel, side="right") - 1, 0, 4).astype(str) + "|" \
            + pd.cut(d.p3, ep, labels=False).astype(str)
        d["bk"] = pd.cut(d.scale, [0, 5, 20, 50, 200, 1e9]).astype(str)
    rows = []
    for v in (8, 12):
        o = X[["movie_title", "cinema_ids", "h"]].merge(pd.read_csv(ROOT / f"results/v{v}/oof.csv"), on=["movie_title", "cinema_ids", "h"], how="left")
        X[f"rh{v}"] = o.oof_final.values / X.scale.values / X.cal_mult.values
        s = T[["id"]].merge(pd.read_csv(ROOT / f"results/v{v}/submission.csv"), on="id", how="left")
        T[f"rh{v}"] = s.total_ticket.values / T.scale.values / T.cal_mult.values
        ref = X.groupby("cell")[f"rh{v}"].mean()
        T[f"gap{v}"] = T[f"rh{v}"] - T.cell.map(ref)
        X[f"gapin{v}"] = X[f"rh{v}"] - X.cell.map(ref)
    X["y_r"] = X.total_ticket / X.scale / X.cal_mult
    refy = X.groupby("cell").y_r.mean()
    X["ygap"] = X.y_r - X.cell.map(refy)
    tab = pd.DataFrame({
        "test_share": T.bk.value_counts(normalize=True),
        "v8 test gap": T.groupby("bk").gap8.mean(), "v12 test gap": T.groupby("bk").gap12.mean(),
        "v8 train gap": X.groupby("bk").gapin8.mean(), "train LABEL gap": X.groupby("bk").ygap.mean(),
    }).round(3)
    pd.set_option("display.width", 200)
    print("Mean(r_hat - cell mean of OOF r_hat), cell = h x rel quintile x p3 bin, by absolute scale bucket")
    print("(train LABEL gap = same statistic for the true r: how much absolute size matters inside train at fixed cell)")
    print(tab.to_string())
    tab.to_csv(out / "gap_by_bucket.csv")

    plt = style()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    order = ["(0.0, 5.0]", "(5.0, 20.0]", "(20.0, 50.0]", "(50.0, 200.0]", "(200.0, 1000000000.0]"]
    x = np.arange(len(order))
    for i, (c, lab) in enumerate([("v8 test gap", "v8 test rows"), ("v12 test gap", "v12 test rows"),
                                  ("v8 train gap", "v8 OOF rows"), ("train LABEL gap", "true r, train rows")]):
        ax.bar(x + (i - 1.5) * .2, tab.loc[order, c], width=.2, label=lab)
    ax.set_xticks(x, ["<=5", "5-20", "20-50", "50-200", ">200"])
    ax.axhline(0, color="#8a8984", lw=.8)
    ax.set_ylabel("mean gap vs same relative cell"); ax.set_xlabel("absolute scale bucket")
    ax.set_title("Models treat small absolute size as weakness at fixed relative level"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "gap_by_bucket.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
