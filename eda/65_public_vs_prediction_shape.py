"""What in the *test predictions* of already-submitted versions tracks their real public score?

Uses only files that already exist (results/vN/submission.csv + reported public scores).
No new submission, no hidden labels. Question: is the public ranking explained by how much
zero/low mass each version puts on test rows (pull-shift lambda), not by model family?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import fig_dir, style

PUBLIC = {1: .45625, 2: .45061, 3: .45731, 4: .44809, 5: .40865, 6: .40119, 8: .39991, 10: .40823, 12: .41060}
# lambda applied to the non-LightGBM component, read from each builder (grep CFG.LAMBDA / LAM_CHOICE)
LAM_FM = {1: 0, 2: 0, 3: 0, 4: .5, 5: .5, 6: .5, 8: .5, 9: .5, 10: .5, 11: None, 12: 0}


def main():
    out = fig_dir("65_public_vs_shape")
    te = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    s = te.scale.to_numpy()
    leb = te.date_show.between("2026-03-21", "2026-03-27").to_numpy()
    small = s <= 20
    rows = []
    subs = {}
    for v in range(1, 13):
        p = ROOT / f"results/v{v}/submission.csv"
        if not p.exists():
            continue
        sub = te[["id"]].merge(pd.read_csv(p), on="id", how="left", validate="one_to_one")
        assert sub.total_ticket.notna().all()
        r = sub.total_ticket.to_numpy() / s
        subs[v] = r
        rows.append(dict(
            v=v, public=PUBLIC.get(v), lam_fm=LAM_FM.get(v),
            zero=np.mean(r[~leb] < 1e-9), low=np.mean(r[~leb] < 0.05),
            zero_small=np.mean(r[small & ~leb] < 1e-9), zero_h10=np.mean(r[(te.h == 10).to_numpy() & ~leb] < 1e-9),
            mean_r=r[~leb].mean(), med_r=np.median(r[~leb]), mean_r_small=r[small & ~leb].mean(),
            mean_r_leb=r[leb].mean()))
    d = pd.DataFrame(rows).set_index("v")
    pd.set_option("display.width", 200)
    print(d.round(4).to_string())
    d.to_csv(out / "shape_by_version.csv")

    k = d.dropna(subset=["public"])
    print("\nSpearman with public (n=%d):" % len(k))
    for c in ["zero", "low", "zero_small", "zero_h10", "mean_r", "med_r", "mean_r_small", "mean_r_leb"]:
        print(f"  {c:13s} {k[c].corr(k.public, method='spearman'):+.3f}")
    # same comparison restricted to the post-Lebaran-analog era (v5+): Lebaran fix dominates earlier jumps
    k5 = k[k.index >= 5]
    print("\nv5+ only (n=%d), Pearson:" % len(k5))
    for c in ["zero", "low", "zero_small", "mean_r", "mean_r_small"]:
        print(f"  {c:13s} {k5[c].corr(k5.public):+.3f}")

    # pairwise: where do v8 and v12 disagree, and in which direction
    r8, r12 = subs[8], subs[12]
    dif = r12 - r8
    seg = pd.DataFrame(dict(h=te.h, small=small, dif=dif, z8=r8 < 1e-9, z12=r12 < 1e-9))
    print("\nv12 - v8 on test (pred/scale), non-Lebaran:")
    print(f"  mean diff {dif[~leb].mean():+.4f} | v12 zero & v8>0: {np.mean((r12 < 1e-9) & (r8 > 1e-9)):.4f}"
          f" | v8 zero & v12>0: {np.mean((r8 < 1e-9) & (r12 > 1e-9)):.4f}")
    print(seg[~leb].groupby("h").agg(diff=("dif", "mean"), z8=("z8", "mean"), z12=("z12", "mean")).round(4).T.to_string())

    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for c, a, lab in [("zero", ax[0], "share of test rows predicted 0 (non-Lebaran)"),
                      ("zero_small", ax[1], "share predicted 0, scale<=20"),
                      ("mean_r", ax[2], "mean pred/scale (non-Lebaran)")]:
        a.scatter(k[c], k.public, color="#2a78d6")
        for v, row in k.iterrows():
            a.annotate(f"v{v}", (row[c], row.public), fontsize=8, xytext=(3, 3), textcoords="offset points")
        a.set_xlabel(lab); a.set_ylabel("public MASE")
    ax[0].set_title("Public vs predicted zero mass")
    ax[1].set_title("Small pairs")
    ax[2].set_title("Public vs level")
    fig.tight_layout(); fig.savefig(out / "public_vs_shape.png")

    fig, ax = plt.subplots(figsize=(7, 3.6))
    for v in [5, 6, 8, 10, 12]:
        z = pd.Series(subs[v] < 1e-9)[~leb].groupby(te.h[~leb].to_numpy()).mean()
        ax.plot(z.index, z.values, marker="o", label=f"v{v} ({PUBLIC[v]:.4f})")
    ax.set_xlabel("horizon h"); ax.set_ylabel("share predicted 0"); ax.legend(); ax.set_title("Predicted zero share by horizon")
    fig.tight_layout(); fig.savefig(out / "zero_share_by_h.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
