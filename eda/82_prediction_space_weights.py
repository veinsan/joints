"""A validation weight that reproduces public scores: test composition in prediction space.

eda/81: if test labels follow v13's OOF calibration cell by cell (h x first-sale day x scale bucket x
predicted-r bin), the expected non-Lebaran MASE of submitted versions is already 0.386-0.397, about
the public level. The old TW weights only match scale x first-sale day. Here:
 1. physically constrained level check: public_v = 0.939 E_v + 0.061 L, L = Lebaran-row MASE, must
    be one plausible constant across versions (per world a, m);
 2. PTW weights for OOF rows = test share / OOF share of the same cell (cells from v13 predictions,
    a function of features, so the same definition works for every version);
 3. does PTW-MASE rank the submitted versions like the public (TW did not)?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eda"))
from common import KEY, fig_dir, style
from evaluate import test_weights_fd

m81 = __import__("81_public_score_inference")
PUBLIC = {5: .40865, 6: .40119, 8: .39991, 10: .40823, 12: .41060, 13: .40244}
LEB_SHARE = 4417 / 72611


def main():
    out = fig_dir("82_prediction_space_weights")
    g = pd.read_csv(ROOT / "outputs/eda/81_public_inference/grid.csv")
    vs = list(PUBLIC)
    pub = np.array([PUBLIC[v] for v in vs])
    rows = []
    for _, r in g.iterrows():
        e = np.array([r[f"E_v{v}"] for v in vs])
        L = (pub - (1 - LEB_SHARE) * e) / LEB_SHARE
        rows.append(dict(a=r.a, m=r.m, L_mean=L.mean(), L_sd=L.std(), E_v13=r.E_v13))
    W = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("Implied Lebaran-row MASE per world (must be one constant in a plausible range, roughly 0.3-1.2):")
    sel = W[W.a.isin([-1.0, -0.5, 0.0, 0.5]) & W.m.round(2).isin([0.8, 0.9, 1.0, 1.1, 1.2, 1.4])]
    print(sel.pivot(index="a", columns="m", values="L_mean").round(2).to_string())
    print("\nspread (sd across versions) of the implied L:")
    print(sel.pivot(index="a", columns="m", values="L_sd").round(3).to_string())
    W.to_csv(out / "implied_lebaran.csv", index=False)

    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    o13 = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    r_oof = o13.oof_final.values / x.scale.values
    edges = np.unique(np.quantile(r_oof, np.linspace(0, 1, 13)))
    leb = t.date_show.between("2026-03-21", "2026-03-27").values
    s13 = t[["id"]].merge(pd.read_csv(ROOT / "results/v13/submission.csv"), on="id", how="left").total_ticket.values
    kx = m81.cells(x, r_oof, edges)
    kt = m81.cells(t[~leb].reset_index(drop=True), s13[~leb] / t.scale.values[~leb], edges)
    ratio = kt.value_counts(normalize=True) / kx.value_counts(normalize=True)
    ptw = kx.map(ratio).fillna(0).values
    ptw = ptw / ptw.mean()
    tw = test_weights_fd(x, t)
    y, s = x.total_ticket.values, x.scale.values
    print(f"\nPTW coverage: test rows in cells with OOF support {kt.isin(kx.unique()).mean():.4f}; ESS {ptw.sum()**2/(ptw**2).sum():.0f} of {len(ptw)}")
    res = []
    for v in vs:
        oo = x[KEY + ["h"]].merge(pd.read_csv(ROOT / f"results/v{v}/oof.csv"), on=KEY + ["h"], how="left")
        p = oo.oof_final.values
        if np.isnan(p).any():
            print(f"v{v}: OOF incomplete, skipped"); continue
        e = np.abs(y - p) / s
        res.append(dict(version=f"v{v}", public=PUBLIC[v], TW=np.average(e, weights=tw), PTW=np.average(e, weights=ptw)))
    R = pd.DataFrame(res).set_index("version")
    R["PTW_plus_lebaran"] = (1 - LEB_SHARE) * R.PTW + LEB_SHARE * 0.6
    print(R.round(5).to_string())
    for c in ["TW", "PTW"]:
        print(f"  {c}: spearman with public {R[c].corr(R.public, method='spearman'):+.2f}, pearson {R[c].corr(R.public):+.2f}")
    R.to_csv(out / "versions.csv")
    np.save(out / "ptw_weights.npy", ptw)
    # what does PTW emphasise? weight share by segment vs TW
    seg = pd.DataFrame({"y_zero": y == 0, "h45": x.h.values <= 5, "s<=50": s <= 50,
                        "pred_r<0.3": r_oof < .3, "growth": y > x.y3.values})
    print("\nWeight share by segment (TW vs PTW):")
    print(pd.DataFrame({"TW": [tw[seg[c]].sum() / tw.sum() for c in seg], "PTW": [ptw[seg[c]].sum() / ptw.sum() for c in seg]},
                       index=seg.columns).round(3).to_string())
    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for a_, c in zip(ax, ["TW", "PTW"]):
        a_.scatter(R[c], R.public)
        for v, row in R.iterrows():
            a_.annotate(v, (row[c], row.public), fontsize=8)
        a_.set(xlabel=f"{c}-MASE (OOF)", ylabel="public", title=f"{c}: spearman {R[c].corr(R.public, method='spearman'):+.2f}")
    fig.tight_layout(); fig.savefig(out / "ptw_vs_public.png")


if __name__ == "__main__":
    main()
