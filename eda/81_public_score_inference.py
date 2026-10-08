"""Which test-label world explains the public scores of versions that were actually submitted?

No new submission. Public scores of submitted notebook versions are data (allowed, like eda/39).
Label model for every non-Lebaran test row: the OOF calibration of v13 (distribution of true r = y/s
given the cell h x first-sale day x scale bucket x bin of predicted r), then deformed by two global
parameters:
    a : logit shift of the zero mass        (a < 0 -> fewer zeros in the test period)
    m : multiplier of the positive part     (m > 1 -> test films sell more relative to D1-D3)
Expected MASE of each version's submission on non-Lebaran rows is computed under each (a, m).
Lebaran rows are identical for v5+ and only add a constant, so the fit uses v5..v13 only.
public_v ~ c + E_v(a, m); the free constant c absorbs Lebaran and the public-subset level.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, fig_dir, style

PUBLIC = {5: .40865, 6: .40119, 8: .39991, 10: .40823, 12: .41060, 13: .40244}
U = (np.arange(64) + .5) / 64
SB = [0, 5, 20, 50, 200, 1e9]


def cells(d, r_hat, edges):
    fd = np.where(d.y1 > 0, 1, np.where(d.y2 > 0, 2, 3)).clip(1, 2)
    sb = np.searchsorted(SB, d.scale.values, side="left").clip(1, 5)
    pb = np.clip(np.searchsorted(edges, r_hat, side="right") - 1, 0, len(edges) - 2)
    return pd.Series(d.h.astype(str).values + "|" + fd.astype(str) + "|" + sb.astype(str) + "|" + pb.astype(str))


def main():
    out = fig_dir("81_public_inference")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    r_true, r_oof = x.total_ticket.values / x.scale.values, o.oof_final.values / x.scale.values
    edges = np.unique(np.quantile(r_oof, np.linspace(0, 1, 13)))
    kx = cells(x, r_oof, edges)
    # per cell: zero mass and positive-part quantiles (pooled fallback by dropping the pred bin)
    grp = pd.DataFrame({"k": kx, "k2": kx.str.rsplit("|", n=1).str[0], "r": r_true})
    def summarize(g):
        pos = g.r[g.r > 0].values
        return pd.Series({"p0": (g.r == 0).mean(), "n": len(g), "q": np.quantile(pos, U) if len(pos) >= 5 else None})
    C = grp.groupby("k").apply(summarize, include_groups=False)
    C2 = grp.groupby("k2").apply(summarize, include_groups=False)
    C3 = grp.assign(h=grp.k.str[:2].str.rstrip("|")).groupby("h").apply(summarize, include_groups=False)

    leb = t.date_show.between("2026-03-21", "2026-03-27").values
    T = t[~leb].reset_index(drop=True)
    subs = {}
    for v in PUBLIC:
        s = t[["id"]].merge(pd.read_csv(ROOT / f"results/v{v}/submission.csv"), on="id", how="left")
        subs[v] = s.total_ticket.values[~leb] / T.scale.values
    kt = cells(T, subs[13], edges)
    p0 = np.empty(len(T)); Q = np.empty((len(T), len(U)))
    for i, (k, k2) in enumerate(zip(kt, kt.str.rsplit("|", n=1).str[0])):
        ok = lambda D, kk: kk in D.index and isinstance(D.loc[kk].q, np.ndarray) and D.loc[kk].n >= 30
        row = C.loc[k] if ok(C, k) else C2.loc[k2] if ok(C2, k2) else C3.loc[k.split("|")[0]]
        p0[i], Q[i] = row.p0, row.q
    print(f"label model built: test rows {len(T)}, mean zero mass {p0.mean():.3f}, mean positive median {np.median(Q[:, 32]):.3f}")

    def expected(a, m):
        pz = 1 / (1 + np.exp(-(np.log(np.clip(p0, 1e-4, 1 - 1e-4) / (1 - np.clip(p0, 1e-4, 1 - 1e-4))) + a)))
        # label quantile function on grid U: zero below pz, scaled positive quantiles above
        uu = (U[None, :] - pz[:, None]) / (1 - pz[:, None])
        idx = np.clip((uu * len(U)).astype(int), 0, len(U) - 1)
        lab = np.where(U[None, :] < pz[:, None], 0.0, np.take_along_axis(Q, idx, 1) * m)
        return {v: float(np.abs(lab - p[:, None]).mean()) for v, p in subs.items()}

    vs = list(PUBLIC)
    pub = np.array([PUBLIC[v] for v in vs])
    rows = []
    for a in np.arange(-3, 1.01, .25):
        for m in np.arange(.7, 1.81, .05):
            E = expected(a, m)
            e = np.array([E[v] for v in vs])
            c = np.mean(pub - e)
            rmse = np.sqrt(np.mean((pub - c - e) ** 2))
            rows.append(dict(a=round(a, 2), m=round(m, 2), c=c, rmse=rmse, zero_mass=None,
                             spearman=pd.Series(e).corr(pd.Series(pub), method="spearman"), **{f"E_v{v}": E[v] for v in vs}))
    R = pd.DataFrame(rows)
    R.to_csv(out / "grid.csv", index=False)
    pd.set_option("display.width", 220)
    base = R[(R.a == 0) & (np.isclose(R.m, 1.0))].iloc[0]
    print("\nIn-distribution world (a=0, m=1): expected non-Lebaran MASE per version vs public")
    print(pd.DataFrame({"expected": [base[f'E_v{v}'] for v in vs], "public": pub}, index=[f"v{v}" for v in vs]).round(4).to_string())
    print(f"  rmse after constant {base.rmse:.5f}, spearman {base.spearman:+.2f}, constant {base.c:+.4f}")
    best = R.sort_values("rmse").head(12)
    print("\nBest-fitting worlds:"); print(best[["a", "m", "c", "rmse", "spearman"]].round(5).to_string(index=False))
    b = best.iloc[0]
    print("\nBest world: expected vs public"); print(pd.DataFrame({"expected": [b[f'E_v{v}'] for v in vs], "public": pub,
          "fitted": [b.c + b[f'E_v{v}'] for v in vs]}, index=[f"v{v}" for v in vs]).round(4).to_string())
    # what the best world says about a simple global rescale of v13 (sanity, not a submission)
    plt = style()
    piv = R.pivot(index="a", columns="m", values="rmse")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    im = ax[0].imshow(piv.values, aspect="auto", origin="lower", cmap="Blues_r",
                      extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()])
    ax[0].set(xlabel="m (positive-part multiplier)", ylabel="a (zero logit shift)", title="RMSE of public fit (lower = better)")
    ax[0].grid(False); fig.colorbar(im, ax=ax[0])
    ax[1].scatter([base[f'E_v{v}'] + base.c for v in vs], pub, label="a=0, m=1")
    ax[1].scatter([b[f'E_v{v}'] + b.c for v in vs], pub, label=f"best a={b.a}, m={b.m}")
    for v, xx, yy in zip(vs, [b[f'E_v{v}'] + b.c for v in vs], pub):
        ax[1].annotate(f"v{v}", (xx, yy), fontsize=8)
    lo, hi = pub.min() - .003, pub.max() + .003
    ax[1].plot([lo, hi], [lo, hi], color="#8a8984", ls="--"); ax[1].set(xlabel="fitted", ylabel="public"); ax[1].legend()
    fig.tight_layout(); fig.savefig(out / "public_inference.png")


if __name__ == "__main__":
    main()
