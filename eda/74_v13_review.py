"""v13 review against v8/v10/v12 on identical OOF rows and weights; test-prediction shape. Saved files only."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, style
from evaluate import test_weights_fd

PUBLIC = {8: .39991, 10: .40823, 12: .41060, 13: .40244}
VERS = [8, 10, 12, 13]


def main():
    out = fig_dir("74_v13_review")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    y, s = x.total_ticket.to_numpy(), x.scale.to_numpy()
    films = base_title(x.movie_title).to_numpy()
    keys = KEY + ["h"]
    P, S = {}, {}
    for v in VERS:
        o = x[keys].merge(pd.read_csv(ROOT / f"results/v{v}/oof.csv"), on=keys, how="left", validate="one_to_one")
        assert o.oof_final.notna().all() and np.allclose(o.y, y) and np.allclose(o.scale, s)
        P[v] = o.oof_final.to_numpy()
        if v == 13:
            comps = {c: o[c].to_numpy() for c in o if c.startswith(("comp_", "oof_b1", "oof_r1"))}
            quiet = o.quiet_film.to_numpy().astype(bool)
        sub = t[["id"]].merge(pd.read_csv(ROOT / f"results/v{v}/submission.csv"), on="id", how="left", validate="one_to_one")
        assert sub.total_ticket.notna().all() and sub.total_ticket.ge(0).all()
        S[v] = sub.total_ticket.to_numpy()

    err = {v: np.abs(y - p) / s for v, p in P.items()}
    rows = [dict(version=f"v{v}", TW=np.average(e, weights=w), MASE=e.mean(),
                 quiet_rows_TW=np.average(e[quiet], weights=w[quiet]), public=PUBLIC.get(v)) for v, e in err.items()]
    for c, p in comps.items():
        e = np.abs(y - p) / s
        rows.append(dict(version=f"v13 {c}", TW=np.average(e, weights=w), MASE=e.mean(), quiet_rows_TW=np.average(e[quiet], weights=w[quiet])))
    tab = pd.DataFrame(rows).set_index("version")
    pd.set_option("display.width", 200)
    print("Identical OOF rows and bfd weights (quiet_rows_TW = grouped-CV OOF on quiet films, NOT the quiet-lens refit):")
    print(tab.round(5).to_string())
    tab.to_csv(out / "scores.csv")

    rng = np.random.default_rng(2026)
    boot = {}
    for ref in (8, 10, 12):
        d = (err[13] - err[ref]) * w
        g = pd.DataFrame({"f": films, "d": d, "w": w}).groupby("f").sum()
        ix = rng.integers(0, len(g), (10000, len(g)))
        b = g.d.values[ix].sum(1) / g.w.values[ix].sum(1)
        boot[f"v13-v{ref}"] = dict(delta=float(d.sum() / w.sum()), ci95=np.quantile(b, [.025, .975]).round(5).tolist(),
                                    films_better=f"{int((g.d < 0).sum())}/{len(g)}",
                                    signed_err_corr=float(np.corrcoef((y - P[13]) / s, (y - P[ref]) / s)[0, 1]))
    print("\nPaired film bootstrap (10k):"); print(json.dumps(boot, indent=1))

    masks = {"zero": y == 0, "positive": y > 0, "growth_vs_D3": y > x.y3.to_numpy(),
             "positive_no_growth": (y > 0) & (y <= x.y3.to_numpy()), "D4": x.h.to_numpy() == 4, "D4-D5": x.h.to_numpy() <= 5}
    seg = pd.DataFrame({v: {k: np.average(err[v][m], weights=w[m]) for k, m in masks.items()} for v in VERS})
    seg["weight_share"] = {k: w[m].sum() / w.sum() for k, m in masks.items()}
    print("\nSegment TW-MASE (within segment):"); print(seg.round(5).to_string())
    seg.to_csv(out / "segments.csv")

    leb = t.date_show.between("2026-03-21", "2026-03-27").to_numpy()
    sm = t.scale.to_numpy() <= 20
    ts = t.scale.to_numpy()
    shape = pd.DataFrame({f"v{v}": dict(zero=np.mean(S[v][~leb] / ts[~leb] < 1e-9), mean_r=np.mean(S[v][~leb] / ts[~leb]),
                                        mean_r_small=np.mean(S[v][sm & ~leb] / ts[sm & ~leb]),
                                        mean_abs_change_vs_v8=np.mean(np.abs(S[v] - S[8]) / ts),
                                        lebaran_identical_to_v12=bool(np.allclose(S[v][leb], S[12][leb])))
                          for v in VERS}).T
    print("\nTest prediction shape:"); print(shape.to_string())
    by_h = pd.DataFrame({f"v{v}": pd.Series(S[v][~leb] / ts[~leb]).groupby(t.h.to_numpy()[~leb]).mean() for v in VERS})
    print("\nMean test pred/scale by horizon (non-Lebaran):"); print(by_h.round(4).T.to_string())
    shape.to_csv(out / "test_shape.csv")
    (out / "bootstrap.json").write_text(json.dumps(boot, indent=1))

    plt = style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    seg.loc[["zero", "positive", "growth_vs_D3", "D4"], VERS].plot.bar(ax=ax[0], rot=0)
    ax[0].set(title="Within-segment TW-MASE, identical rows", ylabel="MASE")
    by_h.plot(ax=ax[1], marker="o"); ax[1].set(title="Test mean pred/scale by horizon", xlabel="h")
    fig.tight_layout(); fig.savefig(out / "v13_review.png")
    print(f"\nfigures -> {out}")


if __name__ == "__main__":
    main()
