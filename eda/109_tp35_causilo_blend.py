"""Is a TabPFN-3.5 + Causilo blend worth packing both models under 200 MB? Saved v15 OOF only.

Compression is now allowed as long as the submitted weights file is <= 200 MB. TabPFN-3.5 int7 takes
169 MB, so Causilo (148 MB float32) only fits if it is compressed too. Before spending GPU time, this
measures on the v15 cohort-fold OOF (identical rows, folds and weights for every component):
  1. median blends  w_tp * tp35 + w_ca * causilo_h + w_lgb * lgb  (the v15 blend form);
  2. quantile-function blend: average the 49 quantiles of tp35 and causilo_h, then the mixture median
     (a distribution-level ensemble rather than an average of medians);
  3. paired bootstrap over release weeks (the fold unit) and over films for the best candidate vs v15.
Temporal predictions of v15 were not saved, so only the cohort lens is available here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, style
from evaluate import test_weights_fd

TAB_QS = np.round(np.arange(0.02, 1.0, 0.02), 2)
PULL_A, ZERO_EPS = -1.32, 0.02


def quant_median(Q, lam):
    p0 = np.array([np.interp(ZERO_EPS, q, TAB_QS, left=0.0, right=1.0) for q in Q])
    lg = np.log(np.clip(p0, 1e-4, 1 - 1e-4) / (1 - np.clip(p0, 1e-4, 1 - 1e-4)))
    p = 1 / (1 + np.exp(-(lg + lam * PULL_A)))
    u = np.clip(p0 + (1 - p0) * (0.5 - p) / (1 - p), TAB_QS[0], TAB_QS[-1])
    val = np.array([np.interp(ui, TAB_QS, qi) for ui, qi in zip(u, Q)])
    return np.where(p >= 0.5, 0.0, np.clip(val, 0, None))


def main():
    out = fig_dir("109_tp35_causilo")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v15/oof.csv"), on=KEY + ["h"], how="left", validate="one_to_one")
    y, s = x.total_ticket.values, x.scale.values
    assert np.allclose(o.y, y) and np.allclose(o.scale, s)
    cm = x.cal_mult.values
    Qn = np.load(ROOT / "results/v15/oof_quantiles.npz")
    tw = lambda p: np.average(np.abs(y - p) / s, weights=w)
    comps = {n: o[f"comp_{n}"].values for n in ["lgb", "tp35_int7", "causilo_h", "fast35", "tabm"]}
    v15 = o.oof_final.values
    print(f"v15 final TW {tw(v15):.5f} | " + " | ".join(f"{n} {tw(p):.5f}" for n, p in comps.items()))

    rows = []
    for a in np.round(np.arange(0, 1.01, .1), 1):
        for b in np.round(np.arange(0, 1.01 - a, .1), 1):
            c = round(1 - a - b, 1)
            p = a * comps["tp35_int7"] + b * comps["causilo_h"] + c * comps["lgb"]
            rows.append(dict(w_tp35=a, w_causilo=b, w_lgb=c, TW=tw(p), MASE=np.mean(np.abs(y - p) / s)))
    B = pd.DataFrame(rows).sort_values("TW")
    pd.set_option("display.width", 200)
    print("\nmedian blends (top 10):"); print(B.head(10).round(5).to_string(index=False))
    B.to_csv(out / "median_blends.csv", index=False)

    qres = []
    for a in (1.0, 0.8, 0.7, 0.6, 0.5):
        Q = a * Qn["tp35_int7_Q"] + (1 - a) * Qn["causilo_h_Q"]
        for lam in (0.0, 0.5):
            p = quant_median(np.sort(Q, axis=1), lam) * cm * s
            qres.append(dict(w_tp35=a, lam=lam, TW=tw(p), MASE=np.mean(np.abs(y - p) / s)))
    QB = pd.DataFrame(qres).sort_values("TW")
    print("\nquantile-function blends tp35 + causilo_h:"); print(QB.round(5).to_string(index=False))
    QB.to_csv(out / "quantile_blends.csv", index=False)

    best = B.iloc[0]
    pb = best.w_tp35 * comps["tp35_int7"] + best.w_causilo * comps["causilo_h"] + best.w_lgb * comps["lgb"]
    d = (np.abs(y - pb) - np.abs(y - v15)) / s * w
    rng = np.random.default_rng(2026)
    res = {}
    for unit, g in [("release week", o.week.values), ("film", base_title(x.movie_title).values)]:
        G = pd.DataFrame({"g": g, "d": d, "w": w}).groupby("g").sum()
        ix = rng.integers(0, len(G), (10000, len(G)))
        b_ = G.d.values[ix].sum(1) / G.w.values[ix].sum(1)
        res[unit] = dict(delta=d.sum() / w.sum(), ci_low=np.quantile(b_, .025), ci_high=np.quantile(b_, .975),
                         units_better=f"{int((G.d < 0).sum())}/{len(G)}")
    R = pd.DataFrame(res).T
    print(f"\nbest median blend {dict(best[['w_tp35', 'w_causilo', 'w_lgb']])} vs v15 final:"); print(R.round(5).to_string())
    print("per fold TW:", {k: (round(np.average((np.abs(y - pb) / s)[o.fold == k], weights=w[o.fold == k]), 4),
                             round(np.average((np.abs(y - v15) / s)[o.fold == k], weights=w[o.fold == k]), 4)) for k in range(5)})
    R.to_csv(out / "bootstrap.csv")
    plt = style()
    piv = B.pivot_table(index="w_tp35", columns="w_causilo", values="TW")
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(piv.values, origin="lower", cmap="Blues_r")
    ax.set_xticks(range(piv.shape[1]), piv.columns); ax.set_yticks(range(piv.shape[0]), piv.index)
    for (i, j), v_ in np.ndenumerate(piv.values):
        if np.isfinite(v_):
            ax.text(j, i, f"{v_:.4f}", ha="center", va="center", fontsize=6)
    ax.set(xlabel="w causilo_h", ylabel="w tp35_int7 (rest LightGBM)", title=f"Cohort TW of median blends (v15 final {tw(v15):.4f})"); ax.grid(False)
    fig.tight_layout(); fig.savefig(out / "blend_grid.png")


if __name__ == "__main__":
    main()
