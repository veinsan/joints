"""Does training on raw + thinned worlds remove the test-period pull shift (no lambda needed)?

eda/68: absolute tickets/show lookup gives a test-period D3-stop logit offset of -1.35; a
cluster-relative yardstick -0.21 to -0.44. Here the SAME absolute lookup (tickets/show deciles x D1
weekday) is built from raw + thinned train-sim proxies (pi 0.35 / 0.5 / 0.7). If the quiet market is
the whole story, the test offset should be near zero and the curves should overlap.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eda"))
from common import fig_dir, load, release_dates, simulate, style

m68 = __import__("68_pull_absolute_vs_relative")
m89 = __import__("89_thinned_features_prepro")
logit = lambda p: np.log(p / (1 - p))
sig = lambda t: 1 / (1 + np.exp(-t))


def main():
    out = fig_dir("90_pull_after_thinning")
    D = load()
    d1 = release_dates(D["train"])
    hist, _ = simulate(D["train"], d1)
    d1s = d1[d1.index.isin(hist.movie_title.unique())]
    raw = m68.stops(D["train"], d1s)
    worlds = [m68.stops(m89.thin_tx(D["train"], pi, s), d1s) for pi, s in [(0.35, 1), (0.5, 2), (0.7, 3)]]
    T = m68.stops(D["hist"], D["hist"].groupby("movie_title").date_show.min())
    res, curves = {}, {}
    for name, A in [("raw", raw), ("raw + thinned", pd.concat([raw] + worlds, ignore_index=True)),
                    ("thinned only", pd.concat(worlds, ignore_index=True))]:
        e = np.quantile(np.log(A.tps), np.linspace(0, 1, 11))
        kb = lambda d: np.clip(np.searchsorted(e, np.log(d.tps), side="right") - 1, 0, 9).astype(str) + "|" + np.minimum(d.d1.dt.dayofweek, 4).astype(str)
        look = A.z.groupby(kb(A)).mean()
        pe = np.clip(pd.Series(kb(T)).map(look).fillna(A.z.mean()).values, 1e-3, 1 - 1e-3)
        off = brentq(lambda b: sig(logit(pe) + b).sum() - T.z.sum(), -8, 8)
        ll = -np.mean(T.z * np.log(pe) + (1 - T.z) * np.log(1 - pe))
        res[name] = dict(train_pairs=len(A), train_stop=A.z.mean(), test_expected=pe.mean(), test_observed=T.z.mean(),
                         test_logit_offset=off, test_logloss=ll)
        dec = np.clip(np.searchsorted(e, np.log(T.tps), side="right") - 1, 0, 9)
        curves[name] = (A.z.groupby(np.clip(np.searchsorted(e, np.log(A.tps), side="right") - 1, 0, 9)).mean(), T.z.groupby(dec).mean())
    R = pd.DataFrame(res).T
    pd.set_option("display.width", 200)
    print(R.round(4).to_string())
    R.to_csv(out / "offsets.csv")
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for a_, (n, (tr_c, te_c)) in zip(ax, curves.items()):
        a_.plot(tr_c.index, tr_c.values, marker="o", label="train world(s)"); a_.plot(te_c.index, te_c.values, marker="o", label="test period")
        a_.set(title=f"{n}: offset {res[n]['test_logit_offset']:+.2f}", xlabel="tickets/show decile (of training set)", ylabel="P(no sale D3)")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out / "pull_after_thinning.png")


if __name__ == "__main__":
    main()
