"""14 - Small pairs (scale <= 20) are 13% of test rows but 3% of train-sim and carry the highest error.

Part A - what do small pairs look like (zero rate, retention, best constant)?
Part B - candidate fixes, each scored on the SAME wide-release rows with the same folds (TW-MASE):
  1. test-composition sample weights
  2. add limited-release films (excluded from test) to TRAINING only - real small pairs
  3. binomial thinning augmentation: copy of a train film with every ticket kept w.p. p
     (simulates the same film in a low season). The thinned pairs are compared VISUALLY with real
     small pairs before being trusted - theory says it should look alike, check that it does.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import KEY, OUT, base_title, fig_dir, load, release_dates, simulate, style
from evaluate import folds, report, test_weights
from features import build, dataset, feature_cols, make_ctx
from model import LEVEL, NEVER, PARAMS

plt = style()
F = fig_dir("14_small")
d = load()
tr = d["train"]
Xtr, Xte = dataset()
cols = [c for c in feature_cols(Xtr) if c not in NEVER and (c not in LEVEL or c in ("log_s", "f_logT"))]
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
r = Xtr.total_ticket / Xtr.scale

print("=== A. anatomy by scale bucket (train-sim) ===")
b = pd.cut(Xtr.scale, [0, 5, 20, 50, 100, 1e9])
A = pd.DataFrame({"rows": b.value_counts().sort_index(), "zero_rate": (Xtr.total_ticket == 0).groupby(b, observed=False).mean(),
                  "median_r": r.groupby(b, observed=False).median(), "mean_r": r.groupby(b, observed=False).mean(),
                  "n_hist<3": (Xtr.n_hist < 3).groupby(b, observed=False).mean()})
print(A.round(3))
zt = (Xtr.total_ticket == 0).groupby([Xtr.scale <= 20, Xtr.h]).mean().unstack(0)
zt.columns = ["s>20", "s<=20"]
print("zero rate by horizon:\n", zt.round(3).T)
Xte_small = Xte.scale <= 20
print(f"test small pairs: {Xte_small.mean():.3f} of rows; n_hist<3 among them {(Xte.n_hist[Xte_small] < 3).mean():.3f};"
      f" films covering 80% of them: {(Xte[Xte_small].movie_title.value_counts().cumsum() / Xte_small.sum() < .8).sum()}")


def fit_w(Xa, Xv):
    """LightGBM L1 on r/cal with weight cal_mult * sw (sw = extra sample weight, 1 by default)."""
    y = Xa.total_ticket / Xa.scale / Xa.cal_mult
    m = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": 500}, random_state=2026).fit(Xa[cols], y, sample_weight=Xa.cal_mult * Xa.sw)
    return np.clip(m.predict(Xv[cols]), 0, None) * Xv.cal_mult.values * Xv.scale.values


def cv(extra=None, w=None, name=""):
    """Train on other folds (+ extra rows carrying the fold of their source film), score held-out wide rows."""
    o = np.zeros(len(Xtr))
    for k in range(5):
        a, v = fold != k, fold == k
        Xa = Xtr[a].assign(sw=w[a] if w is not None else 1.0)
        if extra is not None:
            e = extra[extra.fold != k]
            Xa = pd.concat([Xa, e.assign(sw=e.sw if "sw" in e else 1.0)], ignore_index=True)
        o[v] = fit_w(Xa, Xtr[v])
    return o, report(Xtr, o, w_te, name)


print("\n=== B. fixes (same folds, same eval rows) ===")
res = {}
o0, res["baseline"] = cv(name="baseline LightGBM")
o1, res["test weights"] = cv(w=w_te, name="1. test-composition sample weights")
wsq = np.sqrt(w_te)
o1b, res["sqrt test weights"] = cv(w=wsq / wsq.mean(), name="1b. sqrt(test weights)")

# 2. limited releases
first = tr.groupby("movie_title").date_show.min()
running = first[first == first.min()].index
D_all = release_dates(tr, 0.5, 0).drop(running, errors="ignore")
D_wide = release_dates(tr).drop(running, errors="ignore")
D_lim = D_all.drop(D_wide.index, errors="ignore")
hs, ts = simulate(tr, D_lim)
L = build(hs, ts, make_ctx(hs, d["hol"], d["movies"], d["price"], tr)).assign(fold=-1)
print(f"limited-release extra: {L.movie_title.nunique()} films, {len(L)} rows, share s<=20 {(L.scale <= 20).mean():.3f}")
o2, res["+ limited releases"] = cv(extra=L, name="2. + limited-release films (train only)")

# 3. thinning augmentation
rng = np.random.default_rng(2026)
aug = []
wide_titles = Xtr.movie_title.unique()
fold_of_title = dict(zip(Xtr.movie_title, fold))
for p in (0.1, 0.25):
    t = tr[tr.movie_title.isin(wide_titles)].copy()
    t["total_ticket"] = rng.binomial(t.total_ticket.values, p)
    t["occupation_rate"] *= p
    t = t[t.total_ticket > 0]
    t["src"] = t.movie_title
    t["movie_title"] = t.movie_title + f" #thin{p}"
    D_t = pd.Series(D_wide.reindex(wide_titles).values, index=[m + f" #thin{p}" for m in wide_titles]).dropna()
    h2, t2 = simulate(t.drop(columns="src"), D_t.rename("d1"))
    X2 = build(h2, t2, make_ctx(h2, d["hol"], d["movies"], d["price"], tr))
    X2["fold"] = X2.movie_title.str.replace(r" #thin.*$", "", regex=True).map(fold_of_title)
    aug.append(X2)
T = pd.concat(aug, ignore_index=True)
Ts = T[T.scale <= 20]
Rs = Xtr[Xtr.scale <= 20]
print(f"thinned extra: {len(T)} rows, share s<=20 {(T.scale <= 20).mean():.3f}")
chk = pd.DataFrame({
    "real s<=20": [(Rs.total_ticket == 0).mean(), (Rs.total_ticket / Rs.scale).median(), (Rs.n_hist < 3).mean(), Rs.p3.median()],
    "thinned s<=20": [(Ts.total_ticket == 0).mean(), (Ts.total_ticket / Ts.scale).median(), (Ts.n_hist < 3).mean(), Ts.p3.median()],
    "test s<=20": [np.nan, np.nan, (Xte[Xte_small].n_hist < 3).mean(), Xte[Xte_small].p3.median()]},
    index=["zero rate D4-10", "median r", "share n_hist<3", "median p3"])
print("PREPRO CHECK thinned vs real small pairs:\n", chk.round(3))
o3, res["+ thinning"] = cv(extra=T, name="3. + thinning augmentation (p=0.1,0.25)")
o4, res["weights + limited"] = cv(extra=L, w=w_te, name="1+2. test weights + limited releases")

fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
ax[0].plot(zt.index, zt["s<=20"], marker="o", label="real s<=20")
ax[0].plot(zt.index, zt["s>20"], marker="o", label="real s>20")
zt2 = (Ts.total_ticket == 0).groupby(Ts.h).mean()
ax[0].plot(zt2.index, zt2.values, marker="s", ls="--", label="thinned s<=20")
ax[0].set(title="Zero rate by horizon", xlabel="D"); ax[0].legend()
bins = np.linspace(0, 3, 40)
ax[1].hist(Rs.p3.clip(upper=3), bins, alpha=.5, density=True, label="real s<=20")
ax[1].hist(Ts.p3.clip(upper=3), bins, alpha=.5, density=True, label="thinned s<=20")
ax[1].hist(Xte[Xte_small].p3.clip(upper=3), bins, alpha=.5, density=True, label="test s<=20")
ax[1].set(title="p3 = y3/s for small pairs"); ax[1].legend()
ax[2].barh(list(res)[::-1], list(res.values())[::-1], color="#2a78d6")
ax[2].set(title="TW-MASE (lower is better)", xlim=(min(res.values()) - .01, max(res.values()) + .005))
fig.savefig(F / "small_pairs.png")
print(f"figures -> {F}")
