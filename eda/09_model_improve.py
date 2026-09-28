"""09 - Improvement experiments, each judged on the SAME held-out films (true-D1 samples only).

A. CV noise floor (fold-shuffle repeats)      B. D1-jitter augmentation (+/-1 day anchors)
C. out-of-fold cinema target encoding         D. CatBoost / XGBoost (MAE) vs LightGBM, and blend
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
import catboost as cb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from common import OUT, fig_dir, load, release_dates, simulate, style
from features import build, feature_cols, make_ctx
from model import LEVEL, NEVER, fit_predict, groups, target

plt = style()
F = fig_dir("09_improve")
d = load()
tr = d["train"]
first = tr.groupby("movie_title").date_show.min()
D1 = release_dates(tr).drop(first[first == first.min()].index, errors="ignore")


def sim(shift):
    hs, ts = simulate(tr, D1 + pd.Timedelta(days=shift))
    X = build(hs, ts, make_ctx(hs, d["hol"], d["movies"], d["price"], tr))
    return X.assign(aug=shift)


X = sim(0)
AUG = pd.concat([sim(-1), sim(1)], ignore_index=True)
g, ga = groups(X), groups(AUG)
cols = [c for c in feature_cols(X) if c not in NEVER + ["aug"] and (c not in LEVEL or c in ("log_s", "f_logT"))]
M = lambda p, idx: float(np.mean(np.abs(X.total_ticket.values[idx] - p[idx]) / X.scale.values[idx]))
ALL = np.arange(len(X))
N = 600


def cv(fn, seed=0):
    ug = np.array(sorted(g.unique()))
    rng = np.random.RandomState(seed)
    fold_of = dict(zip(ug, rng.permutation(len(ug)) % 5))
    f = g.map(fold_of).values
    o = np.zeros(len(X))
    for k in range(5):
        a, b = np.where(f != k)[0], np.where(f == k)[0]
        o[b] = fn(a, b, set(ug[[fold_of[u] != k for u in ug]]))
    return o


def lgb_fn(extra=None, use_aug=False, cols_=None):
    def fn(a, b, train_groups):
        Xa = X.iloc[a]
        if use_aug:
            Xa = pd.concat([Xa, AUG[ga.isin(train_groups).values]], ignore_index=True)
        return fit_predict(Xa, X.iloc[b], cols_ or cols, True, n_estimators=N)[0]
    return fn


print("=== A. noise floor: LightGBM r/cal, 3 different fold assignments ===")
base = [M(cv(lgb_fn(), s), ALL) for s in range(3)]
print("  MASE:", np.round(base, 4), " std:", round(np.std(base), 4))
res = {"lgb base": np.mean(base)}

print("\n=== B. D1 jitter augmentation (train on D1-1, D1, D1+1 of train films) ===")
aug = [M(cv(lgb_fn(use_aug=True), s), ALL) for s in range(3)]
print("  MASE:", np.round(aug, 4), " mean diff vs base:", round(np.mean(aug) - np.mean(base), 4))
res["lgb + D1 jitter"] = np.mean(aug)

print("\n=== C. OOF cinema target encoding (mean residual r/cal vs film median) ===")


def te_fn(a, b, train_groups):
    Xa, Xb = X.iloc[a].copy(), X.iloc[b].copy()
    rc = target(Xa)
    resid = rc - rc.groupby([Xa.movie_title, Xa.h]).transform("median")
    # inner OOF on the training part so the encoding is not fitted on its own rows
    inner = np.zeros(len(Xa))
    ig = groups(Xa).values
    for ia, ib in GroupKFold(5).split(Xa, groups=ig):
        mm = resid.iloc[ia].groupby(Xa.cinema_ids.iloc[ia]).agg(["sum", "count"])
        enc = mm["sum"] / (mm["count"] + 200)
        inner[ib] = Xa.cinema_ids.iloc[ib].map(enc).fillna(0).values
    mm = resid.groupby(Xa.cinema_ids).agg(["sum", "count"])
    Xa["cin_te"], Xb["cin_te"] = inner, Xb.cinema_ids.map(mm["sum"] / (mm["count"] + 200)).fillna(0).values
    return fit_predict(Xa, Xb, cols + ["cin_te"], True, n_estimators=N)[0]


te = [M(cv(te_fn, s), ALL) for s in range(3)]
print("  MASE:", np.round(te, 4), " mean diff vs base:", round(np.mean(te) - np.mean(base), 4))
res["lgb + cinema TE"] = np.mean(te)

print("\n=== D. other GBDTs (MAE objective, same target & weights) ===")


def cb_fn(a, b, _):
    Xa, Xb = X.iloc[a], X.iloc[b]
    m = cb.CatBoostRegressor(loss_function="MAE", iterations=1500, learning_rate=0.06, depth=7, random_seed=2026,
                             verbose=0, thread_count=-1)
    m.fit(Xa[cols], target(Xa), sample_weight=Xa.cal_mult)
    return np.clip(m.predict(Xb[cols]), 0, None) * Xb.scale.values * Xb.cal_mult.values


def xgb_fn(a, b, _):
    Xa, Xb = X.iloc[a], X.iloc[b]
    m = xgb.XGBRegressor(objective="reg:absoluteerror", n_estimators=800, learning_rate=0.03, max_depth=7,
                         subsample=0.8, colsample_bytree=0.7, min_child_weight=20, random_state=2026, n_jobs=-1)
    m.fit(Xa[cols], target(Xa), sample_weight=Xa.cal_mult)
    return np.clip(m.predict(Xb[cols]), 0, None) * Xb.scale.values * Xb.cal_mult.values


o_l, o_c, o_x = cv(lgb_fn(), 0), cv(cb_fn, 0), cv(xgb_fn, 0)
for n, o in [("lightgbm", o_l), ("catboost", o_c), ("xgboost", o_x), ("blend mean", (o_l + o_c + o_x) / 3),
             ("blend median", np.median([o_l, o_c, o_x], 0))]:
    print(f"  {n:14s} {M(o, ALL):.4f}")
    res[n] = M(o, ALL)
print("  OOF corr lgb/cb/xgb:", np.round(np.corrcoef([o_l / X.scale, o_c / X.scale, o_x / X.scale]), 3).tolist())
pd.DataFrame({"lgb": o_l, "cb": o_c, "xgb": o_x}).to_parquet(OUT / "oof_models.parquet")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.barh(list(res)[::-1], list(res.values())[::-1], color="#2a78d6")
ax.axvline(res["lgb base"], color="#e34948", lw=1)
ax.set(title="GroupKFold MASE (lower is better)", xlim=(min(res.values()) - .01, max(res.values()) + .01))
fig.savefig(F / "improve.png")
print(f"figures -> {F}")
