"""16 - Grouped feature ablation on TW-MASE (2 fold assignments to see through fold noise).

Each group is added on top of the v1 base set; a group is kept only if it helps on BOTH fold seeds.
Then the kept set is scored with limited-release extra training rows, and against the
test-period proxy direction from eda/13.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from evaluate import folds, report, test_weights
from features import dataset, feature_cols
from model import LEVEL, NEVER, PARAMS

plt = style()
F = fig_dir("16_ablation")
Xtr, Xte, Xlim = dataset()
w_te = test_weights(Xtr, Xte)
GROUPS = {
    "programme change": ["n_wed_passed", "n_thu_passed"],
    "first day / show trend": ["first_day", "sh_r31", "f_sh_r31"],
    "market relative": ["occ_rel", "tps_rel", "logT_rel", "mk_n"],
    "cinema competition": ["c_new_n", "c_new_sh", "c_new_sh_rel", "c_new_sh_vs_own", "c_new_tx_vs_own"],
}
ABS = ["mk_occ", "mk_tps", "mk_logT", "c_new_tx"]
new = sum(GROUPS.values(), [])
base = [c for c in feature_cols(Xtr) if c not in NEVER + new + ABS and (c not in LEVEL or c in ("log_s", "f_logT"))]


def cv(cs, seed, extra=None):
    fold = folds(Xtr, seed=seed)
    o = np.zeros(len(Xtr))
    for k in range(5):
        Xa = Xtr[fold != k]
        if extra is not None:
            Xa = pd.concat([Xa, extra], ignore_index=True)
        m = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": 500}, random_state=2026).fit(
            Xa[cs], Xa.total_ticket / Xa.scale / Xa.cal_mult, sample_weight=Xa.cal_mult)
        v = fold == k
        o[v] = np.clip(m.predict(Xtr[v][cs]), 0, None) * Xtr.cal_mult.values[v] * Xtr.scale.values[v]
    return o


res = {}
for seed in (2026, 7):
    print(f"--- fold seed {seed} ---")
    res[("base", seed)] = report(Xtr, cv(base, seed), w_te, "base (v1 feature set)")
    for n, gcols in GROUPS.items():
        res[(n, seed)] = report(Xtr, cv(base + gcols, seed), w_te, f"+ {n}")
tab = pd.Series(res).unstack()
delta = tab.sub(tab.loc["base"], axis=1)
print("\nTW-MASE delta vs base (negative = better):\n", delta.round(4))
keep = [n for n in GROUPS if (delta.loc[n] < 0).all()]
print("kept groups (help on both fold seeds):", keep)
final = base + sum((GROUPS[n] for n in keep), [])
out = {}
for seed in (2026, 7):
    out[("kept", seed)] = report(Xtr, cv(final, seed), w_te, f"kept groups, seed {seed}")
    o = cv(final, seed, extra=Xlim)
    out[("kept + limited", seed)] = report(Xtr, o, w_te, f"kept groups + limited releases, seed {seed}")
print(pd.Series(out).unstack().round(4))
pd.Series(final).to_csv(OUT / "cache" / "features_v3.csv", index=False)
np.save(OUT / "cache" / "oof_v3_lgb.npy", o)

fig, ax = plt.subplots(figsize=(7, 3.6))
m = delta.mean(axis=1).drop("base")
ax.barh(m.index, m.values, color=["#1baf7a" if v < 0 else "#e34948" for v in m.values])
ax.axvline(0, color="k", lw=.6)
ax.set(title="TW-MASE change when adding a feature group (mean of 2 fold seeds)", xlabel="delta (negative = better)")
fig.savefig(F / "ablation.png")
print(f"figures -> {F}")
