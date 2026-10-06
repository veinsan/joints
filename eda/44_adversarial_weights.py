"""44 - Adversarial validation: does a full covariate re-weighting of the train-sim explain the LB offset?

Composition-correct weights (bucket x first day) put every version ~0.065 below its LB, while the test-period
proxy shows the test period is not harder at D3 (eda/43). Either the test differs from the train-sim in other
covariates (film size, cluster mix, shape of D1-D3 ...) or the relation y|x shifts at D4-D10 (concept shift).
A train-vs-test classifier on pair/film shape features (no dates, no season/market features) gives density-ratio
weights p/(1-p); the kappa-world TW under those weights, compared with the LB of v3-v6, tells which one it is.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from common import OUT, base_title, fig_dir, style
from evaluate import kappa_worlds, test_weights, test_weights_fd
from features import dataset

plt = style()
F = fig_dir("44_adversarial")
Xtr, Xte, Xlim = dataset()
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
z0 = np.load(OUT / "cache" / "hurdle_oof.npz")
SHAPE = ["log_s", "p1", "p2", "p3", "n_hist", "first_day", "sh1", "sh2", "sh3", "occ1", "occ2", "occ3", "sh_trend", "tps3",
         "f_logT", "fnc1", "fnc3", "fp1", "fp3", "f_nc_trend", "f_per_cin", "f_occ", "share", "p3_rel", "cin_size", "cin_nfilms",
         "fmt", "fmt_share", "rating", "d1_dow", "h"]
Z = pd.concat([Xtr[SHAPE].assign(t=0, g=base_title(Xtr.movie_title).values), Xte[SHAPE].assign(t=1, g=base_title(Xte.movie_title).values)],
              ignore_index=True)
ug = np.array(sorted(Z.g.unique()))
fo = dict(zip(ug, np.random.RandomState(7).permutation(len(ug)) % 5))
f = Z.g.map(fo).values
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, verbose=-1, random_state=2026,
         deterministic=True, force_row_wise=True)
pr = np.zeros(len(Z))
for k in range(5):
    m = lgb.LGBMClassifier(**P).fit(Z[SHAPE][f != k], Z.t[f != k])
    pr[f == k] = m.predict_proba(Z[SHAPE][f == k])[:, 1]
print(f"adversarial AUC (film-grouped CV): {roc_auc_score(Z.t, pr):.3f}")
imp = pd.Series(lgb.LGBMClassifier(**P).fit(Z[SHAPE], Z.t).booster_.feature_importance("gain"), index=SHAPE).sort_values(ascending=False)
print("top shift features (gain share):", (imp / imp.sum()).head(8).round(3).to_dict())
ptr = np.clip(pr[: len(Xtr)], 0.02, 0.98)
w_adv = ptr / (1 - ptr)
w_adv = np.clip(w_adv, 0, np.quantile(w_adv, 0.99))
w_adv /= w_adv.mean()
print(f"effective sample size {w_adv.sum() ** 2 / (w_adv ** 2).sum():.0f} of {len(w_adv)}")

LB = {3: 0.45731, 4: 0.44809, 5: 0.40865, 6: 0.40119}
Wk = {k: kappa_worlds(y, s, cm, z0["p0"], z0["Q"], kappa=k) for k in (0.0, 0.5, 1.0)}
ws = {"bucket": test_weights(Xtr, Xte), "bucket x fd": test_weights_fd(Xtr, Xte), "adversarial": w_adv}
rows = []
for v, lb in LB.items():
    p = pd.read_csv(Path(__file__).resolve().parents[1] / f"results/v{v}/oof.csv").oof_final.values
    for wn, w in ws.items():
        for k, W in Wk.items():
            rows.append(dict(v=v, LB=lb, weights=wn, kappa=k, tw=np.mean([np.average(np.abs(t - p) / s, weights=w) for t in W])))
R = pd.DataFrame(rows)
R["offset"] = R.LB - R.tw
print("\n=== LB - validation (v5, v6 have the Lebaran fix; v3, v4 carry ~+0.038 Lebaran error) ===")
print(R.pivot_table(index=["weights", "kappa"], columns="v", values="offset").round(4))
print("\n=== LB change vs validation change v5 -> v6 and v3 -> v4 ===")
for wn in ws:
    for k in Wk:
        t = R[(R.weights == wn) & (R.kappa == k)].set_index("v").tw
        print(f"  {wn:12s} kappa {k}: v3->v4 {t[4] - t[3]:+.4f} (LB -0.0092) | v5->v6 {t[6] - t[5]:+.4f} (LB -0.0075)")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].hist(pr[: len(Xtr)], bins=50, alpha=.6, density=True, label="train-sim"); ax[0].hist(pr[len(Xtr):], bins=50, alpha=.6, density=True, label="test")
ax[0].set(title="P(test | shape features)"); ax[0].legend()
ax[1].barh(imp.index[:10][::-1], (imp / imp.sum()).values[:10][::-1], color="#2a78d6"); ax[1].set(title="adversarial importance")
fig.savefig(F / "adversarial.png")
np.save(OUT / "cache" / "w_adv.npy", w_adv)
print(f"figures -> {F}")
