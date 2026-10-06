"""32 - TabPFN-3.5 vs TabPFN v2, and median blend vs distribution blend, on the LB-calibrated metric.

Inputs: eda/31 OOF quantiles of TabPFN-3.5 (4 members, CPU), the v5 Kaggle OOF (to recover the TabPFN v2
component: oof_final = 0.7 * lgbmix + 0.3 * tab_v2), and freshly fitted LightGBM hurdle parts with the
eda/30 settings (classifier 31 leaves / 300 trees / min_child 100 on BASE + cinema competition).
Metric: TW-MASE in the real world and in the kappa = 0.5 world (the one that matched v3, v4, v5 on the LB).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from evaluate import folds, test_weights
from features import dataset

plt = style()
F = fig_dir("32_tabpfn35")
C = OUT / "cache"
Xtr, Xte, Xlim = dataset()
BASE = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
        'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
        'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
        'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
        'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
ZC = BASE + ["c_new_n", "c_new_sh", "c_new_sh_rel", "c_new_sh_vs_own", "c_new_tx_vs_own"]
QS = np.round(np.arange(0.05, 1.0, 0.05), 2)
TQS = np.round(np.arange(0.02, 1.0, 0.02), 2)
A_SHIFT, LAM, HW, EPS = -1.32, 0.5, 0.75, 0.02
fold = folds(Xtr)
w_te = test_weights(Xtr, Xte)
y, s, cm = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values
rr = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
sig = lambda t: 1 / (1 + np.exp(-t))
lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
bp = lambda **k: dict(learning_rate=0.05, feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, verbose=-1,
                      deterministic=True, force_row_wise=True, random_state=2026, **k)

cache = C / "lgb_parts_v6.npz"
if cache.exists():
    z = np.load(cache); l1, p0, Q = z["l1"], z["p0"], z["Q"]
else:
    l1, p0, Q = np.zeros(len(Xtr)), np.zeros(len(Xtr)), np.zeros((len(Xtr), len(QS)))
    for k in range(5):
        Xa, v = pd.concat([Xtr[fold != k], Xlim], ignore_index=True), fold == k
        l1[v] = np.clip(lgb.LGBMRegressor(objective="l1", **bp(num_leaves=63, n_estimators=800, min_child_samples=100, ))
                        .set_params(learning_rate=0.03).fit(Xa[BASE], rr(Xa), sample_weight=Xa.cal_mult).predict(Xtr[v][BASE]), 0, None)
        p0[v] = lgb.LGBMClassifier(**bp(num_leaves=31, n_estimators=300, min_child_samples=100)).fit(Xa[ZC], Xa.total_ticket == 0).predict_proba(Xtr[v][ZC])[:, 1]
        pos = Xa[Xa.total_ticket > 0]
        Q[v] = np.sort(np.column_stack([lgb.LGBMRegressor(objective="quantile", alpha=float(q), **bp(num_leaves=63, n_estimators=300, min_child_samples=100))
                                        .fit(pos[BASE], rr(pos)).predict(Xtr[v][BASE]) for q in QS]), axis=1)
        print(f"lgb fold {k} done", flush=True)
    np.savez(cache, l1=l1, p0=p0, Q=Q)

z0 = np.load(C / "hurdle_oof.npz")
worlds = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    ps = sig(lg(z0["p0"]) + 0.5 * A_SHIFT)
    un = (y == 0) & (rng.random(len(y)) < 1 - ps / np.clip(z0["p0"], 1e-6, None))
    dr = np.array([np.interp(u, QS, q) for u, q in zip(rng.random(len(y)), z0["Q"])])
    worlds.append(np.where(un, np.clip(dr, 0, None) * cm * s, y))


def sc(pred):
    return (float(np.average(np.abs(y - pred) / s, weights=w_te)),
            float(np.mean([np.average(np.abs(w - pred) / s, weights=w_te) for w in worlds])))


def mixmed(p, Qm, qs, lam):
    pp = sig(lg(p) + lam * A_SHIFT)
    t = np.clip((0.5 - pp) / (1 - pp + 1e-9), qs[0], qs[-1])
    return np.where(pp >= 0.5, 0.0, np.clip([np.interp(ti, qs, qi) for ti, qi in zip(t, Qm)], 0, None))


def tab_parts(TQ):
    tp0 = np.array([np.interp(EPS, q, TQS, left=0.0, right=1.0) for q in TQ])
    u = tp0[:, None] + (1 - tp0[:, None]) * QS[None, :]
    return tp0, np.clip(np.array([np.interp(ui, TQS, qi) for ui, qi in zip(u, TQ)]), 0, None)


def tab_median(TQ, lam):
    tp0, tq = tab_parts(TQ)
    return mixmed(tp0, tq, QS, lam)


TQ = np.load(C / "tab35_oofQ.npy")
v5 = pd.read_csv(OUT.parent / "results/v5/oof.csv")
tab_v2 = (v5.oof_final.values - 0.7 * v5.oof_lgbmix.values) / 0.3
lgbmix = ((1 - HW) * l1 + HW * mixmed(p0, Q, QS, LAM)) * cm * s
rows = {"LightGBM mix (v6 settings)": sc(lgbmix),
        "TabPFN v2 median lambda 0.5 (from v5)": sc(tab_v2),
        "TabPFN-3.5 median lambda 0": sc(tab_median(TQ, 0.0) * cm * s),
        "TabPFN-3.5 median lambda 0.5": sc(tab_median(TQ, LAM) * cm * s),
        "v5 final (Kaggle OOF)": sc(v5.oof_final.values)}
tp0, tq = tab_parts(TQ)
ws = np.linspace(0, 1, 11)
ca, cb = [], []
for w in ws:
    ca.append(sc((1 - w) * lgbmix + w * tab_median(TQ, LAM) * cm * s))
    cb.append(sc(((1 - HW) * l1 + HW * mixmed((1 - w) * p0 + w * tp0, (1 - w) * Q + w * tq, QS, LAM)) * cm * s))
ia, ib = int(np.argmin([c[1] for c in ca])), int(np.argmin([c[1] for c in cb]))
rows[f"median blend, w_tab={ws[ia]:.1f}"] = ca[ia]
rows[f"distribution blend, w_tab={ws[ib]:.1f}"] = cb[ib]
R = pd.DataFrame(rows, index=["TW_real", "TW_kappa0.5"]).T
print(R.round(4))
print("median-blend curve (kappa):", dict(zip(ws.round(1), [round(c[1], 4) for c in ca])))
print("distribution-blend curve (kappa):", dict(zip(ws.round(1), [round(c[1], 4) for c in cb])))
p0t = np.array([np.interp(EPS, q, TQS, left=0, right=1) for q in TQ])
print(f"TabPFN-3.5 implied P(zero): mean {p0t.mean():.3f} vs observed zero rate {np.mean(y == 0):.3f} | LightGBM p0 mean {p0.mean():.3f}")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
ax[0].barh(R.index, R["TW_kappa0.5"], color="#2a78d6"); ax[0].set(title="TW-MASE, kappa=0.5 world (LB-calibrated)", xlim=(R["TW_kappa0.5"].min() - .005, R["TW_kappa0.5"].max() + .003))
ax[1].plot(ws, [c[1] for c in ca], marker="o", label="median blend"); ax[1].plot(ws, [c[1] for c in cb], marker="s", label="distribution blend")
ax[1].set(title="TW-MASE kappa=0.5 vs weight on TabPFN-3.5", xlabel="w_tab"); ax[1].legend()
fig.savefig(F / "tabpfn35.png")
print(f"figures -> {F}")
