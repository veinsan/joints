"""19 - Segments the train cannot validate (Lebaran, Ramadan, Xmas break = 30% of test rows).

1. What do v1/v2 predict there (level, predicted-zero share)?
2. Stakes: if the true level in a segment were k x what the model assumes, how much MASE is lost?
   Simulated on OOF rows (true y scaled by k, prediction unchanged), then weighted by the segment's
   share of test rows -> shows which segment can explain the ~0.07 CV->LB residual.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, fig_dir, style
from features import dataset

plt = style()
F = fig_dir("19_segments")
Xtr, Xte, _ = dataset()
Xte = Xte.sort_values("id").reset_index(drop=True)
SEG = {"lebaran": ("2026-03-21", "2026-03-29"), "ramadan": ("2026-02-19", "2026-03-20"), "xmas": ("2025-12-20", "2026-01-04")}
seg = pd.Series("normal", index=Xte.index)
for k, (a, b) in SEG.items():
    seg[Xte.date_show.between(a, b)] = k
subs = {v: pd.read_csv(OUT.parent / f"results/{v}/submission.csv").sort_values("id").total_ticket.values for v in ("v1", "v2")}

print("=== 1. predictions per segment ===")
rows = []
for s_ in ["normal", "xmas", "ramadan", "lebaran"]:
    m = (seg == s_).values
    r = {v: p[m] / Xte.scale.values[m] for v, p in subs.items()}
    rows.append(dict(segment=s_, rows=m.sum(), share=m.mean(), small_share=(Xte.scale.values[m] <= 20).mean(),
                     v1_mean_r=r["v1"].mean(), v2_mean_r=r["v2"].mean(), v2_pred_zero=(r["v2"] < .05).mean()))
T = pd.DataFrame(rows).set_index("segment")
print(T.round(3))

print("\n=== 2. stakes: MASE lost if the true segment level is k x the modelled level ===")
oof = np.load(OUT / "cache" / "oof_lgb.npy") if (OUT / "cache" / "oof_lgb.npy").exists() else None
if oof is None or len(oof) != len(Xtr):
    from sklearn.model_selection import GroupKFold
    from features import feature_cols
    from model import LEVEL, NEVER, fit_predict, groups
    cols = [c for c in feature_cols(Xtr) if c not in NEVER and (c not in LEVEL or c in ("log_s", "f_logT"))]
    oof = np.zeros(len(Xtr))
    for a, b in GroupKFold(5).split(Xtr, groups=groups(Xtr)):
        oof[b] = fit_predict(Xtr.iloc[a], Xtr.iloc[b], cols, True, n_estimators=500)[0]
y, s = Xtr.total_ticket.values, Xtr.scale.values
base = np.mean(np.abs(y - oof) / s)
ks = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
stake = {}
for k in ks:
    # pretend reality is k x (Binomial/Poisson-free scaling keeps zeros at zero: pulls do not change)
    e_k = np.mean(np.abs(k * y - oof) / s)
    e_fix = np.mean(np.abs(k * y - k * oof) / s)
    stake[k] = dict(uncorrected=e_k, corrected=e_fix, loss=e_k - e_fix)
S = pd.DataFrame(stake).T
print(S.round(4))
print("impact on overall test MASE (loss x segment share):")
imp = pd.DataFrame({s_: S.loss * T.loc[s_, "share"] for s_ in ["xmas", "ramadan", "lebaran"]})
print(imp.round(4))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
ax[0].bar(T.index, T.share, color="#2a78d6"); ax[0].set(title="Share of test rows per segment")
for s_ in imp.columns:
    ax[1].plot(imp.index, imp[s_], marker="o", label=s_)
ax[1].axhline(0.07, color="#e34948", ls="--", lw=1, label="unexplained CV->LB gap")
ax[1].set(title="MASE lost if true level = k x model", xlabel="k"); ax[1].legend()
fig.savefig(F / "segments.png")
print(f"figures -> {F}")
