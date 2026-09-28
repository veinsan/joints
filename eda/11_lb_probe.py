"""11 - Leaderboard probing: turn the public LB into a diagnostic instrument (3 submissions/day).

MASE is a plain mean of per-row terms |y - yhat| / s, so the LB score is ADDITIVE over rows.
Changing predictions only on a segment S moves the public score by exactly
    dL = (1/N_pub) * sum_{i in S, public} ( |y_i - k*yhat_i| - |y_i - yhat_i| ) / s_i
-> the sign of dL tells whether S is under- (k>1 helps) or over-predicted; three k values give a
convex curve whose argmin is the segment's best multiplier. Segments are chosen where train
could NOT teach the model (Lebaran, Ramadan, Xmas break) - exactly where the private LB is at risk.

Also: an all-zero submission scores mean(y/s) over the public rows = the test-period average
retention ratio (vs 0.61 in train-sim), a direct drift measurement with zero modelling.

Usage: python eda/11_lb_probe.py [path/to/notebook/submission.csv]
This script (optionally) trains the local LightGBM, writes base + probe CSVs to outputs/probes/, and prints
the protocol. Probe answers are fed back as Settings.SEG_MULT in the notebook.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, load, release_dates, simulate
from features import build, feature_cols, make_ctx
from model import LEVEL, NEVER, fit_predict

P = OUT / "probes"
P.mkdir(parents=True, exist_ok=True)
d = load()
tr, th, te = d["train"], d["hist"], d["test"]
first = tr.groupby("movie_title").date_show.min()
D1 = release_dates(tr).drop(first[first == first.min()].index, errors="ignore")
hs, ts = simulate(tr, D1)
Xtr = build(hs, ts, make_ctx(hs, d["hol"], d["movies"], d["price"], tr))
Xte = build(th, te, make_ctx(th, d["hol"], d["movies"], d["price"], tr))
cols = [c for c in feature_cols(Xtr) if c not in NEVER and (c not in LEVEL or c in ("log_s", "f_logT"))]
if len(sys.argv) > 1:
    # preferred: probe around the exact submission.csv produced by the Kaggle notebook
    pred = pd.read_csv(sys.argv[1]).set_index("id").total_ticket.reindex(te.id).values
else:
    pred, _ = fit_predict(Xtr, Xte, cols, True, seeds=(2026, 2027, 2028), n_estimators=800)
Xte["pred"] = pred

seg = {
    "lebaran": Xte.date_show.between("2026-03-21", "2026-03-29"),
    "ramadan": Xte.date_show.between("2026-02-19", "2026-03-20"),
    "xmas": Xte.date_show.between("2025-12-20", "2026-01-04"),
}
seg["normal"] = ~(seg["lebaran"] | seg["ramadan"] | seg["xmas"])
print("=== segments (share of test rows, mean pred/scale) ===")
for k, m in seg.items():
    print(f"  {k:8s} rows={m.sum():6d} share={m.mean():.3f}  mean r_hat={(Xte.pred / Xte.scale)[m].mean():.3f}"
          f"  median r_hat={(Xte.pred / Xte.scale)[m].median():.3f}")


def save(name, p):
    pd.DataFrame({"id": te.id, "total_ticket": np.clip(p, 0, None)}).to_csv(P / f"{name}.csv", index=False)
    return name


plan = [save("00_base", Xte.pred.values), save("01_all_zero", np.zeros(len(te)))]
for k in ["lebaran", "ramadan", "xmas"]:
    for mult in [0.7, 1.5]:
        plan.append(save(f"seg_{k}_x{mult}", np.where(seg[k], Xte.pred * mult, Xte.pred)))
plan.append(save("glob_x0.9", Xte.pred.values * 0.9))
plan.append(save("glob_x1.1", Xte.pred.values * 1.1))
print("\n=== probe files ->", P)
for n in plan:
    print("  ", n)
print("""
Protocol (3 subs/day):
 day 1: 00_base, 01_all_zero, seg_lebaran_x1.5
 day 2: seg_lebaran_x0.7 (or x2.0 if x1.5 improved), seg_ramadan_x0.7, seg_ramadan_x1.5
 day 3: seg_xmas_x0.7/x1.5, glob_x0.9 or glob_x1.1
Read-out for segment S with scores L(0.7), L(1), L(1.5): fit a parabola / piecewise line in k,
take the argmin k*, set Settings.SEG_MULT[segment] = k* in the notebook.
Guard-rails: only accept a multiplier if dL > 0.002 (above public-LB noise), and never probe the
'normal' segment beyond a global scalar - that is what CV already measures, and fitting it to the
public LB would overfit the public split.""")
