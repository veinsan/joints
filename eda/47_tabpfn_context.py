"""47 - Does TabPFN-3.5 gain from a larger context? per-horizon (~6.5k rows) vs pooled horizons (~45k, h as feature).

v6 fits one TabPFN-3.5 per horizon, so each sees ~1/7 of the training rows. In-context learners usually keep
improving with more context, and neighbouring horizons share structure. Fold 0, horizons 4 and 8, 2 ensemble
members (CPU budget), identical queries; also a 'neighbour' context (h-1, h, h+1). Scored by the median MASE of
r on those rows (real) and the same in the kappa world. Run with .venv-tp9.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from common import OUT, base_title

C = OUT / "cache"
Xtr, Xlim = pd.read_parquet(C / "Xtr.parquet"), pd.read_parquet(C / "Xlim.parquet")
FEATS = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
         'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
         'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
         'f_share_cohort', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
         'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem', 'h']
TAB_QS = list(np.round(np.arange(0.02, 1.0, 0.02), 2))
g = base_title(Xtr.movie_title).values
ug = np.array(sorted(set(g)))
fold_of = dict(zip(ug, np.random.RandomState(2026).permutation(len(ug)) % 5))
FOLD = np.array([fold_of[x] for x in g])
ckpt = str(C / "tabpfn35" / "tabpfn-v3.5-20260909.safetensors")
r = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
Xa = pd.concat([Xtr[FOLD != 0], Xlim], ignore_index=True)
out = {}
for hz in (4, 8):
    Xb = Xtr[(FOLD == 0) & (Xtr.h == hz)]
    for name, ctx in (("per-horizon", Xa.h == hz), ("neighbour h-1..h+1", Xa.h.between(hz - 1, hz + 1)), ("pooled all h", Xa.h > 0)):
        t0 = time.time()
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, model_path=ckpt, device="cpu", n_estimators=2,
                                                       random_state=2026, ignore_pretraining_limits=True)
        m.fit(Xa[ctx.values][FEATS].values.astype(np.float32), r(Xa)[ctx.values])
        Q = np.sort(np.asarray(m.predict(Xb[FEATS].values.astype(np.float32), output_type="quantiles", quantiles=TAB_QS)).T, axis=1)
        out[(hz, name)] = Q
        np.save(C / f"tabctx_h{hz}_{name.split()[0]}.npy", Q)
        print(f"h={hz} {name:20s} ctx {int(ctx.sum()):6d} rows, {time.time() - t0:5.0f}s", flush=True)
print("done")
