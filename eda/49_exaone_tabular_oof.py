"""49 - EXAONE-Tabular regressor (LG AI Research, 84.5 MB, <= 200 MB) out-of-fold quantiles, run with .venv-tp9.

The competition caps model weights at 200 MB; TabPFN-3.5 (876 MB) and TabPFN-3 (233 MB) exceed it. EXAONE-Tabular
(`LG-AI-Research/EXAONE-Tabular`, revision 093ad1c2, regressor-v1_default) is the strongest compliant candidate
(TabArena Elo above TabPFN-3 in its report). Same folds, features, per-horizon contexts (other folds + limited
releases) and target r = y / s / cal_mult as eda/31; 4 ensemble members, float32 on CPU. Quantiles via
src/exaone_q.py (the package itself returns only a point estimate). Cached per fold.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, base_title
from exaone_q import EXAONEQuantileRegressor

C = OUT / "cache"
Xtr, Xlim = pd.read_parquet(C / "Xtr.parquet"), pd.read_parquet(C / "Xlim.parquet")
FEATS = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
         'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
         'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
         'f_share_cohort', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
         'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
TAB_QS = list(np.round(np.arange(0.02, 1.0, 0.02), 2))
g = base_title(Xtr.movie_title).values
ug = np.array(sorted(set(g)))
fold_of = dict(zip(ug, np.random.RandomState(2026).permutation(len(ug)) % 5))
FOLD = np.array([fold_of[x] for x in g])
r = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
t0 = time.time()
for k in range(5):
    out_f = C / f"exaone_oofQ_fold{k}.npy"
    if out_f.exists():
        print(f"fold {k} cached"); continue
    Xa = pd.concat([Xtr[FOLD != k], Xlim], ignore_index=True)
    Xb = Xtr[FOLD == k]
    Q = np.zeros((len(Xb), len(TAB_QS)))
    for hz in range(4, 11):
        ia, ib = (Xa.h == hz).values, (Xb.h == hz).values
        m = EXAONEQuantileRegressor.from_pretrained(device="cpu", ensemble_count=4, compute_dtype="float32", seed=2026,
                                                    revision="093ad1c2613e1e95d26df8610451d2e02ed53cbe")
        m.fit(Xa[ia][FEATS].values.astype(np.float32), r(Xa)[ia])
        Q[ib] = m.predict_quantiles(Xb[ib][FEATS].values.astype(np.float32), TAB_QS)
        print(f"fold {k} h {hz} done ({time.time() - t0:.0f}s)", flush=True)
    np.save(out_f, Q)
full = np.zeros((len(Xtr), len(TAB_QS)))
for k in range(5):
    full[FOLD == k] = np.load(C / f"exaone_oofQ_fold{k}.npy")
np.save(C / "exaone_oofQ.npy", full)
print("saved", full.shape)
