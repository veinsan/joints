"""35 - TabICL v2 (jingang/TabICL, tabicl-regressor-v2-20260212) out-of-fold quantiles, run with .venv-tp9.

Second tabular foundation model for the ensemble: a different architecture/prior from TabPFN, so errors
should be less correlated. ~6x faster than TabPFN-3.5 on CPU, so it can also take the POOLED context
(all horizons, ~45k rows, horizon as a feature) instead of 7 per-horizon models; both are scored.
Same folds, features and training rows (other folds + limited releases) as the notebook.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tabicl import TabICLRegressor
from common import OUT, base_title

C = OUT / "cache"
Xtr, Xlim = pd.read_parquet(C / "Xtr.parquet"), pd.read_parquet(C / "Xlim.parquet")
FEATS = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
         'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating', 'g_Horror', 'g_Drama', 'g_Action', 'g_Comedy',
         'g_Romance', 'g_Animation', 'g_Family', 'g_Thriller', 'g_Mystery', 'n_genre', 'n_cast', 'comp_n',
         'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult', 'hol_in_hist', 'n_comp_open',
         'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new', 'price_wkd', 'price_prem']
QS = list(np.round(np.arange(0.02, 1.0, 0.02), 2))
g = base_title(Xtr.movie_title).values
ug = np.array(sorted(set(g)))
fold_of = dict(zip(ug, np.random.RandomState(2026).permutation(len(ug)) % 5))
FOLD = np.array([fold_of[x] for x in g])
ck = hf_hub_download("jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt", revision="4dcd344ece2c00be9e831fdd35bed57b5ad83e19")
r = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
mk = lambda: TabICLRegressor(n_estimators=4, model_path=ck, device="cpu", random_state=2026)
t0 = time.time()

out = C / "tabicl_oofQ.npy"
if not out.exists():
    oof = np.zeros((len(Xtr), len(QS)))
    for k in range(5):
        Xa, Xb = pd.concat([Xtr[FOLD != k], Xlim], ignore_index=True), Xtr[FOLD == k]
        Q = np.zeros((len(Xb), len(QS)))
        for hz in range(4, 11):
            ia, ib = (Xa.h == hz).values, (Xb.h == hz).values
            m = mk().fit(Xa[ia][FEATS].values.astype(np.float32), r(Xa)[ia])
            Q[ib] = np.asarray(m.predict(Xb[ib][FEATS].values.astype(np.float32), output_type="quantiles", alphas=QS))
        oof[FOLD == k] = np.sort(Q, axis=1)
        print(f"per-horizon fold {k} done ({time.time() - t0:.0f}s)", flush=True)
    np.save(out, oof)

# pooled-context variant on fold 0 only (cost check + quality signal)
k = 0
Xa, Xb = pd.concat([Xtr[FOLD != k], Xlim], ignore_index=True), Xtr[FOLD == k]
t1 = time.time()
m = mk().fit(Xa[FEATS].values.astype(np.float32), r(Xa))
Qp = np.sort(np.asarray(m.predict(Xb[FEATS].values.astype(np.float32), output_type="quantiles", alphas=QS)), axis=1)
np.save(C / "tabicl_pooled_fold0Q.npy", Qp)
print(f"pooled fold 0: context {len(Xa)} rows, {time.time() - t1:.0f}s")
