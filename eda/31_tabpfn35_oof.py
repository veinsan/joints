"""31 - Out-of-fold quantiles of TabPFN-3.5 per horizon, on CPU (run with .venv-tp9: tabpfn 9.0.0).

Same folds, same features and same training rows (other folds + limited releases) as the notebook, but
4 ensemble members instead of 8 to fit a laptop CPU (~4 min per fold x horizon). Results are cached per
fold so the run can resume. Scored in eda/32 against TabPFN v2 and used to choose the blend method.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion
from common import OUT, base_title

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
ckpt = hf_hub_download("Prior-Labs/tabpfn_3_5", "tabpfn-v3.5-20260909.safetensors",
                       revision="06bf2ba35c80a92a3b9abb436b99cf49e7a0365e", local_dir=str(C / "tabpfn35"))
r = lambda X: (X.total_ticket / X.scale / X.cal_mult).values
t0 = time.time()
for k in range(5):
    out_f = C / f"tab35_oofQ_fold{k}.npy"
    if out_f.exists():
        print(f"fold {k} cached")
        continue
    Xa = pd.concat([Xtr[FOLD != k], Xlim], ignore_index=True)
    Xb = Xtr[FOLD == k]
    Q = np.zeros((len(Xb), len(TAB_QS)))
    for hz in range(4, 11):
        ia, ib = (Xa.h == hz).values, (Xb.h == hz).values
        m = TabPFNRegressor.create_default_for_version(ModelVersion.V3_5, model_path=ckpt, device="cpu", n_estimators=4,
                                                       random_state=2026, ignore_pretraining_limits=True)
        m.fit(Xa[ia][FEATS].values.astype(np.float32), r(Xa)[ia])
        Q[ib] = np.asarray(m.predict(Xb[ib][FEATS].values.astype(np.float32), output_type="quantiles", quantiles=TAB_QS)).T
        print(f"fold {k} h {hz} done ({time.time() - t0:.0f}s)", flush=True)
    np.save(out_f, np.sort(Q, axis=1))
oof = np.zeros((len(Xtr), len(TAB_QS)))
for k in range(5):
    oof[FOLD == k] = np.load(C / f"tab35_oofQ_fold{k}.npy")
np.save(C / "tab35_oofQ.npy", oof)
print("saved", C / "tab35_oofQ.npy", oof.shape)
