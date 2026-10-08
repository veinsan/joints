"""EXP 056: lookup OOF median r per bucket (film, h, pasangan) tanpa hindsight.
Menjawab: berapa struktur bucket pasangan yang benar2 generalisasi antar fold?
Lookup fold-safe = median dari 4 fold lain per sel bucket, diterapkan ke fold val.
Bucket film: kohort (desil ukuran nasional) x d1_dow; bucket pasangan: days_present x scale_decile.
python temp/exp_056_lookup_oof/lookup.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

w = pd.read_parquet("temp/exp_046_fe_decay/output/train_feat.parquet")
w["r"] = w.y / w.scale
w["scale_b"] = pd.qcut(w.scale, 8, labels=False, duplicates="drop")
w["occ_b"] = pd.qcut(w.occ13, 5, labels=False, duplicates="drop")

def oof_lookup(keys, shrink_parent=None):
    oof = np.zeros(len(w))
    for k in range(5):
        tr, va = w.fold.values != k, w.fold.values == k
        med_tr = w[tr].groupby(keys).r.median()
        default = w[tr].r.median()
        if shrink_parent:
            par = w[tr].groupby(shrink_parent).r.median()
        key_va = w.loc[va, keys].copy() if isinstance(keys, list) else pd.DataFrame({0: w.loc[va, keys]})
        pred = key_va.apply(lambda row: med_tr.get(tuple(row.values) if isinstance(keys, list) else row.values[0], np.nan), axis=1)
        if shrink_parent:
            pk = w.loc[va, shrink_parent].copy()
            ppred = pk.apply(lambda row: par.get(tuple(row.values) if isinstance(shrink_parent, list) else row.values[0], default), axis=1)
            pred = pred.fillna(ppred)
        else:
            pred = pred.fillna(default)
        oof[va] = pred.values
    return float(np.mean(np.abs(w.r.values - oof)))


print("lookup OOF (h, days_present):", round(oof_lookup(["h", "days_present"]), 4))
print("lookup OOF (h, days_present, scale_b):", round(oof_lookup(["h", "days_present", "scale_b"]), 4))
print("lookup OOF (cohort, d1_dow, h, days_present, scale_b):", round(oof_lookup(["cohort", "d1_dow", "h", "days_present", "scale_b"]), 4))
print("lookup OOF (cohort, d1_dow, h, days_present, scale_b, occ_b):", round(oof_lookup(["cohort", "d1_dow", "h", "days_present", "scale_b", "occ_b"]), 4))
