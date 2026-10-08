"""EXP 054: oracle hindsight pada partisi fitur terhalus yang realistis.

Prediktor optimal (MAE) untuk kelas model apapun yang memakai fitur x = median in-sample per sel
partisi x. Jika median per (film, h, bucket fitur pasangan D1-D3) saja masih >= 0.25,
maka TIDAK ADA model (XGB/apa pun, prepro apapun) yang bisa mencapai CV < 0.25 secara jujur.
python temp/exp_054_fine_oracle/oracle.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

w = pd.read_parquet("temp/exp_046_fe_decay/output/train_feat.parquet")
w["r"] = w.y / w.scale


def oracle_mase(keys, min_n=1, shrink_global=True):
    """Median in-sample per sel; sel < min_n pakai median parent (tanpa bucket pasangan)."""
    g = w.groupby(keys).r.transform("median")
    n = w.groupby(keys).r.transform("size")
    par = w.groupby(keys[:2]).r.transform("median")  # parent: film x h
    pred = np.where(n >= min_n, g, par)
    return float(np.mean(np.abs(w.r - pred)))


print("oracle (film,h)                          :", round(oracle_mase(["movie_title", "h"]), 4))
print("oracle (film,h,days_present)             :", round(oracle_mase(["movie_title", "h", "days_present"]), 4))
print("oracle (film,h,days_present,first_day)   :", round(oracle_mase(["movie_title", "h", "days_present", "first_day"]), 4))
w["sc_b"] = pd.qcut(w.scale, 6, duplicates="drop")
print("oracle (film,h,scale_bucket)             :", round(oracle_mase(["movie_title", "h", "sc_b"]), 4))
print("oracle (film,h,days_present,scale_bucket):", round(oracle_mase(["movie_title", "h", "days_present", "sc_b"]), 4))
w["sh_b"] = pd.qcut(w.show_r31, 4, duplicates="drop")
print("oracle (film,h,days_present,scale,show)  :", round(oracle_mase(["movie_title", "h", "days_present", "sc_b", "sh_b"]), 4))
w["oc_b"] = pd.qcut(w.occ13, 5, duplicates="drop")
print("oracle (film,h,days,scale,show,occ)      :", round(oracle_mase(["movie_title", "h", "days_present", "sc_b", "sh_b", "oc_b"]), 4))
# oracle (pair, h) - butuh drift: median per pasangan (hindsight murni, 7 hari)
print("oracle (pair,h) [drift hindsight]        :", round(float(np.mean(np.abs(w.r - w.groupby(["movie_title", "cinema_ids", "h"]).r.transform("median")))), 4))
