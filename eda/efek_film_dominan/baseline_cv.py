"""EDA 011: baseline MASE out-of-fold dengan skema validasi yang direkomendasikan.

Skema: StratifiedGroupKFold(5), group = judul dasar film (varian IMAX/3D satu grup),
strata = bulan D1. Baseline tanpa model:
  B1 = 0
  B2 = scale * median(y/scale | h)
  B3 = scale * median(y/scale | h, hari D1)
  B4 = scale * median(y/scale | h, desil nat_tix_d3 film)   (cek nilai fitur agregat nasional)
  B5 = scale * median(y/scale | h, desil occ_rel)            (okupansi relatif pasar)
python eda/efek_film_dominan/baseline_cv.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
P = Path("eda/window_seleksi_d3/output")
wide = pd.read_parquet(P / "train_windows_wide.parquet")
tgt = pd.read_parquet(P / "train_windows_target.parquet")
SEED = 2026
res = {}


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


key = ["movie_title", "cinema_ids"]
w = wide.copy()
w["base_title"] = w.movie_title.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True)
w["nat_tix_d3"] = w.groupby("movie_title").total_ticket_d3.transform("sum")
w["occ13"] = w[["occupation_rate_d1", "occupation_rate_d2", "occupation_rate_d3"]].mean(axis=1)
# okupansi relatif pasar (film lain dengan D1 dalam 28 hari terakhir), hanya data masa lalu
films = w.groupby("movie_title").agg(D1=("D1", "first"), occ=("occ13", "median"))
mk = {m: films[(films.D1 <= r.D1) & (films.D1 > r.D1 - pd.Timedelta(days=28)) & (films.index != m)].occ.median()
      for m, r in films.iterrows()}
w["occ_rel"] = w.occ13 / w.movie_title.map(mk)
t = tgt.merge(w[key + ["base_title", "nat_tix_d3", "occ_rel"]], on=key)
t["r"] = t.total_ticket / t.scale if "scale" in t else None
t = t.merge(w[key + ["scale"]], on=key) if "scale" not in t else t
t["r"] = t.total_ticket / t.scale
t["d1dow"] = t.D1.dt.dayofweek
t["month"] = t.D1.dt.month

fl = w.drop_duplicates("base_title")[["base_title", "D1"]]
fl["m"] = fl.D1.dt.month
sgk = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
fold_of = {}
for k, (_, vi) in enumerate(sgk.split(fl, fl.m, fl.base_title)):
    for b in fl.base_title.iloc[vi]:
        fold_of[b] = k
t["fold"] = t.base_title.map(fold_of)


def mase(df, pred):
    return float(np.mean(np.abs(df.total_ticket - pred) / df.scale))


def run(name, keys_fn):
    """keys_fn(df, ref) -> Series kunci string sejajar df; median y/scale per kunci dari fold latih."""
    scores = []
    oof = np.zeros(len(t))
    for k in range(5):
        trn, val = t[t.fold != k], t[t.fold == k]
        if keys_fn is None:
            pred = np.zeros(len(val))
        else:
            med = trn.r.groupby(keys_fn(trn, trn).values).median()
            pr = pd.Series(keys_fn(val, trn).values).map(med)
            pr = pr.fillna(pd.Series(val.d.values).map(trn.groupby("d").r.median()))
            pred = pr.values * val.scale.values
        oof[val.index] = pred
        scores.append(mase(val, pred))
    s = np.array(scores)
    p(name, {"oof": round(mase(t, oof), 4), "fold_mean": round(float(s.mean()), 4), "fold_std": round(float(s.std()), 4),
             "folds": [round(float(x), 4) for x in s]})
    return oof


def q_edges(col, n=10):
    return np.unique(np.quantile(col.dropna(), np.linspace(0, 1, n + 1)[1:-1]))


def kq(df, ref, col, tf=lambda x: x):
    q = np.digitize(tf(df[col].fillna(ref[col].median())), q_edges(tf(ref[col])))
    return df.d.astype(str) + "_" + pd.Series(q, index=df.index).astype(str)


run("B1_zero", None)
run("B2_h", lambda df, ref: df.d.astype(str))
run("B3_h_d1dow", lambda df, ref: df.d.astype(str) + "_" + df.d1dow.astype(str))
run("B4_h_natTix", lambda df, ref: kq(df, ref, "nat_tix_d3", np.log1p))
run("B5_h_occRel", lambda df, ref: kq(df, ref, "occ_rel"))
p("n_rows", int(len(t)))
p("n_films", int(t.movie_title.nunique()))
p("rows_per_fold", t.fold.value_counts().sort_index().to_dict())
(OUT / "baseline_cv.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
