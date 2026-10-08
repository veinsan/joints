"""EDA 007: adversarial validation window train vs test (fitur D1-D3), pergeseran per bulan.

python eda/okupansi_relatif_pasar/adversarial.py
Butuh output eda/window_seleksi_d3 (jalankan panel_vs_test.py dulu).
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
P = Path("eda/window_seleksi_d3/output")
wide = pd.read_parquet(P / "train_windows_wide.parquet")
tw = pd.read_parquet(P / "test_windows_wide.parquet")
res = {}
SEED = 2026


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


def feats(w):
    f = pd.DataFrame(index=w.index)
    for c in ["total_ticket", "occupation_rate", "total_show"]:
        for k in (1, 2, 3):
            f[f"{c}_d{k}"] = w[f"{c}_d{k}"]
    f["scale"] = w.scale
    f["tix_trend"] = (w.total_ticket_d3 + 1) / (w.total_ticket_d1 + 1)
    f["show_trend"] = (w.total_show_d3 + 1) / (w.total_show_d1 + 1)
    f["tps_d3"] = w.total_ticket_d3 / w.total_show_d3.clip(lower=1)
    # kapasitas kursi tersirat: tiket / (okupansi/100) / show
    f["seats_per_show_d3"] = w.total_ticket_d3 / (w.occupation_rate_d3 / 100).clip(lower=1e-3) / w.total_show_d3.clip(lower=1)
    g = w.groupby("movie_title")
    f["nat_tix_d3"] = g.total_ticket_d3.transform("sum")
    f["n_pairs"] = g.total_ticket_d3.transform("size")
    f["nat_tps_d3"] = f.nat_tix_d3 / g.total_show_d3.transform("sum").clip(lower=1)
    f["d1_dow"] = w.D1.dt.dayofweek
    return f


Xa, Xb = feats(wide), feats(tw)
X = pd.concat([Xa, Xb], ignore_index=True)
y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
groups = np.r_[wide.movie_title.values, tw.movie_title.values]
oof = np.zeros(len(X))
imp = pd.Series(0.0, index=X.columns)
for trn, val in GroupKFold(5).split(X, y, groups):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, random_state=SEED, verbose=-1, deterministic=True)
    m.fit(X.iloc[trn], y[trn])
    oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    imp += pd.Series(m.booster_.feature_importance("gain"), index=X.columns)
p("adv_auc_all", round(roc_auc_score(y, oof), 4))
p("adv_top_gain", (imp / imp.sum()).sort_values(ascending=False).head(10).round(3).to_dict())

# tanpa fitur level absolut (hanya bentuk/rasio)
shape_cols = ["tix_trend", "show_trend", "d1_dow", "occupation_rate_d1", "occupation_rate_d2", "occupation_rate_d3"]
oof2 = np.zeros(len(X))
for trn, val in GroupKFold(5).split(X, y, groups):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1, deterministic=True)
    m.fit(X.iloc[trn][shape_cols], y[trn])
    oof2[val] = m.predict_proba(X.iloc[val][shape_cols])[:, 1]
p("adv_auc_shape_plus_occ", round(roc_auc_score(y, oof2), 4))
shape_only = ["tix_trend", "show_trend", "d1_dow"]
oof3 = np.zeros(len(X))
for trn, val in GroupKFold(5).split(X, y, groups):
    m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, verbose=-1, deterministic=True)
    m.fit(X.iloc[trn][shape_only], y[trn])
    oof3[val] = m.predict_proba(X.iloc[val][shape_only])[:, 1]
p("adv_auc_shape_only", round(roc_auc_score(y, oof3), 4))

# kapasitas kursi tersirat per klaster: stabil antara train dan test? (identitas klaster)
for name, w, f in [("train", wide, Xa), ("test", Xb.assign(cinema_ids=tw.cinema_ids), Xb)]:
    pass
sa = Xa.assign(c=wide.cinema_ids.values)
sb = Xb.assign(c=tw.cinema_ids.values)
ok_a = (wide.occupation_rate_d3 > 1).values
ok_b = (tw.occupation_rate_d3 > 1).values
ca = sa[ok_a].groupby("c").seats_per_show_d3.median()
cb = sb[ok_b].groupby("c").seats_per_show_d3.median()
j = pd.concat([ca, cb], axis=1, keys=["train", "test"]).dropna()
p("seats_per_show_by_cinema_corr_train_test", round(float(j.corr(method="spearman").iloc[0, 1]), 3))
p("seats_per_show_median_train_test", [round(float(ca.median()), 1), round(float(cb.median()), 1)])

# test per bulan D1
tb = Xb.assign(m=tw.D1.dt.to_period("M").astype(str).values)
t = tb.groupby("m").agg(tps_d3=("tps_d3", "median"), occ_d3=("occupation_rate_d3", "median"),
                        scale=("scale", "median"), n_pairs=("n_pairs", "median"))
print("test per bulan D1\n", t.round(2).to_string())
ta = Xa.assign(m=wide.D1.dt.to_period("M").astype(str).values)
t2 = ta.groupby("m").agg(tps_d3=("tps_d3", "median"), occ_d3=("occupation_rate_d3", "median"),
                         scale=("scale", "median"), n_pairs=("n_pairs", "median"))
print("train per bulan D1\n", t2.round(2).to_string())
res["test_by_month"] = t.round(2).to_dict(); res["train_by_month"] = t2.round(2).to_dict()
# film test besar-kecil: jumlah film dengan n_pairs < 10
p("frac_films_npairs_lt10 train/test", [round(float((wide.groupby("movie_title").size() < 10).mean()), 3),
                                          round(float((tw.groupby("movie_title").size() < 10).mean()), 3)])
(OUT / "adversarial.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
