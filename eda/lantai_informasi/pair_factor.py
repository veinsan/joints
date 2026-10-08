"""EXP 049: EDA prediktabilitas faktor pasangan (hambatan utama menurut EXP-047).

Definisi faktor pasangan PF = mean_h r_h(pair) / film_h_median(h), r = y/scale, hanya baris y>0.
Pertanyaan: seberapa banyak PF bisa diduga dari D1-D3 (share, occ, shows, kota, klaster)?
Metode: (a) korelasi Spearman PF vs kandidat prediktor; (b) estimasi plafon MASE bila PF
diganti prediksi OOF dari regresi kuantile sederhana dalam film (gbm kecil hanya utk EDA,
bukan pipeline); (c) dekomposisi varians PF: antar-klaster vs antar-film.
python temp/exp_049_eda_pair_factor/eda.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
w = pd.read_parquet("temp/exp_046_fe_decay/output/train_feat.parquet")  # level pasangan x h
w = w[["movie_title", "cinema_ids", "city_name", "h", "y", "scale", "cal_ratio",
       "tix_d1", "tix_d2", "tix_d3", "show_d3", "occ13", "days_present", "first_day",
       "share_d3", "occ13_vs_film", "tps_d3", "nat_tix_d3", "n_pairs", "cin_seats",
       "occ_d3", "seats_per_show", "city_share_d3", "city_n_cin", "footprint_r31",
       "nat_tix_r31", "tix_r31", "show_r31", "fold", "base_title", "fmt", "cohort"]].copy()

w["r"] = w.y / w.scale
film_h_med = w.groupby(["movie_title", "h"]).r.transform("median")
w["pf_row"] = np.where((w.y > 0) & (film_h_med > 0), w.r / film_h_med, np.nan)

# PF per pasangan (mean baris valid) + jumlah baris valid
pf = w.groupby(["movie_title", "cinema_ids"]).agg(
    pf=("pf_row", "mean"), n_ok=("pf_row", "count"),
    n_pos=("y", lambda s: int((s > 0).sum())), r_mean=("r", "mean"),
    occ13=("occ13", "first"), share_d3=("share_d3", "first"), tps_d3=("tps_d3", "first"),
    occ13_vs_film=("occ13_vs_film", "first"), days_present=("days_present", "first"),
    first_day=("first_day", "first"), show_d3=("show_d3", "first"), cin_seats=("cin_seats", "first"),
    city_share_d3=("city_share_d3", "first"), city_n_cin=("city_n_cin", "first"),
    footprint_r31=("footprint_r31", "first"), scale=("scale", "first"), fold=("fold", "first"),
    city_name=("city_name", "first")).reset_index()
pf = pf[pf.n_ok >= 3]  # butuh minimal 3 hari valid
print("pasangan dengan PF terukur:", len(pf), "dari", w.groupby(["movie_title", "cinema_ids"]).ngroups)

# (a) korelasi Spearman
preds = ["occ13", "share_d3", "tps_d3", "occ13_vs_film", "days_present", "first_day",
         "show_d3", "cin_seats", "city_share_d3", "city_n_cin", "footprint_r31", "scale"]
print("\nSpearman PF vs prediktor D1-D3:")
for p in preds:
    s = pf[["pf", p]].corr(method="spearman").iloc[0, 1]
    print(f"  {p:16s} {s:+.3f}")

# log PF agar simetris
pf["log_pf"] = np.log(pf.pf.clip(0.05, 20))
print("\nspread log PF: std", round(float(pf.log_pf.std()), 3), "| IQR",
      round(float(pf.pf.quantile(0.75) / pf.pf.quantile(0.25)), 2))

# (c) dekomposisi: klaster sama, film beda -> seberapa stabil kekuatan klaster?
# gunakan pasangan yang muncul >= 3 film
cnt = pf.groupby("cinema_ids").size()
multi = pf[pf.cinema_ids.isin(cnt[cnt >= 4].index)]
var_within = multi.groupby("cinema_ids").log_pf.var().mean()
var_total = multi.log_pf.var()
print("\ndekomposisi log PF (klaster >= 4 film): var_total", round(float(var_total), 3),
      "var_within klaster", round(float(var_within), 3),
      "-> R2 klaster", round(1 - var_within / var_total, 3), f"n={len(multi)}")

# kekuatan klaster lintas film: korelasi PF antar film (split film ganjil/genap by hash)
rng = np.random.default_rng(2026)
films = pf.movie_title.unique()
half = set(rng.choice(films, len(films) // 2, replace=False))
pf["_g"] = np.where(pf.movie_title.isin(half), "a", "b")
cin_g = pf.groupby(["cinema_ids", "_g"]).log_pf.mean().unstack()
cin_g = cin_g.dropna()
print("korelasi kekuatan klaster antar dua kelompok film:", round(float(cin_g.a.corr(cin_g.b)), 3),
      "n klaster", len(cin_g))

# (b) plafon: ganti level pasangan model dengan PF prediksi dari D1-D3 fitur (EDA GBM kecil)
import sys

sys.path.insert(0, ".")
import xgboost as xgb
from scripts.utils.seed_utils import gbdt_params, seed_everything

seed_everything(2026)
X = pf[preds + ["city_name"]].copy()
X["city_name"] = X.city_name.astype("category").cat.codes.astype(float)
X = X.astype(float)
oof = np.zeros(len(pf))
fold = pf.fold.values
for k in range(5):
    ti, vi = fold != k, fold == k
    m = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=400, max_depth=5,
                         learning_rate=0.1, tree_method="hist", device="cuda", **gbdt_params("xgb", 2026))
    m.fit(X[ti], pf.log_pf[ti])
    oof[vi] = m.predict(X[vi])
r2 = 1 - np.mean((pf.log_pf - oof) ** 2) / np.var(pf.log_pf)
print("\nR2 prediksi log PF dari fitur D1-D3 (EDA gbm):", round(float(r2), 3))
pf["pf_hat"] = np.exp(oof)

# simulasi dampak: pakai OOF XGB EXP-045, kalikan pred per pasangan dgn pf_hat/pf_model
o = pd.read_parquet("temp/exp_045_xgb_lock/output/xgb_gpu_v4_cal/oof.parquet")
o["rel"] = o.y / o.scale
o["pred_rel"] = o.pred / o.scale
pm = o.groupby(["movie_title", "cinema_ids"]).pred_rel.mean()
key = pf.set_index(["movie_title", "cinema_ids"])
o = o.merge(pf[["movie_title", "cinema_ids", "pf_hat"]], on=["movie_title", "cinema_ids"], how="inner")
o = o.merge(pm.rename("pred_rel_mean"), on=["movie_title", "cinema_ids"], how="left")
adj = (o.pf_hat / o.pred_rel_mean.clip(lower=1e-6)).values
pred_adj = o.pred.values * np.clip(adj, 0.2, 5)
mase0 = float(np.mean(np.abs(o.y - o.pred) / o.scale))
mase1 = float(np.mean(np.abs(o.y - pred_adj) / o.scale))
print(f"\nMASE OOF asli {mase0:.4f} -> dengan faktor pasangan PF-hat {mase1:.4f} (baris cocok {len(o)})")
