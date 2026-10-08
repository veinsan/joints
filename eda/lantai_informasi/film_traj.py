"""EXP 050: EDA prediktabilitas trajektori film nasional D4-D10 dari D1-D3.

Definisi: film_rel_h = national_tix_h(cal-adjusted) / national_tix_mean D1-D3(cal-adjusted), h=4..10.
Pertanyaan: (1) seberapa akurat ekstrapolasi log-linear vs tabel kohort F_h vs campuran;
(2) fitur D1-D3 apa yang memprediksi penyimpangan film dari kohort;
(3) implikasi MASE prediktor deterministik trajektori x scale (tanpa model).
python temp/exp_050_eda_film_traj/eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from scripts.utils.seed_utils import seed_everything

seed_everything(2026)
import numpy as np
import pandas as pd
import xgboost as xgb
from scripts.utils.seed_utils import gbdt_params

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
w = pd.read_parquet("temp/exp_046_fe_decay/output/train_feat.parquet")

# level film per h: jumlah nasional cal-adjusted per umur
film = w.groupby(["movie_title", "h"]).agg(
    nat_h=("nat_tix_adj_d3", "first"), y_sum=("y", "sum"), scale_sum=("scale", "sum"),
    nat_d1=("nat_tix_adj_d1", "first"), nat_d2=("nat_tix_adj_d2", "first"), nat_d3=("nat_tix_adj_d3", "first"),
    n_cin_d1=("n_cin_d1", "first"), n_cin_d3=("n_cin_d3", "first"), nat_r31=("nat_tix_r31", "first"),
    occ=("nat_occ13", "first"), tps=("nat_tps_d3", "first"), genre=("genre", "first"),
    age=("age_rating", "first"), d1=("D1", "first"), fold=("fold", "first"), cohort=("cohort", "first"),
    f_h=("f_h", "first"), base_title=("base_title", "first"), fmt=("fmt", "first")).reset_index()
film["nat_13"] = film[["nat_d1", "nat_d2", "nat_d3"]].mean(axis=1)
film["rel_h"] = film.y_sum / film.nat_13.clip(lower=1)
film["age_h"] = film.h  # umur = h

# (1) pembanding ekstrapolasi
d = np.log1p(film[["nat_d1", "nat_d2", "nat_d3"]].values)
dd = np.arange(3)
slope = ((d - d.mean(axis=1)[:, None]) * (dd - dd.mean())).sum(axis=1) / ((dd - dd.mean()) ** 2).sum()
film["slope"] = slope  # satu baris per film x h, slope sama untuk semua h
film["pred_lin"] = np.clip(np.exp(film.slope.values * (film.h.values - 3)), 0.1, 6.0) * film.nat_d3.values / film.nat_13.values
film["pred_cohort"] = film.f_h.values
# blend: bobot kohort vs lin dari fold latih (dikira-kira 50/50 dulu, dan versi shrink slope 0.5)
film["pred_blend"] = 0.5 * film.pred_cohort + 0.5 * film.pred_lin

for name in ["pred_cohort", "pred_lin", "pred_blend"]:
    err = np.abs(np.log(film.rel_h.clip(1e-3, 10) / film[name].clip(1e-3, 10)))
    print(f"MAE log trajektori film {name}: {err.mean():.3f} (median {np.median(err):.3f})")

# per h
tab = film.assign(err=np.abs(np.log(film.rel_h.clip(1e-3, 10) / film.pred_blend.clip(1e-3, 10))))
print("\nMAE log per h (blend):")
print(tab.groupby("h").err.mean().round(3).to_string())

# (2) apa yang memprediksi penyimpangan film dari kohort: target dev = log(rel_h / pred_cohort) di h rata2
film["dev"] = np.log(film.rel_h.clip(1e-3, 10) / film.pred_cohort.clip(1e-3, 10))
f1 = film[film.h == 7].copy()  # satu baris per film
f1["slope_sig"] = np.sign(f1.slope) * np.sqrt(np.abs(f1.slope))
f1["footprint"] = f1.n_cin_d3 / f1.n_cin_d1.clip(lower=1)
f1["is_horror"] = f1.genre.fillna("").str.contains("Horror").astype(int)
f1["d1_dow"] = f1.d1.dt.dayofweek
preds = ["slope_sig", "nat_r31", "footprint", "occ", "tps", "is_horror", "d1_dow", "cohort", "n_cin_d1"]
X = f1[preds].copy()
X["age"] = f1.age.fillna("NA").astype("category").cat.codes
X = X.astype(float)
y = f1.dev.values
oof = np.zeros(len(f1))
for k in range(5):
    ti, vi = f1.fold.values != k, f1.fold.values == k
    m = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=400, max_depth=4,
                         learning_rate=0.1, tree_method="hist", device="cuda", **gbdt_params("xgb", 2026))
    m.fit(X[ti], y[ti])
    oof[vi] = m.predict(X[vi])
r2 = 1 - np.mean((y - oof) ** 2) / np.var(y)
print(f"\nR2 deviasi film dari kohort (h=7) dari fitur D1-D3: {r2:.3f}")
imp = pd.Series(m.feature_importances_, index=X.columns).sort_values(ascending=False)
print("importance:", {k: round(v, 3) for k, v in imp.items()})

# (3) implikasi MASE: prediktor deterministik trajektori x level pasangan scale
p = w.merge(film[["movie_title", "h", "pred_lin", "pred_blend"]], on=["movie_title", "h"])
pred_y = np.clip(p.f_h.values * p.cal_ratio.values, 0, None) * p.scale.values
print(f"\nMASE deterministik (cohort F_h x cal x scale): {np.mean(np.abs(p.y - pred_y) / p.scale):.4f}")
pred_y2 = np.clip(p.pred_lin.values * p.cal_ratio.values, 0, None) * p.scale.values
print(f"MASE deterministik (lin x cal x scale): {np.mean(np.abs(p.y - pred_y2) / p.scale):.4f}")
pred_y3 = np.clip(p.pred_blend.values * p.cal_ratio.values, 0, None) * p.scale.values
print(f"MASE deterministik (blend x cal x scale): {np.mean(np.abs(p.y - pred_y3) / p.scale):.4f}")

# survival film: apakah jumlah klaster bersistaya pada h bisa diprediksi dari D1-D3?
film["footprint"] = film.n_cin_d3 / film.n_cin_d1.clip(lower=1)
sv = film.groupby(["movie_title", "h"]).agg(rel_h=("rel_h", "first"), slope=("slope", "first"),
                                            nat_r31=("nat_r31", "first"), footprint=("footprint", "first"),
                                            occ=("occ", "first"), fold=("fold", "first")).reset_index()
# survival proxy: rel_h kecil < 0.05 = mati
sv["alive"] = (sv.rel_h > 0.05).astype(int)
for h in [6, 8, 10]:
    s = sv[sv.h == h]
    base = s.alive.mean()
    # AUC-ish: rank slope/footprint vs alive
    auc_sl = (s.groupby("alive").slope.mean().diff().iloc[-1] > 0)
    print(f"h={h}: P(film hidup)={base:.2f}; mean slope hidup={s[s.alive==1].slope.mean():+.3f} vs mati={s[s.alive==0].slope.mean():+.3f}; "
          f"footprint hidup={s[s.alive==1].footprint.mean():.2f} vs mati={s[s.alive==0].footprint.mean():.2f}")
