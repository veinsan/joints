"""EDA 113: pasangan tanpa D1 (masuk bioskop D2/D3). Subtipe, perilaku D4-D10, dan galat v8 per subtipe."""
import sys

import numpy as np
import pandas as pd

sys.argv = ["blend5.py", "--tag", "late2"]
exec(open("temp/exp_048_blend5/blend5.py", encoding="utf-8").read().split("# dunia kappa A (v8)")[0])
v8 = np.load("temp/exp_058_tsfm/output/v8_oof.npy")
X = pd.read_parquet("temp/exp_040_v8port/output/Xtr.parquet", columns=["sh1", "sh2", "sh3", "fnc1", "fnc3", "d1_dow", "f_logT"])
D = xk.assign(p=v8, ys=y_shift, sh2=X.sh2.values, sh3=X.sh3.values, fgrow=(X.fnc3 / X.fnc1.clip(lower=1)).values)
L = D[D.y1 == 0].copy()
L["tipe"] = np.where(L.y2 > 0, "D2+D3", "hanya_D3")
L["e"] = np.abs(L.total_ticket - L.p) / L.scale
pair = L.groupby(["movie_title", "cinema_ids"]).agg(tipe=("tipe", "first"), y2=("y2", "first"), y3=("y3", "first"), s=("scale", "first"),
                                                     tot=("total_ticket", "sum"), ptot=("p", "sum"), e=("e", "mean"), sh3=("sh3", "first"),
                                                     fgrow=("fgrow", "first"), nz=("total_ticket", lambda v: (v > 0).sum()))
pair["level_ratio"] = (pair.tot / 7) / pair.y3.clip(lower=1)
print(pair.groupby("tipe").agg(n=("e", "size"), MASE=("e", "mean"), s_med=("s", "median"), y3_med=("y3", "median"),
                               hari_aktif_med=("nz", "median"), lv_q25=("level_ratio", lambda v: v.quantile(.25)),
                               lv_med=("level_ratio", "median"), lv_q75=("level_ratio", lambda v: v.quantile(.75))).round(3).to_string())
print("\nhari aktif D4-D10 (sebaran):", pair.nz.value_counts(normalize=True).sort_index().round(3).to_dict())
# rasio level D4-D10 vs y3 (pasangan aktif) dan vs pred
pair["pred_ratio"] = (pair.ptot / 7) / pair.y3.clip(lower=1)
act = pair[pair.nz >= 5]
print("pasangan aktif >=5 hari: median level/y3 aktual", round(act.level_ratio.median(), 3), "pred", round(act.pred_ratio.median(), 3))
# siapa yang galatnya terbesar
pair["sbin"] = pd.cut(pair.s, [0, 2, 5, 10, 30, 1e9])
print(pair.groupby(["tipe", "sbin"], observed=True).agg(n=("e", "size"), MASE=("e", "mean"), lv=("level_ratio", "median"),
                                                          pr=("pred_ratio", "median"), aktif=("nz", "mean")).round(2).to_string())
pair.to_csv("eda/pasangan_terlambat/pairs.csv")
