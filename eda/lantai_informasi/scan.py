"""EXP 047: analisis error OOF XGB GPU baseline (EXP 045) untuk menargetkan FE berikutnya.
Pertanyaan: (1) kontribusi MASE per bucket scale/h/zero/dow; (2) film mana penyukit terbesar;
(3) berapa MASE bila level film dikoreksi ke benar (swap oracle); (4) sama untuk faktor pasangan.
python temp/exp_047_error_scan/scan.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

o = pd.read_parquet("temp/exp_045_xgb_lock/output/xgb_gpu_v4_cal/oof.parquet")
o["err"] = (o.pred - o.y).abs() / o.scale
o["rel"] = o.y / o.scale
o["pred_rel"] = o.pred / o.scale
print("MASE total:", round(o.err.mean(), 4))


def bucket_report(name, bins, labels=None):
    b = pd.cut(o[name], bins, labels=labels) if labels else pd.cut(o[name], bins)
    g = o.groupby(b, observed=True).agg(n=("err", "size"), mase=("err", "mean"), share=("err", lambda s: s.sum() / o.err.sum()))
    print(f"\n== {name} ==\n", g.round(3).to_string())


bucket_report("scale", [0.99, 1, 3, 10, 30, 100, 1e9], ["1", "1-3", "3-10", "10-30", "30-100", ">100"])
bucket_report("h", [3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
print("\n== target nol vs positif ==")
print(o.groupby(o.y == 0).agg(n=("err", "size"), mase=("err", "mean"), share=("err", lambda s: s.sum() / o.err.sum())).round(3).to_string())

# bias arah untuk target positif
p = o[o.y > 0]
print("\ntarget>0: median rel", round(p.rel.median(), 3), "median pred_rel", round(p.pred_rel.median(), 3))
print("target>0: MAE di bawah prediksi (under):", round((p.pred_rel > p.rel).mean(), 3))

# film: kontribusi error + koreksi level film (oracle swap)
f = o.groupby("movie_title").agg(n=("err", "size"), mase=("err", "mean"), c=("err", "sum"),
                                 rel_med=("rel", "median"), pred_med=("pred_rel", "median"))
f["share"] = f.c / o.err.sum()
print("\ntop 10 film penyumbang error:")
print(f.sort_values("c", ascending=False).head(10)[["n", "mase", "share", "rel_med", "pred_med"]].round(3).to_string())

# swap oracle level film: pred *= rel_med_true / pred_med_true (per film)
m = o.movie_title.values
corr = f.rel_med.reindex(m).values / f.pred_med.clip(lower=1e-6).reindex(m).values
pred_fix = o.pred.values * corr
print("\nMASE bila level film dikoreksi sempurna (median):", round(float(np.mean(np.abs(o.y - pred_fix) / o.scale)), 4))

# swap oracle faktor pasangan: pred *= mean_rel_pair_true / mean_rel_pair_pred
pf_t = o.groupby(["movie_title", "cinema_ids"]).rel.transform("mean")
pf_p = o.groupby(["movie_title", "cinema_ids"]).pred_rel.transform("mean")
pred_fix2 = o.pred.values * (pf_t / pf_p.clip(lower=1e-6)).values
print("MASE bila faktor pasangan dikoreksi sempurna:", round(float(np.mean(np.abs(o.y - pred_fix2) / o.scale)), 4))

# keduanya
pred_fix3 = pred_fix * (pf_t / pf_p.clip(lower=1e-6)).values
print("MASE bila keduanya dikoreksi:", round(float(np.mean(np.abs(o.y - pred_fix3) / o.scale)), 4))

# pair factor dari D1-D3 vs realisasi: seberapa stabil share pasangan?
w = pd.read_parquet("temp/exp_045_xgb_lock/output/train_feat.parquet")
sh13 = (w[["tix_d1", "tix_d2", "tix_d3"]].sum(axis=1)) / w.nat_tix_d1.clip(lower=1)
o2 = o.merge(w[["movie_title", "cinema_ids", "tix_d1", "tix_d2", "tix_d3", "nat_tix_d3", "days_present"]].assign(sh13=sh13),
             on=["movie_title", "cinema_ids"], how="left")
print("\nkorelasi err dengan share D1-D3:", round(float(o2[["err", "sh13"]].corr().iloc[0, 1]), 3))
print(o2.groupby(pd.cut(o2.sh13, [-0.01, 0.001, 0.005, 0.02, 0.05, 0.2, 1]), observed=True).agg(
    n=("err", "size"), mase=("err", "mean")).round(3).to_string())
print("\nkorelasi err dengan days_present:", round(float(o2[["err", "days_present"]].corr().iloc[0, 1]), 3))
