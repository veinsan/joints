"""EXP 051: verifikasi lantai informasi (ceiling) pada data v4.

Pertanyaan penentu: bisakah CV < 0.25 dicapai dengan prepro apapun?
 1. O1 langsung: median r per (film, h) in-sample (trajektori film benar, tanpa drift pasangan).
 2. Drift pasangan: d = mean_h log(r / film_med_h). Reliabilitas split-half (h ganjil vs genap)
    = batas atas fraksi sinyal drift yang bisa diprediksi oleh fitur sempurna sekalipun
    (fitur hanya melihat D1-D3, tetap tidak bisa melampaui reliabilitas).
 3. Simulasi lantai: pred = film_med_h x drift_hat dengan drift_hat = prediksi sempurna dari
    sinyal ber-reliabilitas terukur (shrink optimal): bandingkan 0.27 (tanpa drift) vs 0.19 (drift benar).
 4. Batas atas peningkatan dari drift: berapa MASE jika drift diketahui dengan korelasi rho terhadap drift benar.
python temp/exp_051_ceiling_v4/ceiling.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
w = pd.read_parquet("temp/exp_046_fe_decay/output/train_feat.parquet")
w["r"] = w.y / w.scale

# 1. O1: median r per film x h (in-sample, oracle)
med = w.groupby(["movie_title", "h"]).r.transform("median")
o1 = float(np.mean(np.abs(w.r - med)))
print(f"O1 median r per (film,h) sebagai prediksi r: MAE pada r = {o1:.4f}")
print("   (MASE identik karena |y-pred|/scale = |r - pred_r|)")

# 2. drift pasangan d = mean_h log(r/film_med_h), hanya baris r>0 dan film_med>0
w["film_med"] = med
v = w[(w.r > 0) & (w.film_med > 0)].copy()
v["lr"] = np.log(v.r / v.film_med)
v["par"] = v.h % 2  # ganjil/genap
d_odd = v[v.par == 1].groupby(["movie_title", "cinema_ids"]).lr.mean().rename("d_odd")
d_even = v[v.par == 0].groupby(["movie_title", "cinema_ids"]).lr.mean().rename("d_even")
dd = pd.concat([d_odd, d_even], axis=1).dropna()
rho = float(dd.d_odd.corr(dd.d_even))
print(f"\nreliabilitas split-half drift pasangan: {rho:.3f} (n={len(dd)})")
# reliabilitas dari rata-rata ganjil+genap ~ 2rho/(1+rho) (Spearman-Brown)
rel_full = 2 * rho / (1 + rho)
print(f"reliabilitas drift penuh (7 hari, Spearman-Brown): {rel_full:.3f}")

# 3. lantai dengan drift ter-shrink sempurna: prediksi drift_true dari estimasi bernoise
# model: d_true ~ N(0, s2), d_obs = d_true + n, var noise diestimasi dari 1-rel
# prediktor optimal E[d|d_obs] = shrink x d_obs dengan shrink = s2/(s2+n2) = rel
rng = np.random.default_rng(2026)
d_true = rng.normal(0, 1, 200000)
d_obs = d_true + rng.normal(0, np.sqrt(max(1e-9, 1 / max(rel_full, 1e-3) - 1)), 200000)
shrink = float(np.cov(d_true, d_obs)[0, 1] / np.var(d_obs))
print(f"shrink optimal estimasi drift dari 7 hari observasi: {shrink:.3f}")

# 4. dampak ke MASE: pakai kerangka multiplikatif pada baris positif
# pred r = film_med x exp(shrink x d_hat) vs film_med x exp(d_true) vs film_med
pos = w[(w.r > 0) & (w.film_med > 0)]
drift_full = v.groupby(["movie_title", "cinema_ids"]).lr.mean().rename("d_full")
pos = pos.merge(drift_full, on=["movie_title", "cinema_ids"], how="left")
pos["d_full"] = pos.d_full.fillna(0)
for label, dmul in [("tanpa drift (O1+)", 0.0), ("drift ter-shrink optimal", shrink), ("drift benar (hindsight)", 1.0)]:
    pred_r = pos.film_med.values * np.exp(dmul * pos.d_full.values)
    m = float(np.mean(np.abs(pos.r.values - pred_r)))
    print(f"baris positif: MAE r {label}: {m:.4f} (n={len(pos)})")

# baris nol: pred 0 optimal, hitung kontribusi
z = w[w.r == 0]
print(f"baris nol: {len(z)} ({len(z)/len(w):.1%}), kontribusi MAE bila pred 0 = 0")
print("\nKesimpulan lantai = kombinasi O1-positif ter-shrink + nol; bandingkan dgn O1 penuh di atas.")
