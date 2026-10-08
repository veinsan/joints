"""EDA 008: apakah film dipangkas berdasarkan performa relatif terhadap pasar saat itu?

Indeks pasar dihitung hanya dari film lain yang D1 <= D1 film ini (data D1-D3 mereka selesai
sebelum atau bersamaan D3 film ini), jadi tersedia juga di test tanpa informasi masa depan.

python eda/okupansi_relatif_pasar/relatif_pasar.py
Butuh output eda/window_seleksi_d3 (jalankan panel_vs_test.py dulu).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
P = Path("eda/window_seleksi_d3/output")
wide = pd.read_parquet(P / "train_windows_wide.parquet")
tgt = pd.read_parquet(P / "train_windows_target.parquet")
tw = pd.read_parquet(P / "test_windows_wide.parquet")
hol = pd.read_csv("data/holidays.csv", parse_dates=["date"])
res = {}
LOOKBACK = 28


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


def add_market(w):
    w = w.copy()
    w["tps_d3"] = w.total_ticket_d3 / w.total_show_d3.clip(lower=1)
    w["occ13"] = w[["occupation_rate_d1", "occupation_rate_d2", "occupation_rate_d3"]].mean(axis=1)
    films = w.groupby("movie_title").agg(D1=("D1", "first"), occ=("occ13", "median"), tps=("tps_d3", "median"))
    mk_occ, mk_tps = {}, {}
    for m, r in films.iterrows():
        prev = films[(films.D1 <= r.D1) & (films.D1 > r.D1 - pd.Timedelta(days=LOOKBACK)) & (films.index != m)]
        mk_occ[m] = prev.occ.median() if len(prev) >= 3 else np.nan
        mk_tps[m] = prev.tps.median() if len(prev) >= 3 else np.nan
    w["mk_occ"] = w.movie_title.map(mk_occ)
    w["mk_tps"] = w.movie_title.map(mk_tps)
    # indeks pasar per klaster: pasangan film lain di klaster sama
    cl = w[["movie_title", "cinema_ids", "D1", "occ13"]]
    j = cl.merge(cl, on="cinema_ids", suffixes=("", "_o"))
    j = j[(j.D1_o <= j.D1) & (j.D1_o > j.D1 - pd.Timedelta(days=LOOKBACK)) & (j.movie_title_o != j.movie_title)]
    ck = j.groupby(["movie_title", "cinema_ids"]).occ13_o.median().rename("mk_occ_cin")
    w = w.merge(ck.reset_index(), how="left")
    w["occ_rel"] = w.occ13 / w.mk_occ
    w["occ_rel_cin"] = w.occ13 / w.mk_occ_cin
    w["tps_rel"] = w.tps_d3 / w.mk_tps
    # peringkat film dalam kohort D1 yang sama minggu ini
    return w


w = add_market(wide)
tgt["zero"] = (tgt.total_ticket == 0).astype(int)
agg = tgt.groupby(["movie_title", "cinema_ids"]).agg(n_zero=("zero", "sum"), tix=("total_ticket", "sum")).reset_index()
w = w.merge(agg)
w["r_mean"] = w.tix / 7 / w.scale
w["m"] = w.D1.dt.to_period("M").astype(str)
ok = w.mk_occ.notna() & w.mk_occ_cin.notna()
p("pairs_with_market_index", f"{int(ok.sum())}/{len(w)}")
ww = w[ok]
cols = ["occ13", "occ_rel", "occ_rel_cin", "tps_d3", "tps_rel"]
sp = ww[cols + ["n_zero", "r_mean"]].corr(method="spearman")[["n_zero", "r_mean"]].loc[cols].round(3)
print("spearman global\n", sp.to_string())
res["spearman_global"] = sp.to_dict()
# per bulan: apakah relasi relatif lebih stabil?
rows = []
for m, g in ww.groupby("m"):
    c = g[cols + ["n_zero"]].corr(method="spearman")["n_zero"].loc[cols]
    rows.append(c.rename(m))
bym = pd.DataFrame(rows).round(3)
print("spearman dengan n_zero per bulan D1\n", bym.to_string())
res["spearman_by_month"] = bym.to_dict()

# ambang okupansi: fraksi nol di D8-D10 per bin occ13 absolut, dipisah per bulan dengan pasar rendah/tinggi
ww = ww.assign(mk_low=ww.mk_occ < ww.mk_occ.median())
bins = [0, 4, 6, 8, 10, 14, 20, 30, 100]
t = ww.groupby([pd.cut(ww.occ13, bins), "mk_low"], observed=True).n_zero.mean().unstack().round(2)
print("rata-rata jumlah nol (0-7) per bin okupansi, kolom = pasar rendah?\n", t.to_string())
res["zero_by_occ_bin_market"] = {str(k): {str(i): x for i, x in v.items()} for k, v in t.to_dict().items()}

# test: indeks pasar
wt = add_market(tw)
p("test_occ_rel_median", round(float(wt.occ_rel.median()), 3))
p("train_occ_rel_median", round(float(ww.occ_rel.median()), 3))
p("test_occ13_median", round(float(wt.occ13.median()), 3))
p("train_occ13_median", round(float(ww.occ13.median()), 3))

# test: horizon mencakup libur (Natal, tahun baru, Imlek, Nyepi, Idulfitri)
hd = set(hol[hol.holiday_tipe == "holiday"].date)
fl = tw.drop_duplicates("movie_title")[["movie_title", "D1"]]
fl["n_hol_D4_D10"] = fl.D1.apply(lambda d: sum((d + pd.Timedelta(days=k)) in hd for k in range(3, 10)))
fl["n_hol_D1_D3"] = fl.D1.apply(lambda d: sum((d + pd.Timedelta(days=k)) in hd for k in range(0, 3)))
p("test_films_with_holiday_in_D4_D10", int((fl.n_hol_D4_D10 > 0).sum()))
print(fl[(fl.n_hol_D4_D10 > 0) | (fl.n_hol_D1_D3 > 0)].sort_values("D1").to_string(index=False))
ft = wide.drop_duplicates("movie_title")[["movie_title", "D1"]]
ft["n_hol_D4_D10"] = ft.D1.apply(lambda d: sum((d + pd.Timedelta(days=k)) in hd for k in range(3, 10)))
p("train_films_with_holiday_in_D4_D10", int((ft.n_hol_D4_D10 > 0).sum()))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for lo, g in ww.groupby("mk_low"):
    z = g.groupby(pd.cut(g.occ13, bins), observed=True).n_zero.mean()
    ax[0].plot(range(len(z)), z.values, marker="o", label=f"pasar rendah={lo}")
ax[0].set_xticks(range(len(bins) - 1)); ax[0].set_xticklabels([str(b) for b in bins[1:]])
ax[0].set_xlabel("okupansi rata-rata D1-D3 (batas atas bin)"); ax[0].set_ylabel("jumlah hari nol D4-D10"); ax[0].legend()
ax[1].hist(np.log(ww.occ_rel.clip(1e-2)), bins=60, alpha=.5, density=True, label="train")
ax[1].hist(np.log(wt.occ_rel.dropna().clip(1e-2)), bins=60, alpha=.5, density=True, label="test")
ax[1].set_title("log occ_rel"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "relatif_pasar.png", dpi=110); plt.close()
(OUT / "relatif_pasar.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
