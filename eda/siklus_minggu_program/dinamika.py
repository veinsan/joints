"""EDA 005: pola nol (film dicabut), siklus minggu program, efek kalender, pendorong decay.

python eda/siklus_minggu_program/dinamika.py
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
hol = pd.read_csv("data/holidays.csv", parse_dates=["date"])
pr = pd.read_csv("data/ticket_prices.csv")
res = {}


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


tgt["zero"] = (tgt.total_ticket == 0).astype(int)
tgt["dow"] = tgt.date_show.dt.dayofweek  # 0 Senin
tgt["d1dow"] = tgt.D1.dt.dayofweek

# 1. nol: horizon vs hari target. Apakah cliff ikut hari kalender (Rabu/Kamis) atau umur film?
zt = tgt.pivot_table(index="d1dow", columns="d", values="zero", aggfunc="mean").round(3)
print("zero rate, baris = hari D1 (0=Senin), kolom = d\n", zt.to_string())
zt.to_csv(OUT / "zero_by_d1dow_d.csv")
zw = tgt.pivot_table(index="d1dow", columns="dow", values="zero", aggfunc="mean").round(3)
print("zero rate, baris = hari D1, kolom = hari target\n", zw.to_string())

# 2. pola nol: sekali nol tetap nol? (film dicabut permanen) vs bolong-bolong
z = tgt.pivot_table(index=["movie_title", "cinema_ids"], columns="d", values="zero")
revive = ((z.diff(axis=1) == -1).sum(axis=1) > 0).mean()
p("frac_pairs_revive_after_zero", round(float(revive), 4))
allzero = (z.sum(axis=1) == 7).mean()
p("frac_pairs_all_zero_D4_D10", round(float(allzero), 4))
last_pos = z.apply(lambda r: max([d for d in r.index if r[d] == 0] + [3]), axis=1)
p("last_positive_day_dist", last_pos.value_counts(normalize=True).sort_index().round(3).to_dict())

# 3. pendorong nol dan rasio: fitur D1-D3
w = wide.copy()
w["occ_mean"] = w[["occupation_rate_d1", "occupation_rate_d2", "occupation_rate_d3"]].mean(axis=1)
w["show_trend"] = (w.total_show_d3 + 1) / (w.total_show_d1 + 1)
w["tix_trend"] = (w.total_ticket_d3 + 1) / (w.total_ticket_d1 + 1)
w["tix_per_show_d3"] = w.total_ticket_d3 / w.total_show_d3.clip(lower=1)
agg = tgt.groupby(["movie_title", "cinema_ids"]).agg(n_zero=("zero", "sum"), r_sum=("total_ticket", "sum"))
w = w.merge(agg.reset_index())
w["r_mean"] = w.r_sum / 7 / w.scale
w["zero_d10"] = w.merge(tgt[tgt.d == 10][["movie_title", "cinema_ids", "zero"]]).zero.values
for c in ["occupation_rate_d3", "occ_mean", "show_trend", "tix_trend", "scale", "total_show_d3", "tix_per_show_d3"]:
    q = pd.qcut(w[c].rank(method="first"), 10, labels=False)
    t = w.groupby(q).agg(lo=(c, "min"), hi=(c, "max"), zero_frac=("n_zero", lambda s: s.mean() / 7),
                         zero_d10=("zero_d10", "mean"), r_mean_med=("r_mean", "median")).round(3)
    print(f"\n== decile {c}\n", t.to_string())
    res[f"decile_{c}"] = t.to_dict()
    cor = w[[c, "r_mean", "n_zero"]].corr(method="spearman").round(3)
    p(f"spearman_{c}", {"r_mean": cor.loc[c, "r_mean"], "n_zero": cor.loc[c, "n_zero"]})

# 4. kalender: efek weekend/libur pada rasio terhadap hari sebelumnya untuk pasangan yang masih tayang
cal = hol.rename(columns={"date": "date_show"})
t2 = tgt.merge(cal, on="date_show", how="left")
pos = t2[t2.total_ticket > 0]
p("median_r_by_daytipe_x_holiday", pos.groupby(["day_tipe", "holiday_tipe"]).r.median().round(3).to_dict().__repr__())
m = pos.pivot_table(index="d", columns="day_tipe", values="r", aggfunc="median").round(3)
print("median r (positif saja) per d x day_tipe\n", m.to_string())

# 5. harga: rasio weekend/weekday per kota vs uplift weekend kota
pp = pr.pivot(index="city_name", columns="price_day", values="ceil")
pp["wkend_ratio"] = pp.Weekend / pp.Weekday
p("price_weekday_desc", pp.Weekday.describe().round(0).to_dict())
p("price_wkend_ratio_desc", pp.wkend_ratio.describe().round(3).to_dict())

# 6. plot
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for dd, g in tgt.groupby("d1dow"):
    if g.movie_title.nunique() < 10:
        continue
    zz = g.groupby("d").zero.mean()
    ax[0].plot(zz.index, zz.values, marker="o", label=f"D1 dow={dd}")
ax[0].set_title("fraksi nol per horizon"); ax[0].legend()
q = pd.qcut(w.occupation_rate_d3, 10, labels=False)
for dd in (4, 7, 8, 10):
    zz = tgt[tgt.d == dd].merge(w[["movie_title", "cinema_ids"]].assign(q=q)).groupby("q").zero.mean()
    ax[1].plot(zz.index, zz.values, marker="o", label=f"d={dd}")
ax[1].set_title("fraksi nol vs desil occupation_rate_d3"); ax[1].legend()
ax[2].scatter(np.log1p(w.show_trend), w.r_mean, s=2, alpha=.3)
ax[2].set_ylim(0, 3); ax[2].set_xlabel("log1p(show_d3/show_d1)"); ax[2].set_ylabel("mean y/scale D4-D10")
plt.tight_layout(); plt.savefig(OUT / "dinamika.png", dpi=110); plt.close()
(OUT / "dinamika.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
