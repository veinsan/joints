"""EDA 009: indeks permintaan nasional harian dan efek kalender (hari, libur, libur sekolah, pasca Lebaran).

Model: log(total tiket semua film) ~ hari-dalam-minggu + libur + tren minggu, lalu lihat residual.
Juga efek kalender per horizon pada rasio y/scale di panel window.

python eda/efek_kalender/kalender.py
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
tr = pd.read_csv("data/train.csv", parse_dates=["date_show"])
hol = pd.read_csv("data/holidays.csv", parse_dates=["date"])
res = {}


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


nat = tr.groupby("date_show").agg(tix=("total_ticket", "sum"), shows=("total_show", "sum"),
                                  n_film=("movie_title", "nunique")).reset_index()
nat = nat.merge(hol.rename(columns={"date": "date_show"}), how="left")
nat["dow"] = nat.date_show.dt.dayofweek
nat["week"] = nat.date_show.dt.to_period("W-SUN").astype(str)
nat["ltix"] = np.log(nat.tix)
# efek hari: rata-rata log per dow relatif mingguan (hilangkan tren minggu)
nat["wk_mean"] = nat.groupby("week").ltix.transform("mean")
nat["dev"] = nat.ltix - nat.wk_mean
norm = nat[(nat.holiday_tipe == "normal")]
dow_eff = norm.groupby("dow").dev.median()
p("dow_effect_multiplier(normal days, vs week mean)", np.exp(dow_eff).round(3).to_dict())
nat["resid"] = nat.dev - nat.dow.map(dow_eff)
hold = nat[nat.holiday_tipe == "holiday"][["date_show", "holiday_name", "dow", "resid"]]
hold["mult"] = np.exp(hold.resid).round(3)
print("libur: multiplier relatif hari yang sama pada minggu normal\n", hold.to_string(index=False))
res["holiday_mult"] = hold.assign(date_show=hold.date_show.astype(str)).to_dict("records")

# residual terbesar (hari spesial tak tercatat: cuti bersama, libur sekolah)
top = nat.reindex(nat.resid.abs().sort_values(ascending=False).index).head(20)[["date_show", "dow", "holiday_name", "resid"]]
top["mult"] = np.exp(top.resid).round(3)
print("hari dengan residual terbesar\n", top.to_string(index=False))

# libur sekolah (Jun-Jul 2025): hari kerja normal, rata-rata dev vs dow effect per minggu
nat["weekday_normal"] = (nat.dow <= 3) & (nat.holiday_tipe == "normal")
wk = nat[nat.weekday_normal].groupby("week").agg(start=("date_show", "min"), weekday_vs_weekend=("dev", "mean"))
# rasio hari kerja terhadap rata-rata minggu: tinggi berarti hari kerja relatif ramai (libur sekolah)
print("rata-rata dev hari Senin-Kamis normal per minggu (tinggi = hari kerja ramai)\n",
      wk.assign(start=wk.start.dt.date).round(3).to_string())
res["weekday_dev_by_week"] = {str(k): round(v, 3) for k, v in wk.weekday_vs_weekend.items()}

# level mingguan (musiman) dan jumlah film tayang
wl = nat.groupby("week").agg(start=("date_show", "min"), tix=("tix", "sum"), n_film=("n_film", "mean"))
print("total tiket mingguan\n", wl.assign(start=wl.start.dt.date).to_string())

fig, ax = plt.subplots(2, 1, figsize=(14, 7))
ax[0].plot(nat.date_show, nat.tix / 1e3)
for _, r in hol[hol.holiday_tipe == "holiday"].iterrows():
    if r.date <= nat.date_show.max():
        ax[0].axvline(r.date, color="r", alpha=.3)
ax[0].set_title("total tiket nasional per hari (ribu), garis merah = libur")
ax[1].bar(nat.date_show, nat.resid, color=np.where(nat.dow >= 5, "tab:orange", "tab:blue"))
ax[1].set_title("residual log setelah efek minggu dan hari (oranye = Sabtu/Minggu)")
plt.tight_layout(); plt.savefig(OUT / "kalender.png", dpi=110); plt.close()
nat.to_csv(OUT / "national_daily.csv", index=False)
(OUT / "kalender.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
