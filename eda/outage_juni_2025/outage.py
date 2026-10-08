"""EDA 010: hari dengan data hilang (outage) di train dan test_history.

Indikator: jumlah klaster yang melapor per hari dan rasio tiket terhadap median 14 hari.
python eda/outage_juni_2025/outage.py
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
th = pd.read_csv("data/test_history.csv", parse_dates=["date_show"])
res = {}


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


def daily(df):
    d = df.groupby("date_show").agg(n_cin=("cinema_ids", "nunique"), n_rows=("movie_title", "size"),
                                    tix=("total_ticket", "sum"), shows=("total_show", "sum"),
                                    n_film=("movie_title", "nunique")).asfreq("D")
    d = d.fillna(0)
    d["tix_rel"] = d.tix / d.tix.rolling(15, center=True, min_periods=5).median()
    d["cin_rel"] = d.n_cin / d.n_cin.rolling(15, center=True, min_periods=5).median()
    d["tps"] = d.tix / d.shows.replace(0, np.nan)
    return d


d = daily(tr)
sus = d[(d.cin_rel < 0.8) | (d.tix_rel < 0.4)]
print("train hari mencurigakan\n", sus.round(3).to_string())
print("\ntrain 2025-05-30 s/d 2025-06-20\n", d.loc["2025-05-30":"2025-06-20"].round(3).to_string())
print("\ntrain 2025-08-20 s/d 2025-09-05\n", d.loc["2025-08-20":"2025-09-05"].round(3).to_string())
p("train_suspect_days", [str(x.date()) for x in sus.index])

# per klaster: berapa klaster yang hilang di hari outage dibanding hari sebelumnya
for day in sus.index[:10]:
    prev = tr[(tr.date_show >= day - pd.Timedelta(days=7)) & (tr.date_show < day)].cinema_ids.unique()
    now = tr[tr.date_show == day].cinema_ids.unique()
    print(f"{day.date()}: klaster aktif 7 hari sebelumnya={len(prev)}, hari ini={len(now)}, hilang={len(set(prev) - set(now))}")

# test_history: per tanggal jumlah klaster per film relatif terhadap D3 film tsb
t = th.copy()
d1 = t.groupby("movie_title").date_show.transform("min")
t["d"] = (t.date_show - d1).dt.days + 1
fc = t.groupby(["movie_title", "d"]).cinema_ids.nunique().unstack()
fc["ratio_d2_d1"] = fc[2] / fc[1]
fc["ratio_d3_d2"] = fc[3] / fc[2]
dd = th.groupby("date_show").agg(n_cin=("cinema_ids", "nunique"), tix=("total_ticket", "sum"), n_film=("movie_title", "nunique"))
dd["tps"] = dd.tix / th.groupby("date_show").total_show.sum()
# tanggal test dengan jumlah klaster relatif rendah terhadap tanggal tetangga dengan film yang sama
low = []
for m, g in t.groupby("movie_title"):
    c = g.groupby("date_show").cinema_ids.nunique()
    if len(c) == 3 and c.min() < 0.6 * c.max():
        low.append((m, [int(x) for x in c.values], [str(x.date()) for x in c.index]))
p("test_films_cluster_drop_within_D1_D3", low[:30])
p("test_history_n_dates", int(dd.shape[0]))

fig, ax = plt.subplots(2, 1, figsize=(14, 6))
ax[0].plot(d.index, d.n_cin); ax[0].set_title("train: jumlah klaster melapor per hari")
ax[1].plot(d.index, d.tix_rel); ax[1].axhline(0.4, color="r"); ax[1].set_title("train: tiket / median 15 hari")
plt.tight_layout(); plt.savefig(OUT / "outage.png", dpi=110); plt.close()
d.to_csv(OUT / "train_daily.csv")
dd.to_csv(OUT / "test_history_daily.csv")
(OUT / "outage.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
