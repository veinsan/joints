"""EDA 116: deteksi gangguan pencatatan di periode test dari test_history. Per tanggal: rasio klaster tercatat per film
terhadap jejak maksimum film itu di D1-D3, rasio show per klaster, dan jumlah film yang tercakup."""
import numpy as np
import pandas as pd

h = pd.read_csv("data/test_history.csv", parse_dates=["date_show"])
f = h.groupby(["movie_title", "date_show"]).agg(nc=("cinema_ids", "nunique"), sh=("total_show", "sum"), tk=("total_ticket", "sum")).reset_index()
mx = f.groupby("movie_title").nc.transform("max")
f["nc_rel"] = f.nc / mx
f["shpc"] = f.sh / f.nc
f["shpc_rel"] = f.shpc / f.groupby("movie_title").shpc.transform("max")
big = f[mx >= 30]
d = big.groupby("date_show").agg(film=("movie_title", "nunique"), nc_rel_med=("nc_rel", "median"), nc_rel_min=("nc_rel", "min"),
                                 shpc_rel_med=("shpc_rel", "median"))
pd.set_option("display.width", 200)
print("tanggal mencurigakan (median rasio klaster < 0.85 atau show/klaster < 0.6):")
print(d[(d.nc_rel_med < 0.85) | (d.shpc_rel_med < 0.6)].round(3).to_string())
print("\nsebaran median rasio klaster per tanggal:", d.nc_rel_med.describe().round(3).to_dict())
# tanggal di periode test tanpa satu pun baris test_history (tidak bisa dicek)
allday = pd.date_range("2025-10-01", "2026-03-29")
t = pd.read_csv("data/test.csv", parse_dates=["date_show"])
tgt = t.groupby("date_show").size()
miss = [x.date() for x in allday if x not in d.index]
print("\ntanggal tanpa film D1-D3 (tidak terpantau):", len(miss), "| baris target pada tanggal itu:", int(tgt.reindex(pd.to_datetime(miss)).fillna(0).sum()))
d.to_csv("eda/outage/test_dates.csv")
