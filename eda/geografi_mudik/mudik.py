"""EDA 115: geografi mudik. Uplift Lebaran 2025 per klaster (train) dan sebaran D1-D3 film slate Lebaran 2026 (test_history).
Prediksi pengali film = rata-rata tertimbang (tiket D1-D3 per klaster) dari uplift klaster 2025."""
import numpy as np
import pandas as pd

tr = pd.read_csv("data/train.csv", parse_dates=["date_show"])
h = pd.read_csv("data/test_history.csv", parse_dates=["date_show"])
bt = lambda s: s.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D)\)\s*$", "", regex=True).str.strip()  # noqa: E731
h["base"] = bt(h.movie_title)
# uplift klaster 2025: rata-rata harian minggu Lebaran (1-7 Apr) / rata-rata harian minggu normal setelahnya (21 Apr - 18 Mei)
leb = tr[tr.date_show.between("2025-04-01", "2025-04-07")].groupby("cinema_ids").total_ticket.sum() / 7
nor = tr[tr.date_show.between("2025-04-21", "2025-05-18")].groupby("cinema_ids").total_ticket.sum() / 28
up = (leb / nor).replace([np.inf], np.nan).dropna()
city = tr.drop_duplicates("cinema_ids").set_index("cinema_ids").city_name
print("uplift klaster 2025: median", round(up.median(), 2), "| Jabodetabek", round(up[up.index.map(city).isin(
    ["JAKARTA", "BOGOR", "DEPOK", "TANGERANG", "BEKASI"])].median(), 2), "| lainnya", round(up[~up.index.map(city).isin(
    ["JAKARTA", "BOGOR", "DEPOK", "TANGERANG", "BEKASI"])].median(), 2))
print("10 kota uplift terendah/tertinggi (median klaster):")
uc = up.groupby(up.index.map(city)).median().sort_values()
print(uc.head(6).round(2).to_dict(), uc.tail(6).round(2).to_dict())
S = h[h.date_show.between("2026-03-18", "2026-03-20")]
slate = S.groupby("base").total_ticket.sum().nlargest(6).index
truth = {"DANUR: THE LAST CHAPTER": 6.469, "TUNGGU AKU SUKSES NANTI": 6.335, "SUZZANNA: SANTET DOSA DI ATAS DOSA": 7.042,
         "SENIN HARGA NAIK": 4.649, "NA WILLA": 6.791, "PELANGI DI MARS": 2.022}
rows = []
for f in slate:
    w = S[S.base == f].groupby("cinema_ids").total_ticket.sum()
    w = w[w.index.isin(up.index)]
    jab = S[(S.base == f)].assign(j=lambda d: d.city_name.isin(["JAKARTA", "BOGOR", "DEPOK", "TANGERANG", "BEKASI"]))
    rows.append(dict(film=f[:24], uplift_tertimbang=np.average(up[w.index], weights=w), uplift_log_tertimbang=np.exp(np.average(np.log(up[w.index]), weights=w)),
                     porsi_jabodetabek=jab[jab.j].total_ticket.sum() / jab.total_ticket.sum(), rasio_aktual=truth[f]))
R = pd.DataFrame(rows).sort_values("rasio_aktual")
print(R.round(3).to_string(index=False))
print("Spearman vs aktual:", {c: round(R[c].corr(R.rasio_aktual, method="spearman"), 2) for c in ["uplift_tertimbang", "uplift_log_tertimbang", "porsi_jabodetabek"]})
up.to_csv("eda/geografi_mudik/uplift_klaster_2025.csv")
