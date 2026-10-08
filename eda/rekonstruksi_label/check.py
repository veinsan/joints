"""EDA 112: apakah label D4-D10 bisa direkonstruksi dari informasi lain di paket?
(a) baris test_history pada tanggal target dari judul lain dengan judul dasar sama (varian format) di klaster sama;
(b) baris test_history pada tanggal target untuk pasangan yang SAMA (seharusnya nol);
(c) di train: apakah total_ticket = occupation_rate x total_show x kursi/show (kursi per show konstan per klaster-format)?"""
import numpy as np
import pandas as pd

t = pd.read_csv("data/test.csv", parse_dates=["date_show"])
h = pd.read_csv("data/test_history.csv", parse_dates=["date_show"])
tr = pd.read_csv("data/train.csv", parse_dates=["date_show"])
bt = lambda s: s.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D)\)\s*$", "", regex=True).str.strip()  # noqa: E731
t["base"], h["base"] = bt(t.movie_title), bt(h.movie_title)
# (b) pasangan sama, tanggal target
same = t.merge(h, on=["movie_title", "cinema_ids", "date_show"], how="inner")
print("(b) baris test yang punya baris test_history pasangan-tanggal sama:", len(same))
# (a) judul dasar sama, judul beda, klaster sama, tanggal target
x = t.merge(h[["base", "movie_title", "cinema_ids", "date_show", "total_ticket"]].rename(columns={"movie_title": "mt_h"}),
            on=["base", "cinema_ids", "date_show"], how="inner")
x = x[x.mt_h != x.movie_title]
print("(a) baris test dengan varian format lain teramati di klaster+tanggal yang sama:", len(x), f"({len(x) / len(t):.2%})",
      "| judul:", x.movie_title.nunique())
print(x.groupby(["movie_title", "mt_h"]).size().sort_values(ascending=False).head(10).to_string())
# (c) relasi kapasitas di train
tr = tr[(tr.total_show > 0) & (tr.occupation_rate > 0)].copy()
tr["seats"] = tr.total_ticket / (tr.occupation_rate / 100) / tr.total_show
g = tr.groupby(["cinema_ids", "movie_title"]).seats
tr["seat_cv"] = g.transform("std") / g.transform("mean")
print("(c) kursi per show tersirat: median CV dalam pasangan", round(float(tr.drop_duplicates(["cinema_ids", "movie_title"]).seat_cv.median()), 3))
cin = tr.groupby("cinema_ids").seats.median()
pred = (tr.occupation_rate / 100) * tr.total_show * tr.cinema_ids.map(cin)
print("    rekonstruksi tiket dari okupansi x show x kursi median klaster: median galat relatif",
      round(float(np.median(np.abs(pred - tr.total_ticket) / tr.total_ticket)), 3))
