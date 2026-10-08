"""Bangun sampel D1-D10 dari train.csv meniru aturan host.

Aturan (hasil EDA 003):
- D1 film = tanggal wide release: tanggal pertama jumlah klaster aktif >= FRAC_WIDE * maksimum klaster film.
- Pasangan (film, klaster) dipakai bila punya transaksi di D3 (aturan seleksi test, 100% vs 0%).
- Target D4-D10: tanggal tanpa baris = 0.
- Film dipakai bila D10 <= tanggal akhir train dan D1 bukan tanggal awal train (left-censored).
- Film dibuang bila D1-D10 bersinggungan dengan outage data 2025-06-07 s/d 2025-06-16 (EDA 010:
  banyak klaster tidak melapor, 13 Jun 0 klaster), karena target nol di sana palsu.

Dipakai sebagai modul: from build_windows import build_windows
"""
from pathlib import Path

import numpy as np
import pandas as pd

FRAC_WIDE = 0.5
OUTAGE = (pd.Timestamp("2025-06-07"), pd.Timestamp("2025-06-16"))


def film_d1(df, frac=FRAC_WIDE):
    nat = df.groupby(["movie_title", "date_show"]).cinema_ids.nunique().rename("n_cin").reset_index()
    mx = nat.groupby("movie_title").n_cin.transform("max")
    d1 = nat[nat.n_cin >= frac * mx].groupby("movie_title").date_show.min()
    return d1.rename("D1")


def long_to_windows(hist, d1, horizon_rows=None, end_date=None):
    """hist: baris transaksi; d1: Series film -> D1. Mengembalikan (hist_wide, target_long)."""
    h = hist.merge(d1.reset_index(), on="movie_title")
    h["d"] = (h.date_show - h.D1).dt.days + 1
    pre = h[h.d < 1]
    h13 = h[(h.d >= 1) & (h.d <= 3)]
    pairs = h13[h13.d == 3][["movie_title", "cinema_ids"]].drop_duplicates()
    h13 = h13.merge(pairs)
    wide = h13.pivot_table(index=["movie_title", "cinema_ids", "city_name"], columns="d",
                           values=["total_ticket", "occupation_rate", "total_show"], fill_value=0)
    wide.columns = [f"{a}_d{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    for c in ["total_ticket", "occupation_rate", "total_show"]:
        for k in (1, 2, 3):
            if f"{c}_d{k}" not in wide:
                wide[f"{c}_d{k}"] = 0.0
    wide["scale"] = np.maximum(wide[["total_ticket_d1", "total_ticket_d2", "total_ticket_d3"]].sum(1) / 3, 1)
    wide = wide.merge(d1.reset_index(), on="movie_title")
    return wide, pre


def build_windows(tr, frac=FRAC_WIDE, exclude_outage=True):
    d1 = film_d1(tr, frac)
    start, end = tr.date_show.min(), tr.date_show.max()
    d1 = d1[(d1 > start) & (d1 + pd.Timedelta(days=9) <= end)]
    # buang film yang sudah tayang luas sejak awal train (left-censored)
    first = tr.groupby("movie_title").date_show.min()
    d1 = d1[first.reindex(d1.index) > start]
    if exclude_outage:
        hit = (d1 <= OUTAGE[1]) & (d1 + pd.Timedelta(days=9) >= OUTAGE[0])
        d1 = d1[~hit]
    wide, pre = long_to_windows(tr, d1)
    # target D4-D10
    tgt = wide[["movie_title", "cinema_ids", "D1"]].copy()
    tgt = tgt.loc[tgt.index.repeat(7)].reset_index(drop=True)
    tgt["d"] = np.tile(np.arange(4, 11), len(wide))
    tgt["date_show"] = tgt.D1 + pd.to_timedelta(tgt.d - 1, unit="D")
    tgt = tgt.merge(tr[["movie_title", "cinema_ids", "date_show", "total_ticket", "occupation_rate", "total_show"]],
                    how="left", on=["movie_title", "cinema_ids", "date_show"])
    tgt[["total_ticket", "occupation_rate", "total_show"]] = tgt[["total_ticket", "occupation_rate", "total_show"]].fillna(0)
    return wide, tgt, pre


def test_windows(th):
    d1 = th.groupby("movie_title").date_show.min().rename("D1")
    wide, _ = long_to_windows(th, d1)
    return wide


if __name__ == "__main__":
    D = Path("data")
    tr = pd.read_csv(D / "train.csv", parse_dates=["date_show"])
    wide, tgt, pre = build_windows(tr)
    print("films:", wide.movie_title.nunique(), "pairs:", len(wide), "target rows:", len(tgt))
