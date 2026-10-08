"""EDA murni: sensitivitas gap retensi terhadap perkiraan tanggal D1 train."""

from pathlib import Path
import importlib.util

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("windows_v2", ROOT / "temp/exp_020_data_fix/build_windows_v2.py")
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)
tr = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
th = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
d1 = v2.film_d1(tr)
first = tr.groupby(v2.base_title(tr.movie_title)).date_show.min()
first_for_title = first.reindex(v2.base_title(pd.Series(d1.index))).to_numpy()
d1 = d1[(d1 > tr.date_show.min()) &
        (d1 + pd.Timedelta(days=10) <= tr.date_show.max()) &
        (first_for_title > tr.date_show.min())]
d1 = d1[~((d1 - pd.Timedelta(days=1) <= v2.OUTAGE[1]) &
          (d1 + pd.Timedelta(days=10) >= v2.OUTAGE[0]))]


def transition(data, dates):
    h = data.merge(dates.rename("D1").reset_index(), on="movie_title", how="inner")
    h["d"] = (h.date_show - h.D1).dt.days + 1
    day2 = h[h.d == 2][["movie_title", "cinema_ids", "total_ticket", "total_show"]].copy()
    day2["tps"] = day2.total_ticket / day2.total_show
    day3 = h[h.d == 3][["movie_title", "cinema_ids"]].drop_duplicates().assign(active_d3=True)
    p = day2.merge(day3, on=["movie_title", "cinema_ids"], how="left")
    p["active_d3"] = p.active_d3.fillna(False)
    p["bucket"] = pd.cut(p.tps, [0, 8, 15, 25, 40, float("inf")],
                         labels=["<=8", "8-15", "15-25", "25-40", ">40"], include_lowest=True)
    return p.groupby("bucket", observed=True).active_d3.agg(["size", "mean"])


test_base = th.groupby(v2.base_title(th.movie_title)).date_show.min()
titles = pd.Index(th.movie_title.unique(), name="movie_title")
test_d1 = pd.Series(test_base.reindex(v2.base_title(pd.Series(titles))).to_numpy(), index=titles, name="D1")
tables = []
for offset in [-1, 0, 1]:
    s = transition(tr, d1 + pd.Timedelta(days=offset))
    s["period"] = f"train_D1_{offset:+d}"
    tables.append(s.reset_index())
first_title = tr.groupby("movie_title").date_show.min()
no_preview = d1[first_title.reindex(d1.index).to_numpy() == d1.to_numpy()]
release_audit = pd.DataFrame({"movie_title": d1.index, "D1": d1.to_numpy(),
    "first_transaction": first_title.reindex(d1.index).to_numpy()})
release_audit["lead_days"] = (release_audit.D1 - release_audit.first_transaction).dt.days
release_audit.to_csv(OUT / "release_audit.csv", index=False)
print("release audit: titles", len(release_audit), "no_preview", int(release_audit.lead_days.eq(0).sum()),
      "preview_1plus", int(release_audit.lead_days.gt(0).sum()), "median_lead", release_audit.lead_days.median())
official = pd.DataFrame([
    ("JALAN PULANG", "2025-06-19", "https://lsf.go.id/en/film/jalan-pulang/494"),
    ("SORE ISTRI DARI MASA DEPAN", "2025-07-10", "https://lsf.go.id/en/film/sore-istri-dari-masa-depan/507"),
    ("BELIEVE - TAKDIR, MIMPI, KEBERANIAN", "2025-07-24", "https://movimax.co.id/article/BELIEVE--TAKDIR-MIMPI-KEBERANIAN"),
    ("PANGGIL AKU AYAH", "2025-08-07", "https://lsf.go.id/film/panggil-aku-ayah/297"),
    ("JADI TUH BARANG", "2025-09-18", "https://lsf.go.id/film/jadi-tuh-barang/319"),
], columns=["movie_title", "official_wide_release", "source_url"])
official["official_wide_release"] = pd.to_datetime(official.official_wide_release)
verified = official.merge(release_audit, on="movie_title", how="left")
verified["match"] = verified.D1.eq(verified.official_wide_release)
verified.to_csv(OUT / "official_release_check.csv", index=False)
print("official release exact matches", int(verified.match.sum()), "of", len(verified))
s = transition(tr, no_preview)
s["period"] = "train_no_preview"
tables.append(s.reset_index())
s = transition(th, test_d1)
s["period"] = "test"
tables.append(s.reset_index())
result = pd.concat(tables, ignore_index=True)
result["dropout"] = 1 - result["mean"]
result.to_csv(OUT / "sensitivity.csv", index=False)
print(result.round(3).to_string(index=False))
