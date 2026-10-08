"""EDA: audit label cuti bersama yang tersedia sebelum cutoff terhadap kalender paket."""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

# Dokumen pemerintah yang terbit <= 30 September 2025, hanya untuk audit, bukan fitur.
# 2025: https://www.setneg.go.id/baca/index/presiden_tetapkan_keputusan_baru_mengenai_hari_cuti_bersama_asn_tahun_2025
# Tambahan 18 Agustus: https://www.setneg.go.id/baca/index/sambut_hut_ke_80_kemerdekaan_ri_pemerintah_tetapkan_18_agustus_2025_sebagai_cuti_bersama
# 2026: https://jdih.kemnaker.go.id/peraturan/detail/2723/keputusan-bersama-menteri-agama-menteri-ketenagakerjaan-dan-menteri-pendayagunaan-aparatur-negara-dan-reformasi-birokrasi-republik-indonesia-nomor-2-tahun-2025
cuti_2025 = ["2025-04-02", "2025-04-03", "2025-04-04", "2025-04-07",
             "2025-05-13", "2025-05-30", "2025-06-09", "2025-08-18", "2025-12-26"]
cuti_2026 = ["2026-02-16", "2026-03-18", "2026-03-20", "2026-03-23", "2026-03-24"]
cuti = pd.DataFrame({"date": pd.to_datetime(cuti_2025 + cuti_2026)})
hol = pd.read_csv(ROOT / "data/holidays.csv", parse_dates=["date"])
train = pd.read_csv(ROOT / "data/train.csv", parse_dates=["date_show"])
hist = pd.read_csv(ROOT / "data/test_history.csv", parse_dates=["date_show"])
test = pd.read_csv(ROOT / "data/test.csv", parse_dates=["date_show"])

table = cuti.merge(hol, on="date", how="left")
for label, df in [("train", train), ("test_history", hist), ("test_target", test)]:
    g = df.groupby("date_show").agg(rows=("date_show", "size"))
    if "total_ticket" in df:
        g["tickets"] = df.groupby("date_show").total_ticket.sum()
    table = table.merge(g.add_prefix(label + "_").reset_index().rename(columns={"date_show": "date"}),
                        on="date", how="left")
table = table.fillna({col: 0 for col in table if col.endswith(("_rows", "_tickets"))})
table.to_csv(OUT / "official_cuti_vs_package.csv", index=False)

target_cuti = test[test.date_show.isin(cuti.date)]
hist_cuti = hist[hist.date_show.isin(cuti.date)]
summary = {
    "official_cuti_dates_in_package": int(table.holiday_tipe.notna().sum()),
    "dates_marked_normal": int(table.holiday_tipe.eq("normal").sum()),
    "dates_marked_holiday": int(table.holiday_tipe.eq("holiday").sum()),
    "test_target_rows_on_cuti": len(target_cuti),
    "test_target_share_on_cuti": float(len(target_cuti) / len(test)),
    "test_target_pairs_on_cuti": int(target_cuti.groupby(["movie_title", "cinema_ids"]).ngroups),
    "test_history_rows_on_cuti": len(hist_cuti),
    "train_rows_on_cuti": int(train.date_show.isin(cuti.date).sum()),
    "march_20_23_24_test_rows": int(test.date_show.isin(pd.to_datetime(["2026-03-20", "2026-03-23", "2026-03-24"])).sum()),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(table.to_string(index=False))
print(json.dumps(summary, indent=2))
