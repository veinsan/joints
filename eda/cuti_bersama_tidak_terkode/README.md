# Cuti bersama resmi tidak terwakili di `holiday_tipe`

**Satu kalimat:** semua 14 tanggal cuti bersama resmi yang jatuh dalam rentang `holidays.csv` diberi label `normal`; 2.392 dari 72.611 baris target test (3,29%) jatuh pada tanggal tersebut.

## Pertanyaan

Apakah kalender paket sudah mengkodekan cuti bersama yang diumumkan sebelum batas data eksternal 30 September 2025?

## Metode

Jalankan `python eda/cuti_bersama_tidak_terkode/calendar_audit.py` dari root. Tanggal 2025 berasal dari [keputusan cuti bersama 2025 yang diumumkan Januari](https://www.setneg.go.id/baca/index/presiden_tetapkan_keputusan_baru_mengenai_hari_cuti_bersama_asn_tahun_2025) dan [tambahan 18 Agustus yang diumumkan 7 Agustus](https://www.setneg.go.id/baca/index/sambut_hut_ke_80_kemerdekaan_ri_pemerintah_tetapkan_18_agustus_2025_sebagai_cuti_bersama). Tanggal 2026 berasal dari [SKB yang ditetapkan 19 September 2025](https://jdih.kemnaker.go.id/peraturan/detail/2723/keputusan-bersama-menteri-agama-menteri-ketenagakerjaan-dan-menteri-pendayagunaan-aparatur-negara-dan-reformasi-birokrasi-republik-indonesia-nomor-2-tahun-2025); [laporan pada hari pengumuman](https://www.antaranews.com/berita/5120681/pemerintah-tetapkan-delapan-hari-cuti-bersama-tahun-2026) mencantumkan tanggalnya. Ini audit data; tabel sumber tidak diubah.

## Hasil

| Ukuran | Hasil |
|---|---:|
| Tanggal cuti resmi dalam rentang `holidays.csv` | 14 |
| Ditandai `holiday_tipe=normal` | 14 |
| Baris target test pada tanggal cuti | 2.392 / 72.611 (3,29%) |
| Pasangan test dengan minimal satu hari target pada cuti | 1.580 / 10.373 (15,23%) |
| Baris target test pada 20, 23, 24 Maret 2026 | 1.443 |
| Baris transaksi train pada tanggal cuti | 5.385 |

Tanggal 20, 23, dan 24 Maret 2026 adalah cuti bersama Idulfitri, tetapi `holidays.csv` menandainya `normal`; tanggal 21 dan 22 Maret diberi `holiday`. Rincian per tanggal di `output/official_cuti_vs_package.csv`. Sembilan Juni 2025 hanya punya satu baris train karena outage, jadi data tanggal itu tidak dapat dipakai mengukur efek cuti.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Fitur kalender | Pertimbangkan indikator terpisah untuk cuti bersama yang telah diumumkan sebelum cutoff, terutama 20/23/24 Maret | tinggi |
| Validasi | Laporkan segmen target pada cuti dan rangkaian Idulfitri 2026; train tidak punya analog Lebaran di awal horizon | tinggi |
| Riset | Catat tanggal publikasi SKB dan URL resmi bila suatu saat kalender eksternal digunakan | tinggi |

`holiday_tipe=normal` mungkin memang bermakna "bukan libur nasional" sehingga hasil ini tidak membuktikan kesalahan data. Besarnya efek cuti pada penjualan tiket belum diestimasi. Ini tidak menjalankan eksperimen modelling.
