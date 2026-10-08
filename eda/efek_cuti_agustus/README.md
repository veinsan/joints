# Cuti bersama 18 Agustus tampak pada intensitas tiket

**Satu kalimat:** pada cuti bersama 18 Agustus 2025, tiket nasional dalam dataset 2,56 kali rata-rata geometrik dua Senin tetangga, sementara jumlah show hanya 0,95 kali; indikator `holiday_tipe` menyebut tanggal itu `normal`.

## Pertanyaan

Apakah tanggal cuti bersama yang tidak diberi kategori khusus dalam `holidays.csv` memiliki pola transaksi berbeda?

## Metode

Jalankan `python eda/efek_cuti_agustus/august_cuti.py` dari root. Bandingkan 11, 18, dan 25 Agustus 2025, ketiganya Senin. Selain total jaringan, gunakan pasangan film x klaster yang muncul pada ketiga hari dan hitung rasio tiket, show, dan tiket/show terhadap rata-rata geometrik dua Senin tetangga. Pemeriksaan kedua memakai pasangan yang muncul pada Senin dan Selasa di ketiga minggu untuk membandingkan rasio Senin/Selasa. Penetapan cuti 18 Agustus diumumkan pada [7 Agustus 2025 oleh Setneg](https://www.setneg.go.id/baca/index/sambut_hut_ke_80_kemerdekaan_ri_pemerintah_tetapkan_18_agustus_2025_sebagai_cuti_bersama).

## Hasil

| Ukuran | Hasil |
|---|---:|
| Tiket 11 / 18 / 25 Agustus | 162.697 / 397.281 / 148.197 |
| Rasio tiket 18 Agustus terhadap rata-rata geometrik tetangga | 2,56 |
| Rasio show harian dengan pembanding sama | 0,95 |
| Pasangan yang hadir pada tiga Senin | 209 dari 11 judul dasar |
| Median rasio tiket/show per pasangan | 3,03 |
| Median rasio show per pasangan | 0,82 |
| Judul dengan median tiket/show lebih tinggi | 10/11; 6/6 untuk judul dengan >=10 pasangan |
| Panel lengkap enam hari Senin/Selasa | 189 pasangan |
| Median rasio Senin/Selasa tiket/show, minggu cuti relatif minggu tetangga | 2,28 |

Output terperinci ada di `output/daily.csv`, `by_film.csv`, `balanced_six_day_ratios.csv`, dan `summary.json`.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Fitur kalender | Uji indikator cuti bersama terpisah dari hari libur nasional pada tahap modelling nanti, bila sesuai batas data eksternal | tinggi |
| Validasi | Laporkan performa pada target yang jatuh di cuti dan pada rangkaian Idulfitri | tinggi |

Ini sinyal dari satu kejadian yang juga merupakan akhir pekan panjang HUT RI. Perbandingan film x klaster aktif mengurangi, tetapi tidak menghilangkan, perubahan komposisi dan kurva umur film. Tidak ada estimasi kausal, tidak ada eksperimen modelling, dan besar lonjakan ini tidak boleh langsung dipakai sebagai multiplier untuk cuti 2026.
