# EDA 099: fase Februari-Maret 2026 dan momentum Ramadan/Lebaran

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/fase_ramadan_2026/ramadan_audit.py` dari root. Analisis hanya memakai input D3 pasangan yang dipilih untuk test, tanpa target D4-D10 dan tanpa fitting model. Definisi D3 tiket/show lemah ialah <=8, sama dengan audit rezim sebelumnya.

| Fase tanggal D3 | Pasangan | Film dasar | Median tiket/show | Proporsi lemah | Median okupansi |
|---|---:|---:|---:|---:|---:|
| Oktober-Januari | 7.112 | 94 | 23,67 | 13,65% | 15,61% |
| 1-18 Februari | 1.066 | 17 | 11,33 | 34,62% | 7,52% |
| 19 Februari-17 Maret | 1.564 | 23 | 7,60 | 52,94% | 4,85% |
| 18-20 Maret | 631 | 6 | 22,60 | 6,97% | 13,39% |

Pelemahan sudah muncul sebelum Ramadan resmi: minggu D3 6-8 Februari memiliki 46,85% pasangan lemah. Dalam 33 kota dengan minimal 10 pasangan pada kedua fase, 30 kota meningkat dari Oktober-Januari ke 1-18 Februari; 31 kota meningkat lagi dari 1-18 Februari ke 19 Februari-17 Maret. Minggu D3 13 Maret terdiri hanya dari tiga film dan 92,82% pasangan lemah; lonjakan itu sangat tergantung slate film. Enam film D1 18 Maret berbalik kuat tepat sebelum Idulfitri. Semua fase membandingkan film yang berbeda, sehingga tidak membuktikan efek kausal Ramadan atau Lebaran.

[Cinema XXI pada 24 April 2025](https://cinema21.co.id/newsroom/di-tengah-kelesuan-jumlah-penonton-cinema-xxi-berhasil-jaga-ebitda-positif-di-kuartal-i-2025-dan-bukukan-pendapatan-rp9292-miliar) menyatakan aktivitas menonton turun selama Ramadan 2025 dan rebound pada Lebaran dengan lima rilis. Ini konteks industri, bukan identifikasi operator dataset. [Kemenag menetapkan 1 Ramadan 1447 H pada 19 Februari 2026](https://aceh.kemenag.go.id/baca/pemerintah-tetapkan-1-ramadan-1447-h-jatuh-pada-19-februari-2026?audio=1) melalui sidang 17 Februari 2026, setelah cutoff eksternal 30 September 2025. Tanggal resmi itu hanya untuk audit retrospektif. [SKB libur 2026](https://jdih.kemnaker.go.id/peraturan/detail/2723/keputusan-bersama-menteri-agama-menteri-ketenagakerjaan-dan-menteri-pendayagunaan-aparatur-negara-dan-reformasi-birokrasi-republik-indonesia-nomor-2-tahun-2025) sudah ditetapkan 19 September 2025, sehingga jarak ke libur Idulfitri dan jendela Ramadan perkiraan dapat dihitung tanpa bocor waktu.

Output lengkap ada di `output/phase_summary.csv`, `weekly_summary.csv`, `film_by_phase.csv`, dan `city_by_phase.csv`.

Jejak kalender sekolah juga perlu versi waktu: [keputusan Disdik DKI 89/2025](https://edu.jakarta.go.id/cdn/files/photos/cms/2025-06-25_09%3A11%3A38_275c35de-fe11-e719-f1bb-d17171c08a68_pdf.pdf) sudah merencanakan libur semester 20 Desember-3 Januari dan libur sekitar Ramadan/Idulfitri sebelum cutoff, tetapi [perubahan 113/2026](https://edu.jakarta.go.id/cdn/files/photos/cms/kalender20252026.pdf) terbit 12 Februari 2026. Jadwal DKI tidak otomatis berlaku pada kota lain; perubahan setelah cutoff hanya dipakai untuk audit retrospektif.
