# Deep analysis 7 Oktober 2026 (dasar v13)

Sembilan script baru (`eda/65` sampai `eda/73`). Semuanya hanya membaca data resmi, prediksi yang sudah tersimpan, atau tabel lookup. Tidak ada model yang dilatih untuk evaluasi, tidak ada submisi, tidak ada label target uji. Setiap script mencetak angka dan menyimpan gambar di `outputs/eda/<nama>/`. Jalankan dengan `rtk proxy .venv/bin/python eda/<script>.py`.

## Ringkasan satu paragraf

Offset validasi ke leaderboard (0,062 sampai 0,067) sama untuk semua model, sehingga sumbernya ada pada sesuatu yang dipakai semua model. Sumber terkuat yang ditemukan adalah **fitur level absolut** (`log_s`, `f_logT`). Pasar periode uji hanya separuh pasar train: referensi klaster median 75 vs 146 tiket per hari. Model membaca pasangan uji yang kecil karena pasar sepi sebagai pasangan yang sedang mati. Pada satu-satunya label di dalam periode uji (D3 dari D1 dan D2), tolok ukur absolut memprediksi terlalu rendah dengan median 0,32 sampai 0,38 skala pada pasangan berskala 5 sampai 50 (sepertiga pasangan uji). Tolok ukur relatif terhadap film lain yang tayang bersamaan tidak bias. Grouped CV tidak bisa melihat ini karena pergeseran pasar tidak ada di dalam train, dan secara lokal grouped CV justru sedikit memilih fitur absolut. Pola ini sama dengan catatan lama "proxy memilih fitur relatif, GroupKFold menolak": yang selama ini salah adalah hakimnya, bukan fiturnya.

## Per script

| Script | Pertanyaan | Hasil | Keputusan |
| :--- | :--- | :--- | :--- |
| `65_public_vs_prediction_shape` | Apakah urutan public mengikuti porsi nol prediksi uji? | v12 justru memprediksi nol lebih sedikit dari v8 (3,36% vs 4,03% baris uji non-Lebaran). Di v5+ (n = 5) korelasinya terbalik dari hipotesis (lebih banyak nol, public lebih baik; Pearson -0,76, lemah). OOF hanya memprediksi nol untuk 2,8% baris padahal 30% target nol. | Hipotesis "v12 kalah karena nol" ditolak. Tuas nol bekerja lewat prediksi positif kecil, bukan nol keras. |
| `66_calendar_residual_test_period` | Apakah kalender struktural membuat dinamika D1 sampai D3 sama di semua periode? | Ramadan: pergeseran +0,04 [-0,06; +0,14] dan +0,035, jadi sesuai train (34 sampai 36 film). Akhir tahun: positif (+0,19/+0,12) tetapi n = 9 sampai 10. | `RAMADAN_F = 1,0` kini didukung 34 film, bukan 2. |
| `67_pull_transfer_d3_to_target` | Apakah kecenderungan mencopot di D3 terbawa ke nol D4 sampai D10? | Klaster: korelasi 0,30, kemiringan terkoreksi reliabilitas **0,09** (116 klaster, reliabilitas 0,74). Klaster x bulan 0,10. Bulan (n = 6) tidak reliabel. | Pergeseran D3 tidak bisa dipindahkan utuh ke horizon target; λ = 0,5 tidak punya dasar transfer. |
| `68_pull_absolute_vs_relative` | Pergeseran kebijakan atau pasar sepi? | Offset logit periode uji: tps absolut **-1,35**, okupansi -1,34, relatif film -0,39, relatif klaster **-0,21**, dengan log-loss train setara (0,227 vs 0,230). Porsi berhenti per bulan 5 sampai 17% di kedua periode walaupun okupansi turun ke separuh. | Sebagian besar "pergeseran kebijakan" adalah artefak tolok ukur absolut. |
| `69_relative_level_target` | Apakah target D4 sampai D10 lebih cocok diukur dengan level absolut atau relatif? | Distribusi `rel_clu` identik train vs uji (median 0,26 vs 0,23), `log_s` bergeser -0,67. Lookup absolut menyiratkan porsi nol uji 0,42 dan median r 0,30; lookup relatif 0,30 dan 0,38. Sekitar 25% baris uji ada di sel "kecil absolut, normal relatif" yang jarang di train. | Ganti level absolut dengan level relatif. |
| `70_relative_level_validation` | Validasi di luar distribusi | **Proxy periode uji:** MAE absolut 0,402, relatif 0,370 (nasional 0,366); bias median +0,116 vs 0,000; skala 5 sampai 20: +0,32 vs 0; skala 20 sampai 50: +0,38 vs 0. MAE di train: absolut 0,340, relatif 0,347. **Split pasar sepi di train:** MASE 0,356 vs 0,348, bias +0,046 vs +0,018. | Lensa keputusan v13: proxy periode uji + pasar sepi; grouped CV dan temporal hanya batas biaya. |
| `71_submitted_models_abs_extrapolation` | Apakah model yang disubmit benar-benar melakukan ekstrapolasi absolut? | Pada sel (h x kuintil relatif x bin p3) yang sama, r_hat uji lebih rendah dari OOF: skala 5 sampai 20 -0,105 (v8) / -0,088 (v12), 20 sampai 50 -0,075 / -0,060, 50 sampai 200 -0,061 / -0,040. Label train di sel yang sama hanya -0,01 sampai -0,02. | Bukti pendukung (kovariat lain juga berbeda, jadi bukan bukti tunggal). |
| `72_test_holiday_factors` | Libur periode uji diukur dari `test_history` | Natal: selisih median +0,06 (masuk) dan -0,09 (keluar), sesuai asumsi. Isra Mikraj dan Nyepi juga sesuai. Hanya Tahun Baru menyimpang (+0,45/+0,33) dari 3 film, dan rasionya tidak bisa membedakan malam tahun baru yang lemah dari 1 Januari yang kuat. | Kalender tidak diubah. Angka 0,61 di `eda/66` untuk transisi Rabu ke Kamis akhir tahun tercampur malam tahun baru. |
| `73_relative_features_prepro_check` | Implementasi notebook dan pemeriksaan hasil preprocessing | KS train vs uji: `log_s` 0,231 menjadi `rel_nat` 0,042 / `rel_clu` 0,043; `f_logT` 0,236 menjadi `f_rel` 0,105. Fallback referensi klaster 0,06 sampai 0,10%. Lookup pasar sepi: 0,3374 menjadi 0,3328. Gambar contoh klaster: referensi mengikuti puncak Natal dan penurunan Februari sampai Maret. | Fitur siap dipakai; implementasi yang sama disalin ke notebook dan diuji brute-force di sana. |

## Yang diterapkan di v13 (`notebooks/build_v13.py`)

1. **Fitur level relatif.** `rel_nat`, `rel_clu`, dan `f_rel` menggantikan `log_s` dan `f_logT`, juga di klasifier nol. Referensinya adalah jendela D1 sampai D3 film lain (train dan `test_history`) dengan D1 paling jauh 14 hari, tanpa judul dasar yang sama. Tidak memakai label D4 sampai D10, jadi tidak ada kebocoran target. Versi IMAX dan 3D tidak bisa menjadi pembandingnya sendiri.
2. **Lensa baru.** (a) Proxy periode uji berbasis model untuk set fitur absolut vs relatif. (b) Lensa pasar sepi: dilatih tanpa 40% film dengan referensi nasional terendah, lalu dinilai pada film itu. Statistik klaster untuk lensa ini juga dihitung tanpa film tersebut.
3. **Aturan adopsi R1 (ditulis sebelum hasil).** Proxy periode uji membaik, pasar sepi membaik, grouped TW dan temporal masing-masing tidak lebih buruk dari +0,003. Bila gagal, notebook kembali ke B1 (v12).
4. **Koreksi pencopotan.** Nilai `a` diukur ulang dengan fitur yang sama dengan model. λ dipilih per model dengan minimax atas TW, temporal, dan pasar sepi. Dunia κ hanya dicetak.
5. **Model.** LightGBM (kontrol), TabPFN-2.5 Quantiles, **TabPFN-3.5 Fast FP16** (unduh revisi `06bf2ba3`, SHA `14e0b812…`, matriks dikonversi ke FP16 di notebook, 167,2 MB), dan TabM. Paling banyak satu TabPFN. Ukuran komponen diukur dari model CV, dan kombinasi yang melewati 200 MB dikeluarkan dari grid **sebelum** dipilih.
6. **A3 (kalender input) tidak diuji ulang**, karena sudah gagal ambang di v11 dan v12.
7. **Audit.** `oof.csv` berisi prediksi B1, R1, setiap komponen, flag pasar sepi, dan blend. `oof_quantiles.npz` berisi p0/Q LightGBM dan kuantil setiap model (OOF dan pasar sepi). Ada juga tabel segmen nol, positif, tumbuh, positif tanpa tumbuh, dan D4 sampai D5.

## Pemeriksaan yang sudah dijalankan

- Notebook dibangun, semua cell kode lolos `ast.parse`, `ruff --select F` bersih, tanpa docstring, impor hanya di cell Libraries.
- Dry run lokal Bagian 1 sampai 7 dan cell pandas Bagian 8 (33 detik; cell yang melatih model dilewati). Hasilnya: pemeriksaan brute-force referensi klaster lolos (80 baris), semua fitur relatif finite, KS sama dengan `eda/73`, lensa pasar sepi 51 dari 126 film (referensi 151 busy, 102 quiet, 72 uji).
- Di notebook, offset tolok ukur relatif klaster adalah -0,44, bukan -0,21 seperti `eda/68`, karena definisi referensinya berbeda (tps D1 sampai D3 semua film terlihat vs D1 sampai D2 pasangan proxy). Narasi notebook memakai angka notebook. Sisa selisih ada di dua desil terbawah, jadi sebagian pergeseran mungkin nyata. Karena itu koreksi λa tetap diuji.
- Pemeriksaan crash jalur model pada subset kecil data nyata (14 film, 5 pohon, TabPFN 1 anggota, TabM 1 epoch, CPU): lihat bagian status di bawah.

## Batasan yang harus dibaca bersama hasil Kaggle

- Lensa pasar sepi masih lebih lunak dari periode uji (referensi quiet 102 vs uji 72), sehingga perbaikan di lensa ini adalah batas bawah.
- Proxy periode uji menilai horizon 1 hari (D3), bukan D4 sampai D10.
- `eda/71` bukan bukti tunggal, karena kovariat lain (kalender, jadwal, cakupan) juga berbeda antara baris uji dan train.
- Bila R1 lolos, perbedaan grouped CV kemungkinan kecil atau sedikit memburuk. Itu memang diperkirakan, bukan tanda gagal.
- Tidak ada klaim skor public atau private. Selisih validasi di bawah sekitar 0,002 adalah noise bootstrap per film.
- TabPFN-3.5 Fast adalah model 8 layer, bukan versi 24 layer yang dipakai v8. Performanya belum diketahui sampai run Kaggle.

## Hasil run Kaggle v13 (direview 8 Oktober 2026, `eda/74_v13_review.py`)

- **Hipotesis level relatif tidak terbukti di model.** R1 gagal 3 dari 4 syarat dan notebook kembali ke B1. Proxy periode uji: MAE 0,4763 vs 0,4759 (bias memang turun 0,055 ke 0,020, tetapi MAE sama). Lensa pasar sepi: 0,3224 vs 0,3119 (lebih buruk). Temporal: 0,3089 vs 0,3058 (September +0,010). Grouped TW justru membaik (0,3274 vs 0,3319), kebalikan dari prediksi lookup. Yang terbukti hanya sisi pencopotan: klasifier relatif lebih terkalibrasi di periode uji (log-loss 0,206 vs 0,252, a = -0,32 vs -1,32).
- **Final:** B1 + LightGBM 0,2 / TabPFN-3.5 Fast FP16 0,5 (λ = 0,5) / TabM 0,3. Bobot 198,4 MB desimal (189,2 MiB).
- **Pada baris dan bobot yang sama:** TW v8 0,32513, v10 0,32468, v12 0,32572, v13 0,32673. v13 - v12 +0,0010 [-0,0026; +0,0046], v13 - v8 +0,0016 [-0,0014; +0,0046]. Tidak ada beda yang signifikan.
- **Lensa lain di notebook:** temporal 0,2994 (v12 0,3019), lensa pasar sepi 0,3050 (LightGBM saja 0,3119).
- **Prediksi uji v13 paling dekat ke v8** (perubahan absolut rata-rata 0,038 vs 0,048 untuk v12), dengan profil horizon yang mirip v8. Baris Lebaran identik dengan v8 dan v12.
