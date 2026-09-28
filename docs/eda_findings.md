# Temuan EDA dan Keputusan Preprocessing

Semua angka di bawah berasal dari script di `eda/` (jalankan dengan `.venv/bin/python eda/NN_*.py`,
gambar tersimpan di `outputs/eda/`). Modul bersama: `src/common.py` (load, rekonstruksi D1, simulasi,
MASE), `src/features.py` (fitur), `src/model.py` (LightGBM).

## 01_data_audit.py: integritas dan cara panitia membangun test

| Temuan | Angka | Keputusan |
| :--- | :--- | :--- |
| Data bersih | 0 NA, 0 duplikat (tanggal, klaster, film), 1 klaster = 1 kota | tidak perlu imputasi |
| **Aturan seleksi pasangan** | 10.373/10.373 pasangan test laku di D3; 1.450 pasangan yang terakhir laku di D1/D2 semuanya dibuang | simulasi train WAJIB memakai aturan "laku di D3" |
| D1 = tanggal rilis resmi | 10 film test sudah ada di train sebagai pratinjau (1 sampai 10 klaster) | D1 direkonstruksi dari lonjakan cakupan, bukan transaksi pertama |
| D1 hampir selalu Rabu/Kamis | Rabu 67, Kamis 65, Jumat 24, Sabtu 5, Selasa 2 | fitur `d1_dow` + normalisasi kalender |
| Klaster baru | 4 klaster test tidak ada di train (602 baris) | fitur klaster boleh NaN |

## 02_target_scale.py: apa yang sebenarnya dinilai MASE

- MASE = MAE pada rasio `r = y / s`. Median `r` = 0,42, rata-rata 0,61 (menceng kanan) sehingga
  **objective harus L1**; konstanta terbaik `k*s` dengan k = 0,42 (MASE 0,504), naive `s` = 0,731.
- 31% target nol; porsi nol naik dari 10% (D4) ke 54% (D10) karena film dicopot.
- Pasangan kecil (skala < 20) menyumbang galat per baris terbesar tetapi hanya 15% baris.

## 03_release_lifecycle.py: rekonstruksi D1 dan verifikasi terhadap test

- Aturan D1 "cakupan pertama >= 50% puncak" tanpa filter: cakupan D3 median 29 klaster vs test 70,
  Rabu/Kamis 71% vs 81%. **Secara teori benar, secara distribusi salah.**
- Penyebab: test hanya berisi **rilis luas**. 56 judul di `movies.csv` (Bollywood terbatas, konser
  K-pop, nobar) tidak ada di train maupun test; film 2D terkecil di test dibuka di 28 klaster.
- Setelah filter film dasar >= 25 klaster pada D1 dan D1 dihitung di level judul dasar (2D/3D/IMAX
  berbagi D1): cakupan 48 sampai 51, Rabu/Kamis 78%. Sisa selisih berasal dari musim (lihat 07).
- Kurva nasional median (/rata-rata D1 sampai D3): D4 0,84, D7 0,38, D10 0,17. Momentum D3/D1
  hanya berkorelasi 0,18 dengan retensi D4 sampai D10: prediksi tidak bisa sekadar ekstrapolasi tren.

## 04_calendar.py: efek kalender dan periode yang tidak ada di train

- Pengali hari: Sen-Jum 0,78 sampai 0,85, Sabtu 1,29, Minggu 1,28.
- Libur di hari kerja = level akhir pekan (Hari Buruh 1,85x, Waisak 1,88x, Kenaikan 1,98x hari yang
  sama); libur di akhir pekan hampir tidak menambah.
- Libur sekolah: hari kerja +15%.
- **30% baris test ada di periode tanpa padanan di train**: Ramadan 17%, Natal/Tahun Baru 7%, Lebaran 6%.
  Pohon tidak bisa ekstrapolasi, sehingga kalender dimasukkan sebagai **pengali target struktural**
  `c_mult = c(t) / mean c(D1..D3)`.
- Volume D1 sampai D3 film rilis Ramadan setengah dari film lain, TETAPI film yang D1 sampai D3-nya
  melintasi awal Ramadan (CRIME 101, BLADES OF THE GUARDIANS) tidak menunjukkan penurunan rasio.
  Kesimpulan: penurunan volume adalah efek seleksi (film besar menghindari Ramadan), bukan efek
  kalender, sehingga `RAMADAN_F = 1.0`.
- Perbaikan hasil verifikasi: pada versi pertama, cuti bersama 18 sampai 20 Maret 2026 (hari mudik di
  akhir Ramadan) dianggap "hari libur tinggi" sehingga prediksi minggu Lebaran justru turun. Uplift
  libur kini dimatikan di dalam Ramadan.

## 05_cinema_city.py: klaster

- Ukuran klaster timpang (p10 539, p90 5.345 tiket/hari); 5 klaster terbesar = 23% tiket.
- Residual klaster (r - median film) punya korelasi split-half 0,29, tetapi target encoding OOF
  memperburuk CV (lihat 09), jadi tidak dipakai.
- Harga tiket kota berkorelasi lemah dengan ukuran klaster (0,15); dipakai sebagai fitur ringan.

## 06_movie_meta.py: metadata film

- Retensi film (mean D4-10 / mean D1-3) median 0,43; Family 0,12, Romance 0,17, Adventure 0,78,
  IMAX 0,58. Sinyal ada tetapi kecil; dipakai sebagai flag genre dan rating.

## 07_preprocessing_check.py: verifikasi preprocessing

- Skala dari `build` identik dengan fungsi `hitung_skala` resmi; urutan `id` test terjaga.
- Adversarial validation **harus di-group per film**: split acak memberi AUC palsu 1,0 karena fitur
  level film konstan dalam satu film.
- AUC grouped: fitur level absolut 0,82, fitur relatif 0,66. Periode test adalah musim sepi
  (okupansi median 10% vs 22%, tiket/pertunjukan 18,6 vs 32,9) sehingga fitur level absolut dibuang
  kecuali `log_s` dan `f_logT`.

## Pembersihan data (ditemukan di 08)

- Outage pelaporan di train: 7, 9, 10, 16 Juni 2025 (1 sampai 67 klaster) dan 13 Juni hilang total.
  Zero-fill menciptakan nol palsu. Baris target di tanggal tersebut dibuang dan film yang D1 sampai
  D3-nya kena outage tidak dijadikan sampel.

## 08_baselines_cv.py / 09_model_improve.py: validasi dan eksperimen

| Varian | GroupKFold | Temporal |
| :--- | :--- | :--- |
| median r/c per (horizon, hari D1) | 0,422 | 0,388 |
| LightGBM r, semua fitur | 0,363 | 0,342 |
| LightGBM r/c, fitur relatif + log_s + f_logT (+ kalender resmi) | **0,358** | **0,328** |
| + augmentasi D1 +-1 hari | +0,007 (lebih buruk) | |
| + target encoding klaster OOF | +0,0025 (lebih buruk) | |
| CatBoost MAE / XGBoost MAE | 0,372 / 0,379 | |
| blend LGB+CB+XGB | 0,366 | |

Noise floor CV (3 penugasan fold berbeda): std 0,003.

## 10_tabpfn.py: model pretrained

TabPFN v2 (Prior-Labs/TabPFN-v2-reg, Nature 9 Jan 2025, lolos batas 30 Sep 2025). Foundation model
deret waktu (Chronos-Bolt, TimesFM, Moirai) tidak dipakai karena tiap seri hanya punya 3 titik.
TabPFN-2.5 (Nov 2025) tidak boleh dipakai karena terbit setelah batas.

| Model (split temporal, sub-sampel 5.000 baris) | MASE |
| :--- | :--- |
| LightGBM 3 seed | 0,3373 |
| TabPFN v2, median prediktif, konteks 5.000 | 0,3301 |
| 0,3 LightGBM + 0,7 TabPFN | **0,3291** |

Korelasi prediksi 0,92. Di notebook, konteks 10.000 di GPU T4 dan bobot blend dipilih otomatis dari
argmin kurva temporal.

## 11_lb_probe.py: probing leaderboard

Skor MASE aditif per baris; mengalikan prediksi satu segmen dengan k menggeser skor publik secara
terukur. Protokol 3 submisi/hari dan file probe ada di `outputs/probes/`.
Jalankan ulang dengan `python eda/11_lb_probe.py path/ke/submission.csv` agar probe berbasis submisi
notebook. Hasil k* dimasukkan ke `Settings.SEG_MULT`.
