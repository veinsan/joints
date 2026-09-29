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


---

# v3: Mengapa CV 0,30 tetapi public LB 0,45 (script 12 sampai 19)

## 12_gap_analysis.py: jarak CV -> LB sebagian besar adalah komposisi

| Estimasi | LightGBM | v2 (TabPFN-3.5) |
| :--- | :--- | :--- |
| OOF biasa | 0,328 | 0,300 |
| OOF dibobot ke komposisi skala uji | 0,414 | 0,377 |
| Density-ratio adversarial | 0,398 | - |
| Public LB | ~0,456 (v1) | 0,4506 |

Uji memuat 13% baris ber-skala <= 20 (train-sim 3%); galat bucket itu 0,6 sampai 2,6. Metrik keputusan v3 =
**TW-MASE** (`src/evaluate.py`).

## 13_proxy_test_period.py: validasi berlabel di dalam periode uji

Tugas proxy D1,D2 -> D3 bisa dihitung di `test_history`. Galat per bucket skala train CV vs periode uji
hampir sama; setelah dibobot ke skala uji, CV (0,498) bahkan sedikit di atas periode uji (0,474).
Level shift kecil: k* = 1,06 di uji (Februari 1,16), gain hanya 0,002.

## 14_small_pairs.py: pasangan kecil

- Bobot komposisi uji di training: memperburuk (0,4094 -> 0,4136).
- Film rilis terbatas sebagai data latih tambahan: membantu (-0,0035), konsisten di 2 seed fold.
- Thinning binomial: **cek prepro gagal** - zero-rate pasangan kecil hasil thinning 0,52 vs asli 0,76
  (thinning meniru penonton sedikit, bukan bioskop mencopot film). TW-MASE memburuk (0,4171).
- Pasangan s <= 20: 76% target nol, model ~ prediksi nol; galat bucket ini hampir tak tereduksi.

## 15/16: fitur tambahan

Kompetisi per klaster pada tanggal target (film baru di jendela D1-D3 di klaster yang sama, data resmi):
zero-rate naik monoton 0,13 -> 0,73 menurut kuintilnya, distribusi train/uji sama. Tetapi pada ablation
2 seed fold, semua grup (jadwal program, first-day, pasar relatif, kompetisi) berada di dalam noise
(~0,0035). LightGBM jenuh di sisi fitur.

## 17/18: arti "kecil" bergeser

Pada skala absolut yang sama, zero-rate D3 di periode uji jauh lebih kecil (20-50: 0,39 train vs 0,11 uji).
Di train, kecil = film gagal; di uji, kecil = pasar sepi. Skala relatif pasar menyejajarkan distribusi
(`log_s` bergeser 0,67, `pair_vs_mkt` 0,14). Bukti validasi bertentangan: proxy periode uji memilih fitur
"both" (0,467 vs 0,4735), tetapi GroupKFold (+0,0045) dan transfer ramai->sepi dalam train (+0,012)
memilih "abs". Bulan sepi train tidak cukup sepi (zero-rate hanya turun 0,78 -> 0,72). Keputusan: default
`abs`, varian `both` diuji A/B di LB.

## 19_segment_stakes.py: segmen kalender tanpa padanan di train (30% baris)

Sisa jarak ~0,07 membutuhkan level asli 2 sampai 3x dari asumsi model di Ramadan dan/atau Lebaran
(k=3: Ramadan +0,066, Lebaran +0,024). v2 memprediksi ~0 untuk 54% baris Ramadan. Tidak ada probing
leaderboard: asumsi segmen harus diputuskan dari validasi dan logika domain di dalam notebook.

## Model pretrained: kepatuhan batas 30 Sep 2025

| Model | Revisi / versi | Tanggal | Status |
| :--- | :--- | :--- | :--- |
| TabPFN v2 reg | HF `213f8e38`, `tabpfn==2.1.4` | 11 Jun 2025 / 11 Sep 2025 | dipakai v3 |
| LimiX-16M | HF `4fd2dbad` | 1 Sep 2025 | patuh, belum diuji |
| TabDPT 1.1 | HF `514eadca`, `tabdpt==1.1.5` | Agu/Sep 2025 | gagal jalan (faiss) |
| TabPFN-2.5 | - | 6 Nov 2025 | melanggar |
| TabPFN-3.5 (dipakai v2) | `tabpfn-v3.5-20260909` | Sep 2026 | **melanggar** |

---

# v4: Sumber offset LB tetap ~0,07 (script 20 sampai 25)

Titik LB: v1 0,45625; v2 0,45061 (TW 0,377); v3 0,45731 (TW 0,388). TW-MASE memberi peringkat yang sama
dengan LB tetapi offset ~0,07 muncul di kedua model -> kesalahan sistematis yang diwarisi dari train.

## 20_pull_policy_shift.py: bioskop di periode uji lebih jarang mencopot film

| tiket/show D1-D2 | berhenti di D3 (train) | berhenti di D3 (uji) |
| :--- | :--- | :--- |
| <= 7,8 | 0,50 | 0,22 |
| 7,8-14,8 | 0,26 | 0,06 |
| 14,8-25 | 0,085 | 0,03 |

Bukan musim: November/Desember 2025 punya level pasar setara bulan train (153/259 tiket per klaster-hari)
tetapi zero-rate D3-nya tetap rendah. Bukan data bolong: pola kehadiran D1-D3 train vs uji mirip
(111: 78% vs 80%), lubang di luar outage Juni median 0,25%.

## 21_pull_odds_correction.py: besar pergeseran

Rekalibrasi logit P(D3 = 0) di periode uji: a = -1,27 (odds x 0,28), sd antar split-half 0,075,
konsisten per bulan. Di D3 sendiri koreksi tidak menurunkan MASE (nol hanya 8%).

## 22_d3_propensity_transfer.py: jembatan D3 -> D4-D10 lemah

Di train, kecenderungan mencopot di D3 per klaster x bulan hampir tidak memprediksi pencopotan D4-D10
(korelasi 0,13, slope 0,02). Pergeseran D3 tidak bisa diasumsikan berlaku penuh di horizon target.
Juga dicek: hipotesis "pencopotan karena kapasitas layar" tidak didukung (penurunan terjadi dengan atau
tanpa film baru di klaster).

## 23_holiday_calibration.py: kalender libur tidak meremehkan

Grid HOL_LEVEL x SCHOOL_WD: TW-MASE datar 0,4062-0,4108 (noise); bias di baris target hari libur ~0.
Parameter kalender tetap.

## 24_pull_shift_decision.py: model hurdle + analisis keputusan

Hurdle (klasifier p0 + 19 regresi kuantil, median campuran) vs L1, TW-MASE train OOF:
L1 0,4087 -> hurdle 0,3874 (-0,021, jauh di atas noise 0,0035). Dunia bergeser (pencopotan dibatalkan
sesuai a): L1 memburuk ke 0,4479, hurdle lambda=1 0,4211.

## 25_hurdle_lambda_proxy.py: validasi di periode uji membalik

Proxy D3 periode uji (held-out per film, 10 split): hurdle murni kalah dari L1 di 10/10 split
(0,481-0,485 vs 0,473). Penyebab: hurdle lebih baik pada baris yang laku (0,479 vs 0,492) tetapi jauh
lebih buruk pada baris yang ternyata nol (0,491 vs 0,275). Blend menang di kedua sisi:

| proxy D3 | L1 | blend 50% hurdle lambda 0 | lambda 0,5 | lambda 1 |
| :--- | :--- | :--- | :--- | :--- |
| periode uji held-out | 0,4734 | 0,4728 | 0,4701 | 0,4680 |
| train-sim OOF | 0,4219 | 0,4172 | - | - |

Tugas utama (TW-MASE), baris lambda x kolom bobot hurdle -> minimax regret atas dunia nyata, dunia
bergeser, dan proxy periode uji memilih **bobot hurdle 0,75, lambda 0,5** (regret maks ~0,008).

## Model pretrained lain yang dicek (patuh batas 30 Sep 2025)

- LimiX-16M: output point prediction, butuh flash-attn 2.8 (tidak mendukung T4), retrieval > RTX 4090 -> tidak layak.
- TabDPT 1.1.5: gagal dengan faiss terbaru.
- GPU lokal Intel Arc: PyTorch XPU butuh `intel-compute-runtime` + `level-zero-loader` (sudo), belum terpasang.
