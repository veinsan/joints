# Pipeline model (bentuk terakhir: `notebooks/build_v12.py`)

## Fitur (Section 7, fungsi `build`; dipakai identik untuk train-sim, rilis terbatas, dan uji)

1. **Bentuk kurva pasangan:** `log_s`, `p1-p3` (y_i/scale), `n_hist`, `sh_trend` (show D3 vs max D1-D2),
   `share` (pangsa pasangan dalam film), `p3_rel`, `p1_rel`.
2. **Agregat film nasional:** `fnc1-3` (cakupan klaster), `fp1-3`, `f_logT`, `f_nc_trend`, `fmt`, `fmt_share`,
   `d1_dow`, `rating`, flag 9 genre, `n_genre`, `n_cast`.
3. **Kalender target:**
   - `h`, `dow`, `t_hol`, `t_school`, `t_ramadan`, `cal_mult`, `hol_in_hist`;
   - kalender sekolah Jawa Barat untuk 12 kota Jabar sejak v10;
   - libur nasional dan cuti bersama SKB 2025/2026;
   - uplift libur dimatikan di dalam Ramadan, karena cuti bersama akhir Ramadan adalah masa mudik.
4. **Jadwal rilis film lain:** `comp_n`, `f_share_cohort`, `n_comp_open`.
5. **Kompetisi klaster pada tanggal target** (film lain yang sedang di jendela D1-D3 di klaster yang sama):
   `c_new_n`, `c_new_sh`, `c_new_sh_rel`, `c_new_sh_vs_own`, `c_new_tx_vs_own`. **Hanya dipakai di klasifier
   nol.**
6. **Klaster dan kota:** `cin_size`, `cin_nfilms`, `cin_new`, `price_wkd`, `price_prem`. Statistik klaster
   fold-local sejak v10.
7. Kandidat yang **tidak dipakai** (gagal adopsi): fitur kalender input `cal_p1-3`/`cal_log31` (A3), metadata
   resmi, prediksi show dan tiket per show, panel bioskop.

`BASE_FEATS` = 47 fitur. `ZFEATS` = `BASE_FEATS` + 5 fitur kompetisi.

## Kalender struktural

- Profil hari: Senin 0,811, Selasa 0,789, Rabu 0,811, Kamis 0,783, Jumat 0,848, Sabtu 1,290, Minggu 1,282.
- Libur di hari kerja dinaikkan ke level 1,29 (seperti Sabtu).
- Libur sekolah: hari kerja x1,15.
- Ramadan 19 Feb - 20 Mar 2026: faktor 1,0. Penurunan volume film Ramadan adalah efek seleksi, bukan efek
  kalender (`eda/04`).

## LightGBM: L1 + hurdle (komponen kontrol)

- **L1:** 5 seed, lr 0,03, 63 daun, 800 pohon, feature_fraction 0,7, bagging 0,8.
- **Klasifier nol** `p0 = P(r = 0)` dengan `ZFEATS`: 31 daun, 300 pohon, lr 0,05.
- **19 regresi kuantil** τ = 0,05-0,95 pada baris positif.
- **Median campuran:** p0 digeser `logit p0' = logit p0 + λ a` (a = -1,32 dari proxy, λ = 0,5).
  - Bila p0' ≥ 0,5, prediksi 0.
  - Selain itu, kuantil ke-(0,5 - p0')/(1 - p0') dari bagian positif.
- **Mix** = 0,25 L1 + 0,75 hurdle. λ dan bobot hurdle dibekukan sejak v10.

## Komponen kuantil (v12)

Semua mengeluarkan 49 kuantil r (τ = 0,02-0,98). p0 dibaca sebagai porsi kuantil di bawah `ZERO_EPS = 0,02`,
lalu median campuran yang sama dipakai dengan **λ dipilih per model** (v12: keduanya λ = 0).

- **TabPFN-2.5 Quantiles:** `Prior-Labs/tabpfn_2_5`, `tabpfn-v2.5-regressor-v2.5_quantiles.ckpt`, rev
  `6c45f3a6`, SHA-256 `6dd4dbcd…`, 40,8 MB, `tabpfn==9.0.0`, `ModelVersion.V2_5`. Satu model per horizon
  (tanpa `h`), 8 anggota.
- **TabM:** `tabm==0.0.3`, `rtdl_num_embeddings==0.0.12`.
  - Arsitektur: k = 32, piecewise-linear embeddings 48 bin versi B, d_emb 16.
  - Input diisi median lalu `QuantileTransformer`; semuanya di-fit **hanya pada inner-training**. Fitur konstan
    (`t_ramadan`, `cin_new`) dibuang.
  - Pinball loss 49 kuantil dibobot `cal_mult`, dirata-rata atas k submodel.
  - Optimizer AdamW lr 2e-3, wd 3e-4, batch 256, maks 40 epoch, patience 5. Early stopping pada 10% film
    bagian latih.
  - Satu model untuk semua horizon (`h` sebagai fitur).

## Blending

- Grid simpleks 0,1 atas komponen.
- Kombinasi lolos bila, terhadap LightGBM saja:
  - TW turun ≥ 0,001;
  - temporal tidak memburuk;
  - κ tidak naik > 0,002.
- Dipilih **minimax regret** atas tiga lensa (TW, temporal, κ).
- Selalu dicetak juga: pilihan label asli saja dan referensi bebas κ.

## Pascaproses

- **Analog Lebaran 2025** (`eda/28`) untuk 4.417 baris (7 judul slate, 21-27 Mar 2026):
  - prediksi = scale x ρ (0,5) x akar(rasio klaster x rasio nasional), dengan rasio = total Lebaran 2025 hari
    ke-k / rata-rata D1-D3 slate;
  - hari 1 = 0,75 x hari 2;
  - `eda/39` (inferensi dari LB nyata v4 ke v5): rescale 0,8-1,25 hanya mengubah < 0,003, jadi ρ dipertahankan.
- Prediksi di-clip ≥ 0.

## Paket bobot dan reproduksi

- `model_weights.pkl` berisi:
  - LightGBM, terkompresi zlib;
  - state TabM, QuantileTransformer, median, daftar fitur, dan bin;
  - checkpoint TabPFN (bila dipakai);
  - setting, bobot blend, λ per model, revisi, dan SHA-256.
- Pemeriksaan ≤ 200 MB dilakukan sebelum inferensi panjang.
- Checkpoint dicari dulu di input Kaggle, lalu di dalam pkl, baru diunduh dari HF pada revisi terkunci. Selalu
  diverifikasi SHA-256.
- `STRICT = True`: kegagalan komponen menghentikan notebook. Prediksi tidak diubah diam-diam.
- `FM_MODE`: `'tabpfn25q'` (default), `'tabpfn_v2'` (rev `213f8e38`, 11 Juni 2025, SHA `2ab5a07d…`, 44,4 MB,
  pilihan konservatif sebelum cutoff), atau `'none'`.
- Runtime v12 di T4: LightGBM ladder ~20 menit, TabPFN-2.5 CV + temporal 985 detik, TabM 193 detik, A/B
  kalender ~21 menit, fit final dan inferensi 943 detik.
