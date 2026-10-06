# Timeline versi v1 sampai v12

## Istilah metrik

| Istilah | Arti |
| :--- | :--- |
| TW (bucket) | OOF dibobot ke komposisi bucket skala uji. Metrik lama sampai v7. |
| TW (bfd) | Bobot bucket skala x hari pertama laku. Metrik yang benar sejak v8. |
| κ / dunia κ | Dunia sintetis tempat sebagian target nol "dibatalkan" sesuai pergeseran kebijakan pencopotan periode uji. Lihat `03_data_dan_pipeline/validasi_dan_metrik.md`. |
| Public | Skor public LB yang dilaporkan user. |

## Ringkasan skor

| Versi | Komponen utama | Validasi (saat itu) | TW (bfd), dihitung ulang dari `results/vN/oof.csv` | Public |
| :--- | :--- | :--- | :--- | :--- |
| v1 | LightGBM + CatBoost + XGBoost + TabPFN v2 | OOF ~0,33 | - | **0,45625** |
| v2 | TabPFN-3.5 + GBDT | TW (bucket) 0,377 | - | **0,45061** |
| v3 | TabPFN v2 per horizon (bobot 1,0) | TW (bucket) 0,3879 | 0,3274 | **0,45731** |
| v4 | Hurdle LightGBM + pull shift + TabPFN v2 (0,5) | TW 0,3911 / κ 0,4082 | 0,3318 | **0,44809** |
| v5 | v4 + fitur kompetisi klaster + analog Lebaran | TW 0,3894 / κ 0,4066 | 0,3307 | **0,40865** |
| v6 | Hurdle LightGBM + TabPFN-3.5 (0,5/0,5) | TW 0,3846 / κ 0,4009 | 0,3258 | **0,40119** |
| v7 | v6 + TabPFN-3.5 fine-tuned (bobot 0) | sama dengan v6 | sama | tidak disubmit (identik v6) |
| v8 | v6 + bobot bfd + TabPFN 16 anggota (0,4/0,6) | TW (bfd) 0,3251 / κ 0,3384 | 0,3251 | **0,39991 (terbaik)** |
| v9 | LightGBM 0,6 + EXAONE 0,4 (TabPFN-2.6 bobot 0) | TW 0,3261 / κ 0,3401 | 0,3261 | tidak dilaporkan |
| v10 | Tangga ablation, statistik fold-local, kalender Jabar, EXAONE 0,7 | TW 0,3247 / κ 0,3408 | 0,3247 | **0,40823** |
| v11 | GBDT (LightGBM, CatBoost, XGBoost) + Causilo, final LightGBM saja | TW 0,3319 / κ 0,3410 | 0,3319 | tidak dilaporkan |
| v12 | LightGBM 0,5 + TabPFN-2.5 Quantiles 0,4 + TabM 0,1 | TW 0,3257 / temporal 0,3019 / κ 0,3401 | 0,3257 | **0,41060** |

## Detail per versi

### v1 (public 0,45625)

- Fondasi pipeline:
  - Rekonstruksi D1 dan simulasi aturan panitia (`src/common.py`).
  - Fitur relatif dan kalender (`src/features.py`).
  - Target `r = y / (s * c_mult)` dengan pengali kalender struktural.
- Model LightGBM, CatBoost, dan XGBoost (objective MAE) ditambah TabPFN v2 (`tabpfn==2.0.9`). Blend dipilih
  dari kurva temporal.
- Validasi: GroupKFold per film plus split temporal. OOF ~0,33 vs LB 0,456, karena komposisi uji belum
  disadari.

### v2 (public 0,45061)

- TabPFN-3.5 (`tabpfn==9.0.0`) menggantikan TabPFN v2.
- OOF 0,300, TW 0,377, tetapi jarak ke LB tetap ~0,07.

### v3 (public 0,45731, lebih buruk)

- Saat itu kami mengira batas tanggal berlaku untuk pretrained, sehingga kembali ke TabPFN v2 (revisi Juni
  2025) per horizon.
- Metrik keputusan baru **TW-MASE** (`src/evaluate.py`).
- Film rilis terbatas (36 film, 1.435 baris) dipakai sebagai data latih tambahan.
- Blend memilih TabPFN 100%.

### v4 (public 0,44809)

- **Model hurdle:** klasifier `p0 = P(r = 0)` ditambah 19 regresi kuantil bagian positif, lalu median campuran.
- **Pergeseran kebijakan pencopotan.** Dari proxy berlabel periode uji (D1, D2 ke D3): pada tiket/show yang
  sama, bioskop periode uji 2-4x lebih jarang mencopot film. Odds x0,27, logit a = -1,32.
  - Diterapkan sebagai `λ * a` dengan λ = 0,5.
  - Bobot hurdle 0,75 (sisanya L1).
  - Keduanya dipilih dengan minimax regret atas dunia nyata, dunia bergeser, dan proxy.
- Blend TabPFN v2 0,5.

### v5 (public 0,40865; lompatan terbesar -0,039)

- Fitur kompetisi per klaster pada tanggal target (film lain di jendela D1-D3-nya), hanya di klasifier.
- **Analog Lebaran 2025** untuk 7 judul slate Lebaran 2026 (D4-D10 = Lebaran hari 1-7, 6,1% baris uji):
  - total klaster Lebaran 2025 (teramati di train sejak 1 Apr 2025) x ρ = 0,5;
  - Lebaran hari 1 = 0,75 x hari 2;
  - rata-rata geometrik rasio klaster dan rasio nasional.
- Dengan TW (bucket) di dunia κ = 0,5, offset v5 ke LB hanya +0,002. Ini kemudian ternyata **kebetulan**
  (lihat v8).

### v6 (public 0,40119)

- User memutuskan batas tanggal hanya untuk data, sehingga kembali ke TabPFN-3.5 (8 anggota, per horizon).
- Blend median 0,5/0,5. TabPFN-3.5 OOF TW (bucket) 0,3815.

### v7 (tidak disubmit)

- Ditambah TabPFN-3.5 fine-tuned (CRPS, lr 1e-5, 600 detik per fold): TW 0,3968 vs ICL 0,3840, korelasi galat
  0,994.
- Bobot 0, sehingga submisi **identik byte-per-byte** dengan v6.

### v8 (public 0,39991, terbaik)

- **Koreksi metrik validasi.** Bobot bucket saja memberi late starter (laku pertama di D3) porsi 4,2%, padahal
  di uji 1,8%. Bobot baru = bucket skala x hari pertama laku.
  - Dengan bobot yang benar, **semua versi berjarak 0,062-0,067 dari LB**.
  - Artinya kecocokan "κ = 0,5 cocok dengan LB" di v5/v6 adalah dua kesalahan yang saling meniadakan.
- TabPFN-3.5 16 anggota. Blend 0,4 LightGBM / 0,6 TabPFN.
- Masalah: checkpoint TabPFN-3.5 berukuran 876 MB.

### v9 (public tidak dilaporkan)

- Kepatuhan 200 MB: TabPFN-3.5 diganti TabPFN-2.6 (51,6 MB) dan **EXAONE-Tabular** (LG AI Research, 84,5 MB).
  - EXAONE butuh kelas turunan untuk membaca 999 kuantil.
  - Backend attention dialihkan dari FlashAttention, yang tidak ada di T4.
- Hasil:
  - EXAONE TW 0,3258, TabPFN-2.6 0,3298, LightGBM 0,3322.
  - Blend LightGBM 0,6 + EXAONE 0,4, dipilih di dunia κ yang bias ke LightGBM.
  - Bobot 109,6 MB dengan checkpoint tertanam.

### v10 (public 0,40823)

- Mengikuti audit 5 Okt (`docs/experiment_audit_2026-10-05.md`):
  - statistik klaster **fold-local** (kebocoran B0-B1 terukur: TW +0,0007);
  - backtest temporal Juli/Agustus/September;
  - tangga ablation dengan aturan adopsi tertulis.
- Kandidat:
  - A1: fitur prediksi perubahan show dan tiket per show, nested cross-fitting;
  - A2: metadata resmi (`has_number`, `has_subtitle`, `is_reissue`, `cast_overlap`).
  - Keduanya **gagal** (A2 gagal karena September +0,0035).
- Kalender sekolah **Jawa Barat** (22% baris uji: libur 29 Des - 10 Jan) diterapkan sebagai koreksi fakta.
- Aturan baru: pemilihan blend terbaik yang lolos syarat, verifikasi SHA-256, fail loudly (`STRICT`).
- EXAONE dipilih dengan bobot 0,7 berdasarkan label asli saja, padahal κ terbaik di 0,4. Public memburuk.

### v11 (public tidak dilaporkan; submisi = LightGBM saja)

- Awalnya: kandidat fitur input kalender (A3) dan blend minimax regret 3 lensa.
- Lalu atas permintaan user: GBDT (LightGBM, CatBoost `MultiQuantile`, XGBoost `reg:quantileerror`) ditambah
  **Causilo** (nums-ai, klaim nomor 1 TabArena, 148,4 MB).
- Hasil:
  - A3: TW 0,3319 ke 0,3310, temporal 0,3058 ke 0,3027, tetapi gagal ambang 0,0015.
  - CatBoost 0,3367 dan XGBoost 0,3342, korelasi galat dengan LightGBM 0,993-0,997.
  - Causilo 0,3954 (lihat catatan di `05_eksperimen/model_yang_dicoba.md`).
  - Tidak ada blend yang lolos.

### v12 (public 0,41060)

- Mengikuti revisi Astra:
  - fitur B1 dibekukan;
  - urutan LightGBM (kontrol), **TabPFN-2.5 Quantiles** (40,8 MB), lalu **TabM**;
  - λ dipilih per model;
  - kalender diuji terakhir pada blend terpilih.
- Hasil:
  - TabPFN-2.5 TW 0,3274 (model tunggal terbaik), TabM 0,3354, LightGBM 0,3319.
  - **Keduanya memilih λ = 0**, karena pull shift memperburuk mereka.
  - Blend LightGBM 0,5 / TabM 0,1 / TabPFN 0,4, korelasi galat 0,984.
  - Kalender: temporal 0,3019 ke 0,2989 tetapi TW tidak berubah, sehingga tidak diadopsi.
- Dibanding v11: -0,0062, bootstrap per film [-0,0108; -0,0023], 84/126 film membaik.
- Public tetap memburuk (0,41060).

### Pola penting dari skor public

- Perubahan public mengikuti perubahan **model yang kuat di periode uji** (TabPFN-3.5), bukan perbaikan
  validasi:
  - v5 ke v6 (tambah TabPFN-3.5): public -0,0075, padahal validasi hanya memprediksi -0,003.
  - v8 ke v10/v12 (TabPFN-3.5 diganti model patuh 200 MB): public +0,008 sampai +0,011, padahal validasi
    datar.
- Lompatan besar hanya datang dari koreksi struktur: pull shift (v4) dan analog Lebaran (v5).
