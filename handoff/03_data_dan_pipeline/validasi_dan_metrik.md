# Validasi dan metrik keputusan

## Fold

- 5 fold **dikelompokkan per judul dasar film** (`base_title`), dengan seed 2026 dan permutasi judul dasar.
- Semua format (2D/3D/IMAX) satu film berada di fold yang sama.
- **Jangan pernah** split acak per baris, pasangan, atau horizon. `eda/58` membuktikan split yang bocor
  menurunkan MASE semu dari 0,358 ke 0,252.

## Lensa validasi (sejak v10, dipakai sampai v12)

1. **TW-MASE grouped CV, label asli (lensa utama).**
   - Bobot baris = rasio porsi uji/latih per sel (bucket skala x hari pertama laku), dengan bucket skala
     0-5-20-50-100-200-500-inf.
   - Fungsi: `test_weights_fd` di `src/evaluate.py`, dan `TW` di notebook.
   - Bobot lama per bucket saja (`TW_OLD`) masih dicetak sebagai pembanding.
   - 28 baris uji (sel skala >500, laku pertama D2) tidak punya padanan di train dan diabaikan.
2. **Backtest temporal** untuk film rilis Juli, Agustus, September 2025.
   - Data latih hanya film yang D10-nya selesai sebelum awal bulan (62, 93, dan 130 film latih).
   - Statistik klaster hanya dari transaksi sebelum cutoff.
   - Kalender dan fitur jadwal tetap dibangun sekali, jadi ini **bukan pipeline kausal ketat**.
3. **Stabilitas per fold** dan **bootstrap per film** (2.000 kali, bobot tetap). Satuan informasi independen
   adalah film: 126 judul dasar, effective sample size sekitar 101.
4. **Dunia κ (uji stres).**
   - Setiap target nol "dibatalkan" dengan peluang 1 - p0'/p0, dengan logit p0' = logit p0 + κ a dan κ = 0,5.
   - Target yang dibatalkan diganti sampel dari distribusi positif OOF, rata-rata atas 3 seed.
   - Generator memakai **p0/Q LightGBM B1**, sehingga **tidak netral** bagi TabPFN/TabM. Karena itu selalu
     dicetak pilihan tanpa lensa κ.

## Aturan adopsi kandidat (ditulis sebelum hasil, sejak v10)

Kandidat diadopsi bila **semua** syarat terpenuhi terhadap baseline:
- TW turun ≥ 0,0015;
- rata-rata temporal turun;
- tidak ada bulan temporal yang naik lebih dari 0,003;
- membaik di ≥ 3 dari 5 fold;
- κ tidak naik lebih dari 0,002.

Bila selisih tipis atau kesimpulan berubah antar-lensa, pertahankan versi yang lebih sederhana.

## Proxy berlabel periode uji

- `test_history` memungkinkan tugas D1, D2 ke D3 **di dalam periode uji** (`eda/13`).
- Ini satu-satunya label periode uji. Dipakai untuk:
  - mengukur pergeseran kebijakan pencopotan (a = -1,32);
  - memeriksa apakah periode uji lebih sulit. Hasilnya tidak: nyata 0,474 vs harapan 0,479 (`eda/43`).
- Batasannya: proxy memilih pasangan yang laku di D2, bukan D3, dan horizonnya 1 hari, bukan D4-D10.

## Hubungan validasi dan LB (penting)

- Dengan bobot bucket saja dan κ = 0,5, offset v5/v6 ke LB hanya sekitar 0,002. Itu **kebetulan**: bobot
  bucket melebih-bobotkan late starter.
- Dengan bobot yang benar (bfd), semua versi v3-v6 berjarak **0,062-0,067** ke LB. Offset ini tidak
  bergantung model; dengan bobot adversarial sekitar 0,04.
  - Penjelasan yang paling masuk akal: film breakout yang tak terprediksi di set uji. Galat level film
    mendominasi dan tidak bisa ditaksir dari fitur D1-D3 (`eda/38`).
  - Alternatif: TabPFN-3.5 lebih tahan terhadap pergeseran periode uji.
- Validasi lokal **tidak memprediksi arah public** v8 ke v10 ke v12 (lokal datar, public memburuk). Gunakan
  validasi untuk keputusan, tetapi jangan klaim angka public.
