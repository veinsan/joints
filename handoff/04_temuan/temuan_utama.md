# Temuan utama (yang paling menentukan)

## 1. Seleksi test = laku di D3; D1 = rilis resmi; hanya rilis luas

- Fakta ini dipakai di simulasi; tanpa itu, sampel latih tidak mirip test (`eda/01`, `eda/03`).
- Dengan aturan cakupan D1 tanpa filter, cakupan D3 median 29 klaster vs uji 70. Setelah filter ≥ 25 klaster
  dan D1 per judul dasar: 48-51.

## 2. MASE memberi hadiah pada median dan pasangan kecil

- MASE = MAE pada `r = y/s`, sehingga objective harus L1 atau kuantil.
- Baseline:
  - konstanta k·s dengan k = 0,42: MASE 0,504;
  - naive y = s: 0,731;
  - median per (h, hari D1): TW 0,433-0,495.
- Pasangan skala ≤ 20: **13% baris uji** (train 3%), MASE per baris 0,6-2,6, menyumbang sekitar 0,11 dari
  ~0,40.
- Late starter (laku pertama di D3): 1,8% massa uji tetapi sekitar 16% galat. Bimodal: hilang atau laku penuh
  di skala kecil. Spesialis tidak membantu (`eda/42`).

## 3. Pergeseran kebijakan pencopotan di periode uji (temuan v4)

- Pada tiket/show yang sama, porsi pasangan berhenti di D3 periode uji jauh lebih kecil:

  | Tiket per show D1-D2 | Berhenti di D3 (train) | Berhenti di D3 (uji) |
  | :--- | :--- | :--- |
  | ≤ 7,8 | 0,50 | 0,22 |
  | 7,8-14,8 | 0,26 | 0,06 |

- Logit odds a = -1,32 (odds x0,27), stabil per bulan, termasuk Nov/Des yang level pasarnya setara train.
- Bukan efek musim, bukan data bolong.
- Di train, propensitas D3 hampir tidak memprediksi pencopotan D4-D10 (korelasi 0,13), sehingga besar
  pergeseran di horizon target tidak pasti. λ = 0,5 dipilih dengan minimax regret.
- LB v3 ke v4 mendukung arahnya: public -0,0092. Tetapi v4 juga mengubah model, jadi λ tidak terisolasi.
- **Pergeseran ini membantu LightGBM tetapi merusak model kuantil lain:**

  | Model | λ = 0 | λ = 0,5 |
  | :--- | :--- | :--- |
  | TabPFN v2 | 0,3876 | 0,3951 |
  | TabPFN-2.6 | 0,3263 | 0,3298 |
  | TabPFN-2.5 Quantiles | 0,3274 | 0,3386 |
  | TabM | 0,3354 | 0,3398 |

  Jangan wariskan koreksi ini otomatis.

## 4. Lebaran (6,1% baris uji): satu-satunya lompatan besar dari koreksi struktur

- Slate Lebaran 2026 dirilis 18 Mar. D1-D3 ada di akhir Ramadan (okupansi 15-23%), D4-D10 adalah minggu
  Lebaran.
- Model statistik memprediksi rasio sekitar 0,79, setara okupansi 13% di minggu puncak tahunan. Itu tidak masuk
  akal secara fisik.
- Lebaran 2025 teramati di train: 608 ribu tiket/hari, 2,8x normal.
- Analog top-down dengan ρ = 0,5 menghasilkan rasio sekitar 1,77, dan public turun **-0,039** (v4 ke v5).
- `eda/39`: dengan mempertimbangkan noise, level sekarang konsisten dengan perubahan LB nyata. Rescale tidak
  berguna.

## 5. Komposisi validasi: bobot bucket saja menyesatkan

- Bobot per bucket skala melebih-bobotkan late starter (4,2% vs 1,8% di uji). Bobot yang benar adalah bucket x
  hari pertama laku.
- Dengan bobot yang benar, offset LB - validasi = 0,062-0,067 untuk **semua** versi (`eda/42`, `44`).
- Periode uji tidak lebih sulit di proxy D3 (`eda/43`), dan reweight adversarial hanya menurunkan offset ke
  sekitar 0,04.

## 6. Galat didominasi level film, dan level film tidak terprediksi dari D1-D3

- Lima film menyumbang 28% galat (LILO & STITCH, SAYAP SAYAP PATAH 2, SORE ISTRI DARI MASA DEPAN, UNTIL DAWN,
  WEAPONS). Semuanya under-predicted, karena breakout terus tumbuh.
- Oracle (diagnosis, bukan skor), dengan TW (bfd) v8 0,3251:

  | Bagian yang diketahui benar | TW |
  | :--- | :--- |
  | Multiplier film x horizon | 0,263 |
  | Multiplier klaster x tanggal | 0,202 |
  | Hanya tahu baris nol | 0,274 |

- `eda/38`: model level film pada 26 fitur film memberi R² CV 0,00. Koreksi apa pun memperburuk.
- `eda/52`: talent (casts/producer) berkorelasi lemah (casts ρ 0,27 fold-local). Untuk film yang lebih awal
  saja tidak signifikan, dan koreksi memperburuk.
- `eda/57`: **54,5% galat ada pada target positif yang tumbuh dibanding D3**.
  - Korelasi miss film dengan perubahan attendance masa depan 0,46, dengan perubahan show 0,33.
  - Dengan tren panel D1-D3 hanya 0,07.

## 7. Nol bukan berarti bioskop tutup atau film pasti ditarik permanen

- 99,5% target nol terjadi di klaster-tanggal yang masih punya transaksi film lain.
- Dari 7.988 pasangan dengan tujuh horizon lengkap, 4.400 pernah nol, dan 337 kembali positif (`eda/51`,
  `eda/54`).

## 8. Duplikasi dan kebocoran

- Tidak ada duplikat kunci. Signature (y1-y3, sh1-sh3) yang identik hanya 26/8.113 pasangan train dan 81/10.373
  uji. Wajar, jangan dihapus.
- Kebocoran sebenarnya dari **split**. Pada baris dan ukuran training yang sama:

  | Split | MASE |
  | :--- | :--- |
  | Film disjoint | 0,358 |
  | Bocor: klaster lain dari film yang sama | 0,260 |
  | Bocor: horizon lain dari pasangan yang sama | 0,252 |

- Statistik klaster global sempat bocor sedikit (TW +0,0007; korelasi peringkat global vs fold-local 0,9989).

## 9. Model: TabPFN-3.5 tampak lebih kuat di uji daripada yang ditunjukkan validasi

- v5 ke v6 (tambah TabPFN-3.5): public -0,0075, padahal validasi hanya -0,003.
- v8 ke v10/v12 (TabPFN-3.5 diganti EXAONE / TabPFN-2.5 + TabM, keduanya patuh 200 MB): public +0,008/+0,011,
  padahal validasi datar.
- Ini **satu-satunya pola yang konsisten** antara versi dan public. Biaya kepatuhan 200 MB kemungkinan sekitar
  0,01 di public. Belum terbukti, karena v10/v12 juga mengubah statistik klaster dan kalender Jabar. Kalender
  Jabar hanya mengubah 1,56% baris uji, jadi terlalu kecil untuk menjelaskannya.

## 10. Kalender

- Profil hari: Sabtu 1,29, Minggu 1,28, hari kerja 0,78-0,85.
- Libur hari kerja = level Sabtu.
- Libur sekolah: hari kerja +15%.
- Grid kalibrasi libur datar (`eda/23`).
- Amplitudo mingguan periode uji: estimasi tidak meyakinkan, karena decay dan DOW terkonfundasi di jendela 3
  hari (`eda/45`).
- Fitur input dinormalisasi kalender (`cal_p*`): konsisten sedikit membaik (grouped -0,0024; hurdle
  0,3340 ke 0,3328; v11 TW -0,0009, temporal -0,0031), tetapi selalu di bawah ambang adopsi.
- Kalender sekolah per provinsi: hanya **Jawa Barat** yang jendela hari kerjanya berbeda (libur 29 Des - 10
  Jan; sumber SE Kadisdik Jabar 14995, 20 Jun 2025).
