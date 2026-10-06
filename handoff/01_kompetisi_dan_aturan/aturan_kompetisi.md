# Aturan kompetisi (ringkasan `docs/context.md`)

## Tugas

- Untuk setiap pasangan (film, klaster bioskop), prediksi `total_ticket` harian **D4 sampai D10** dari
  penjualan **D1 sampai D3**.
- D1-D3 adalah tiga tanggal berturut-turut yang sama untuk semua klaster satu film. "D1 tidak selalu tanggal
  pertama film punya transaksi" (ada pratayang).
- Hari tanpa transaksi bernilai 0. Setiap pasangan di `test.csv` punya tepat 7 baris (D4-D10).

## Metrik: MASE

- `s_p = max(mean(y_D1..D3), 1)` per pasangan. MASE = rata-rata `|y - yhat| / s_p` atas semua baris.
- Setara dengan MAE pada rasio `r = y/s`, sehingga prediksi optimal adalah **median bersyarat**.
- Komposisi baris uji, misalnya porsi pasangan kecil, langsung menentukan skor.

## Data resmi

| Berkas | Baris | Isi |
| :--- | :--- | :--- |
| `train.csv` | 138.959 | Transaksi harian 1 Apr - 30 Sep 2025 (`date_show`, `cinema_ids`, `city_name`, `movie_title`, `total_ticket`, `occupation_rate`, `total_show`) |
| `test_history.csv` | 32.323 | Transaksi D1-D3 dari 163 judul uji (Okt 2025 - Mar 2026) |
| `test.csv` | 72.611 | 10.373 pasangan x 7 hari, untuk 160 judul / 140 judul dasar dan 121 klaster di 69 kota |
| `movies.csv` | 397 | `original_title`, `age_rating`, `genre`, `producer`, `director`, `writer`, `casts` |
| `holidays.csv` | 366 | Kalender hari, libur nasional |
| `ticket_prices.csv` | 207 | Harga tiket per kota x Weekday/Friday/Weekend |

## Aturan penting

- **Data eksternal:**
  - Hanya boleh bila informasi atau versinya publik **paling lambat 30 September 2025**, dan tanggal itu harus
    bisa dibuktikan.
  - Data yang terbit atau diperbarui setelah tanggal itu tidak boleh, walaupun membahas kejadian lama.
  - Panitia bisa meminta URL, penerbit, tanggal, dan cara penggunaan.
- **Pretrained model publik diperbolehkan**, misalnya dari HuggingFace. Aturan tidak menyatakan eksplisit
  apakah batas tanggal juga berlaku untuk checkpoint; lihat keputusan user.
- **Dilarang:** AutoML; LLM untuk inference; AI API calling untuk inference; "AI Agent dalam kompetisi"
  (cakupan belum jelas). LLM boleh membantu menulis kode.
- **Notebook:**
  - Cell pertama berisi `%pip install` dengan versi spesifik dan `-q`.
  - Seed/`random_state` di semua proses acak.
  - Alur runtut: Data Acquisition, EDA, Preprocessing, Cleaning, Feature Engineering, Modeling, Evaluation.
  - Narasi dan visualisasi di tiap langkah.
  - Sumber dan cara pakai pretrained dijelaskan.
- **Submisi akhir:**
  - Satu ZIP `[Nama Tim].zip` berisi `[Nama Tim].ipynb` dan berkas bobot `[Nama Tim].pkl/.pt/...`.
  - **Bobot model maksimal 200 MB.**
- **Penilaian penyisihan:** 90% private leaderboard, 10% kualitas dokumentasi notebook.
- **Reproduksi:** 10 kandidat finalis di-re-run oleh panitia. Selisih signifikan dengan private LB berarti
  diskualifikasi.
- **Batas submisi Kaggle:** 3 per hari.

## Jadwal

| Tahap | Tanggal |
| :--- | :--- |
| Penyisihan Kaggle | 28 Sep - 12 Okt 2026 |
| Pengumpulan notebook + model | 13 Okt 2026 (23:59 WIB) |
| Pengumuman finalis (Top 10) | 23 Okt 2026 |
| TM final | 25 Okt 2026 |
| Batas file presentasi | 31 Okt 2026 |
| Final offline di FMIPA UGM | 1 Nov 2026 |

## Perkembangan Top 1 public yang dilaporkan user

0,41380, lalu 0,39525, 0,37453, 0,34972, dan **0,34456** (terakhir).
