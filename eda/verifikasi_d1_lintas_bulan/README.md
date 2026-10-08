# Audit D1 rilis luas lintas bulan

## Pertanyaan

Apakah D1 hasil rekonstruksi train cocok dengan tanggal rilis luas publik di lebih banyak film dan semua bulan train? Bagaimana menangani tanggal sumber yang saling bertentangan?

## Metode

Jalankan dari root:

```powershell
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/verifikasi_d1_lintas_bulan/candidates.py
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/verifikasi_d1_lintas_bulan/audit_sources.py
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/verifikasi_d1_lintas_bulan/tabayyun_timeline.py
```

Tanggal dari sumber publik dicatat manual beserta URL dan lokasi klaim pada `verified_sources.csv`. Judul tersebut kemudian dicocokkan dengan kandidat D1 dari [audit sebelumnya](../verifikasi_tanggal_rilis/README.md). Audit memakai data train saja, tanpa fitting model.

## Temuan

- Empat belas judul baru mencakup April sampai September 2025. Pada 12 judul, semua klaim tanggal yang dicatat cocok dengan D1 v2. Dua judul punya sumber saling bertentangan, tetapi masing-masing memiliki dukungan independen untuk D1 v2. Dua belas dari 14 punya transaksi lebih dari tujuh hari sebelum D1; median jeda 12,5 hari, maksimum 32 hari. Pada lima judul audit sebelumnya, D1 juga cocok. Total 17 dari 19 judul unik yang diaudit cocok tanpa konflik sumber yang ditemukan.
- [Tabayyun pada halaman LSF](https://lsf.go.id/film/tabayyun/407) memuat dua tanggal: `Tanggal Tayang` 30 April dan narasi `tayang mulai` 8 Mei. [Indonesian Film Center](https://www.indonesianfilmcenter.com/filminfo/detail/17149/tabayyun) mencantumkan 8 Mei. [Laporan gala premiere](https://journeyofindonesia.com/entertainment/film/tabayyun-sajikan-kisah-cinta-dan-luka-masa-lalu-dan-ending-sesuai-harapan/) menyebut acara 30 April dan rilis serentak 8 Mei.
- Data Tabayyun pada 30 April berisi 12 show di 2 `cinema_ids` dan 2.199 tiket. Pada 8 Mei terdapat 600 show di 76 `cinema_ids` dan 11.623 tiket. Sebelum 8 Mei hanya ada 20 show di sembilan tanggal, jauh lebih kecil daripada hari rilis luas. Pola ini mendukung D1 v2 = 8 Mei dan menunjukkan bahaya memilih tanggal pertama, gala, atau metadata terstruktur tunggal tanpa memeriksa cakupan jaringan.
- [LSF Jodoh 3 Bujang bahasa Indonesia](https://lsf.go.id/film/jodoh-3-bujang/195) menyebut 26 Juni 2025, sedangkan [versi Inggrisnya](https://lsf.go.id/en/film/jodoh-3-bujang/515) menulis 26 Juni 2023 meski tahun produksi dan STLS 2025. Ini sangat mungkin salah ketik; kedua klaim tetap tercatat dalam audit.

Data rinci: `output/all_candidate_titles.csv`, `output/source_claim_audit.csv`, `output/film_summary.csv`, `output/tabayyun_daily.csv`.

## Implikasi

Untuk membentuk window train, pakai rilis luas, bukan transaksi pertama atau tanggal gala. Pada judul dengan klaim rilis yang bertentangan, cek perubahan jumlah show dan klaster bioskop serta sumber independen. Sumber rilis historis di sini dipakai untuk audit retrospektif; bila kelak dijadikan fitur prediksi, tanggal publikasi dan batas informasi 30 September 2025 harus diverifikasi per sumber.

Audit ini dipilih secara sengaja, bukan sampel acak dari seluruh film. Hasil 17/19 tanpa konflik tidak boleh ditafsirkan sebagai akurasi populasi. Tidak ada eksperimen modelling.
