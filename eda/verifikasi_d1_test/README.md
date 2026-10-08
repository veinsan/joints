# Audit D1 test terhadap tanggal tayang Indonesia

## Metode

Jalankan dari root:

```powershell
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/verifikasi_d1_test/candidates.py
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/verifikasi_d1_test/audit_sources.py
```

Script pertama menyusun D1 per judul dasar dari `test_history.csv`. Klaim sumber yang dipilih manual disimpan di `verified_sources.csv` dengan URL dan konteks. Audit ini retrospektif atas definisi tanggal; sumber dari setelah batas informasi 30 September 2025 tidak menjadi fitur model.

## Temuan

Delapan film dari tiap bulan Oktober 2025 sampai Maret 2026 diperiksa. Enam punya seluruh klaim tanggal cocok dengan D1 `test_history`. Dua menunjukkan perbedaan antara sumber tanggal umum dan tanggal tayang Indonesia:

| Film | D1 test | Sumber lokal | Klaim berbeda |
|---|---|---|---|
| Mercy | 21 Jan 2026 | [Cinema XXI](https://m.21cineplex.com/movies/26MERY), [trailer Cinema 21](https://www.youtube.com/watch?v=Z-i2BjOaizw) | [LSF](https://lsf.go.id/film/mercy/944): 23 Jan |
| Hoppers | 4 Mar 2026 | [Cinema XXI](https://m.21cineplex.com/id/movies/26HOPS), [trailer Walt Disney Studios Indonesia](https://www.youtube.com/watch?v=NLrq58WkCdE) | [Disney.id](https://www.disney.id/movies/hoppers): 6 Mar |

Sumber lokal yang cocok untuk film lain: [Tukar Takdir](https://movimax.co.id/article/TUKAR-TAKDIR), [Agak Laen: Menyala Pantiku!](https://www.youtube.com/watch?v=KGWkDj_vMOM), [Timur](https://lsf.go.id/film/timur/783), [Alas Roban](https://lsf.go.id/film/alas-roban/877), [Titip Bunda di Surga-Mu](https://m.21cineplex.com/movies/16TBDS), dan [Danur: The Last Chapter](https://lsf.go.id/film/danur-last-chapter/995).

Perbedaan dua hari tampak konsisten dengan tanggal umum/internasional yang berbeda dari rilis Indonesia, tetapi penyebab metadata tersebut belum dikonfirmasi. Untuk pembentukan train window dan validasi, gunakan tanggal rilis luas Indonesia. Pada test, D1 sudah tersedia dalam history dan tidak perlu diganti oleh tanggal web. Delapan film dipilih sengaja, sehingga proporsi cocok bukan estimasi akurasi seluruh film. Tidak ada eksperimen modelling.
