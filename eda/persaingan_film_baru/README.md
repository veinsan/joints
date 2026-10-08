# Persaingan film baru tidak menjelaskan ekor kecil D3

## Metode

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/persaingan_film_baru/opening_supply.py` dari root. Script menghitung jumlah film usia D1-D3 yang bertransaksi pada `cinema_ids` dan tanggal sama. D1 train mengikuti builder v2; D1 test berasal dari `test_history`. Untuk perbandingan pasangan, hanya film-klaster yang aktif D3 masuk. Output agregat, strata, dan baris pasangan tersedia di `output/`.

## Hasil

| Ukuran pada pasangan D3 | Train | Test history |
|---|---:|---:|
| Pasangan | 7.988 | 10.373 |
| Median film baru pada klaster-hari | 3 | 3 |
| Median show film baru lain | 10 | 11 |
| Tiket/show D3 <=8 | 5,71% | 21,32% |

Kelima bucket jumlah pesaing baru teramati (0, 1, 2, 3, dan >=4) memiliki proporsi tiket/show D3 <=8 lebih tinggi di test. Jika campuran bucket test diberi tingkat train, hasilnya **6,09%**, masih jauh dari **21,32%** test. Pada 467 strata `cinema_ids` x bucket pesaing dengan sedikitnya lima pasangan di tiap periode, cakupan test 91,03%; tingkat tertimbang 6,08% train vs 21,06% test.

[Studi penjadwalan multiplex oleh Eliashberg dkk.](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Demand-driven-scheduling-of-movies-in-a-multiplex.pdf) menjelaskan bahwa proyeksi permintaan, pilihan layar, dan jadwal show saling berkaitan. Karena itu, hubungan pasokan dan tiket dalam data ini bersifat deskriptif, bukan efek kausal. Data `test_history` hanya berisi tiga hari pertama film dan baris bertransaksi; film yang lebih tua serta show tanpa penjualan tidak tercakup. Temuan ini menolak penjelasan sederhana berupa perubahan *campuran film baru yang teramati*, bukan semua bentuk persaingan bioskop.

Untuk validasi nanti, laporkan segmen menurut jumlah film baru teramati serta klaster bioskop, tetapi jangan menganggap reweighting ini sebagai estimasi skor test D4-D10. Tidak ada eksperimen modelling.
