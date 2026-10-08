# Tanggal D1 adalah rilis luas, bukan transaksi pertama

**Satu kalimat**: D1 hasil rekonstruksi v2 cocok dengan 5/5 tanggal rilis publik independen, sementara transaksi pertama film yang sama muncul 7 sampai 33 hari lebih awal.

## Pertanyaan

Apakah gap retensi train-test muncul karena salah menentukan D1 pada train, dan apakah D1 v2 punya dukungan eksternal?

## Metode

Jalankan `python eda/verifikasi_tanggal_rilis/sensitivity.py` dari root. Script menguji D1 v2, D1-1, D1+1, serta film tanpa preview untuk transisi D2 ke D3, lalu membandingkan D1 v2 dengan lima tanggal publik yang tercatat sebelum 30 September 2025. Output ada di `output/sensitivity.csv`, `output/release_audit.csv`, dan `output/official_release_check.csv`.

## Hasil

| Film | D1 v2 | Transaksi pertama | Sumber rilis |
|---|---|---|---|
| JALAN PULANG | 19 Juni | 4 Juni | [LSF](https://lsf.go.id/en/film/jalan-pulang/494) |
| SORE ISTRI DARI MASA DEPAN | 10 Juli | 28 Juni | [LSF](https://lsf.go.id/en/film/sore-istri-dari-masa-depan/507) |
| BELIEVE | 24 Juli | 3 Juli | [Movimax](https://movimax.co.id/article/BELIEVE--TAKDIR-MIMPI-KEBERANIAN) |
| PANGGIL AKU AYAH | 7 Agustus | 5 Juli | [LSF](https://lsf.go.id/film/panggil-aku-ayah/297) |
| JADI TUH BARANG | 18 September | 11 September | [LSF](https://lsf.go.id/film/jadi-tuh-barang/319) |

Pada subset 137 judul yang aman untuk pergeseran satu hari, 111 punya transaksi sebelum D1 v2, dengan median selisih 5 hari. Untuk tiket/show D2 <=8, tingkat berhenti train pada D1 v2 53,8%, D1 dimajukan satu hari 29,2%, dan test 23,1%. Perbandingan D1 dimajukan memakai fase penayangan berbeda, sehingga ini uji sensitivitas, bukan alternatif yang terbukti benar.

## Interpretasi

Transaksi sebelum rilis luas lazim sebagai pratinjau. Definisi D1 berdasarkan transaksi pertama per film atau per klaster akan menukar fase preview dengan pembukaan resmi dan mengubah distribusi retensi. Lima kecocokan eksternal mendukung D1 v2, tetapi tidak membuktikan semua judul benar.

## Saran FE / validasi

| Saran | Alasan | Prioritas |
|---|---|---|
| Pertahankan D1 rilis luas untuk pembentukan window train | Cocok dengan 5/5 tanggal publik dan mekanisme test | tinggi |
| Pisahkan preview sebelum D1 dari tiga hari observasi resmi | Preview dapat muncul berminggu-minggu sebelumnya | tinggi |
| Laporkan ketidakpastian D1 untuk judul berjangkauan kecil atau rilis bertahap | Lima verifikasi bukan semua film | sedang |

## Tingkat keyakinan dan status

Keyakinan tinggi untuk lima film yang diperiksa, sedang untuk seluruh train. Tidak ada eksperimen modelling.
