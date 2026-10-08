# Pergeseran retensi D3 antara train dan test_history

**Satu kalimat**: setelah penyamaan 36 strata permintaan awal, ukuran film, dan hari rilis, tingkat transaksi D3 pada pasangan aktif D2 lebih tinggi 11,50 poin di test_history daripada train (CI 95% bootstrap film 7,89 sampai 15,72 poin).

## Pertanyaan

Apakah pasangan kecil pada periode uji berhenti bertransaksi secepat pasangan dengan permintaan awal setara di train?

## Metode

Jalankan `python eda/pergeseran_retensi_d3/d3_policy_shift.py` dari root. Script memakai D1 hasil builder v2, membentuk seluruh pasangan aktif D1-D3 sebelum filter D3, lalu membandingkan transisi D2 ke D3. Standardisasi menggunakan komposisi test di strata tiket per show D2, ukuran film, dan hari rilis. Bootstrap 1.000 kali dilakukan per judul dasar. Output angka ada di `output/summary.json`, tabel CSV, dan `output/dropout_by_tps.png`.

## Hasil

| Ukuran | Train | Test history |
|---|---:|---:|
| Pasangan aktif D2 | 8.835 | 11.074 |
| Median tiket per show D2 | 27,45 | 14,00 |
| Tetap aktif D3, mentah | 88,61% | 91,55% |
| Tetap aktif D3, komposisi test | 80,72% | 92,22% |
| Berhenti sebelum D3 pada tiket per show D2 <=8 | 53,9% | 23,1% |

Pasangan test.csv cocok 100% dengan pasangan test_history yang aktif D3. Transisi D1 ke D2 pada tiket per show D1 <=8 juga berbeda: 70,3% train dan 87,9% test.
Pada pemeriksaan kedua yang juga menyamakan jumlah show D2, selisih tetap 9,72 poin pada 43 strata yang mencakup 97,56% pasangan test aktif D2.
Untuk tiket per show D2 <=8, tingkat berhenti per bulan train berkisar 39,6-65,6%, sedangkan seluruh bulan test berkisar 14,9-37,7%. Rincian jumlah pasangan dan bulan ada di `output/by_month_tps.csv`.

## Interpretasi

Test berisi pasangan lemah yang lebih sering tetap bertransaksi. Perubahan ini tidak dijelaskan oleh campuran tiket awal, ukuran film, atau hari rilis saja. Penyebab operasional tidak dapat ditentukan dari berkas. Tidak ada target D4-D10 test, sehingga pergeseran retensi D3 tidak boleh langsung dikonversi menjadi koreksi target tujuh hari.
Besarnya gap bergantung pada rekonstruksi D1 train. Pada uji sensitivitas, memajukan D1 train satu hari menurunkan tingkat berhenti bucket tiket/show <=8 dari 53,8% menjadi 29,2%, sementara test 23,1%. Namun itu membandingkan fase hari tayang yang berbeda; lihat `temp/eda_076_d1_sensitivity/`.

## Saran FE / validasi

| Saran | Alasan | Prioritas |
|---|---|---|
| Diagnostik transisi D1/D2 ke D3 di test_history, dipisah dari validasi target utama | Ada label yang tersedia secara sah pada horizon observasi | tinggi |
| Laporan CV per skala dan kekuatan awal pasangan | Rezim retensi berbeda terutama di pasangan lemah | tinggi |
| Audit fitur okupansi, show, dan kekuatan relatif pasar hanya dari D1-D3 | Memisahkan tingkat pasar dari kondisi pasangan | tinggi |

## Tingkat keyakinan dan status

Keyakinan tinggi pada aturan seleksi D3 dan perbedaan dengan D1 v2; sedang pada besar pergeseran yang sebenarnya karena D1 train direkonstruksi; rendah pada atribusi penyebab dan implikasi ke target D4-D10. Tidak ada eksperimen modelling dalam analisis ini.
