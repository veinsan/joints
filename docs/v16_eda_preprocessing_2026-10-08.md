# Audit v15 dan eksperimen v16

v15 tetap baseline public terbaik: **0,39918**, berdasarkan laporan user. Target **0,33662** membutuhkan perbaikan absolut **0,06256**, atau **15,67%**. Belum ada hasil eksperimen baru yang membuktikan target itu tercapai. Tidak ada probing, pengiriman submission, training model lokal, atau perubahan notebook v15 dalam pekerjaan ini.

## Hasil yang benar-benar dijalankan

`eda/96_v15_validation_audit.py` memeriksa data mentah dan prediksi OOF v15. `eda/97_v15_quantile_diagnostics.py` merekonstruksi prediksi kuantil v15 dan memeriksa geometri target. Keduanya hanya membaca data/prediksi tersimpan, menghasilkan tabel dan gambar di `outputs/eda/96_v15_validation_audit/` dan `outputs/eda/97_v15_quantile_diagnostics/`.

| Ukuran | Hasil |
| --- | ---: |
| MASE resmi, OOF tanpa pembobotan | 0,31567 |
| TW-MASE, komposisi skala × hari penjualan pertama disamakan dengan test | 0,33149 |
| Public, dilaporkan user | 0,39918 |
| Film dasar / minggu rilis di validasi | 126 / 23 |
| Effective sample size bobot pada tingkat minggu | 20,55 |
| Bootstrap minggu 95% untuk TW-MASE | [0,26972; 0,40564] |
| Bootstrap minggu 95% perubahan v15 minus LightGBM | [-0,02041; -0,00370] |
| Error yang berasal dari lima film terbesar | 28,26% |

Bootstrap mengambil ulang seluruh minggu, bukan baris bioskop. Interval tersebut menggambarkan variasi kohort train **bersyarat pada prediksi yang sudah dipilih**; tidak memasukkan ketidakpastian pemilihan model, training ulang, atau pergeseran distribusi ke test. Itu bukan interval prediksi public/private. Keunggulan v15 terhadap LightGBM didukung perbandingan berpasangan ini, tetapi nilai absolut 0,33149 tidak dapat diperlakukan sebagai estimasi private yang presisi.

## EDA: error terbesar ada di pertumbuhan penjualan

| Kondisi aktual target | Porsi bobot evaluasi | TW-MASE segmen | Kontribusi MASE total |
| --- | ---: | ---: | ---: |
| Penjualan > D3 | 16,93% | 1,07659 | 0,18232 |
| Positif tetapi tidak melebihi D3 | 40,29% | 0,24755 | 0,09975 |
| Nol | 42,77% | 0,11554 | 0,04942 |

Kelompok pertumbuhan menyumbang **55,0%** error. Gain v15 atas LightGBM pada kelompok ini hanya 0,00136, jauh lebih kecil daripada gain 0,01855 pada target nol. Artinya, peningkatan model sejauh ini lebih berhasil pada penurunan/pencopotan daripada film yang tumbuh. Ini diagnosis memakai label OOF, **bukan fitur yang tersedia saat inference**. Tidak boleh mengalikan semua prediksi karena sudah mengetahui kelompok yang sulit dari label validasi.

D4 dan D5 memiliki TW-MASE 0,42083 dan 0,45235; D8–D10 sekitar 0,264–0,266. Bulan rilis Mei paling sulit (0,47133), dibanding Juli–September sekitar 0,286–0,302. Angka bulanan ini berasal dari **cohort OOF**, bukan hasil backtest temporal. Backtest v15 yang hanya memakai Juli–September memang mencakup bulan yang secara OOF lebih mudah; nilai temporal 0,2971 tidak otomatis membuktikan kemampuan pada musim sulit.

Median skala pasangan train 180,33 versus test 91,67. Pembobotan skala membantu menyamakan komposisi, tetapi tidak menjamin pola target bersyarat atau musim libur juga sama. Kesalahan tidak hanya terjadi pada skala terkecil: bucket 20–50 menyumbang 0,06664 dari TW-MASE total, 50–100 menyumbang 0,06093, dan 100–200 menyumbang 0,05824.

## Preprocessing yang terverifikasi

- Data transaksi train/history tidak memiliki missing value, duplikat kunci, tiket negatif, atau okupansi di luar 0–100.
- Ada 237 baris train dan 222 history dengan okupansi nol tetapi tiket positif. Jangan menganggapnya penjualan nol; pembulatan/pencatatan okupansi dapat berbeda. Jangan mengestimasi kapasitas dengan membagi okupansi nol tanpa guard.
- Skala test dari cache sama dengan `max(sum(D1:D3)/3, 1)` pada history mentah. **Pembagi tetap 3**, bukan jumlah hari yang memiliki transaksi.
- Setiap pasangan test memiliki tujuh target. Pasangan D3 dan pasangan test sama persis: tidak ada pasangan hilang di salah satunya.
- Ada 55 baris history dari format film yang tidak muncul sebagai format target. Ini bukan duplikat atau baris otomatis tidak berguna; tanggalnya tetap sesuai jendela judul dasar.
- Setelah pengecualian outage lama, hanya 83 target nol terjadi ketika tidak ada satu pun transaksi lain di klaster/tanggal yang sama. Itu 0,49% dari target nol dan hanya **0,18% dari error tertimbang**. Status ini tidak membuktikan outage. Menghapus semuanya tetap tidak dapat menjelaskan gap leaderboard sebesar 0,06256.
- Tanggal 11 Juni masih memiliki 98 baris target. Belum ada bukti memadai untuk menambahkannya ke daftar outage secara otomatis.

**Perbaikan konkret:** `train_part()` v15 menambahkan seluruh `Xlim` ke setiap fold. Akibatnya, 371 / 350 / 273 / 119 / 322 baris limited masing-masing masuk training walaupun minggu rilisnya berada di fold validasi. Tidak ada overlap judul dasar, tetapi klaim bahwa seluruh minggu validasi tidak terlihat training menjadi tidak benar. Dalam 20 dari 23 minggu validasi terdapat overlap limited.

v16 menyaring limited berdasarkan minggu dan judul dasar validasi, serta mengeluarkan film limited tersebut dari statistik klaster. Baseline dan kandidat dihitung ulang dengan aturan identik. Dampak skor perbaikan ini **belum diukur** dan tidak boleh diklaim positif sebelum run. Cohort split juga tetap berbeda dari validasi yang sepenuhnya maju menurut waktu.

Rekonstruksi D1 lama masih menggunakan puncak cakupan sepanjang riwayat. Ini asumsi simulator yang memakai informasi setelah D3, bukan tanggal rilis resmi yang diberikan panitia. v16 mempertahankannya agar ablation terbatas; keterbatasan ini tetap perlu dinyatakan. Fitur pesaing juga memakai jendela D1–D3 resmi film lain, termasuk pada tanggal sesudah origin film sendiri; validitasnya bersandar pada seluruh `test_history` yang memang diberikan untuk batch kompetisi, bukan skenario produksi online.

## Mengapa kandidat v16 adalah target log1p

Target model saat ini `r = y / scale / cal_mult` mencapai 60,90 dan memiliki sekitar 29,95% nol. Rata-rata skew per horizon adalah 19,77, versus 1,54 sesudah `log1p`. Rentang IQR setelah standardisasi naik dari 0,60 menjadi 1,26 simpangan baku, sehingga rentang tengah tidak terlalu tertekan oleh ekor besar.

Transformasi monoton mempertahankan kuantil populasi. `expm1` atas keluaran kuantil log kemudian dikembalikan ke skala tiket sebelum menghitung MASE. Ini **tidak** menjamin model terlatih lebih baik; model dapat kehilangan kemampuan menangani pertumbuhan besar. TabPFN memiliki preprocessing internal sendiri, sehingga hasil Causilo log1p terdahulu juga tidak membuktikan manfaat yang sama pada TabPFN.

Rekonstruksi kuantil TabPFN v15, termasuk kalender Jawa Barat dan lambda 0,5, cocok dengan `comp_tp35_int7` tersimpan hingga selisih maksimum **3,64e-12**. Pemeriksaan tidak menggunakan `cal_mult` cache lama secara buta.

Sensitivitas lambda dengan blend tetap menunjukkan TW 0,32892 / 0,32902 / 0,33149 untuk lambda 0 / 0,25 / 0,5. Itu evaluasi ulang pada OOF yang sudah dipakai memilih model; bukan gain baru yang independen. Lambda 0,5 sebelumnya dipilih memakai lensa temporal juga. v16 **tidak** mengganti lambda berdasarkan tabel ini.

## Aturan eksperimen v16

Notebook: `notebooks/v16.ipynb`. Builder: `notebooks/build_v16.py`; builder membaca sel v15 tanpa menulis ulang v15, sedangkan notebook hasilnya berdiri sendiri di Kaggle.

1. Hitung ulang kontrol LightGBM 0,1 + TabPFN-3.5 int7 target mentah 0,9, sesudah perbaikan limited.
2. Ganti hanya target TabPFN menjadi log1p; checkpoint, seed, fitur, bobot blend, dan lambda tetap.
3. Pilih log1p hanya bila gain TW ≥ 0,0015, rata-rata temporal tidak memburuk, penurunan setiap bulan ≤ 0,003, menang pada minimal tiga fold, dan batas atas bootstrap minggu untuk delta MASE < 0.
4. Jika gagal salah satu syarat, gunakan kontrol. Kegagalan menjalankan model menghentikan notebook, bukan diam-diam mengganti keluarga model.
5. Simpan `ablation.csv`, `adoption.csv`, `temporal.csv`, `oof.csv`, `oof_quantiles.npz`, `inference_audit.csv`, `model_weights.pkl`, dan `submission.csv`.

September dan bulan lainnya telah digunakan dalam eksperimen terdahulu. Gate temporal membantu menolak kerusakan yang jelas, tetapi **bukan holdout baru yang bebas dari sejarah tuning**. Jangan menyebut gain hasil pemilihan ini sebagai estimasi unbiased. Perbandingan utama v16 adalah kontrol baru versus kandidat baru, bukan skor mutlak v16 versus OOF v15 dengan split berbeda.

## Lebaran dan gap private/public

Override Lebaran mencakup 6,08% baris test, dengan rata-rata prediksi/skala 1,76931. Mengubah seluruh prediksi segmen itu sebesar 10% memiliki batas perubahan MASE total 0,01076 berdasarkan pertidaksamaan segitiga, **tanpa mengetahui arah perubahan atau label**. Kapasitas kursi hanya memberi batas supply, bukan bukti tiket akan terjual.

Lebaran tetap memakai analog v15 agar perbandingan kandidat adil; jangan menyesuaikan `LEB_RHO` untuk mengejar public. v16 menyimpan prediksi sebelum override supaya area yang belum tervalidasi ini terlihat. Tidak ada metode yang dapat menjamin jarak public/private kecil dengan label private tidak tersedia.

## Prioritas berikutnya dan keputusan

**Jalankan v16 di Kaggle dahulu; belum ada dasar menyatakan submission baru lebih baik.** Jika kandidat lolos, tinjau tabel temporal, perubahan pada D4–D5 dan pertumbuhan, serta ukuran bobot aktual sebelum memakai submission hasil run. Jika gagal, pertahankan v15 sebagai baseline public dan jangan menganggap banyak model tambahan otomatis menyelesaikan masalah.

Untuk lompatan sebesar 0,06256, bukti paling bernilai selanjutnya adalah informasi D1–D3 yang menjelaskan lintasan pertumbuhan dan validasi simulator D1, bukan tuning kecil lambda. Data eksternal hanya layak masuk bila tanggal publikasi/arsip sebelum 30 September 2025 dapat dibuktikan dan ablation memperbaiki beberapa kohort. Tidak ada data eksternal baru yang digunakan dalam eksperimen ini. Tidak ada dasar untuk menyatakan 0,33662 sebagai batas akurasi yang sudah diketahui dari dataset ini.

## Verifikasi dan referensi

`eda/98_v16_preprocessing_check.py` memeriksa semua sel kode secara sintaks, eksklusi limited pada kelima fold, transformasi dan inverse dengan stub inference, serta cabang menerima/menolak kandidat. Semua lolos. Ini bukan training GPU, pemeriksaan kuantisasi ulang, atau pembuktian bahwa keseluruhan run Kaggle pasti berhasil. GPU run tetap diperlukan.

```bash
rtk uv run --no-project --with numpy --with pandas --with matplotlib --with pyarrow python eda/96_v15_validation_audit.py
rtk uv run --no-project --with numpy --with pandas --with matplotlib --with pyarrow python eda/97_v15_quantile_diagnostics.py
rtk python3 notebooks/build_v16.py
rtk uv run --no-project --with numpy --with pandas --with matplotlib --with pyarrow python eda/98_v16_preprocessing_check.py
```

- [Forecasting: Principles and Practice, time-series cross-validation](https://otexts.com/fpp3/tscv.html): evaluasi beberapa origin dan horizon dengan training mendahului target evaluasi. Mendukung desain temporal, bukan klaim skor kompetisi.
- [scikit-learn, nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): penggunaan ulang CV untuk memilih dan melaporkan model dapat memberi optimisme seleksi.
- [Source resmi TabPFN regressor](https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/regressor.py): preprocessing dan jalur keluaran regresi/kuantil. Halaman main dapat berubah; eksperimen tetap memakai versi paket dan checkpoint v15 yang dipin, bukan mengadopsi API main.

Dokumen metodologi di atas hanya referensi pengembangan, bukan sumber fitur atau label kompetisi.
