# EDA 100: apakah okupansi satu show merekonstruksi kursi integer?

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/aritmetika_okupansi_kursi/capacity_arithmetic.py` dari root. Script memakai `train.csv` dan `test_history.csv` yang hanya berisi input observasi, tanpa target tersembunyi dan tanpa fitting model.

Untuk baris bertiket positif, satu show, serta okupansi 10% sampai di bawah 100%, hitung kapasitas tersirat `100 * total_ticket / occupation_rate`. Uji kedua bilangan integer yang mengapit kapasitas itu, lalu cek apakah `round(100 * total_ticket / kursi_integer, 2)` mereproduksi angka okupansi tercatat.

| Periode | Baris satu show, okupansi 10-<100% | Rekonstruksi tepat | Jarak kapasitas tersirat ke integer <=0,1 |
|---|---:|---:|---:|
| Train | 4.260 | 5,54% | 20,23% |
| Test history | 555 | 5,41% | 19,28% |

Pada subset okupansi 20-<100%, rekonstruksi tepat hanya 3,36% train (2.678 baris) dan 2,30% test (217). Sebaliknya, okupansi tepat 100% menghasilkan kapasitas sama dengan tiket secara otomatis; 1.589 baris train satu show seperti ini terutama terkait preview dan harus dipisahkan. Bila okupansi adalah pembulatan langsung dari tiket dibagi jumlah kursi integer pada satu show, tingkat rekonstruksi seharusnya jauh lebih tinggi. Maka `occupation_rate` tidak boleh dibalik menjadi jumlah kursi fisik tanpa verifikasi definisi dan proses agregasinya.

Temuan ini tidak mengidentifikasi apakah data asli atau sintetis. Kemungkinan penjelasan termasuk okupansi rata-rata dengan denominator lain, transformasi/agregasi upstream, atau nilai sintetis/jitter. Definisi operasional `total_show` dan `occupation_rate` dari sumber asli belum tersedia. Output `output/summary.json` dan tabel kapasitas satu show disediakan agar audit bisa diulang.
