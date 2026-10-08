# EDA 095: D3 tiket per show dalam jumlah show sama

Pertanyaan: apakah ekor D3 test lebih lemah hanya karena proporsi pasangan bershow sedikit lebih banyak?

Input: pasangan aktif D3 dari EDA 094, yang dibuat dengan builder v2 train dan test_history. Bagi jumlah show menjadi 1, 2, 3-4, 5-8, 9-16, 17+; hitung median tiket/show dan proporsi <=8. Standardisasi deskriptif ke campuran show test, lalu ke strata klaster x bucket show. Tanpa fit model.

Hasil: semua enam bucket menunjukkan median tiket/show test lebih rendah dan porsi <=8 lebih tinggi. Misal pada 5-8 show median tps37 train vs22.6 test dan weak share3.39% vs14.00%; pada17+ show39.76 vs24.31 dan0.61% vs6.92%. Train rate menurut campuran show test menghasilkan 6.54%, jauh dari 21.32% teramati. Pada 452 strata klaster x bucket show dengan >=5 pasangan per periode, cakupan test91.12% dan rate test-weighted train5.61% vs test20.05%.

Interpretasi: pergeseran permintaan per show teramati pada alokasi show sebanding, bukan hanya perubahan jumlah show. Masih dapat tercampur film, musim, occupancy/capacity, dan kebijakan pemutaran. D3 hanya input test, target D4-D10 tersembunyi.
