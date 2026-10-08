# EDA 094: pasokan film baru bersamaan

Pertanyaan: apakah ekor tiket/show D3 yang lebih lemah pada test hanya akibat jumlah film baru yang bersaing pada klaster-tanggal sama?

Metode: rekonstruksi D1 train dengan builder v2 dan D1 test dari test_history. Hanya baris usia film D1-D3 yang masuk pasokan film baru teramati. Pilih pasangan aktif D3; hitung judul dasar lain yang bertransaksi pada klaster-tanggal itu. Standarisasi deskriptif ke campuran jumlah film pesaing test dan pada strata klaster x bucket pesaing. Tanpa fit model.

Hasil: median film baru teramati pada klaster-hari D3 sama 3 di train/test. Median show film baru lain 10 vs11. Proporsi pasangan D3 tiket/show <=8 adalah 5.71% train vs21.32% test. Standardisasi campuran bucket pesaing test dengan rate train memberi 6.09%. Pada 467 strata klaster x bucket dengan setidaknya 5 pasangan tiap periode, mencakup 91.03% pasangan test, rate tertimbang train 6.08% vs test21.06%.

Interpretasi: campuran pesaing film baru yang teramati tidak menjelaskan pergeseran ekor kecil. Dataset test_history hanya memuat D1-D3 dan baris transaksi positif, sehingga pasokan ini tidak mengukur film lama, jadwal tanpa penjualan, atau seluruh kapasitas operator. Perbedaan film, musim, dan kebijakan pemutaran masih mungkin. Paper Eliashberg dkk. 2009 menempatkan permintaan dan alokasi show sebagai keputusan bersama; asosiasi ini tidak mengidentifikasi efek kausal persaingan.
