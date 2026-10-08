# EDA 096: kapasitas vs okupansi D3

Pertanyaan: apakah tiket/show D3 test lebih rendah terutama karena auditorium/kapasitas per show yang lebih kecil atau keterisian yang lebih rendah?

Metode: pasangan D3 EDA094. Untuk okupansi positif <=100, kapasitas/show tersirat = tiket x100 / okupansi / jumlah show. Dekomposisi log tepat: log(tiket/show) = log(okupansi/100) + log(kapasitas/show). Bandingkan agregat, enam bucket show, dan 452 strata klaster x bucket show dengan >=5 pasangan per periode. Tanpa model fit. Okupansi nol positif dikeluarkan hanya dari perhitungan kapasitas/log (8 train,60 test).

Hasil: median okupansi D3 22.46% train vs12.36% test; median kapasitas/show tersirat148.63 vs153.12. Dalam keenam bucket show, median kapasitas test sedikit lebih tinggi. Delta rerata log test-train semua baris valid: tps -0.5673 = okupansi -0.5975 + kapasitas +0.0302. Di strata klaster x bucket show cakupan90.74% test, delta log tertimbang tps -0.5413 (rasio0.582), okupansi -0.5779 (0.561), kapasitas +0.0366 (1.037). Sensitivitas membatasi okupansi1-100% dan kapasitas50-400: delta log tps -0.5287, okupansi -0.5446, kapasitas +0.0159.

Interpretasi: pergeseran tps D3 terutama berkaitan dengan keterisian lebih rendah, bukan rata-rata kapasitas/show lebih kecil. Ini identitas aritmetis berbasis occupation_rate yang dilaporkan dan tidak mengidentifikasi sebab permintaan, harga, program film, atau cara pencatatan. Cinema_ids adalah agregat dan kapasitas/show di sini rata-rata tersirat, bukan ukuran ruang fisik tertentu.
