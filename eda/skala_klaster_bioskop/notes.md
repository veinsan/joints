# EDA 082: jejak jaringan dan sumber data

Pertanyaan: seberapa konsisten jumlah klaster, kota, tiket, dan show internal dengan laporan publik Cinema XXI?

Metode: hitung klaster unik, kota unik, tiket/show bulanan, serta cakupan September 2025. Angka publik hanya digunakan sebagai pembanding, tidak untuk menambah fitur atau label.

Hasil: train punya 117 `cinema_ids` pada 67 `city_name`, test_history 121 pada 69. Setiap ID hanya dipetakan ke satu kota. Di September 2025, 117 ID dan 67 kota muncul setiap hari (median), total 6.188 show/hari. Baris film x klaster x hari paling besar punya 285 show; agregat seluruh film pada satu klaster mencapai 540 show/hari di Jakarta. Ada 274 baris train dengan >100 show untuk satu film dan klaster per hari. Laporan publik Cinema XXI per 30 September 2025 menyebut 261 lokasi dan 1.369 layar di 67 kota/kabupaten pada bagian utama; angka 61 pada boilerplate Inggris usang. Kecocokan kota dan audit ekspansi EDA103 memperkuat hipotesis basis jaringan XXI, tetapi `cinema_ids` tidak layak diperlakukan sebagai satu gedung fisik dan asal transaksi mentah belum terkonfirmasi.

Kesimpulan: skala klaster amat heterogen; `total_ticket` mentah bercampur ukuran wilayah/lokasi dan permintaan film. Fitur rasio per show dan baseline historis klaster lebih masuk akal dibanding asumsi satu `cinema_ids` = satu gedung. `city_name` dan ID juga tak identik. Tidak ada fitting model.
