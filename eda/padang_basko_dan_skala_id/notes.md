# Catatan EDA 104

Pertanyaan: apakah lokasi XXI baru pada kota yang sudah ada membuat ID baru, atau terserap dalam ID lama?

Metode: gunakan tanggal pembukaan Basko City Mall XXI dari pengelola mal; agregasi seluruh film train menjadi show/tiket per hari-ID; bandingkan 21 hari sebelum vs 21 hari sesudah agar hari pekan seimbang dan outage 7-16 Juni dihindari. Hitung perubahan kota lain dengan jendela yang sama sebagai konteks deskriptif. Tidak ada fitting atau target D4-D10 test.

Temuan: ID Padang kecil yang sudah ada sejak April melonjak tepat 9 Juli, dari 4 ke 25 show; rata-rata +23,43 show/hari setelah pembukaan. ID Padang lain tidak berubah pada rerata show. Rasio show total Padang 1,412 adalah tertinggi dari 66 kota pembanding; peringkat berikutnya 1,048. Ini memperkuat hubungan footprint paket data dengan XXI dan memberi bukti perubahan jaringan dapat muncul pada ID lama. Mekanisme pasti penggabungan lokasi ke ID tetap tidak teridentifikasi.

Batas: baris data tidak mengandung lokasi fisik, sehingga tidak bisa mengisolasi show Basko secara langsung. Jumlah tiket bergantung portofolio film, dan `total_show` sendiri dapat berubah akibat penjadwalan. Kesimpulan tidak boleh digeneralisasi bahwa semua ID selalu agregat banyak lokasi. Harga tiket dan kapasitas kursi Basko bukan target yang teramati di paket.
