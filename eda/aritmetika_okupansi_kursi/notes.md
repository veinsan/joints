# Catatan EDA 100

Pertanyaan provenance: jika total_show=1 dan tiket positif, apakah okupansi dua desimal setara dengan tiket dibagi kapasitas kursi integer?

Metode: hanya baris okupansi 1-100% dan show positif. Fokus nonfull 10-<100% agar okupansi100% tidak menghasilkan kecocokan tautologis. Kedua integer di bawah/atas kapasitas tersirat diuji; rasio balik dibulatkan dua desimal lalu dibanding tepat. Karena rasio monoton terhadap denominator, pasangan integer yang mengapit kapasitas tersirat cukup untuk uji keberadaan denominator yang konsisten. Tidak menggunakan nilai target test D4-D10.

Hasil: satu-show nonfull okupansi>=10 memberi 236/4260 train (5.54%) dan30/555 test (5.41%) rekonstruksi tepat. Jarak pecahan kapasitas tersirat ke integer<=0.1 sebesar20.23% train dan19.28% test, dekat dengan20% yang diharapkan dari pecahan seragam; median jarak0.248 dan0.244. Untuk okupansi>=20 dan<100, 90/2678 train dan5/217 test tepat. Angka semua baris atau satu-show termasuk okupansi100% akan menyesatkan: 1589 train dan17 test satu-show persis penuh merekonstruksi otomatis.

Interpretasi: kolom okupansi kemungkinan bukan rasio tiket / jumlah kursi integer pada unit baris satu-show yang naif. Ini membatasi penggunaan kapasitas tersirat sebagai kursi fisik atau alat memastikan data asli. Alternatif yang belum bisa dibedakan: agregasi/normalisasi upstream, definisi show yang berbeda, rata-rata rasio berbobot, atau pembuatan angka sintetik. Tidak menyimpulkan operator.
