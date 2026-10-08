# EDA 083: label cuti bersama di kalender paket

Pertanyaan: apakah `holidays.csv` menandai tanggal cuti bersama resmi yang sudah diumumkan sebelum 30 September 2025, dan berapa banyak target yang terdampak?

Metode: cocokkan tanggal dari dokumen pemerintah 2025 dan SKB 2026 bertanggal 19 September 2025 dengan `holidays.csv`, lalu hitung baris target/history pada tanggal tersebut. Ini audit EDA, tidak mengubah file paket atau membuat fitur model.

Hasil: ke-14 tanggal cuti bersama dalam rentang kalender semuanya ditandai `holiday_tipe=normal`. Sebanyak 2.392/72.611 (3,29%) baris target berada pada cuti bersama, menyentuh 1.580/10.373 pasangan test (15,23%). Tiga tanggal 20, 23, dan 24 Maret sendiri memuat 1.443 baris target. Pada train, 5.385 baris transaksi berada pada delapan tanggal cuti bersama yang relevan; 9 Juni terdampak outage. Pada test_history ada 1.629 baris pada tanggal cuti bersama.

Kesimpulan: `holidays.csv` tampaknya menandai hari libur nasional saja, bukan cuti bersama. Hal ini bukan kesalahan tabel bila itu definisinya, tetapi metadata kalender kehilangan kategori yang diketahui publik sebelum cutoff 30 September 2025. Khusus 20/23/24 Maret 2026, file menyebut `normal` di sekitar Idulfitri 21/22 Maret. Implikasi FE tinggi, terutama rentang Maret yang tidak ada analog di train. Tidak ada fitur atau model yang dibuat dalam audit ini.
