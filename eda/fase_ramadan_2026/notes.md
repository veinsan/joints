# Catatan EDA 099

Pertanyaan: apakah ekor tiket/show D3 kecil pada Februari-Maret bertepatan secara eksklusif dengan Ramadan, dan apakah jendela ini relevan bagi validasi?

Metode: pasangan test D3 EDA094 dipotong ke fase kalender, dikelompokkan menurut minggu, film dasar, dan kota. Film berformat khusus dinormalisasi ke judul dasar. Tidak ada target test yang dibaca. Tanggal 19 Februari berasal dari keputusan Kemenag 17 Februari 2026 untuk analisis retrospektif.

Temuan: weak rate naik dari 13.65% Oktober-Januari ke34.62% pra-Ramadan Februari lalu52.94% selama 19Feb-17Mar. Median tiket/show masing-masing23.67,11.33,7.60. Weak rate minggu D3 6-8Feb sudah46.85%; awal Ramadan bukan titik perubahan tunggal. Perubahan awal terjadi pada30/33 kota berukuran cukup dan kenaikan fase Ramadan pada31/33 kota, tetapi film tiap fase berbeda. Pekan D3 13Mar berisi181 pasangan/3 film dan92.82% weak. Kohort 6 film D1 18Mar:631 pasangan, median22.6 tiket/show,6.97% weak. Cinema XXI melaporkan Ramadan 2025 secara umum menurunkan moviegoing dan Lebaran menaikkan admission dengan lima film. Ini mendukung hipotesis musim tetapi data sekarang tidak mengidentifikasi efek kausal, operator, atau target D4-D10.

Implikasi: validasi/diagnostik nanti perlu melaporkan segmen Februari awal, Ramadan perkiraan, dan kohort Lebaran secara terpisah; skala test dapat berubah tajam antarslate. Penggunaan tanggal 19Feb resmi Kemenag sebagai fitur eksternal pada cutoff 30Sep2025 adalah lookahead. SKB Idulfitri 2026 tersedia sebelum cutoff, sehingga jarak menuju libur dan jendela Ramadan sekitar 29-30 hari sebelumnya dapat disusun secara ex ante dengan status perkiraan.

Sumber: https://cinema21.co.id/newsroom/di-tengah-kelesuan-jumlah-penonton-cinema-xxi-berhasil-jaga-ebitda-positif-di-kuartal-i-2025-dan-bukukan-pendapatan-rp9292-miliar ; https://aceh.kemenag.go.id/baca/pemerintah-tetapkan-1-ramadan-1447-h-jatuh-pada-19-februari-2026?audio=1 ; https://jdih.kemnaker.go.id/peraturan/detail/2723/keputusan-bersama-menteri-agama-menteri-ketenagakerjaan-dan-menteri-pendayagunaan-aparatur-negara-dan-reformasi-birokrasi-republik-indonesia-nomor-2-tahun-2025
