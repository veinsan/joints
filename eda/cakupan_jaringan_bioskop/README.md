# Cakupan dataset mendekati jaringan bioskop besar, bukan pasar nasional

**Satu kalimat**: total `train.csv` pada April 2025 adalah 14.608.950 tiket, dekat dengan [pengumuman Cinema XXI](https://www.cinema21.co.id/id/newsroom/cinema-xxi-catat-rekor-baru-lebih-dari-14-juta-penonton-pada-april-2025) tentang lebih dari 14 juta penonton, sedangkan agregat film tertentu jauh di bawah angka penonton nasional.

## Pertanyaan

Apakah `total_ticket` mencakup semua bioskop Indonesia, satu jaringan, atau sampel?

## Metode

Jalankan `python eda/cakupan_jaringan_bioskop/coverage.py` dari root. Script menjumlah tiket train per bulan dan pada tanggal tonggak publik untuk tiga film. Data eksternal hanya pembanding provenance, bukan fitur atau label tambahan. Output ada di `output/monthly_tickets.csv`, `output/milestone_comparison.csv`, dan `output/benchmark_summary.json`.

## Hasil

| Pembanding | Dataset | Angka publik |
|---|---:|---:|
| April 2025, semua film | 14.608.950 | Cinema XXI: lebih dari 14 juta, [rilis 2 Mei 2025](https://www.cinema21.co.id/id/newsroom/cinema-xxi-catat-rekor-baru-lebih-dari-14-juta-penonton-pada-april-2025) |
| Q2 2025, semua film | 27.485.375 | Cinema XXI: lebih dari sekitar 28,3 juta, diturunkan dari H1 42,5 juta dan Q2 >2x Q1 pada [rilis 28 Juli 2025](https://www.cinema21.co.id/newsroom/kinerja-stabil-cinema-xxi-bukukan-pendapatan-rp28-triliun-di-semester-i-2025) |
| JUMBO sampai 2 Juni | 6.098.971 | Nasional: sekitar 10,07 juta, [Katadata](https://katadata.co.id/digital/startup/683d1bbd06863/jumbo-resmi-jadi-film-terlaris-di-indonesia-salip-kkn-di-desa-penari) |
| SORE sampai 24 Juli | 1.236.167 | Nasional: lebih dari 1,7 juta, [Kemenekraf](https://ekraf.go.id/news-en/ministry-of-creative-economy-appreciates-the-film-sore-istri-dari-masa-depan-surpassing-more-than-17-million-viewers) |

Train dimulai 1 April, sementara JUMBO mulai tayang 31 Maret. Q2 train juga memiliki outage 7-16 Juni. Perbandingan ini bukan rasio cakupan yang presisi.

## Interpretasi

Data tampaknya merepresentasikan sebagian besar satu jaringan bioskop besar atau jaringan yang serupa, bukan total pasar Indonesia. Kemiripan angka April dengan Cinema XXI dan struktur harga tiga tier mendukung hipotesis XXI, tetapi tidak mengidentifikasi pemilik data secara pasti. Judul nasional dan angka penonton publik tidak boleh digabung sebagai target pada level film di dataset ini.

## Saran FE / validasi

| Saran | Alasan | Prioritas |
|---|---|---|
| Tafsirkan agregat seluruh klaster sebagai skala jaringan, bukan skala nasional | Dataset hanya mencakup sebagian penonton nasional | tinggi |
| Gunakan prior pasar relatif yang dibangun dari paket kompetisi | Eksternal nasional memiliki cakupan dan definisi berbeda | tinggi |
| Jangan menganggap sumber jaringan atau tingkat cakupan konstan antarfilm | Belum ada konfirmasi operator dan tanggal awal train tersensor | sedang |

## Tingkat keyakinan dan status

Keyakinan tinggi bahwa tiket di paket kompetisi bukan total nasional; sedang bahwa cakupannya mirip jaringan Cinema XXI; rendah untuk identitas sumber data asli. Tidak ada eksperimen modelling.
