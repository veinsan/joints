# Catatan EDA 101

Pertanyaan: apakah `total_ticket` mungkin menghitung tiket berbayar, sedangkan okupansi mencakup admissions tambahan seperti tiket gratis? Vista Help Centre menjelaskan perbedaan kedua ukuran ini pada sistem pelaporan bioskop, tetapi operator dan pipeline dataset kompetisi belum teridentifikasi.

Metode: baris satu show, total_ticket 1-400, occupancy 10-<100%. Untuk k=0..10, uji apakah rasio `100*(total_ticket +/- k)/kursi_integer` dapat membulat ke okupansi tercatat dengan kursi50-400. Kategorikan kecocokan tambah saja, kurang saja, keduanya, atau tidak satu pun. Kontrol negatif mengacak okupansi dalam desil jumlah tiket sebanyak100 replikasi agar distribusi marginal tiket/okupansi dan efek batas denominator masih relevan. Tidak ada model prediktif.

Hasil: kecocokan positif-saja dikurangi negatif-saja =+10.71pp train dan+20.90pp test, tetapi rerata kontrol acak +10.31pp [9.10,11.60] dan+20.48pp [17.74,24.15]. Posisi observasi di dalam rentang kontrol. Jadi tanda positif berlebih tidak bisa dianggap bukti keberadaan admissions gratis. Jarak minimal tambahan yang dapat mencocokkan median5 tiket; ini pun tidak membuktikan lima tiket gratis per show karena banyak kandidat denominator tersedia.

Sumber eksternal: https://help.vista.co/hc/en-nz/articles/13519958299033-Marketing-report (definisi Paid Admits, Box Office Admits, dan Occupancy Rate). Penggunaan sumber hanya untuk merumuskan hipotesis; tidak mengidentifikasi penyedia sistem dataset.
