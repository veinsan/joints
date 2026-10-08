# EDA 080: transport komposisi skala pada galat OOF lama

Pertanyaan: seberapa besar rata-rata MASE CV berubah bila distribusi skala pasangan mengikuti test, tanpa menjalankan model baru?

Hasil: OOF XGB terkunci 0,3164 berubah menjadi 0,4048 setelah rata-rata error per bucket skala ditimbang dengan proporsi bucket test; selisih +0,0884, CI bootstrap film 95% +0,0283 sampai +0,2060. OOF TabPFN e9 0,2985 menjadi 0,3770; selisih +0,0784, CI +0,0207 sampai +0,1928. Bucket skala 3-10 memiliki MASE OOF 1,78-1,88; proporsinya 0,94% train vs 4,38% test. Semua OOF berasal dari artefak yang sudah ada, tanpa fitting atau tuning baru.

Kesimpulan: perbedaan komposisi skala saja cukup untuk menaikkan ringkasan galat CV sekitar 0,08 pada dua model lama. Nilai yang ditimbang bukan prediksi skor test: error kondisional pada bucket bisa bergeser, bukti retensi D3 juga menunjukkan regime shift; bucket kecil punya sampel train terbatas. Laporkan CV tertimbang komposisi test sebagai diagnostik tambahan, bukan mengganti skema group-film dan bukan mengoptimalkan model ke test.
