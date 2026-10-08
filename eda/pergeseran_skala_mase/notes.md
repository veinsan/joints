# EDA 074: pergeseran skala MASE

Pertanyaan: apakah skala yang diberi bobot oleh MASE punya distribusi sama di train dan test?

Hasil: window v2 memberi 7.988 pasangan train dan 10.373 pasangan test. Median skala 181,67 vs 91,67. Proporsi skala <=20 adalah 3,2% train vs 13,0% test; Februari test 22,6%, Maret 25,5%, maksimum train bulanan 4,7%. Pada target train D4-D10, proporsi nol di bucket skala 10-20 adalah 75,1%; tidak ada label target test untuk memeriksa angka ini pada periode baru.

Kesimpulan: skor CV biasa didominasi pasangan menengah/besar yang lebih sering muncul di train. Laporkan diagnostik per bucket skala dan per periode, serta sensitivitas terhadap komposisi test. Jangan menganggap tingkat nol train pada pasangan kecil berlaku langsung ke test karena EDA 072 menunjukkan retensi berbeda.
