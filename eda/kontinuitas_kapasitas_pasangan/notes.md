# Catatan EDA 102

Pertanyaan: apakah kapasitas per show tersirat memiliki kontinuitas khusus pasangan film-klaster, atau hanya distribusi umum klaster dan banyak show?

Metode: builder v2 untuk 7.988 train window, builder test untuk10.373 test window. D1 dan D3 harus sama-sama punya tiket>=20, show>=2, okupansi5-<100, kapasitas per show tersirat50-400. Statistik median nilai mutlak log rasio D3/D1. Kontrol permutasi100 kali D3 dalam klaster; lalu dalam klaster x bucket show D3 (2-3,4-7,8-15,>=16). Tidak ada model prediktif.

Hasil: train n6028, teramati0.0679 vs acak klaster+show0.1480 (95% 0.1421-0.1530); test n5720, teramati0.0786 vs acak0.1532 (95% 0.1483-0.1579). Persentase perubahan kapasitas absolut<=10:58.59% dan55.30%. Median perubahan saat show stabil -0.25% train/+0.53% test, ketika dipangkas>=50% -4.56%/-3.89%. Cakupan test lebih kecil akibat filter okupansi/tiket, sehingga pernyataan tidak berlaku langsung bagi ekor lemah.

Interpretasi: ada struktur kapasitas relatif yang melekat pada film-klaster selama pembukaan, melampaui klaster dan jumlah show semata. Ini cocok dengan kontinuitas alokasi auditorium, tetapi rasio bukan kursi fisik integer (EDA100) dan generator data yang mempertahankan kapasitas juga dapat menciptakan pola sama. Jangan menyimpulkan operator atau data asli dari temuan ini. Bila menjadi fitur nanti, gunakan hanya D1-D3 dan sertakan flag kualitas.
