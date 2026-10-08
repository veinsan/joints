# Catatan EDA 106

Pertanyaan awal: seberapa stabil total show ID dan apakah lompatan venue luar biasa? Pemindaian 7 hari pertama justru menemukan sinkronisasi penurunan pada awal Juni yang belum tercatat sebagai outage penuh.

Metode: agregasi seluruh film per hari-ID; median rasio ID dan pasangan film-ID saat transisi 31 Mei→1 Juni, bandingkan dua transisi Sabtu→Minggu lain. Hitung profil stabilitas setelah menandai 1-16 Juni sebagai anomali agar profil tak didominasi periode itu. Impor builder v2 sebagai fungsi baca data untuk menghitung exposure window saat ini; builder tidak diubah.

Temuan: 112/114 ID bersama turun ke <=setengah show 1 Juni, median rasio 0,321. Dua akhir pekan kontrol punya median 1,0 dan 0 ID turun separuh. Row turun 18%, ID hanya 2%, show turun 67%. Target 267 pasangan dari tiga film D1 28 Mei melintasi 1-5 Juni dan saat ini tidak dibuang oleh aturan outage 7-16 Juni. Potensi kontaminasi target validasi tinggi meski porsi seluruh data 3,34% pasangan.

Batas: tidak ada jadwal operasi fisik seluruh venue untuk verifikasi, dan `total_show` dapat mencerminkan penjadwalan yang benar-benar dikurangi. Jangan menyebut 1-5 Juni hilang 100% atau mengganti target dengan imputasi. Analisis ini tidak mengubah kontrak modelling; temuan harus dijadikan audit kualitas sebelum eksperimen berikutnya.
