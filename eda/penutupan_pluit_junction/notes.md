# Catatan EDA 105

Pertanyaan: apakah penutupan lima layar XXI dapat dilacak sebagai penurunan jumlah show ID anonim?

Metode: ambil tanggal efektif penutupan Pluit Junction XXI dari presentasi investor primer. Agregasi seluruh film train per hari-ID. Bandingkan 30 Apr dan 1 Mei serta jendela 14 hari berpasangan hari pekan; peringkat perubahan terhadap ID lain sebagai pembanding deskriptif. Tidak ada fitting atau penggunaan target test.

Temuan: satu ID Jakarta turun 25 show tepat 1 Mei dan menetap sekitar 390 show/hari seminggu sesudahnya. Dua ID Jakarta lain nyaris stabil pada perubahan harian. Penurunan -25 adalah terbesar dari 116 ID pada tanggal itu. ID tersebut juga turun rerata -33,29 show/hari pada 14 hari setelahnya, penurunan terbesar di semua ID. Ini memperkuat interpretasi show sebagai jejak kapasitas jaringan, tetapi bukan pemetaan venue yang terverifikasi.

Batas: pergantian film dan libur 1 Mei dapat mengubah show/tiket; peringkat empiris tidak mengidentifikasi sebab. PDF investor tidak menyebut `cinema_ids`. Status asli vs sintetis tetap terbuka: generator berbasis jadwal jaringan nyata pun dapat menyalin perubahan ini.
