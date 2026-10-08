# EDA 091: preview satu show

Pertanyaan: apakah pola preview 1 show dengan tiket nyaris kapasitas pada Tabayyun adalah kasus tunggal atau mekanisme berulang?

Temuan: 1647/3319 baris preview berisi satu show. Dari 1647 baris itu, 69.03% okupansi persis 100%, dibanding 5.19% satu show pada D1-D3 dan 7.03% sesudahnya. Median tiket satu show preview 142 vs 21 pada D1-D3. Bahkan tanpa Believe, 605/1113 = 54.36% preview satu show persis penuh. Believe memiliki 734 baris preview, 170254 tiket, 21 tanggal, 48 klaster; 532/534 baris preview satu show okupansi 100%. Pada satu klaster Madiun, 20 hari berturut-turut satu show berisi 165-168 tiket dan okupansi 100%. Ada 37 pasangan film-klaster dengan sedikitnya tiga tanggal preview satu show, median tiket >=100, dan rentang tiket <=5; 30 di antaranya Believe.

Penjelasan domain yang teramati: TNI AU dan TNI AL mencatat nobar Believe sebelum rilis luas 24 Juli. Akun resmi film Panggil Aku Ayah mengumumkan special screening 25 kota pada 3 Agustus sebelum rilis 7 Agustus. Peristiwa khusus dapat menjelaskan sebagian screening penuh. Kekonstanan tiket hingga 20 hari di banyak klaster adalah jejak sangat teratur, sehingga perlu audit provenance tambahan; tidak cukup untuk menyatakan keseluruhan dataset sintetis.

Keputusan: keluarkan preview dari window D1-D3 utama; jangan memperlakukan permintaan preview sebagai sinyal umum tanpa memisahkan event/group screening. Jika fitur sejarah pra-rilis dipakai, flag fase dan pemutaran 100% serta uji sensitivitas tanpa film ekstrem. Ini hipotesis FE/validasi, bukan eksperimen modelling.
