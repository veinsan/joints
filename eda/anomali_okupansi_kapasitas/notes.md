# EDA 073: kapasitas tersirat dan okupansi nol

Pertanyaan: apakah tiket, jumlah show, dan okupansi konsisten secara mekanis, dan apakah angka okupansi nol bermakna?

Hasil: kapasitas per show tersirat `100*tiket/(okupansi*show)` berkorelasi Spearman 0,937 pada 114 klaster yang memiliki minimal 30 baris berkualitas di kedua periode. Selisih median absolut kapasitas antarperiode 1,63%. Baris bertiket positif tetapi okupansi nol berjumlah 237/138.959 di train dan 222/32.323 di test_history. Pada 17 September 2025 terdapat 111 dari 816 baris seperti itu, termasuk semua 73 baris bertiket >=20 yang memiliki okupansi nol di seluruh data. Film JADI TUH BARANG memiliki 37/37 klaster berokupansi nol pada tanggal itu dengan median 128 tiket; tanggal tayang resmi menurut LSF adalah 18 September 2025.

Kesimpulan: data mempunyai struktur kapasitas klaster yang stabil, tetapi `occupation_rate=0` dengan tiket positif adalah sentinel atau error pengukuran, bukan keterisian fisik nol. Anomali 17 September tampaknya gangguan pelaporan okupansi parsial; dugaan ini tidak membuktikan sumber asli atau sintesis. Jangan tafsirkan nilai tersebut sebagai sinyal permintaan rendah; pertimbangkan indikator missing dan alternatif tiket per show pada D1-D3. Jangan mengubah target tiket.
