# Anomali okupansi nol dan kapasitas kursi tersirat

**Satu kalimat**: 459 baris bertiket positif memiliki okupansi nol; 111 kasus train terkumpul pada 17 September 2025, termasuk seluruh 73 kasus dengan >=20 tiket, sedangkan kapasitas tersirat klaster tetap stabil lintas periode (Spearman 0,937).

## Pertanyaan

Apakah `occupation_rate=0` menyatakan kursi benar-benar kosong, dan apakah tiga variabel operasional konsisten?

## Metode

Jalankan `python eda/anomali_okupansi_kapasitas/capacity_audit.py` dan `python eda/anomali_okupansi_kapasitas/zero_occupancy.py` dari root. Kapasitas tersirat dihitung sebagai `100*tiket/(okupansi*show)` untuk baris show >=3 dan okupansi >=5; klaster dengan minimal 30 observasi di kedua periode dibandingkan. Script kedua menelusuri semua baris bertiket positif dengan okupansi nol. Output ada di `output/`.

## Hasil

- Kapasitas tersirat median per show: 150,5 train dan 158,6 test_history.
- 114 klaster dengan observasi memadai: Spearman kapasitas median 0,937; median selisih relatif absolut 1,63%.
- Tiket positif dan okupansi nol: 237/138.959 baris train, 222/32.323 test_history.
- 17 September 2025: 111/816 baris train berokupansi nol walau bertiket positif. Semua 73 kasus bertiket >=20 berada pada tanggal ini.
- JADI TUH BARANG pada 17 September: 37/37 baris klaster berokupansi nol, median 128 tiket. [LSF](https://lsf.go.id/en/film/jadi-tuh-barang/392) mencatat premiere 18 September 2025.
- Setelah pembentukan window resmi D1-D3, anomali ini menyentuh 17/7.988 pasangan train (0,21%; 4 dari 17 terkait 17 September) dan 100/10.373 pasangan test (0,96%).

## Interpretasi

Nilai nol itu tidak dapat diperlakukan sebagai okupansi fisik nol. Anomali tiket besar pada satu tanggal mengarah pada masalah pelaporan parsial atau status pertunjukan khusus. Nilai nol bertiket sangat kecil pada hari lain mungkin punya mekanisme berbeda. Data memiliki struktur operasi yang realistis, tetapi ini tidak membuktikan apakah data mentah asli atau hasil sintesis berbasis data asli.

## Saran FE / validasi

| Saran | Alasan | Prioritas |
|---|---|---|
| Flag `occupation_rate==0` ketika tiket >0 pada D1-D3 | Nol merupakan nilai kualitas data, bukan permintaan nol; jumlah pasangan terdampak kecil | sedang |
| Tiket per show sebagai sinyal alternatif pada hari okupansi anomali | Tiket dan show masih positif | sedang |
| Audit khusus film dengan D1-D3 mengenai 17 September 2025 | Hanya 4 pasangan window terdampak | rendah |

## Tingkat keyakinan dan status

Keyakinan tinggi bahwa okupansi nol pada tiket positif bukan nilai fisik. Penyebab data dan status sintesis tetap tidak terverifikasi. Belum diuji dengan eksperimen modelling.
