# Outage data 7 sampai 16 Juni 2025

**Satu kalimat**: pada 7 sampai 16 Jun 2025 sebagian besar klaster tidak melapor (13 Jun 0 dari 116 klaster, 9 Jun 1 klaster, 11 Jun 136 baris vs normal sekitar 700), sehingga 8 film yang window D1-D10-nya bersinggungan harus dibuang agar target nol palsu tidak masuk data latih.

## Pertanyaan

Apakah ada hari dengan data hilang yang akan terbaca sebagai "film dicabut"?

## Metode

`python eda/outage_juni_2025/outage.py`: jumlah klaster dan baris per hari, rasio tiket terhadap median 15 hari, klaster yang hilang dibanding 7 hari sebelumnya, cek pola serupa di test_history.

## Hasil

| Tanggal | Klaster | Baris | Tiket / median 15 hari |
|---|---|---|---|
| 7 Jun | 10 | 10 | 0,08 |
| 9 Jun | 1 | 1 | 0,003 |
| 10 Jun | 67 | 449 | 1,14 |
| 11 Jun | 76 | 136 | 0,63 |
| 13 Jun | 0 | 0 | 0 |
| 14 Jun | 83 | 169 | 0,56 |
| 15 Jun | 87 | 326 | 1,00 |
| 16 Jun | 62 | 404 | 0,40 |

Normal: 116 klaster, sekitar 600 sampai 900 baris. 25 Agu sampai 2 Sep 2025 turun (1 sampai 2 Sep 0,48) tetapi klaster lengkap, jadi permintaan nyata. Plot `output/outage.png`.

## Interpretasi

Masalah pengumpulan data, bukan perilaku penonton. Tidak terlihat di test_history (penurunan jumlah klaster per film D1-D3 wajar).

## Saran FE / modelling

| Saran | Alasan | Prioritas |
|---|---|---|
| Buang window yang bersinggungan 7 sampai 16 Jun (sudah di `eda/window_seleksi_d3/build_windows.py`) | Target nol palsu | tinggi |
| Jangan hitung statistik historis klaster/pasar dari hari outage | Bias rendah | sedang |
| 25 Agu sampai 2 Sep: biarkan (nyata), boleh diberi penanda | Permintaan turun nyata | rendah |

## Tingkat keyakinan dan status

- Keyakinan: tinggi.
- Sudah diterapkan di window (172 film, sebelumnya 180).
