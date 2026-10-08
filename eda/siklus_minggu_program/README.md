# Nol mengikuti siklus minggu program (Rabu/Kamis), bukan umur film

**Satu kalimat**: fraksi nol melonjak saat window melewati Rabu/Kamis minggu berikutnya (film D1 Rabu: 13% di D7 lalu 32% di D8; film D1 Jumat: 28% di D5 lalu 49% di D6 yang jatuh Rabu), nol bersifat menyerap (hanya 5% pasangan hidup lagi), dan scale, show D3, tiket per show, serta okupansi memisahkan fraksi nol dari 69% sampai 2%.

## Pertanyaan

Target D4-D10 punya 32% nol. Kapan dan pada pasangan apa nol muncul?

## Metode

`python eda/siklus_minggu_program/dinamika.py` (butuh output `eda/window_seleksi_d3`). Tabel fraksi nol per (hari D1, horizon) dan per (hari D1, hari target), deteksi pola hidup lagi setelah nol, tabel desil fitur D1-D3 vs fraksi nol dan median y/scale, efek day_tipe/libur.

## Hasil

Fraksi nol per horizon, baris = hari D1:

| D1 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|
| Rabu | 0,06 | 0,10 | 0,11 | 0,13 | **0,32** | 0,44 | 0,49 |
| Kamis | 0,08 | 0,13 | 0,18 | 0,29 | **0,42** | 0,51 | 0,55 |
| Jumat | 0,19 | 0,28 | **0,49** | 0,62 | 0,70 | 0,70 | 0,69 |

- Hari terakhir positif: D3 7,7%, D7 13,1% (hari sebelum Kamis berikutnya untuk film Kamis), D10 45,1%. Pasangan yang hidup lagi setelah nol: 5,1%.
- Desil scale: fraksi nol 0,69 (desil 1) sampai 0,02 (desil 10). total_show_d3 1 sampai 2: nol 0,72; >= 16: 0,06. Tiket per show D3 < 10,75: 0,68; > 76: 0,13. Okupansi D1-D3 < 7,4%: 0,60; > 48%: 0,12.
- Rasio y/scale (hari positif) median: weekend 0,97, weekday 0,53, weekday libur 1,17.
- Plot: `output/dinamika.png`. Angka lengkap: `output/stdout_dinamika.txt`.

## Interpretasi

Bioskop menyusun program mingguan yang berganti saat film baru rilis (Rabu/Kamis). Pasangan lemah dicabut di pergantian itu, bukan perlahan. Ini sejalan dengan praktik industri: okupansi hari pertama di bawah 10% "lampu kuning", di bawah 6% bisa langsung turun layar (docs/research_kasus_serupa.md). Karena metrik MAE, bila peluang nol di atas 50% prediksi optimal adalah 0.

## Saran FE / modelling

| Saran | Alasan | Prioritas |
|---|---|---|
| Fitur jumlah Rabu dan Kamis yang dilewati sejak D1 sampai tanggal target, plus hari sejak pergantian program terakhir | Cliff mengikuti kalender, bukan horizon | tinggi |
| Horizon h dan hari target sebagai fitur, plus hari D1 | Pola nol berbeda per hari D1 | tinggi |
| Level dan tren D1-D3: scale, total_show_d1..3, tren show d3/d1, tiket per show d3, okupansi | Pemisah nol yang monoton | tinggi |
| Model dua bagian (peluang tayang x level bila tayang) sebagai alternatif direct L1 | Target campuran nol menyerap + level | sedang |
| Hari target weekend / libur weekday | Rasio 0,53 vs 0,97 vs 1,17 | tinggi |

## Tingkat keyakinan dan status

- Keyakinan: tinggi (pola konsisten untuk tiga hari D1 dengan jumlah film besar). Hari D1 Senin/Selasa/Sabtu/Minggu sampelnya kecil.
- Sudah divalidasi dengan eksperimen: belum.
