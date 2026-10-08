# Aturan host: pasangan test = ada transaksi di D3, D1 = hari wide release

**Satu kalimat**: pasangan (film, klaster) di test_history masuk test.csv jika dan hanya jika punya transaksi di D3 (10.373 masuk, 1.450 dibuang, aturan ini benar 100%), dan D1 adalah hari wide release setelah preview di 1 sampai 2 klaster; meniru dua aturan ini di train memberi 172 film / 7.745 pasangan dengan proporsi hari hadir D1-D3 92,3/6,0/1,7% vs test 91,1/7,0/1,8%.

## Pertanyaan

`train.csv` hanya riwayat transaksi mentah. Agar data latih meniru test, perlu tahu bagaimana host memilih D1 dan pasangan.

## Metode

- `definisi_window.py`: bandingkan pasangan test_history vs test.csv (hari terakhir, jumlah hari, tiket), hari-dalam-minggu D1 film uji, dan pola klaster aktif per hari film di train (preview vs wide release). 10 film uji yang juga muncul di train dipakai untuk melihat preview sebelum D1.
- `build_windows.py` (modul): D1 = tanggal pertama jumlah klaster aktif >= 50% maksimum film; pasangan dipakai bila ada transaksi di D3; target D4-D10 diisi 0 bila tidak ada baris; buang film left-censored, film yang D10 melewati 30 Sep, dan film yang bersinggungan dengan outage 7 sampai 16 Jun 2025.
- `panel_vs_test.py`: bangun panel, bandingkan distribusi dengan test, baseline in-sample.
- Jalankan dari root: `python eda/window_seleksi_d3/definisi_window.py` lalu `python eda/window_seleksi_d3/panel_vs_test.py`.

## Hasil

| Ukuran | Train window | Test |
|---|---|---|
| Hari D1 terbanyak | Kamis 74, Rabu 57, Jumat 40 | Rabu 64, Kamis 65, Jumat 24 |
| Hari hadir D1-D3 (1/2/3) | 1,7 / 6,0 / 92,3% | 1,8 / 7,0 / 91,1% |
| Pasangan dengan scale = 1 | 0,01% | 0,07% |
| Median tiket D3 / D1 | 0,91 | 0,88 |

- in_test menurut hari terakhir hadir: D1 = 0,0, D2 = 0,0, D3 = 1,0 (`output/stdout_definisi_window.txt`).
- Target: nol per horizon D4..D10 = 9, 14, 21, 29, 43, 51, 55%; median y/scale = 0,92, 0,61, 0,47, 0,36, 0,17, 0, 0.
- Contoh preview: SUPERMAN 8 Jul 1 klaster lalu 9 Jul 116 klaster; JALAN PULANG preview 4 sampai 15 Jun, wide 19 Jun.
- Output: `output/train_windows_wide.parquet`, `output/train_windows_target.parquet`, `output/test_windows_wide.parquet`, `output/rasio_horizon_scale.png`.

## Interpretasi

Host membangun test dengan: pilih film yang wide release di Okt 2025 sampai Mar 2026, ambil 3 hari pertama wide release sebagai history, simpan pasangan yang masih menjual tiket di D3. Pasangan yang sudah berhenti di D1/D2 tidak dinilai. Preview sebelum D1 tidak diberikan untuk film uji (kecuali 10 film yang preview-nya jatuh di periode train).

## Saran FE / modelling

| Saran | Alasan | Prioritas |
|---|---|---|
| Bangun data latih dengan `build_windows.py` (aturan D3 + D1 wide) | Meniru distribusi test hampir persis | tinggi |
| Target dinormalisasi `y / scale` dengan loss L1 (atau objektif lain yang dikoreksi ke median) | Metrik = MAE ter-skala, prediksi optimal median kondisional | tinggi |
| Jangan pakai fitur preview sebelum D1 | Hanya tersedia untuk 10/160 film uji | tinggi |
| Uji sensitivitas definisi D1: FRAC_WIDE 0,3 dan 0,7 | Beberapa film rilis bertahap (FINAL DESTINATION: 81 klaster Jumat lalu 115 Rabu) | rendah |

## Tingkat keyakinan dan status

- Keyakinan: tinggi untuk aturan D3 (deterministik 100%). Tinggi untuk definisi D1 (distribusi cocok), sedang untuk film dengan rilis bertahap.
- Sudah divalidasi dengan eksperimen: baseline EXP-011 memakai window ini.
