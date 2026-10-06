# Data eksternal

Aturan: informasi atau versinya harus publik **paling lambat 30 Sep 2025**, dan tanggal itu harus bisa
dibuktikan. Semua sumber yang dipakai adalah fakta kalender pemerintah. Tabelnya ada di Section 7 notebook.

## Dipakai

| Sumber | Penerbit / tanggal | Dipakai untuk |
| :--- | :--- | :--- |
| SKB 3 Menteri Libur Nasional & Cuti Bersama 2025 (kemenkopmk.go.id) | Kemenko PMK, Okt 2024 (revisi Agu 2025) | Cuti bersama 2025 |
| SKB 3 Menteri 2026 (setneg.go.id) | Kemensetneg, 19 Sep 2025 | Cuti bersama 2026; 1 Syawal 1447 H = 21 Mar 2026; Ramadan 19 Feb - 20 Mar 2026 (diturunkan) |
| Kalender Pendidikan DKI 2024/2025 (Tirto) | 2024 | Libur 28 Jun - 12 Jul 2025 |
| Kalender Pendidikan DKI 2025/2026, Kepdis 89/2025 (Detik) | 7 Agu 2025 | Libur 22-31 Des 2025 |
| SE Kadisdik Jabar 21808/PK.02.01.05/Sekre (PDF mmc.tirto.id) | 10 Jun 2024 | Jabar: libur 30 Jun - 12 Jul 2025 |
| SE Kadisdik Jabar 14995/TU.03/PSMA (komunitasbelajar.id 2025/06; turunan Disdik Ciamis 2025/07, kini 404) | 20 Jun 2025 | Jabar: libur 29 Des 2025 - 10 Jan 2026 |
| Pemberitaan April 2025: 2 juta penonton dalam 3 hari libur Lebaran, 5 juta selama libur Lebaran | Apr 2025 | Pendukung analog Lebaran (bukan fitur) |

## Dicek, tidak dipakai

- **Kalender sekolah provinsi lain** (Banten, Jateng, Jatim, DIY, Bali, Lampung, Sulsel, Sulut, Sultra,
  Sulteng, Kalimantan, Maluku, Papua, NTB/NTT).
  - Libur mereka 20 atau 22 Des sampai 1-4 Jan, sehingga jendela hari kerjanya sama dengan DKI. Tanggal 25-26
    Des dan 1 Jan sudah libur nasional atau cuti bersama.
  - Peta provinsi diambil dari artikel Kontan 8 Des 2025. Tanggal artikelnya setelah cutoff, tetapi hanya
    dipakai untuk memilih provinsi yang perlu dicek, bukan sebagai sumber fitur.
  - 82% baris uji ada di luar Jabodetabek.
- **Cuaca periode uji:** terbit setelah cutoff, jadi dilarang.
- **IMDb, TMDB, rating, review, views trailer, Google Trends untuk film uji:** versinya setelah cutoff atau
  tidak bisa dibuktikan. Dataset IMDb diperbarui setiap hari. Hanya mungkin lewat snapshot Wayback bertanggal
  ≤ 30 Sep 2025, dan tetap tidak berguna karena level film tidak terprediksi dari metadata (`eda/38`).
- **Demografi kota (BPS):** prioritas rendah; `cin_size` sudah menangkap ukuran.
- **Box office historis (filmindonesia.or.id) untuk track record:** berat dikumpulkan, dan learning curve
  serta talent tidak menjanjikan.
- **Identitas franchise, sekuel, distributor, negara, durasi:** sebagian diturunkan dari data resmi (A2,
  gagal). Sisanya butuh sumber bertanggal.

## Prinsip

- "Yang dinilai aturan adalah kapan informasi atau versinya tersedia, bukan tahun filmnya."
- Data resmi lomba tidak terkena batas publikasi.
- Fitur dari `test_history` (misalnya film lain di jendela D1-D3) sah karena itu paket resmi. Validasinya harus
  meniru informasi yang terlihat.
