# Data dan rekonstruksi sampel latih

`train.csv` hanyalah riwayat transaksi, tanpa penanda D1-D10. Sampel latih dibuat ulang dengan meniru cara
panitia membangun `test_history` dan `test`. Kodenya ada di `src/common.py` (`release_dates`, `simulate`) dan
disalin ke Section 6 notebook.

## Fakta cara panitia membangun test (`eda/01`)

- **Seleksi pasangan = laku di D3.**
  - 10.373 dari 10.373 pasangan uji memiliki transaksi D3.
  - 1.450 pasangan yang terakhir laku di D1 atau D2 tidak ada di `test.csv`.
- D1 = tanggal rilis resmi, bukan transaksi pertama: 10 film uji sudah muncul di train sebagai pratayang.
- D1 film uji: Rabu 67, Kamis 65, Jumat 24, Sabtu 5, Selasa 2.
- Data bersih: 0 NA, 0 duplikat (tanggal, klaster, film), satu klaster = satu kota, tidak ada transaksi dengan
  `total_ticket = 0`.
- 4 klaster uji tidak ada di train (602 baris). Fitur klaster boleh NaN.
- Test berisi **rilis luas** saja; film 2D terkecil dibuka di 28 klaster.

## Aturan D1 (`release_dates`)

- D1 = hari pertama cakupan klaster judul dasar mencapai **50% puncak**, dicari di dalam segmen tayang
  kontinu yang memuat puncak. Ini melewati pratayang, misalnya A MINECRAFT MOVIE: pratayang 4-6 Apr, rilis
  9 Apr.
- Judul dasar (2D/3D/IMAX disatukan dengan `base_title`) berbagi D1.
- Filter rilis luas: cakupan D1 judul dasar ≥ 25 klaster.
- Film rilis terbatas (36 film, 1.435 baris) dipakai **hanya sebagai data latih tambahan**. Ini membantu
  sekitar -0,0035 (`eda/14`).
- Audit `eda/53`: aturan ini melihat puncak masa depan.
  - Aturan yang benar-benar kausal (jendela D1-D3 sendiri ≥ 25 klaster setiap hari) memindahkan 6 film,
    mengubah skala 191 pasangan, dan membuang 15 film yang dibuka lebar lalu runtuh.
  - **Test justru memuat pembukaan runtuh seperti itu**: 8 film uji punya hari dengan kurang dari 25 klaster
    (SEND HELP 52, 49, 1). Jadi aturan sekarang konsisten dengan seleksi panitia dan dipertahankan.

## Simulasi (`simulate`)

- History = semua transaksi film pada D1-D3.
- Target = pasangan yang laku di D3, x 7 hari D4-D10, nol diisi eksplisit.
- Film yang D10-nya melewati 30 Sep 2025 dibuang (target tidak lengkap).
- **Outage pelaporan** di train: 7, 9, 10, 16 Juni 2025 (hanya 1-67 klaster melapor) dan 13 Juni hilang total.
  - Film yang D1-D3-nya kena outage tidak dijadikan sampel.
  - Baris target pada tanggal outage dibuang, karena zero-fill akan menciptakan nol palsu.
- Self-check di notebook: film sintetis, dengan klaster yang tidak laku di D3 dibuang dan hari kosong menjadi 0.
- Hasil: **56.372 baris target dari 143 judul (126 judul dasar), 8.113 pasangan**. Uji: 72.611 baris, 10.373
  pasangan.

## Skala dan target

- `scale = max(mean(y1, y2, y3), 1)`, identik dengan `hitung_skala` resmi (ada assertion).
- Target model: `r = y / (scale * cal_mult)`.
  - `cal_mult = c(t) / mean c(D1..D3)` adalah pengali kalender struktural.
  - Loss L1 dibobot `cal_mult` setara MASE.
- Median r = 0,42, rata-rata 0,61. 31% target nol, dan porsi nol naik dari 10% (D4) ke 54% (D10) karena film
  dicopot.

## Perbedaan komposisi train-sim vs uji (`eda/12`, `17`, `44`, `51`)

| Statistik | Train-sim | Uji |
| :--- | :--- | :--- |
| Median skala | 181 | 91,7 |
| Baris skala ≤ 20 | 3,2% | 13,0% |
| Late starter (laku pertama di D3) | 4,2% (bobot bucket) / 1,6% mentah | 1,8% |
| Okupansi median D1-D3 | 21% | 11% |
| Zero-rate D3 pada skala 20-50 | 0,39 | 0,11 |

- Periode uji (Okt 2025 - Mar 2026) adalah **pasar sepi**: film lebih kecil dan okupansi lebih rendah.
- Bioskop **jauh lebih jarang mencopot film** pada tingkat penonton yang sama.
- **30% baris uji ada di periode tanpa padanan di train**: Ramadan 17%, Natal/Tahun Baru 7%, Lebaran 6%.
- Adversarial validation (grouped per film) AUC 0,736. Fitur pembeda utama: `f_occ`, `f_nc_trend`, `f_logT`.
  Split acak memberi AUC palsu 1,0.
