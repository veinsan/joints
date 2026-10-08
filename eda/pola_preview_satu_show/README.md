# Rezim pemutaran preview satu show

## Metode

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/pola_preview_satu_show/preview_audit.py` dari root. Script menggabungkan setiap baris `train.csv` dengan D1 luas hasil builder v2, lalu membandingkan baris sebelum D1, D1-D3, dan sesudah D3. Semua hitungan deskriptif. Output rinci ada di `output/`.

## Hasil

| Fase | Baris satu show | Median tiket | Okupansi tepat 100% |
|---|---:|---:|---:|
| Preview | 1.647 | 142 | 69,03% |
| D1-D3 | 1.252 | 21 | 5,19% |
| Setelah D3 | 4.707 | 24 | 7,03% |

Preview memiliki 3.319 baris pada 134 film; hampir separuhnya hanya satu show. Tanpa **Believe**, 605/1.113 atau 54,36% baris preview satu show tetap berokupansi tepat 100%. Angka ini berbeda tajam dari penayangan reguler.

**Believe - Takdir, Mimpi, Keberanian** memiliki 734 baris preview, 170.254 tiket, 21 tanggal, dan 48 klaster sebelum D1 luas 24 Juli. Dari 534 baris preview satu show, 532 berokupansi tepat 100%. Satu pasangan film-klaster di Madiun tercatat 20 hari berturut-turut, setiap hari satu show dengan 165-168 tiket dan okupansi tepat 100%. Pada 37 pasangan film-klaster yang punya sedikitnya tiga preview satu show, median >=100 tiket, dan rentang tiket <=5, sebanyak 30 adalah Believe.

[TNI AU](https://www.tni-au.mil.id/berita/detail/sekkau-rencanakan-nobar-film-believe) dan [TNI AL](https://www.tnial.mil.id/berita/84287/BANGKITKAN-PATRIOTISME%2C-KODIKLATAL-GELAR-NONTON-BERSAMA-FILM-BELIEVE/) mencatat kegiatan nobar Believe sebelum tanggal rilis umum 24 Juli. [Akun resmi Panggil Aku Ayah](https://linktr.ee/filmpanggilakuayah) juga mengumumkan special screening 25 kota pada 3 Agustus sebelum rilis 7 Agustus. Ini mendukung bahwa preview merupakan jenis pemutaran tersendiri. Angka tiket yang sangat stabil bisa timbul dari block booking, aturan pencatatan, atau konstruksi data. Sumber asli dataset belum tersedia untuk membedakannya.

## Implikasi

Pertahankan pemisahan preview dan rilis luas ketika menyusun train window. Jika sinyal pra-rilis dipakai dalam FE, bedakan screening acara atau rombongan dari penayangan reguler; jangan langsung mengartikan preview yang penuh sebagai permintaan organik. Lakukan analisis sensitivitas per film agar Believe tidak mendominasi kesimpulan. Pola ini membatasi inferensi tentang realisme data preview, tetapi belum cukup untuk memberi label sintetis pada dataset keseluruhan.
