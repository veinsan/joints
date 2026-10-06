# Apa yang salah dan apa yang tidak efektif

## Kesalahan saya (agar tidak diulang)

1. **Menimpa notebook yang sudah di-run user.**
   - `build_v6.py` sempat menulis ke `v5.ipynb`. Dipulihkan lewat `git checkout -- notebooks/v5.ipynb`.
   - Sejak itu: ubah `OUT` dulu, cek notebook tujuan tidak punya output, baru build.
2. **Menjalankan model berjam-jam di laptop user yang sedang memakai baterai** (EXAONE CPU 2.303 detik per
   fold x horizon, TabPFN CPU, dry-run penuh). User menegur.
   - Aturan: hanya EDA/preprocessing lokal. Model dievaluasi di notebook Kaggle.
3. **`pkill -f <pola>` membunuh shell sendiri**, karena polanya cocok dengan command bash itu sendiri, dan
   menyebabkan exit 144. Pakai PID atau pola `[x]yz`.
4. **Klaim terlalu kuat.**
   - "Tidak ada tuas sah > 0,003 yang tersisa" (v8) dikoreksi audit: eksperimen yang gagal hanya menolak
     implementasi yang diuji, bukan seluruh kelas metode.
   - "Validasi κ cocok dengan LB" (v5/v6) ternyata kebetulan dua kesalahan.
5. **Salah baca statistik.** Median rasio skala = 1 dibaca sebagai "skala tidak berubah" (`eda/53` versi
   awal), padahal 125 pasangan berubah. Selalu hitung jumlah pasti.
6. **Audit D1 versi awal belum kausal**: aturan "kausal" masih memilih segmen lewat puncak (`idxmax`).
   Ditemukan oleh Astra.
7. **Smoke test dengan data sintetis yang terlalu bersih.** TabM lolos smoke test tetapi akan crash di data
   nyata, karena kolom konstan ditolak `compute_bins`. Smoke test harus memakai data nyata (subset).
8. **Bug logika seleksi** yang ditemukan Astra:
   - gabungan kandidat bisa menang walau lebih buruk dari kandidat tunggal terbaik;
   - blend memilih TW terbaik dulu lalu cek syarat, sehingga bobot lain yang lolos terlewat;
   - inferensi final gagal bila bobot LightGBM nol;
   - preprocessing TabM melihat data early-stopping.
9. **Bobot validasi per bucket saja (v3-v7)** melebih-bobotkan late starter. Keputusan v4-v7 diambil dengan
   metrik itu.
10. **Dunia κ dibangun dari kuantil LightGBM**, sehingga bias ke LightGBM saat memilih bobot blend. v9 memilih
    LightGBM 0,6 karena itu.
11. **v10 memilih bobot EXAONE 0,7 hanya dari label asli**, padahal κ menyukai sekitar 0,4. Public memburuk ke
    0,40823.
12. **Asumsi awal bahwa batas tanggal berlaku untuk pretrained (v3)** membuat v3 memakai TabPFN v2 yang
    lebih lemah. Public memburuk ke 0,45731, sampai user memutuskan sebaliknya.
13. **Probing LB di awal proyek.** Dibuat berkas submisi probe (`outputs/probes`, `eda/11`), walaupun tidak
    pernah dipakai. User melarangnya, dan berkasnya sudah dihapus.

## Yang tidak efektif (pola umum)

- **Mengganti atau menambah foundation model berdasarkan klaim benchmark** (TabArena: Causilo, TabICL;
  EXAONE). Di data ini kecocokan dengan median campuran dan massa nol lebih penting daripada peringkat
  benchmark.
- **Menambah GBDT lain (CatBoost/XGBoost) atau seed bagging:** korelasi galat 0,99, jadi tidak ada keragaman.
- **Fine-tuning TabPFN dan konteks gabungan:** lebih buruk atau terlalu mahal.
- **Fitur baru dari D1-D3** (panel, show, attendance, momentum klaster, metadata): sebagian terlihat bagus di
  grouped CV tetapi gagal di backtest temporal. Galat utama adalah pertumbuhan **setelah** D3, yang tidak
  terlihat di D1-D3.
- **Kalibrasi atau penyesuaian kecil** (rounding, Platt, retune knob, DOW): semuanya di bawah noise (sekitar
  0,001-0,003).
- **Data eksternal tentang film uji:** tidak tersedia secara sah sebelum cutoff.
- **Mengejar validasi lokal:** validasi v8-v12 datar di sekitar 0,325, sedangkan public bergerak 0,400-0,411.
  Selisih kecil di validasi (< 0,002) tidak memprediksi public.

## Yang terlambat disadari

- **Model yang kuat di periode uji (TabPFN-3.5) lebih berharga daripada perbaikan validasi.** Setelah diganti
  karena batas 200 MB, public memburuk sekitar 0,01. Jawaban panitia soal checkpoint menentukan apakah v8 bisa
  dipakai.
- **Satuan informasi adalah film (126 judul), bukan baris (56 ribu).** Bootstrap per film hampir selalu
  melewati nol untuk perbedaan versi di bawah sekitar 0,003.
