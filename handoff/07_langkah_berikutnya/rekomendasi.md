# Rekomendasi untuk chat berikutnya

## Yang sebaiknya diterapkan

1. **Pertahankan fondasi v12:**
   - simulasi;
   - fold per judul dasar;
   - statistik fold-local;
   - empat lensa (TW bfd, temporal, bootstrap film, κ sebagai uji stres);
   - aturan adopsi tertulis;
   - SHA-256;
   - `STRICT`;
   - batas 200 MB.
2. **Untuk setiap model kuantil baru:** nilai λ = 0 dan λ = 0,5 secara terpisah. Jangan wariskan pull shift.
3. **Prioritas kandidat model** (semuanya ≤ 200 MB): TabPFN-2.5 Quantiles dan TabM sudah ada. Berikutnya:
   - diagnosis Causilo (murah, karena kode sudah ada di `build_v11.py`);
   - TabPFN-2.6 dengan λ = 0;
   - varian checkpoint TabPFN-2.5 lain.

   Evaluasi semuanya pada fitur B1 yang dibekukan.
4. **Bila menambah fitur:** ukur di grouped CV **dan** temporal. Panel dan show/attendance terlihat bagus di
   grouped tetapi gagal temporal.
5. **Smoke test di data nyata** (subset Xtr dari `outputs/cache/Xtr.parquet`), bukan data sintetis.
6. **Bacaan skor:**
   - selisih validasi < 0,002 = noise (bootstrap film);
   - jangan klaim perbaikan public dari validasi;
   - laporkan interval per film.

## Yang sebaiknya dihindari

- Probing leaderboard dalam bentuk apa pun.
- Menjalankan training model lokal.
- Menimpa notebook yang sudah di-run.
- Menambah GBDT lain, seed bagging, fine-tuning TabPFN, rounding, atau kalibrasi kecil. Sudah terbukti noise.
- Menaikkan prediksi global demi film breakout. Optimal MASE adalah median, dan menaikkan prediksi merusak
  banyak film yang tidak tumbuh.
- Data eksternal tentang film uji tanpa snapshot bertanggal ≤ 30 Sep 2025.
- Memilih bobot atau model dari dunia κ saja, atau dari label asli saja. Pakai minimax dan cetak semua lensa.

## Realisme target

- Top 1 public 0,34456 vs kita 0,39991 (terbaik) / 0,41060 (v12).
- Tidak ada tuas sah yang kami temukan sebesar sekitar 0,055.
- Oracle menunjukkan galat terbesar ada pada level film dan pertumbuhan setelah D3, yang tidak terlihat di
  D1-D3.
- Kemungkinan penjelasan gap: komposisi public yang menguntungkan, metode lain, atau penggunaan informasi yang
  tidak bisa kita verifikasi. **Tidak ada dasar untuk menuduh**, dan metode mereka tidak diketahui.
- Fokus pada private yang kokoh (90% nilai) dan dokumentasi (10%).
