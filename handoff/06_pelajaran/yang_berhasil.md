# Apa yang berhasil (dan sebaiknya dipertahankan)

1. **Meniru konstruksi test dengan presisi**: seleksi laku di D3, D1 rilis resmi, rilis luas, zero-fill, dan
   membuang outage. Ini fondasi semua hal lain.
2. **Mencari jarak validasi ke LB dengan EDA struktural, bukan probing:**
   - komposisi skala uji (TW);
   - pergeseran kebijakan pencopotan, terukur dari label periode uji (proxy D1,D2 ke D3);
   - level Lebaran dari analog Lebaran 2025 dan cek kapasitas kursi.

   Dua yang terakhir memberi **lompatan public terbesar**: 0,457 ke 0,448 ke 0,409.
3. **Proxy berlabel periode uji** sebagai satu-satunya bukti di luar train.
4. **Model hurdle + median campuran** untuk target dengan massa nol besar (MASE = median).
5. **TabPFN in-context per horizon** sebagai komponen terkuat. TabPFN-3.5 terbaik di public; TabPFN-2.5
   Quantiles terbaik di validasi di antara model ≤ 200 MB.
6. **Disiplin evaluasi sejak v10:**
   - fold per judul dasar;
   - statistik fold-local;
   - backtest temporal;
   - bootstrap per film;
   - aturan adopsi ditulis sebelum hasil;
   - satu perubahan per kandidat;
   - λ per model;
   - minimax antar-lensa ditambah pembanding bebas κ.

   Ini mencegah adopsi perubahan yang hanya noise.
7. **Notebook reproducible:**
   - versi terkunci;
   - seed;
   - SHA-256 checkpoint;
   - checkpoint tertanam di berkas bobot ≤ 200 MB;
   - `STRICT` fail loudly;
   - cek batas bobot sebelum inferensi panjang;
   - fallback `FM_MODE`.
8. **Review silang (Astra)** menemukan bug nyata sebelum run Kaggle yang mahal. Selalu verifikasi klaim di
   kode, perbaiki, lalu tambahkan self-check.
9. **Build notebook dengan skill `ref/notebook/`** dan narasi Indonesia yang lengkap, untuk nilai dokumentasi
   10%.
