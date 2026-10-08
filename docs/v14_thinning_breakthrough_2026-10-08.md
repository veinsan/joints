# Analisis 8 Oktober 2026: dari offset misterius ke binomial thinning (dasar v14)

v13 public 0,40244 (v8 0,39991, v12 0,41060, top 1 0,34456). Dua belas script baru (`eda/79` sampai `eda/90`). Tidak ada model yang dilatih untuk evaluasi, tidak ada submisi, dan tidak ada label uji. Skor public versi yang sudah disubmit dipakai sebagai data, sesuai preseden `eda/39`. Hasil Astra (`eda/75` sampai `eda/78`, `validation_audit.ipynb`) berupa infrastruktur validasi (geometri fold, split target-like) dan belum memberi sinyal prediktif baru.

## Ringkasan

1. File resmi bersih. `test.csv` terurut per (film, klaster, tanggal), dan `sample_submission` bernilai konstan 100. Tidak ada kebocoran struktural.
2. **Offset validasi ke public bukan bias level label** (`eda/81`, `eda/82`). Bila label uji mengikuti kalibrasi OOF v13 per sel (h x hari pertama laku x skala x bin prediksi), MASE harapan baris non-Lebaran sudah 0,386 sampai 0,397. Dunia dengan level label jauh lebih tinggi (m >= 1,2) membutuhkan galat Lebaran negatif, jadi mustahil. Offset 0,07 adalah komposisi uji di ruang prediksi.
3. **Jalan yang tertutup.**
   - Klaster x tanggal: oracle 0,202 di `eda/37` adalah artefak self-leakage. Dengan label film lain di (klaster, tanggal) yang sama, TW justru memburuk ke 0,355 (`eda/83`).
   - Tekanan kompetisi per tanggal (`eda/85`) tidak punya sinyal.
   - Late starter (`eda/80`) sudah diprediksi dengan benar di median. Galatnya adalah ekor yang tidak bisa diprediksi.
4. **Informasi yang hilang adalah lintasan film setelah D3** (`eda/84`). Leave-one-out film x tanggal dari klaster lain: korelasi 0,57, batas -0,046. Lintasan ini tidak terlihat di data sah, dan dengan statistik D1 sampai D3, |ρ| < 0,18. Batas itu hampir sama dengan selisih ke top 1. Dugaan, **bukan tuduhan**: top 1 mungkin punya cara mengetahui lintasan nasional film uji. Data box office film uji terbit setelah batas 30 September 2025, sehingga tidak sah untuk kita.
5. **Terobosan yang sah: binomial thinning** (`eda/87` sampai `eda/90`). Periode uji berperilaku seperti train dengan pembeli lebih sedikit.
   - **Label periode uji (D3 dari D1 dan D2).** P(D3 = 0) pasangan 20 sampai 50 tiket: train 0,431, uji **0,115**, thinned 0,35 **0,091**. Median rasio D3 pasangan 10 sampai 20: train 0,000, uji **0,356**, thinned 0,35 **0,337**.
   - **Lookup.** Yang dilatih pada data thinned memprediksi periode uji lebih baik: MAE 0,3505 menjadi 0,3346, pasangan kecil 0,531 menjadi 0,500, bias pasangan kecil 0,043 menjadi 0,000.
   - **Komposisi.** Dunia thinned 0,4 mereproduksi komposisi skala uji tanpa bobot: skala 5 sampai 20 = 0,118 (uji 0,114, mentah 0,029), skala di bawah 5 = 0,015 (uji 0,015).
   - **D4 sampai D10 (lookup, grouped 5-fold per film).** Dilatih mentah dan dinilai mentah: 0,350. Dilatih mentah dan dinilai di dunia thinned: **0,380**, kira-kira offset validasi ke public yang selama ini terlihat. Dilatih dengan campuran thinned dan dinilai di dunia thinned: **0,355 (-0,025)**. Biaya di dunia mentah +0,003.
   - **Fitur.** `log_s` KS ke uji 0,231 menjadi 0,063 sampai 0,093, okupansi 0,32 menjadi 0,17. Fitur kalender dan jadwal tidak berubah, dan memang tidak seharusnya.
   - **Pergeseran pencopotan.** Dengan tolok ukur absolut yang sama, offset -1,33 menjadi -0,33 sampai -0,56, dan log-loss periode uji 0,326 menjadi 0,247. Sekitar 75% "kebijakan pencopotan berbeda" adalah pasar yang lebih sepi.
   - Thinning pernah ditolak di `eda/14` karena dibandingkan dengan pasangan kecil train, yang memang pasangan sekarat. Itu pembanding yang salah.
6. **Sinyal kecil yang sah:** amplitudo akhir pekan per klaster (`eda/86`). Reliabilitas split-half 0,91, dan residual akhir pekan rilis Rabu naik monoton per kuintil (-0,26 ke -0,06). Koreksi struktural hanya -0,0007, jadi belum dipakai.

## v14 (`notebooks/build_v14.py` -> `notebooks/v14.ipynb`)

- **Data latih.** Mentah + tiga dunia thinned (π 0,35/0,5/0,7, seed 11/12/13). Setiap dunia dibangun ulang dengan aturan panitia: transaksi ditipiskan, okupansi ikut diskalakan, show tetap, D1 mentah. Statistik histori klaster tetap dari train mentah, seperti data uji.
- **Lensa utama.** Dunia validasi thinned π 0,45 (seed 99), MASE biasa. Pembatas: grouped CV mentah TW dan temporal, masing-masing maksimal +0,005. Bukti tambahan: proxy periode uji dengan model.
- **Konfigurasi.** B1 (resep v13), T1 (+ dunia thinned), T2 (T1 + fitur relatif). Aturan adopsi ditulis sebelum hasil: QW turun minimal 0,003, proxy membaik, dan kedua pembatas terpenuhi.
- **Duplikasi dan kebocoran.** Semua dunia satu film berada di fold yang sama. Dry run memeriksa kelima fold: tidak ada film validasi di dunia mana pun di bagian latihnya. Backtest temporal memakai aturan cutoff film asli.
- **Model.** LightGBM (3 seed L1), TabPFN-3.5 Fast FP16 (konteks per horizon 10.000: semua baris mentah + sampel thinned), dan TabM. Paling banyak satu TabPFN, total bobot dibatasi 200 MB.
- **Pemeriksaan lokal.** Build, `ast`, dan ruff bersih. Dry run Bagian 1 sampai 7 dan pandas Bagian 8 berhasil. Crash check jalur kode Bagian 8 sampai 10 dijalankan dengan TabPFN/TabM diganti stub dan LightGBM 3 pohon, jadi tidak ada model sungguhan yang dilatih di laptop.

## Yang harus dibaca dari hasil Kaggle v14

- QW B1 seharusnya mendekati level public v13 setelah memperhitungkan Lebaran: sekitar 0,39 sampai 0,40 untuk baris non-Lebaran. Bila jauh lebih rendah, dunia thinned belum cukup mirip uji.
- Ekspektasi lookup adalah gain QW sekitar 0,02 sampai 0,025. Bila T1/T2 hanya membaik kurang dari 0,005, efeknya kecil di model nyata.
- Offset `a` proxy untuk raw+thin seharusnya jauh lebih kecil dari -1,32.
- Batasan: dunia thinned tidak mensimulasikan perubahan film-mix, kalender periode uji, atau lintasan film. Validasinya tetap simulasi, meskipun didukung label periode uji di D3.

## Hasil run Kaggle v14 (direview)

- **Thinning gagal di level model.**
  - QW B1 0,3192, T1 0,3187, T2 0,3164. Gain T2 0,0028, di bawah ambang 0,003, dengan CI [-0,0124; +0,0070].
  - Proxy periode uji: bias hilang (0,055 ke 0,003), tetapi MAE memburuk (0,4759 ke 0,4813).
  - Temporal memburuk (T1 +0,0106, T2 +0,0042).
  - Offset pencopotan proxy memang menyusut (-1,32 ke -0,59 sampai -0,75), jadi mekanisme D3 benar, tetapi tidak menjadi akurasi D4 sampai D10.
- **Prediksi kunci gagal.** QW B1 0,319 lebih rendah dari TW mentah 0,332, bukan sekitar 0,39 sampai 0,40. Dunia thinned tidak mereproduksi kesulitan test di horizon target, dan gain lookup (-0,025) tidak terbawa ke LightGBM yang sudah punya fitur histori klaster dan bentuk.
- **Final = LightGBM saja**, karena tidak ada blend yang lolos syarat QW minimal 0,001. Pada baris dan bobot yang sama: TW 0,33176 vs v13 0,32673 (+0,0050 [-0,0007; +0,0111]), v12 +0,0060 [+0,0024; +0,0102], v8 +0,0066 [+0,0021; +0,0114]. Temporal 0,3058 vs v13 0,2994.
