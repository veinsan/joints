# Evaluasi eksperimen v1–v9 dan peluang menuju 0.34456

Audit 5 Oktober 2026. Perhitungan ulang: `eda/51_experiment_audit.py`; tabel, ringkasan JSON, dan grafik: `outputs/eda/51_audit/`. Audit membaca CSV resmi, notebook beserta output tersimpannya, kode pembentukan fitur, dan prediksi OOF/submission. Tidak melatih ulang model atau mengirim submission. Skor public di bawah berasal dari catatan `docs/eda_findings.md`, bukan verifikasi leaderboard langsung. Skor 0.34456 diperlakukan sebagai angka pembanding yang diberikan pengguna; metode tim tersebut tidak diketahui.

**Kesimpulan: 0.34456 belum terbukti mustahil dengan aturan ini, tetapi eksperimen yang ada belum menunjukkan cara mencapainya.** Dari public v8 0.39991, diperlukan pengurangan 0.05535, atau 13.84%. Bukti lebih mendukung memperbaiki simulasi validasi dan sinyal film daripada menambah foundation model. Klaim lama “tidak ada tuas sah >0.003 yang tersisa” terlalu kuat: eksperimen gagal hanya menolak implementasi yang diuji, bukan seluruh kelas metode.

## 1. Apa yang benar-benar membaik?

Semua angka berikut dihitung ulang dari OOF yang di-join dengan `(movie_title, cinema_ids, h)`, dengan pengecekan label, skala, dan pemisahan judul dasar antar-fold. Populasi bersama v3–v9 berisi 56.372 baris dari 126 film dasar. v2 punya 55.916 baris sehingga tidak dimasukkan ke perbandingan populasi identik; v1 tidak menyimpan OOF CSV.

| Versi | MASE asli OOF | Bobot skala | Bobot skala × hari pertama laku | Public, catatan lama |
| --- | ---: | ---: | ---: | ---: |
| v3 | 0.311815 | 0.387861 | 0.327392 | 0.45731 |
| v4 | 0.314521 | 0.391051 | 0.331836 | 0.44809 |
| v5 | 0.312519 | 0.389352 | 0.330697 | 0.40865 |
| v6 | 0.308508 | 0.384613 | 0.325771 | 0.40119 |
| v7 | 0.308508 | 0.384613 | 0.325771 | sama dengan v6 menurut catatan |
| v8 | 0.308051 | 0.384131 | 0.325126 | 0.39991 |
| v9 | 0.307859 | 0.384772 | 0.326065 | belum ditemukan |

- Submission v6 dan v7 identik, selisih maksimum tepat nol.
- Output v9 memilih LightGBM 0.6, EXAONE 0.4, TabPFN-2.6 0.0. Evaluasi TabPFN-2.6 tidak berarti checkpoint tersebut dipakai dalam prediksi akhir.
- v9 sedikit lebih baik pada MASE asli, tetapi sedikit lebih buruk pada kedua pembobotan. Bootstrap berpasangan per film untuk selisih v9−v8 berbobot memberi interval persentil 95% **−0.00267 sampai +0.00456**. Belum ada bukti keunggulan yang kokoh.
- Lompatan public terbesar yang tercatat ialah v4→v5, 0.03944. Notebook v5 dan penerus menerapkan analog Lebaran dari transaksi resmi April 2025; ini perubahan struktur prediksi, bukan sekadar pergantian model. Perbedaan versi juga mencakup perubahan lain, jadi seluruh gain tidak boleh dianggap efek kausal tunggal Lebaran.

## 2. EDA yang menentukan

| Statistik | Train simulasi | Test |
| --- | ---: | ---: |
| Film dasar | 126 | 140 |
| Pasangan film–klaster | 8.113 | 10.373 |
| Median skala D1–D3 | 181.00 | 91.67 |
| Baris dengan skala ≤20 | 3.20% | 12.96% |
| Pasangan/baris mulai laku D3 | 1.61% | 1.82% |
| Median rata-rata okupansi D1–D3, setelah zero-fill | 21.29% | 10.99% |

CSV train/history/test tidak memiliki missing cell atau duplikat kunci film–klaster–tanggal. Tidak ada overlap kunci target test dengan transaksi berlabel train/history. Semua film dasar train-simulasi dan test cocok dengan `movies.csv`. Ini tidak meniadakan risiko leakage turunan fitur, tetapi menolak beberapa kesalahan dasar.

Distribusi test jelas lebih kecil/sepi. Akan tetapi pembobotan skala saja ikut memperbesar proporsi late starter secara tidak tepat. Pada **prediksi v8 yang sama**, mengganti bobot mengubah 0.38413 menjadi 0.32513 tanpa memperbaiki satu pun prediksi. Ada 28 baris test dalam sel `(skala >500, first_day=2)` yang tidak punya dukungan train; pembobotan sekarang mengabaikan massa 0.0386% ini lalu menormalisasi ulang.

Pada v8, menurut bobot skala × first-day:

- Lima film menyumbang **28.25%** error: LILO & STITCH, SAYAP SAYAP PATAH 2: OLIVIA, SORE ISTRI DARI MASA DEPAN, UNTIL DAWN, WEAPONS. Semuanya rata-rata diprediksi terlalu rendah.
- D4–D5 menyumbang **37.48%** error meski hanya sekitar dua dari tujuh horizon.
- Late starter D3 hanya **1.82%** massa bobot tetapi menyumbang **0.05219** MASE, atau **16.05%** error. Ini bukan bukti bahwa menaikkan semua prediksi late starter akan membantu: distribusinya bimodal dan median bisa tetap rendah.
- Target positif menyumbang **84.28%** error; target nol 15.72%. Fokus eksklusif pada klasifier nol akan melewatkan bagian terbesar galat.
- Dari 7.988 pasangan dengan tujuh target lengkap, 4.400 memiliki setidaknya satu hari nol; **337** kemudian kembali positif. Nol bukan bukti pasti pencopotan permanen. Model hazard yang memaksa nol selamanya akan salah untuk sebagian seri.

Galat per film sangat tidak stabil. Bootstrap 2.000 kali per film memberi interval persentil 95% sekitar **0.25881–0.40867** untuk skor berbobot v8. Bobot dipertahankan tetap dan model tidak dilatih ulang; interval ini menunjukkan sensitivitas terhadap komposisi film, **bukan interval prediksi public/private LB** atau koreksi atas pemilihan model berulang.

## 3. Mengapa validasi 0.338 tidak menjamin public 0.344?

1. **Target `kappa_worlds` sebagian dibuat model.** Sebagian nol asli diganti draw kuantil model dengan asumsi pergeseran kebijakan. Notebook memilih bobot blend pada dunia buatan ini. Ini berguna untuk stress test, tetapi bukan label observasi independen; model dengan asumsi serupa dapat diuntungkan.
2. **Proxy D1,D2→D3 bukan D4–D10.** Proxy memilih pasangan yang laku D2, sementara tugas utama memilih pasangan yang laku D3. Pergeseran peluang nol D3 mendukung hipotesis perubahan kondisi pasar, tetapi tidak membuktikan besar atau bentuk koreksi untuk semua horizon.
3. **OOF tidak menguji override Lebaran final.** Seluruh 4.417 baris Lebaran (6.08% test) diganti analog pada tahap submission. v8 dan v9 identik di segmen ini. Skor OOF yang dicetak sebelumnya tidak mengukur kualitas override tersebut.
4. **Fold film bukan backtest temporal.** `make_ctx` menghitung statistik klaster dari seluruh train sebelum fold; sebagian statistik memasukkan transaksi validasi. Profil kalender juga berasal dari eksplorasi keseluruhan train. Audit fold yang jujur perlu menghitung ulang statistik terlatih hanya dari bagian train. Ini masalah estimasi validasi, bukan otomatis penggunaan label test tersembunyi.
5. **Rekonstruksi D1 melihat puncak cakupan sepanjang riwayat.** D1 dan pemilihan sampel dapat bergantung pada masa depan film. Itu perlu diuji sensitivitasnya; tidak boleh dianggap identik dengan prosedur panitia yang tidak sepenuhnya diketahui.
6. **Banyak keputusan memakai fold yang sama.** Puluhan eksperimen menghasilkan selection bias; gain kecil pada fold yang sering dilihat bukan bukti out-of-sample baru. Koreksi residual bertingkat juga perlu nested cross-fitting: target residual training yang dibuat model OOF dapat secara tidak langsung bergantung pada label outer-validasi.

Fitur film lain pada tanggal target boleh tersedia dalam paket resmi `test_history`; jangan otomatis menganggapnya data eksternal terlarang. Namun simulasi validasi perlu mereproduksi persis informasi yang terlihat, dan jangan menyebut evaluasinya sebagai forecast real-time ketat jika memakai informasi setelah cutoff masing-masing film.

## 4. Seberapa besar ruang perbaikannya?

Audit menghitung diagnosis dengan informasi jawaban validasi. Untuk setiap kelompok, multiplier dipilih menggunakan median rasio `y/p` berbobot `w*p/s`, sehingga cocok dengan loss MASE, bukan rasio jumlah tiket yang lebih cocok untuk agregat volume.

| Diagnosis memakai jawaban asli | MASE berbobot v8 |
| --- | ---: |
| Prediksi tersimpan | 0.32513 |
| Multiplier sempurna per film × horizon | 0.26261 |
| Multiplier sempurna per klaster × tanggal | 0.20182 |
| Hanya tahu baris mana yang benar-benar nol | 0.27402 |

**Ini bukan skor model yang dapat dipakai atau ramalan skor test.** Grup kecil pada oracle klaster–tanggal dapat terlalu mudah disesuaikan dengan jawaban. Diagnosis hanya menunjukkan lokasi struktur error, bukan kemampuan mempelajari struktur itu dari D1–D3.

Karena itu 0.34456 bisa konsisten dengan solusi sah yang memiliki rekonstruksi sampel, sinyal film, atau penyesuaian kalender lebih baik. Komposisi public yang menguntungkan juga mungkin. Kita tidak bisa memilih penjelasan tersebut tanpa solusi tim terkait dan label evaluasi. Tidak ada dasar menyimpulkan mereka pasti curang atau pasti memakai model tertentu.

## 5. Jalur berikutnya, berdasarkan bukti

**Prioritas 1: bekukan evaluasi yang dapat dipercaya.** Pertahankan MASE label asli sebagai skor utama; bobot skala dan skala×first-day sebagai dua lensa; kappa hanya stress test. Jalankan rolling holdout berlabel lengkap (misalnya train target sampai akhir Juni, validasi film Juli; lalu sampai Juli→Agustus; sampai Agustus→September). Semua format film tetap satu grup, target training harus sudah tersedia sebelum cutoff, dan statistik klaster/kalender harus fit di training saja. Laporkan delta berpasangan per film, bukan hanya beda rata-rata. Prinsip rolling-origin ini mengikuti [Forecasting: Principles and Practice](https://otexts.com/fpp3/tscv.html).

**Prioritas 2: tambah sinyal metadata resmi yang belum digunakan, mulai dari film.** Builder sekarang mengambil genre, rating, jumlah genre/pemain, tetapi belum identitas producer/director/writer/casts. Dengan pemisahan nama sederhana berdasar koma, test yang memiliki setidaknya satu nama pernah terlihat dalam seluruh transaksi train mencakup:

| Field | Porsi film test | Porsi baris test |
| --- | ---: | ---: |
| Producer | 42.86% | 46.12% |
| Director | 29.29% | 33.03% |
| Writer | 36.43% | 38.60% |
| Casts | 70.00% | 75.04% |

Ini kandidat sinyal, belum bukti improvement. Audit nama/homonym dahulu. Mulai dari encoding identitas yang diregularisasi kuat atau ringkasan retensi film sebelumnya dengan shrinkage; setiap angka berbasis target harus hanya berasal dari film training/past. Jangan melakukan target encoding seluruh data sebelum CV. Uji terhadap baseline identik dengan satu perubahan ini. Data resmi tidak perlu scraping ulasan terbaru atau LLM untuk menilai kualitas film.

**Prioritas 3: uji ulang definisi jendela dan objective level film.** Bandingkan D1 sekarang dengan aturan tanpa puncak masa depan, serta simulasi window observasi setelah preview. Pertahankan bobot sample/window agar film panjang tidak mendominasi. Eksperimen residual `eda/38` menargetkan log rasio jumlah tiket, sedangkan kompetisi menimbang error per pasangan dengan skala masing-masing; kegagalannya tidak menutup peluang objective film yang selaras MASE. Kandidat koreksi film harus dievaluasi secara nested dan dibatasi/shrink, bukan multiplier dari jawaban validasi.

**Prioritas 4: positif awal dan jadwal pertunjukan.** Karena D4–D5 dan target positif dominan, eksperimen kecil yang layak ialah prediksi tambahan perubahan jumlah show dan tiket/show dari label train, dengan fitur inference tetap D1–D3. Kalibrasi hasil akhirnya terhadap MASE; hasil perkalian dua median tidak otomatis median tiket. Ini belum diuji oleh audit, dan tidak boleh menggunakan total_show target asli sebagai fitur test. Fitur kompetisi, momentum klaster, koreksi pasar, dan hurdle umum sudah banyak dicoba; jangan mengulanginya tanpa alasan baru.

Tidak ada estimasi gain numerik yang sah untuk empat langkah ini sebelum backtest. Fine-tuning TabPFN, TabICL, seed bagging, dan context pooling sudah menunjukkan hasil lemah dalam catatan; prioritasnya lebih rendah daripada masalah di atas.

## 6. Kepatuhan terhadap docs/context.md

**Yang jelas:** data resmi diperbolehkan; data eksternal perlu bukti versi tersedia paling lambat 30 September 2025; AutoML dilarang; seed, dependency versi spesifik, dokumentasi, reproduksi, dan batas bobot 200 MB diwajibkan. LLM untuk development code diperbolehkan, inference LLM/API AI dan AI Agent dilarang.

**Pretrained perlu dibedakan dari data eksternal.** Aturan mengizinkan pretrained publik tetapi tidak secara eksplisit menjelaskan apakah cutoff data juga berlaku untuk checkpoint. `eda_findings.md` sendiri semula menyebut model baru melanggar, kemudian mengklaim ada broadcast yang memperbolehkan; teks broadcast resmi itu tidak ada di konteks yang diaudit. Karena itu v6–v9 tidak bisa diberi cap “pasti patuh” hanya dari catatan ini.

- [Dokumentasi resmi Prior Labs](https://docs.priorlabs.ai/models) mencatat TabPFN-2.5 November 2025, 2.6 April 2026, 3.5 September 2026.
- [Laporan EXAONE Tabular](https://arxiv.org/abs/2608.25774) bertanggal 26 Agustus 2026. Model ini foundation model tabular; jangan disamakan begitu saja dengan EXAONE LLM. Sifat tabular tidak otomatis menjawab apakah cutoff checkpoint berlaku.
- Jalur konservatif ialah LightGBM dari nol dan, jika diperlukan, TabPFN v2 dengan revisi sebelum cutoff. [Riwayat checkpoint v2](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/commits/main) menunjukkan revisi `213f8e3` pada 11 Juni 2025, serta upload lain November 2025; nama “v2” saja belum mengunci versi.
- pkl v8 hanya 62.94 MB tetapi checkpoint pendukungnya **876.03 MB**. Mengecilkan pkl dengan mengunduh bobot lain saat rerun tidak membuktikan memenuhi batas keseluruhan bobot. v9 memiliki pkl **109.61 MB**, dan output notebook menyatakan checkpoint EXAONE sudah tertanam. Jangan menggandakan direktori `fm/` ke paket tanpa menghitung ukuran dan kebutuhan sebenarnya.
- Fallback v9 dapat membuang EXAONE saat OOM lalu menormalisasi ulang blend. Itu mengubah prediksi; untuk final, kegagalan komponen harus terlihat dan reproduksi pada hardware sasaran perlu diperiksa.
- Larangan “AI Agent” dan izin “LLM membantu development code” perlu dijelaskan cakupannya oleh panitia. Bantuan audit ini tidak merupakan sertifikasi kepatuhan alur kerja; hindari menafsirkan izin code-assistance sebagai izin menjalankan agent otonom dalam lomba.

Scraping rating, jumlah penonton aktual, Google Trends, atau box office yang dipublikasikan/diperbarui setelah cutoff tidak menjadi sah hanya karena filmnya lebih lama. Sumber yang dipakai untuk membaca metode dalam audit ini bukan fitur model; calon fitur eksternal tetap wajib punya provenance sebelum cutoff.

## Reproduksi audit

```bash
rtk proxy env MPLCONFIGDIR=/tmp/joints-audit-mpl .venv/bin/python eda/51_experiment_audit.py
```

Script memiliki assertion untuk alignment ID, skala resmi, kelengkapan target, label OOF, grup film, nilai prediksi, dan konsistensi oracle. Output menggunakan cache fitur yang sudah ada dan memverifikasi label/skala terhadap OOF; tidak mengklaim seluruh feature pipeline sudah dibangun ulang fold-locally. Grafik: `outputs/eda/51_audit/audit.png`.
