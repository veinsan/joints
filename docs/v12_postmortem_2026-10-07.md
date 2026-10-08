# Audit v12 setelah public 0.41060 — 7 Oktober 2026

Notebook v12 yang sudah dieksekusi, `results/v8,v10,v11,v12`, dan seluruh kelompok dokumen `handoff/` ditinjau. Audit baru hanya membaca prediksi, melakukan EDA, dan mengubah presisi penyimpanan checkpoint. Tidak ada training lokal, submission, atau akses label target test. Notebook yang sudah dieksekusi tidak diubah.

## 1. V12 mengalahkan pembanding yang lebih lemah, belum mengalahkan v8

Semua OOF dicocokkan menggunakan `(movie_title, cinema_ids, h)`, bukan sekadar posisi baris. Label dan skala 56.372 baris identik. TW menggunakan komposisi bucket skala × hari pertama transaksi yang sama.

| Versi | MASE asli | TW-MASE | Public |
|---|---:|---:|---:|
| v8 | 0.308051 | 0.325126 | 0.39991 |
| v10 | 0.307237 | 0.324683 | 0.40823 |
| v11 | 0.313231 | 0.331879 | tidak tersedia |
| v12 | 0.308680 | 0.325720 | 0.41060 |

V12 − v8: TW **+0.000595**; paired bootstrap 10.000 pengulangan per film memberikan interval 95% **[-0.003220, +0.004505]**. Hanya 64/126 film membaik. Korelasi galat bertanda 0.99683. Interval ini bersyarat pada prediksi yang sudah dipilih; tidak mengoreksi pemilihan berulang terhadap banyak eksperimen.

Bootstrap v12 terhadap LightGBM v12 yang terlihat bagus dalam notebook tidak menjawab pertanyaan apakah ia mengalahkan incumbent v8. Perbandingan itu perlu dilaporkan terpisah.

## 2. Temuan paling penting: kemenangan pada nol menutup kekalahan pada pertumbuhan

Berikut MAE berskala **di dalam masing-masing segmen**, menggunakan bobot TW yang sama. Segmen growth adalah bagian dari positive, bukan kelompok tambahan yang boleh dijumlahkan lagi.

| Segmen berdasarkan label validasi | v8 | v10 | v12 |
|---|---:|---:|---:|
| Target nol | 0.119479 | **0.101649** | 0.108489 |
| Target positif | **0.478816** | 0.491368 | 0.488069 |
| Tiket target > tiket D3 | **1.046180** | 1.085804 | 1.071773 |
| Positif, tidak tumbuh dari D3 | **0.240366** | 0.241539 | 0.242751 |

Pada v12, pengurangan kontribusi galat nol **-0.004701** hampir menutup kenaikan galat positif **+0.005295**. Pada D4 saja, TW memburuk dari 0.407664 menjadi 0.414684. Ini bukan masalah yang hanya terlihat pada satu pengganti: v10 juga memperlihatkan pola yang sama.

Hipotesis yang masuk akal: jika test mempunyai pencopotan lebih sedikit, keuntungan pada target nol kurang bernilai dan kerugian pada film yang masih laku lebih terasa. Proxy D1–D2→D3 dalam eksperimen sebelumnya mendukung kemungkinan perubahan kebijakan, tetapi **tidak mengukur zero-rate target D4–D10**. Karena itu belum terbukti bahwa inilah penyebab public turun.

Audit menyimpan stress test yang mengubah proporsi nol sambil mempertahankan galat kondisional. Ini tidak menciptakan label dari kuantil LightGBM, tetapi tetap memakai asumsi kuat bahwa distribusi galat dalam setiap segmen tetap. Angkanya bukan estimasi skor test. Jangan menaikkan semua prediksi atau memilih quantile lebih tinggi hanya karena diagnosis growth: itu bisa merusak film yang menurun.

![Trade-off](../outputs/eda/62_v12_postmortem/zero_growth_tradeoff.png)

## 3. Apa yang berubah pada submission

- Seluruh **4.417 prediksi Lebaran identik**, sampai nilai tiketnya. Perubahan bagian ini tidak menjelaskan selisih public v8→v12.
- Mean prediksi/skala normal: 0.4784→0.4792; Ramadan: 0.3716→0.3942; Natal: 0.6365→0.6335.
- Rata-rata perubahan absolut berskala seluruh test 0.04828. Nilai rata-rata segmen yang mirip bisa menyembunyikan perubahan besar antarfilm.
- Identitas dan label subset public tidak tersedia. Film dengan prediksi paling berubah **bukan berarti** film penyumbang error terbesar.

Penggantian TabPFN-3.5 tetap hipotesis, belum sebab yang terisolasi: versi-versi ini juga mengubah statistik klaster dan kalender. Klaim handoff bahwa model lain pasti tidak berguna, atau selisih kepatuhan pasti 0.01, terlalu kuat.

## 4. EDA baru: lebih banyak baris belum tentu lebih relevan

Hanya 39.488/138.959 transaksi positif mentah menjadi target positif simulasi utama (28.4%). Sisanya tidak otomatis terbuang: sebagian sudah digunakan sebagai history, sampel limited, statistik klaster, kalender, dan analog Lebaran.

Dicoba pembentukan jendela tambahan pada usia film 4, 8, dan 15 hari, tanpa fitting model. Ada 90.336 baris target tambahan, tetapi profilnya tidak memecahkan pergeseran distribusi:

| Awal jendela | Median skala | Proporsi skala ≤20 | Median okupansi |
|---|---:|---:|---:|
| D1 utama | 181.0 | 3.20% | 22.05% |
| Usia 4 | 182.0 | 3.77% | 21.86% |
| Usia 8 | 197.0 | 1.24% | 25.96% |
| Usia 15 | 163.7 | 2.21% | 25.31% |
| Test D1 | 91.7 | 12.96% | 10.99% |

Film yang masih aktif pada usia lanjut justru berokupansi tinggi. **Jangan langsung menggabungkan semua jendela sebagai D1 baru.** Ada 37.388 kunci target yang muncul di lebih dari satu jendela; jika kelak diuji, semua jendela/format film harus dalam fold yang sama, tanggal label harus sudah selesai untuk temporal training, usia asal dipertahankan, dan bobot total augmentasi dibatasi per film. Belum ada bukti peningkatan akurasi dari augmentasi ini.

## 5. Kandidat konkret yang sebelumnya terlalu cepat dicoret karena ukuran

Checkpoint lokal **TabPFN-3.5 Fast** ternyata float32, bukan format minimal. Matriksnya sudah dikonversi ke float16; parameter vektor dan regression borders tetap float32.

| Artefak | Ukuran desimal |
|---|---:|
| Fast asli | 334.18 MB |
| Fast dengan matriks FP16 | **167.21 MB** |
| LightGBM tersimpan v12 | 25.00 MB |
| Perkiraan LGB + Fast FP16 | **192.21 MB** |

Metadata dan seluruh tensor lolos round-trip serialization. Loader TabPFN lokal berhasil memuat arsitektur dan bobot dalam sekitar satu detik, dengan semua parameter finite. Belum dilakukan forward prediction atau evaluasi akurasi. Galat L2 relatif bobot 0.000172 tidak membuktikan galat prediksi kecil.

Fast mempunyai **8 layer**, sementara checkpoint default yang dipakai v8 mempunyai **24 layer**. Ini bukan konversi v8 yang ekuivalen. Konversi FP16 default v8 saja masih sekitar 438 MB sehingga tetap tidak muat. Pengujian Fast beralasan karena keluarga default pernah berhasil pada public, bukan karena performa Fast sudah terbukti.

Paket final lengkap tetap harus diukur ≤200 MB, termasuk model, checkpoint, dan metadata yang disertakan. Checkpoint hasil konversi harus digunakan sejak validasi hingga inferensi final; jangan validasi FP32 lalu diam-diam submit versi konversi. Ambiguitas cutoff checkpoint mengikuti keputusan user sebelumnya dan belum memperoleh konfirmasi baru.

## 6. Eksperimen berikutnya yang layak dijalankan di Kaggle

1. Bekukan preprocessing B1 v12, fold, kalender, data tambahan, dan analog Lebaran. Bandingkan incumbent v12 **serta OOF v8**; v8 menjadi referensi kualitas, bukan otomatis artefak final yang memenuhi batas ukuran.
2. Uji Fast FP16 sebagai satu perubahan komponen. Gunakan adapter quantile yang sudah ada; evaluasi λ=0 dan 0.5, tanpa mewariskan keputusan LightGBM. Bandingkan FP32 versus FP16 pada subset tetap lebih dahulu untuk mengukur efek konversi.
3. Simpan prediksi setiap komponen per fold dan temporal, quantiles/p0, serta prediksi final sebelum override. Ini memungkinkan audit ulang dan ablation bobot tanpa mengulang semua model.
4. Selain total TW/temporal, laporkan target nol, positif, growth, dan D4–D5. Tetapkan stress test sebelum run. Jangan menerima kemenangan yang hanya menggeser galat ke segmen berisiko tanpa menunjukkan pertukarannya.
5. Jangan membangun banyak varian submission untuk menebak label public. Satu kandidat dipilih dari evaluasi yang dibekukan. Bila Fast gagal, kembali ke incumbent yang memenuhi aturan, bukan terus mencari blend minimum pada OOF yang sama.

Target 0.34456 berjarak **0.05535 dari v8** dan **0.06604 dari v12**. Bukti saat ini belum menunjukkan cara mencapai selisih sebesar itu. Fast adalah eksperimen pemulihan kualitas yang masuk akal, bukan janji mengejar top-1. Untuk lompatan selanjutnya dibutuhkan sinyal pertumbuhan yang benar-benar lolos validasi pada film lain; kegagalan fitur sebelumnya tidak membuktikan sinyal tersebut mustahil ditemukan.

## Artefak dan reproduksi

- `eda/62_v12_postmortem.py`: pencocokan OOF, bootstrap film, error segmen, perubahan submission, stress prevalensi nol; CSV/JSON dan dua gambar di `outputs/eda/62_v12_postmortem/`.
- `eda/63_lifecycle_window_audit.py`: EDA jendela lifecycle; CSV/JSON, parquet target dengan asal jendela, dan gambar di `outputs/eda/63_lifecycle_windows/`.
- `eda/64_fast_checkpoint_storage.py`: konversi checkpoint lokal, pemeriksaan tensor, hash, statistik ukuran, dan gambar di `outputs/eda/64_checkpoint_storage/`. Hasil smoke loader ada di `loader_smoke.json`.

Jalankan setiap skrip menggunakan `rtk proxy .venv/bin/python eda/<nama_script>.py`. Skrip 64 memerlukan checkpoint lokal; sumber dapat diatur lewat `--source`. Tidak ada unduhan otomatis. Ketiga gambar analisis serta grafik ukuran sudah dibuka dan diperiksa secara visual.
