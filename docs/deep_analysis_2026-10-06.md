# Deep EDA dan eksperimen preprocessing — 6 Oktober 2026

Analisis ini memakai data resmi, cache fitur, serta OOF/submission yang sudah tersimpan. Angka public **0.40823** untuk v10 dan **0.34456** untuk pemimpin leaderboard berasal dari laporan pengguna, bukan evaluasi ulang label test. Semua angka baru di bawah adalah evaluasi lokal. Tidak ada submission probing, pengambilan label tersembunyi, atau perubahan notebook/submission produksi.

**Kesimpulan setelah seluruh run selesai:** preprocessing panel bioskop terlihat menjanjikan pada grouped CV, tetapi gagal temporal; perbaikannya dengan shrinkage juga gagal. Normalisasi kalender pada input memberikan perbaikan kecil pada dua pembagian film dan tetap membantu ketika dimasukkan ke hurdle. Namun, seluruh interval bootstrap kandidat masih mencakup nol. Ini belum membuktikan gain public atau kemampuan mencapai 0.34456. Delapan script baru telah dijalankan, menghasilkan angka dan gambar; seluruh delapan plot utama juga telah diperiksa secara visual. Total run mencakup 63 fit regresi langsung serta 320 fit classifier/quantile hurdle.

## Cara mereproduksi dan menemukan hasil

Jalankan dari root repository menggunakan environment `.venv` yang sudah ada. Set `MPLCONFIGDIR` supaya matplotlib tidak menulis ke home yang tidak writable. Urutannya penting: 55 menghasilkan fitur untuk 56–61, dan 56 menghasilkan prediksi acuan untuk 59–60.

```bash
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda .venv/bin/python eda/54_missingness_and_duplicates.py
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda .venv/bin/python eda/55_history_preprocessing.py
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda OMP_NUM_THREADS=4 .venv/bin/python eda/56_preprocessing_ablation.py
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda .venv/bin/python eda/57_error_mechanisms.py
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda OMP_NUM_THREADS=4 .venv/bin/python eda/58_split_leakage_control.py
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda OMP_NUM_THREADS=4 .venv/bin/python eda/59_preprocessing_confirmation.py
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda OMP_NUM_THREADS=4 .venv/bin/python eda/60_calendar_hurdle_check.py
rtk proxy env MPLCONFIGDIR=/tmp/joints-eda OMP_NUM_THREADS=4 .venv/bin/python eda/61_film_sample_learning_curve.py
```

Setiap script mencetak angka, menyimpan tabel CSV/JSON, dan menghasilkan PNG. Prediksi ablation tersimpan dalam Parquet supaya hasil bisa diperiksa tanpa retraining. Tidak ada framework AutoML; ini beberapa eksperimen manual dengan perubahan yang ditentukan sebelum run.

| Script | Pertanyaan | Direktori hasil |
|---|---|---|
| `54_missingness_and_duplicates.py` | Nol berarti apa? Ada duplikasi? Seberapa kuat bukti v10 lebih baik? | `outputs/eda/54_missingness/` |
| `55_history_preprocessing.py` | Apakah kalender dan perubahan panel merusak interpretasi tren D1–D3? | `outputs/eda/55_preprocessing/` |
| `56_preprocessing_ablation.py` | Apakah setiap kelompok preprocessing memperbaiki prediksi? | `outputs/eda/56_ablation_2026/` |
| `57_error_mechanisms.py` | Film yang meleset berubah di jumlah show atau tiket per show? | `outputs/eda/57_mechanisms/` |
| `58_split_leakage_control.py` | Seberapa menipu split yang membocorkan film/pasangan? | `outputs/eda/58_split_control/` |
| `59_preprocessing_confirmation.py` | Apakah hasil bertahan di split lain? Apakah shrinkage memperbaiki panel? | `outputs/eda/59_confirmation/` |
| `60_calendar_hurdle_check.py` | Apakah gain kalender bertahan di pipeline hurdle? | `outputs/eda/60_calendar_hurdle/` |
| `61_film_sample_learning_curve.py` | Apakah menambah film berlabel masih membantu? | `outputs/eda/61_film_learning/` |

Hash input dan versi library tersedia di `outputs/eda/deep_analysis_2026-10-06_manifest.json`. Environment lokal menggunakan Python dan library sebagaimana manifest; antara lain LightGBM 4.7.0 dan NumPy 2.5.3, sehingga ini bukan reproduksi byte-identical notebook Kaggle yang memakai versi lain. Peringatan deprecation timedelta dari kombinasi NumPy/pandas tidak menggagalkan run; assertion alignment, target, dan skala tetap diperiksa.

## 1. Target nol bukan sinonim bioskop tutup atau film ditarik permanen

Data mentah train 138.959 baris dan test history 32.323 baris tidak memiliki duplikat kunci `(movie_title, cinema_ids, date_show)` maupun transaksi `total_ticket == 0`. Nol pada target simulasi berasal dari transaksi yang tidak tercatat setelah pengisian grid D4–D10.

Dari 16.884 target nol, **99,508% terjadi pada cinema-date yang masih memiliki transaksi film lain**. Itu menolak hipotesis bahwa sebagian besar nol berasal dari seluruh bioskop berhenti melapor. Namun, data ini sendiri belum membedakan film tidak dijadwalkan, jadwal tanpa penjualan, atau pencatatan film yang hilang.

Di 7.988 pasangan dengan seluruh tujuh horizon tersedia, 4.400 pernah nol; **337 kembali positif setelah nol**. Karena itu, aturan “sekali nol berarti selesai selamanya” akan salah pada sebagian pasangan. Label masa depan pada audit ini hanya dipakai menjelaskan kesalahan, tidak menjadi fitur inferensi.

![Audit missingness dan leverage](../outputs/eda/54_missingness/diagnostics.png)

## 2. Duplikat persis sedikit; kebocoran kelompok jauh lebih besar

Signature enam angka `(y1,y2,y3,sh1,sh2,sh3)` sama lintas film ditemukan pada 26/8.113 pasangan train dan 81/10.373 pasangan test. Kesamaan pada hitungan kecil bisa terjadi secara wajar: bukan bukti catatan ganda. Menghapus semuanya berisiko membuang contoh valid. Bahkan signature ini belum mencakup occupancy, kalender, genre, dan atribut bioskop.

Negative control memakai **baris evaluasi yang sama (1.665)** dan **jumlah baris training yang sama (43.527)**. Statistik bioskop pada ketiga kontrol sama-sama dihitung tanpa film evaluasi. Yang berubah hanya apakah label film/pasangan yang sama boleh masuk training.

| Split | MASE | TW-MASE | Film evaluasi yang terlihat di training |
|---|---:|---:|---:|
| Film disjoint | 0.357658 | 0.366112 | 0 |
| Bocor: bioskop lain dari film sama | 0.259937 | 0.287940 | 25 |
| Bocor: horizon lain dari pasangan sama | 0.251927 | 0.277942 | 25 |

Dua baris terakhir sengaja tidak valid untuk skenario film baru; **bukan kandidat model**. Angka di bawah target leaderboard bisa diperoleh secara lokal hanya dengan evaluasi keliru. Temuan ini tidak menyatakan peserta leaderboard melakukan kebocoran; metode mereka tidak diketahui.

Penanganan yang benar: audit unique key, satukan format 2D/3D/IMAX dari base film dalam fold yang sama, jangan random-split horizon/pasangan, dan hitung statistik yang memakai train hanya di bagian training. Pedoman pemisahan kelompok dan fitting preprocessing di training konsisten dengan [dokumentasi scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html).

![Negative control split](../outputs/eda/58_split_control/split_control.png)

## 3. Kenapa v10 yang terlihat lebih bagus malah turun di public?

Pada 56.372 baris evaluasi yang sama, v8 MASE 0.308051 dan v10 0.307237. Dengan bobot scale × first observed day, v8 0.325126 dan v10 0.324683: gain hanya **0.000443**.

Bootstrap berpasangan dengan resampling **base film**, bukan baris, memberikan rentang 95% selisih v10−v8 **[−0.005397, +0.004462]**. Rentang melintasi nol. Ini ukuran sensitivitas pada komposisi film dari data yang ada; bukan confidence interval skor leaderboard masa depan.

Lima film menyumbang **28,25% weighted error**. Ada 126 base film, dan effective sample size bobot pada tingkat film sekitar **101,44**; 56 ribu baris tidak setara dengan 56 ribu contoh film independen. Korelasi Spearman rata-rata signed miss per film antara v8 dan v10 **0,96936**. Jadi perubahan ensemble masih menyisakan pola kesalahan film yang hampir sama.

Penurunan public yang dilaporkan konsisten dengan lemahnya bukti keunggulan lokal, tetapi belum mengidentifikasi penyebab per film atau periode test. Perubahan prediksi v10 terhadap v8 bukan bukti bahwa perubahan tersebut salah pada label test.

## 4. Tren D1–D3 tercampur perubahan panel bioskop

Total tiket sebuah film berubah karena gabungan perubahan penjualan pada bioskop yang tetap menayangkan, bioskop masuk, dan bioskop keluar. Membagi total D3/D1 tidak memisahkan ketiganya.

Script 55 menghitung tren pada bioskop yang tercatat di D1 **dan** D3, porsi tiket dari bioskop baru, porsi tiket D1 dari bioskop yang hilang di D3, ukuran panel, dan dispersi tren. Semua berasal dari jendela D1–D3, termasuk bioskop yang tidak bertahan ke D3. Ini penting: jika hanya memakai pasangan target yang selamat D3, exit share akan hilang karena selection bias.

Selisih absolut log-trend panel dengan log-trend agregat melebihi 0,2 pada **31/143 judul berformat train dan 19/160 judul berformat test yang memiliki target**. Jumlah judul test di sini berbeda dari seluruh history karena judul tanpa pasangan D3 tidak menghasilkan target.

Contoh SEND HELP: log-trend seluruh bioskop −2,422, panel yang sama +0,257, tetapi panel itu hanya **satu bioskop**. Tidak boleh menyimpulkan film secara nasional sedang tumbuh dari angka tersebut. Contoh ini memotivasi perbaikan shrinkage dan juga menjelaskan mengapa fitur panel yang tampak masuk akal dapat overfit.

## 5. Preprocessing kalender: periksa transformasinya, bukan hanya rumus

Input tiket per hari dibagi faktor kalender hari bersangkutan, kemudian dinormalisasi terhadap rata-rata tiga hari yang sudah disesuaikan. **Label dan denominator MASE resmi tidak diubah.** Fitur mentah tetap tersedia agar model tidak dipaksa mempercayai koreksi kalender.

Pada plot F1 THE MOVIE, tren mentah D2→D3 meningkat, tetapi setelah penyesuaian kalender justru menurun. Perubahan tersebut sesuai tujuan memisahkan efek akhir pekan dari momentum film, tetapi ketepatannya tetap bergantung pada faktor kalender. Contoh LOCKED yang baru mencatat penjualan D3 tetap memiliki D1/D2 nol; preprocessing tidak menciptakan penjualan fiktif.

Pemeriksaan runnable: skala dan target harus identik sebelum/sesudah; mengacak urutan transaksi serta mengganti seluruh nilai tiket D4+ dengan 999.999.999 tidak boleh mengubah fitur baru. Assertion ini lulus. Ini membuktikan transformasi baru tidak membaca tiket masa depan **dengan tanggal D1 yang diberikan tetap**; tidak membuktikan rekonstruksi D1 lama bebas lookahead.

![Input sebelum dan sesudah](../outputs/eda/55_preprocessing/before_after.png)

## 6. Teori diuji: beberapa preprocessing yang masuk akal justru gagal

Baseline screening adalah LGB L1 satu seed, 600 pohon, fitur inti yang sama, statistik bioskop per fold. Semua varian memakai baris, label, scale, dan pembagian film yang sama. Ini **bukan** baseline ensemble v8/v10 dan angkanya tidak boleh dibandingkan langsung dengan public MASE.

Temporal menggunakan film Juli/Agustus/September; training hanya film dengan D10 selesai sebelum awal bulan evaluasi. Keputusan awal: delta grouped TW negatif, rata-rata temporal MASE negatif, dan tidak ada bulan memburuk lebih dari 0,003. Bobot TW hanya untuk pelaporan; loss training tetap mengikuti MASE asli.

| Perubahan tunggal | Grouped TW | Delta grouped TW | Delta mean temporal MASE | Bulan terburuk | Keputusan screening |
|---|---:|---:|---:|---:|---|
| Baseline | 0.348897 | — | — | — | Acuan |
| + kalender input | 0.346538 | −0.002359 | −0.001629 | +0.000127 | Lanjut konfirmasi |
| + panel bioskop | 0.343505 | −0.005392 | +0.007001 | +0.015387 | Tolak |
| + show/TPS/occupancy yang sudah diamati | 0.347927 | −0.000970 | +0.000268 | +0.005211 | Tolak |

![Ablation dan learning curve pohon](../outputs/eda/56_ablation_2026/ablation.png)

Pada pembagian film seed 2027, kalender kembali memperbaiki grouped TW **0.366932 → 0.364268 (−0.002664)**. Perbaikan terjadi pada 73/126 film, tetapi paired bootstrap 95% masih **[−0.007551, +0.002365]**. Jadi arah hasil konsisten, besar bukti belum kuat. Ini pengelompokan lain dari data yang sama, bukan holdout baru yang independen; temporal kalender memakai prediksi 56 yang sama, tidak dihitung sebagai konfirmasi temporal tambahan.

Perhatikan baseline sendiri berubah dari 0.348897 menjadi 0.366932 ketika pembagian training antarfold berubah, walaupun total baris evaluasi tetap sama. Sensitivitas model terhadap film yang tersedia untuk training lebih besar daripada gain kandidat. Memilih satu split dengan skor absolut paling rendah bukan cara membuktikan model lebih baik.

Detail delta kontribusi tersimpan di `56_ablation_2026/calendar_delta_*.csv`. Gain kalender pada split awal berasal dari pasangan first_day=1 (−0.001210) dan first_day=2 (−0.001155); first_day=3 sedikit memburuk (+0.000006). D4–D7 membaik, tetapi D8/D9/D10 menyumbang delta +0.000099/+0.000219/+0.000399. Ini menunjukkan koreksi kalender belum memecahkan late starters ataupun pertumbuhan jangka lebih panjang. Tidak dibuat aturan pemilihan horizon dari hasil ini karena itu akan menjadi optimasi tambahan terhadap OOF yang sama.

Panel diperbaiki dengan satu fitur koreksi: `(panel_log31 − raw_log31) × n/(n+20)`. Nilai 20 ditetapkan sebelum run dan tidak dituning. Plot memastikan koreksi panel kecil ditarik mendekati nol. Hasilnya: delta grouped TW −0.000193, mean temporal **+0.003356**, September **+0.011547**. **Perbaikan teori ini juga ditolak.** Tidak ada alasan melanjutkan sweep konstanta hanya demi mencari seed yang cocok.

![Konfirmasi dan hasil shrinkage](../outputs/eda/59_confirmation/confirmation.png)

## 7. Error besar bukan hanya prediksi nol

Pada v8, target positif menyumbang sekitar **84,28% error**. Untuk baris positif, dekomposisi berikut tepat secara aljabar:

`log(y_h / y_D3) = log(show_h / show_D3) + log(TPS_h / TPS_D3)`.

Show/TPS masa depan dalam analisis ini adalah penjelas setelah kejadian, bukan fitur yang tersedia saat prediksi. “Komponen lebih besar” berarti nilai absolut log-change lebih besar, bukan estimasi efek kausal intervensi jadwal.

| Keadaan target positif vs D3 | Komponen perubahan lebih besar | Porsi seluruh weighted error |
|---|---|---:|
| Tiket turun | Attendance/TPS | 20,67% |
| Tiket turun | Show | 9,12% |
| Tiket naik | Attendance/TPS | 36,23% |
| Tiket naik | Show | 18,26% |

Artinya **54,49% seluruh error** berada pada target positif yang tumbuh dibanding D3. Namun, ini tidak membenarkan menaikkan seluruh prediksi: contoh dipilih berdasarkan label nyata, dan prediktor optimal loss absolut adalah median kondisional, bukan mean. Menaikkan prediksi demi beberapa film breakout dapat merusak banyak film yang tidak tumbuh. Hubungan loss absolut dan median dibahas di [Forecasting: Principles and Practice](https://otexts.robjhyndman.com/fpp3/accuracy.html).

Pada level film, signed miss berkorelasi sekitar 0,458 dengan perubahan attendance masa depan dan 0,326 dengan perubahan show masa depan. Sebaliknya, korelasi dengan tren panel D1–D3 yang dinormalisasi kalender hanya 0,070. Informasi yang menjelaskan kejadian setelahnya belum tentu dapat diprediksi dari tiga hari awal. Ini bukan bukti bahwa prediksi pertumbuhan mustahil; hanya menunjukkan mengapa korelasi target dengan fitur masa depan tidak otomatis menjadi solusi.

Plot lima film dengan error terbesar menunjukkan miss growth yang bertahan lintas horizon. Garisnya **rata-rata berbobot y/scale**, bukan median kondisional yang dioptimalkan model. Jangan menggunakan jarak antar garis ini untuk memilih uplift global.

![Timeline error dan correlation matrix](../outputs/eda/57_mechanisms/mechanisms.png)

## 8. Konfirmasi kalender pada hurdle: gain bertahan, tetapi mengecil

Eksperimen 60 memakai 19 quantile positif, classifier nol, lambda pull-shift tetap 0,5, dan bobot hurdle tetap 0,75. Prediksi direct L1 baseline/kalender diambil dari eksperimen 56 agar tidak melatih ulang model identik. Classifier dan quantile dilatih ulang untuk masing-masing varian; statistik cinema untuk fitur classifier juga dihitung per fold/cutoff. Tidak ada pemilihan ulang bobot, lambda, atau quantile berdasarkan hasilnya.

| Model lokal | Grouped MASE | Grouped TW-MASE | Mean temporal MASE | Mean temporal TW-MASE |
|---|---:|---:|---:|---:|
| Baseline hurdle | 0.314197 | 0.333985 | 0.315517 | 0.304889 |
| Hurdle + kalender input | 0.312749 | 0.332800 | 0.313997 | 0.304040 |
| Delta | −0.001448 | −0.001186 | −0.001520 | −0.000849 |

Grouped TW membaik pada 69/126 film, tetapi paired bootstrap 95% delta **[−0.003767, +0.001376]**. Pada temporal, Juli membaik, Agustus sedikit memburuk, September membaik pada MASE biasa tetapi sedikit memburuk pada TW. Jadi klaimnya hanya perbaikan rata-rata kecil; bukan menang pada semua kondisi.

Ini masih model lokal satu seed, tanpa tambahan limited-release rows dan tanpa retraining EXAONE/TabPFN. Gain **tidak boleh langsung dianggap** sebagai gain ensemble final. Belum dibuat submission dari kandidat ini. Dengan bobot ensemble yang sudah besar pada foundation model, manfaat marginal fitur pada cabang LGB bisa makin kecil.

![Kalender dalam hurdle](../outputs/eda/60_calendar_hurdle/hurdle_check.png)

## 9. Learning curve jumlah film: belum ada bukti bahwa data sudah habis gunanya

Evaluasi September dibekukan. Training hanya film yang label D10-nya selesai sebelum September; tiga pengambilan subset film dipakai untuk 25%, 50%, dan 75% data. Titik 100% hanya sekali karena datanya sama. Statistik bioskop juga dihitung dari film training terpilih, sehingga titik 100% ini tidak persis sama dengan baseline 56 yang mengizinkan semua transaksi historis sebelum cutoff untuk statistik bioskop.

| Jumlah film training | Mean MASE September | Rentang antarsubset | Mean TW-MASE |
|---|---:|---:|---:|
| 26 | 0.335313 | 0.307209–0.388514 | 0.334894 |
| 52 | 0.292562 | 0.285134–0.300517 | 0.301152 |
| 78 | 0.303883 | 0.290240–0.311438 | 0.304121 |
| 104 | 0.306228 | Satu training set | 0.303263 |

Menambah film dari 26 ke 52 membantu rata-rata pada pengambilan ini, tetapi setelah itu kurva tidak monoton. Ini bukti sensitivitas terhadap komposisi dan potensi ketidakcocokan distribusi, **bukan** bukti bahwa 52 adalah jumlah optimal. Memilih subset yang paling bagus berdasarkan September akan menambah selection bias. Eksperimen ini belum mengukur efek menambah data berlabel dari periode berbeda yang benar-benar merepresentasikan test.

Learning curve jumlah pohon di script 56 menunjukkan train MASE rata-rata turun sekitar 0,255 → 0,226 → 0,211 untuk 100/300/600 pohon, sementara validasi sekitar 0,331 → 0,323 → 0,322. Model makin pas pada train tetapi tambahan generalisasi mengecil; menambah iterasi saja belum menjawab miss growth film baru.

![Learning curve film independen](../outputs/eda/61_film_learning/film_learning.png)

## Batas interpretasi dan aturan kompetisi

- Tanggal D1 train masih memakai rekonstruksi lama yang melihat coverage peak sepanjang sejarah. Audit 53 menunjukkan menggantinya dapat mengubah populasi, scale, dan kasus preview. Eksperimen baru membekukan tanggal dan populasi agar perbandingan fitur adil; bias rekonstruksi tersebut belum dihapus.
- Rolling backtest ini memakai konstanta kalender lama dan cache fitur jadwal/jendela film resmi. Statistik bioskop dan ketersediaan target training dibatasi per cutoff, tetapi seluruh pipeline belum merupakan rekonstruksi real-time historis murni. Nilai temporal adalah lensa tambahan, bukan jaminan bebas semua lookahead.
- Faktor kalender global juga pernah dieksplorasi pada dataset ini. Konfirmasi kedua tidak menghapus selection bias akibat banyak eksperimen sebelumnya.
- Bobot scale × first_day hanya mendekatkan komposisi fitur yang teramati. Bobot tersebut tidak menjamin distribusi `y | X` sama pada test. Dalam audit 51 terdapat 28 baris test pada sel bobot tanpa dukungan train.
- Public MASE adalah satu agregat pada label yang tidak tersedia. Tidak dapat menyatakan film test tertentu salah hanya dari angka agregat, selisih submission, atau correlation matrix. Evaluasi fitur dilakukan dengan label training yang sah, bukan submission yang dirancang mengekstrak label tersembunyi.
- Tidak ada sumber eksternal baru atau checkpoint baru yang dipakai di analisis ini. Kelayakan cutoff checkpoint dan aturan AI Agent untuk submission akhir tetap mengacu pada keputusan panitia; kode analisis tidak menyelesaikan ambiguitas tersebut. Batas 200 MB dan reproduksi notebook harus diperiksa pada artefak final, bukan pada CSV OOF.

**EDA penting, tetapi preprocessing yang terlihat benar bukan jaminan model apa pun akan menang.** Pada dataset ini, satuan informasi independennya film; hanya D1–D3 yang terlihat; dan sebagian error besar berasal dari perubahan setelah jendela tersebut. Hasil ablation di atas memperlihatkan bahwa pemeriksaan visual perlu disusul evaluasi target akhir, serta bahwa perbaikan lokal kecil belum menjembatani gap public sekitar 0,06.

## Keputusan praktis setelah batch analisis ini

1. **Jangan mengganti submission incumbent hanya karena grouped CV lebih rendah.** Perbandingan harus berpasangan, populasi/scale sama, temporal diperiksa, dan baseline merupakan pipeline final yang sebenarnya. Bukti keunggulan v10 sebelumnya terlalu lemah.
2. **Pertahankan kalender input sebagai kandidat kecil**, dengan implementasi yang sudah ada di `55_history_preprocessing.py`. Statusnya layak A/B pada ensemble final, belum rekomendasi submit. Target, scale, dan raw input tetap dipertahankan. Jangan menggabungkan sekaligus dengan fitur panel, attendance, atau pergantian bobot agar efeknya masih bisa diatribusikan.
3. **Tolak versi panel penuh, panel shrinkage ini, dan penambahan attendance ini.** Dua perbaikan panel sudah gagal temporal. Ini menolak implementasi yang diuji, bukan membuktikan semua bentuk informasi panel tidak berguna.
4. **Untuk lompatan besar, prioritas bukti adalah ketepatan konstruksi data dan representasi periode test.** Tanggal rilis/D1 resmi train yang dapat diverifikasi akan lebih berguna untuk mengaudit simulasi daripada mencoba banyak transformasi pada tanggal yang masih ambigu. Informasi berlabel tambahan hanya berguna bila legal, tersedia sebelum cutoff, dan benar-benar relevan; learning curve tidak mendukung klaim bahwa menambah sembarang film atau metadata pasti membantu.
5. **Tidak ada alasan untuk mengeklaim “sudah mentok secara matematis”.** Yang selesai adalah audit dan rangkaian hipotesis yang dijalankan di sini: missingness, duplikasi, split, leverage error, kalender, panel, attendance, repair, split confirmation, hurdle confirmation, dan learning curve. Masih ada batas data dan evaluasi yang belum terselesaikan. Tidak diketahui bagaimana peserta 0.34456 membangun solusinya; skor mereka bukan bukti bahwa salah satu trik tertentu akan memberi gain yang sama pada pipeline ini.

Pemeriksaan akhir: seluruh script lolos parse Python; assertion pada run mencakup alignment OOF, grouping film, label/scale yang tidak berubah, ketahanan transformasi terhadap peracunan transaksi masa depan, identitas dekomposisi show × TPS, dan batas perilaku mixture hurdle. Tidak ada notebook, model weights produksi, atau submission yang ditimpa oleh batch ini.
