# Indeks script EDA (`eda/`) dan hasil singkatnya

Jalankan dari root: `.venv/bin/python eda/NN_*.py`. Untuk script yang memakai TabPFN, TabICL, atau EXAONE,
pakai `.venv-tp9`. Gambar ada di `outputs/eda/<nama>/`. Angka lengkap ada di `docs/eda_findings.md`.
Script 11 dihapus karena isinya probing LB.

**Script yang melatih model berat** (jangan dijalankan ulang di laptop): 31, 35, 47, 49, 50. Juga 56, 59, 60,
dan 61 (ratusan fit LightGBM).

| No | Pertanyaan | Hasil / keputusan |
| :--- | :--- | :--- |
| 01 | Integritas dan cara panitia membangun test | Seleksi = laku di D3; D1 resmi; data bersih; 4 klaster baru |
| 02 | Anatomi target/skala | L1/median; 31% nol; k* = 0,42 |
| 03 | Rekonstruksi D1, lifecycle | Filter rilis luas ≥ 25 klaster; kurva D4 0,84, D7 0,38, D10 0,17 |
| 04 | Kalender | Profil hari; libur = Sabtu; sekolah +15%; 30% baris uji tanpa padanan |
| 05 | Klaster/kota | Ukuran timpang; TE klaster memperburuk |
| 06 | Metadata film | Retensi per genre berbeda, sinyal kecil |
| 07 | Cek preprocessing | Skala identik dengan resmi; adversarial harus grouped |
| 08 | Baseline dan desain validasi | LightGBM r/c fitur relatif terbaik (0,358 grouped / 0,328 temporal); noise CV 0,003 |
| 09 | Eksperimen perbaikan | Augmentasi D1 ±1, TE klaster, CatBoost/XGBoost, blend: semua tidak membantu |
| 10 | TabPFN v2 | Blend 0,3 LightGBM + 0,7 TabPFN terbaik di sampel temporal |
| 12 | Jarak CV ke LB | Komposisi skala uji sekitar 2/3 jarak; lahirnya TW-MASE |
| 13 | Proxy D1,D2 ke D3 periode uji | Galat per bucket sama; label periode uji tersedia |
| 14 | Pasangan kecil | Rilis terbatas sebagai data latih tambahan -0,0035; thinning gagal cek prepro |
| 15/16 | Kompetisi klaster, ablation fitur | Sinyal ada, tetapi dalam noise di regresor; dipakai di klasifier |
| 17/18 | Arti "kecil" bergeser | Level absolut terkonfundasi musim; default fitur abs |
| 19 | Segmen tanpa padanan | Ramadan/Lebaran butuh asumsi domain, bukan probing |
| 20 | Offset ~0,07 | Bioskop periode uji jauh lebih jarang mencopot film |
| 21 | Odds correction | a = -1,27 sampai -1,32, stabil per bulan |
| 22 | Propensitas D3 ke D4-D10 | Korelasi 0,13: jembatan lemah |
| 23 | Kalibrasi libur | Grid datar, kalender tetap |
| 24 | Keputusan pull shift | Hurdle 0,3874 vs L1 0,4087 (TW); minimax |
| 25 | λ di proxy | Bobot hurdle 0,75, λ = 0,5 |
| 26 | Kalibrasi rezim uji | Bagian positif terkalibrasi (PIT 0,513, rasio median 1,005) |
| 27 | Fitur klasifier nol | Hanya kompetisi klaster yang membantu (AUC 0,927 ke 0,930) |
| 28 | Analog Lebaran | ρ = 0,5, rasio sekitar 1,7 |
| 29 | Permintaan klaster-hari dari film lain | Ditolak (korelasi residual -0,018) |
| 30 | Tuning hurdle | Klasifier 31 daun lebih baik |
| 31/32 | TabPFN-3.5 OOF dan blend | TabPFN-3.5 > v2; blend median > blend distribusi |
| 33/34 | Heterogenitas shift, kalibrasi Platt | Pergeseran seragam (intersep); Platt tidak membantu |
| 35/36 | TabICL v2 | 0,4564; massa nol buruk; bobot 0 |
| 37 | Oracle galat | Tuas film / klaster-hari / nol masing-masing sekitar 0,055 (oracle) |
| 38 | Residual level film | Tidak terprediksi (R² 0) |
| 39 | Inferensi Lebaran dari LB nyata | Level sudah tepat; rescale < 0,003 |
| 40 | Momentum klaster | Ditolak (korelasi -0,016) |
| 41 | Pembulatan integer | -0,0005, diabaikan |
| 42 | Late starter; bobot bfd | **Bobot validasi dikoreksi**; spesialis gagal |
| 43 | Proxy per segmen | Periode uji tidak lebih sulit |
| 44 | Bobot adversarial | AUC 0,736; offset turun ke sekitar 0,04 saja |
| 45 | DOW periode uji | Tidak meyakinkan, tidak diterapkan |
| 46 | Retune di metrik baru | Knob sudah optimal (gain ≤ 0,0014) |
| 47/47b | Konteks TabPFN | Per-horizon terbaik; pooled 17x lebih lambat |
| 48 | Seed bagging hurdle | Noise |
| 49 | EXAONE OOF CPU | Dihentikan (2.303 detik per fold x horizon) |
| 50 | TabPFN kecil CPU | Dihentikan (user minta tidak run lokal) |
| 51 | Audit eksperimen (Astra) | Bobot dan oracle dihitung ulang; v9 - v8 tidak signifikan |
| 52 | Talent producer/cast | Sinyal lemah, koreksi memperburuk |
| 53 | Aturan D1 | Aturan sekarang konsisten dengan seleksi panitia (test berisi pembukaan runtuh) |
| 54 | Missingness, duplikat (Astra) | 99,5% nol di klaster yang aktif; v10 vs v8 +0,00044 tidak signifikan |
| 55 | Preprocessing D1-D3 (Astra) | Fitur kalender dan panel dibangun; tahan uji peracunan masa depan |
| 56 | Ablation preprocessing (Astra) | Kalender lanjut; panel dan show/TPS/okupansi ditolak (temporal memburuk) |
| 57 | Mekanisme galat (Astra) | 54,5% galat = positif yang tumbuh |
| 58 | Kontrol kebocoran split (Astra) | Split bocor 0,252 vs film disjoint 0,358 |
| 59 | Konfirmasi (Astra) | Kalender konsisten di split kedua; panel shrinkage gagal |
| 60 | Kalender di hurdle (Astra) | 0,333985 ke 0,332800; temporal 0,315517 ke 0,313997 |
| 61 | Learning curve jumlah film (Astra) | Tidak monoton; sensitif terhadap komposisi |
