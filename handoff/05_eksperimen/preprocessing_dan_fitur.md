# Preprocessing, fitur, dan teknik yang dicoba

## Yang dipakai (lolos)

| Teknik | Bukti | Versi |
| :--- | :--- | :--- |
| Simulasi aturan panitia (D1 resmi, rilis luas, laku di D3, zero-fill, buang outage) | Distribusi cakupan dan hari D1 cocok dengan uji | v1 |
| Target r/c (pengali kalender struktural) | Temporal 0,342 ke 0,328 | v1 |
| Fitur level relatif dan hanya `log_s` + `f_logT` absolut | Adversarial AUC turun; CV 0,358 | v1 |
| Rilis terbatas sebagai data latih tambahan | -0,0035, konsisten 2 seed fold | v3 |
| TW-MASE sebagai metrik keputusan | Menjelaskan sekitar 2/3 jarak CV ke LB | v3 |
| Model hurdle (p0 + 19 kuantil) + L1 mix 0,75 | Hurdle TW 0,3874 vs L1 0,4087 (bucket) | v4 |
| Pull shift λ = 0,5 (hanya untuk LightGBM) | Proxy periode uji dan LB v3 ke v4 | v4 |
| Fitur kompetisi klaster (hanya klasifier) | AUC 0,927 ke 0,930; TW hurdle -0,0038 | v5 |
| Analog Lebaran ρ = 0,5 | Public -0,039 | v5 |
| Klasifier 31 daun | TW-κ 0,4058 ke 0,4038 | v6 |
| Bobot validasi bucket x hari pertama laku | Memperbaiki komposisi late starter | v8 |
| Statistik klaster fold-local | Menghapus kebocoran TW +0,0007 | v10 |
| Kalender sekolah Jawa Barat | Koreksi fakta (1,56% baris uji); tidak bisa divalidasi CV | v10 |
| λ dipilih per model kuantil | TabPFN-2.5 dan TabM memilih λ = 0 | v12 |

## Yang ditolak, dengan alasan

| Teknik | Hasil | Mengapa tidak efektif |
| :--- | :--- | :--- |
| Augmentasi D1 ±1 hari | +0,007 (lebih buruk) | Mengaburkan aturan D1 yang presisi |
| Target encoding klaster (OOF) | +0,0025 | Noise; residual klaster split-half hanya 0,29 |
| Bobot komposisi uji saat training | 0,4094 ke 0,4136 | Varians naik; bobot hanya untuk evaluasi |
| Thinning binomial (pasangan kecil sintetis) | 0,4171; cek prepro gagal (zero-rate 0,52 vs 0,76 asli) | Meniru penonton sedikit, bukan bioskop mencopot film |
| Fitur level relatif pasar ("both") | Proxy memilih, GroupKFold dan transfer menolak | Bukti bertentangan; default abs |
| Indeks permintaan klaster-hari dari film lain (`eda/29`) | Korelasi -0,018 | Tidak ada sinyal |
| Rekalibrasi libur (`eda/23`) | Grid datar | - |
| Shift per bucket, kalibrasi Platt (`eda/33-34`) | Tidak membantu | Heterogenitas berasal dari miskalibrasi model proxy |
| Override Natal | Cek kapasitas wajar | Tidak perlu |
| Blend distribusi (rata-rata kuantil) | 0,4039 vs blend median 0,4025 | Kalah tipis |
| Koreksi level film (`eda/38`, talent `eda/52`) | Memperburuk | Tidak terprediksi dari D1-D3 |
| Momentum klaster dari film lain (`eda/40`) | Korelasi -0,016 | - |
| Pembulatan integer | -0,0005 | Dapat diabaikan |
| Spesialis late starter; target r' = y/max(s, y3) | Memburuk | Bimodal; median tetap rendah |
| Penyesuaian DOW periode uji (`eda/45`) | Tidak meyakinkan | Decay dan DOW terkonfundasi |
| A1: prediksi perubahan show dan tiket per show (nested cross-fit) | TW +0,0002; temporal tipis; 1 bulan +0,007 | Gagal aturan adopsi |
| A2: metadata resmi (angka judul, subtitle, reissue, cast overlap) | TW -0,0018 tetapi September +0,0035 | Gagal syarat bulan |
| A3: fitur input dinormalisasi kalender | TW -0,0009, temporal -0,0031 (LightGBM); di blend v12 TW 0, temporal -0,003 | **Selalu di bawah ambang 0,0015; kandidat paling konsisten** |
| Panel bioskop (tren di bioskop yang sama D1 dan D3) | Grouped -0,0054, temporal **+0,0070** | Overfit; shrinkage n/(n+20) juga gagal |
| Show/TPS/okupansi D1-D3 sebagai fitur | Temporal +0,0003, 1 bulan +0,005 | Ditolak |
| Aturan D1 kausal | Membuang pembukaan runtuh yang ada di uji | Aturan sekarang dipertahankan |
| Hurdle weight 1,0 / retune knob (`eda/46`) | Gain ≤ 0,0014 | Knob sudah optimal |
| Pemilihan blend dengan label asli saja (v10) | Pilih EXAONE 0,7; public memburuk | Diganti minimax 3 lensa |

## Teknik validasi yang dibangun

- Proxy D1,D2 ke D3 periode uji (`eda/13`): satu-satunya label periode uji.
- Analisis keputusan minimax regret (`eda/24`, `eda/25`).
- Dunia κ dan kalibrasinya ke LB (`eda/26`, kemudian dikoreksi di v8).
- Bobot adversarial (`eda/44`), backtest temporal (v10), bootstrap per film (`eda/51`), aturan adopsi tertulis
  (v10).
- Kontrol negatif kebocoran split (`eda/58`), learning curve jumlah film (`eda/61`).
