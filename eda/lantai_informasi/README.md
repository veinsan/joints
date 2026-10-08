# Lantai informasi MASE pada CV terkunci (SGKF5, window v2)

EDA 047-056 (2026-09-29). Pertanyaan: bisakah CV MASE < 0.25 dicapai dengan prepro apapun,
model XGB GPU tunggal terkunci, validasi SGKF5 (group judul dasar) terkunci?

## Kesimpulan

**Tidak.** Target 0.25 berada di bawah lantai informasi yang bisa dicapai model jujur pada
validasi ini. Bukti berlapis:

| Bukti | Angka | Arti |
|---|---|---|
| Oracle trajektori film benar, tanpa info pasangan (median in-sample per film x h) | 0.2700 | HINDSIGHT murni; sudah > 0.25 sebelum pasangan dihitung |
| Oracle partisi halus + bucket pasangan (film,h,days,scale,show,occ) | 0.158-0.24 | Hindsight IDENTITAS FILM; tidak transfer antar fold |
| Lookup OOF bucket tanpa identitas film (h,days,scale,cohort,dow,occ) | 0.36-0.43 | Semua struktur yang generalisasi antar fold: lebih buruk dari model |
| Drift pasangan D4-D10 dari fitur D1-D3 | R2 0.003 | Tak terprediksi (ditentukan keputusan program masa depan) |
| Kekuatan klaster lintas film | R2 0.035 (split-half rho 0.33) | Nyaris nol |
| Deviasi film dari kohort (kaki panjang) dari fitur D1-D3 | R2 ~ 0 | Slope nasional hanya lemah prediksi survival |
| Lantai NB dengan intensitas sejati | 0.2297 | Hindsight total; target 0.25 cuma 0.02 di atasnya |

Posisi model: deterministik (kohort x kalender x scale) = 0.4074; XGB GPU terkunci terbaik =
0.3187 (per-horizon); TabPFN-3.5 (inductive bias lebih kuat, fitur sama) = 0.29995.
Model sudah menangkap seluruh struktur bucket yang generalisasi; sisa error = ketidakpastian
trajektori film antar-fold + drift pasangan yang memang tidak teramati di D1-D3.

## Catatan penting

- Fold 1-3 (0.35-0.38) lebih sulit dari fold 0/4 (0.24-0.26): berisi film yang tumbuh
  pasca-D3 (mis. SORE ISTRI DARI MASA DEPAN rel 2.2 vs pred 0.9) - sampel film terlalu
  sedikit (141) untuk mempelajari aturan dari 2-3 kasus.
- h=4-5 terburuk (0.35-0.36): umumnya akhir pekan pertama setelah rilis Rabu/Kamis.
- Baris positif cenderung under-prediksi (median pred_rel 0.555 vs rel 0.63): efek median
  pada distribusi campur nol, bukan kalibrasi salah.
- Metadata eksternal TMDB (Kaggle asaniczka v11) tidak mengcover rilis 2025 dengan baik
  (match 71/264, budget 12/71, tanpa sinyal) - jalur metadata eksternal ditutup untuk CV.

## File

- `scan.py`: dekomposisi error OOF XGB (bucket scale/h/nol, film, oracle swap pasangan 0.2293).
- `pair_factor.py`: prediktabilitas drift pasangan (R2 0.003, dekomposisi klaster).
- `film_traj.py`: prediktabilitas trajektori film, prediktor deterministik 0.4074.
- `ceiling.py`: O1 = 0.2700, reliabilitas drift split-half 0.674/0.805, lantai NB.
- `oracle.py`: oracle partisi halus (hindsight) 0.27 -> 0.158.
- `lookup.py`: lookup OOF tanpa identitas film 0.36-0.43 (penutup argumen).

## implikasi

Untuk menurunkan CV di bawah ~0.30 perlu melonggaran lock: model lain (TabPFN-3.5 sudah
terbukti 0.29995 di fitur yang sama) atau ensemble (dilarang user). Dalam batas lock saat ini,
prepro terbaik = v4 + target kalender + per-horizon (0.3187).
