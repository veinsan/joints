# Pergeseran retensi D3 bertahan dalam genre utama

**Satu kalimat:** retensi pasangan D2 tiket/show <=8 meningkat di test pada kelima genre utama yang memiliki >=50 pasangan di kedua periode, dengan selisih +24,9 sampai +38,4 poin.

## Pertanyaan

Apakah film test memiliki komposisi genre yang berbeda sehingga pergeseran retensi D3 terlihat besar secara agregat?

## Metode

Jalankan `python eda/retensi_menurut_genre/genre_retention.py` dari root. Gabungkan agregat film EDA 088 dengan `movies.csv` menurut judul dasar. Genre utama diambil dari label sebelum koma; hitung retensi berbobot pasangan D2 lemah per genre. Laporkan genre dengan >=50 pasangan D2 lemah pada train dan test. Join metadata mencakup seluruh 214 baris film-periode yang diuji.

## Hasil

| Genre utama | Pasangan train/test | Retensi train | Retensi test | Selisih |
|---|---:|---:|---:|---:|
| Action | 142 / 368 | 51,4% | 85,6% | +34,2 poin |
| Animation | 72 / 102 | 52,8% | 91,2% | +38,4 poin |
| Drama | 278 / 1.180 | 43,5% | 74,3% | +30,8 poin |
| Horror | 143 / 390 | 39,9% | 77,7% | +37,8 poin |
| Thriller | 51 / 325 | 52,9% | 77,8% | +24,9 poin |

Lima genre tersebut mencakup 71,2% pasangan D2 lemah train dan 74,6% test. Tabel lengkap, termasuk rating usia dan genre jarang, tersedia di `output/`.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Validasi | Pantau retensi D3 dan galat menurut genre utama, tetapi jangan anggap genre menjelaskan pergeseran rezim | tinggi |
| Fitur | Metadata genre dapat dipakai sebagai konteks film; indikator intensitas permintaan dan show tetap diperlukan | sedang |

Urutan label multigenre di `movies.csv` menentukan genre utama dan mungkin bukan taksonomi ideal. Film tidak diacak antar periode; hasil observasional ini tidak membuktikan sebab. Target D4-D10 test tidak tersedia. Tidak ada eksperimen modelling.
