# Retensi D3 pasangan lemah meningkat pada banyak film

**Satu kalimat:** untuk film dengan >=10 pasangan D2 bertiket/show <=8, median retensi D3 per film naik dari 45,4% train (38 film) menjadi 81,4% test (90 film), selisih +36,0 poin.

## Pertanyaan

Apakah kenaikan retensi D3 test hanya berasal dari beberapa film yang menghasilkan banyak pasangan lemah?

## Metode

Jalankan `python eda/retensi_per_film/film_retention.py` dari root. D1 train memakai rekonstruksi v2 dan aturan penyaringan proyek; D1 test memakai tanggal pertama test_history per judul dasar. Pada pasangan D2 tiket/show <=8, hitung peluang transaksi D3 per judul dasar. Laporkan film dengan >=10 pasangan agar rate kecil tidak mendominasi. Bootstrap 3.000 kali mengambil ulang film untuk selisih median. Hitung juga konsentrasi pasangan lemah terpilih pada 5/10 film terbesar.

## Hasil

| Ukuran | Train | Test |
|---|---:|---:|
| Film dengan >=10 pasangan D2 lemah | 38 | 90 |
| Median retensi D3 per film | 45,43% | 81,39% |
| Film dengan retensi >=70% | 21,05% | 70,00% |
| Film dengan retensi <=50% | 57,89% | 11,11% |
| Porsi pasangan lemah terpilih dari 10 film terbesar | 41,89% | 26,50% |

Selisih median per film **+35,96 poin**, CI bootstrap 95% **+16,52 sampai +48,78 poin**. Sebanyak 74,44% film test yang memenuhi syarat melampaui kuartil atas retensi film train (67,57%). Seluruh enam bulan test memiliki median retensi per film antara 66,7% dan 86,6%; bulan train antara 28,0% dan 64,3%. Rincian pada `output/by_film.csv`, `by_month.csv`, dan `summary.json`.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Validasi | Periksa retensi D2-D3 test_history menurut film dan bootstrap per judul dasar, bukan hanya menghitung baris independen | tinggi |
| Validasi | Laporkan bucket tiket/show rendah terpisah; beberapa bulan train tidak mendekati distribusi test | tinggi |
| Fitur | Jangan memakai asumsi train bahwa tiket/show <=8 hampir pasti berhenti sebelum D3; D3 yang teramati membawa informasi tentang alokasi show | tinggi |

Data ini tidak menunjukkan target D4-D10 test. Perbedaan komposisi film/genre dan rekonstruksi D1 train belum sepenuhnya dipisahkan, sehingga perubahan kebijakan eksibitor belum terbukti secara kausal. Tidak ada eksperimen modelling.
