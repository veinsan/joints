# Perbandingan satu show D1-D3 train dan test

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/okupansi_penuh_awal_rilis/compare.py` dari root. Script memakai builder v2 sehingga hanya pasangan yang aktif D3 dihitung. Output ada di `output/summary_by_day.csv`, `output/one_show_by_day.csv`, dan `output/by_film.csv`.

| D3 | Train | Test history |
|---|---:|---:|
| Pasangan aktif | 7.988 | 10.373 |
| Baris satu show | 469 (5,87%) | 855 (8,24%) |
| Median tiket baris satu show | 19 | 9 |
| Baris satu show dengan <=8 tiket | 23,24% | 46,78% |
| Okupansi persis 100% pada satu show | 2,77% | 1,40% |

Sebagai pembanding, 69,03% baris preview satu show pada train berokupansi persis 100% (lihat [EDA preview](../pola_preview_satu_show/README.md)). Pola preview penuh tidak menjalar secara umum ke D1-D3 test_history. Akan tetapi, ekor satu show dengan tiket sangat rendah lebih besar di test. Ini memberi alasan untuk memantau segmen satu show pada validasi dan interpretasi FE yang memakai tiket per show. Target D4-D10 test tidak diamati, sehingga arah pergeseran setelah D3 belum diketahui. Tidak ada eksperimen modelling.
