# Pergeseran tiket per show D3 bertahan pada jumlah show sebanding

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/pergeseran_tps_d3_berdasar_show/show_bucket.py` dari root. Input adalah [pasangan D3 dari EDA 094](../persaingan_film_baru/README.md). Semua analisis deskriptif, tanpa fitting model.

| Show D3 | Pasangan train/test | Median tiket/show train/test | Proporsi tiket/show <=8 train/test |
|---|---:|---:|---:|
| 1 | 469 / 855 | 19 / 9 | 23,24% / 46,78% |
| 2 | 1.083 / 1.760 | 22 / 12 | 12,28% / 35,00% |
| 3-4 | 1.484 / 1.936 | 29 / 15,67 | 6,00% / 25,83% |
| 5-8 | 2.859 / 3.393 | 37 / 22,60 | 3,39% / 14,00% |
| 9-16 | 1.269 / 1.461 | 39 / 24,67 | 1,81% / 10,54% |
| >=17 | 824 / 968 | 39,76 / 24,31 | 0,61% / 6,92% |

Setiap bucket menunjukkan penurunan median tiket/show dan kenaikan ekor <=8 di test. Jika campuran jumlah show test dipadukan dengan tingkat train per bucket, hasilnya **6,54%**, dibanding **21,32%** test. Pada 452 strata `cinema_ids` x bucket show yang memiliki minimal lima pasangan pada tiap periode dan mencakup 91,12% pasangan test, tingkat tertimbang train **5,61%** vs test **20,05%**.

Jadi perubahan campuran jumlah show tidak cukup menjelaskan pergeseran D3. Film, musim, okupansi/kursi, dan keputusan pemutaran masih dapat berpengaruh. D3 adalah bagian history yang tersedia; audit ini tidak menyimpulkan perilaku target D4-D10. Output ada di `output/by_show_bucket.csv` dan `output/shared_cluster_show_strata.csv`.
