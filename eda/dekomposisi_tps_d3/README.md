# Ekor tiket D3 terutama terkait okupansi, bukan kapasitas tersirat

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/dekomposisi_tps_d3/occupancy_capacity.py` dari root. Input pasangan D3 berasal dari [EDA 094](../persaingan_film_baru/README.md). Kapasitas kursi per show tersirat dihitung dari `total_ticket * 100 / occupation_rate / total_show` bila okupansi positif dan <=100. Ini rata-rata tersirat pada `cinema_ids` agregat, bukan kapasitas satu gedung.

| Ukuran D3 | Train | Test history |
|---|---:|---:|
| Pasangan | 7.988 | 10.373 |
| Median tiket/show | 33,18 | 18,57 |
| Median okupansi | 22,46% | 12,36% |
| Median kapasitas/show tersirat | 148,63 | 153,12 |
| Okupansi nol bertiket positif | 8 | 60 |

Pada keenam bucket jumlah show, median kapasitas tersirat test sedikit lebih tinggi, tetapi median okupansi jauh lebih rendah. Identitas `log(tiket/show) = log(okupansi/100) + log(kapasitas/show)` memberi selisih rata-rata log test minus train: **-0,567 = -0,598 + 0,030** pada baris dengan okupansi valid. Pada 452 strata `cinema_ids` x bucket show bersama (cakupan 90,74% pasangan test), rasio geometrik test/train masing-masing **0,582** untuk tiket/show, **0,561** untuk okupansi, dan **1,037** untuk kapasitas tersirat.

Sebagai cek terhadap pembulatan atau kapasitas ekstrem, pembatasan `occupation_rate` 1–100% dan kapasitas 50–400 kursi/show masih memberi selisih log tiket/show -0,529, okupansi -0,545, dan kapasitas +0,016. Dengan demikian kapasitas tersirat yang lebih kecil tidak menjelaskan penurunan tiket/show D3.

Penjelasan ini aritmetis, bukan identifikasi sebab. Okupansi sendiri dihitung dari tiket dan kapasitas oleh sumber data; sebab keterisian yang lebih rendah bisa melibatkan campuran film, musim, pilihan show, atau aturan pencatatan. Delapan dan 60 baris dengan okupansi nol tetapi tiket positif dikeluarkan dari dekomposisi log. Tidak ada eksperimen modelling.
