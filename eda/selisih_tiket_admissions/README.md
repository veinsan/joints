# EDA 101: uji selisih tiket dan admissions dalam okupansi satu show

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/selisih_tiket_admissions/numerator_gap.py` dari root. Input hanya `train.csv` dan `test_history.csv`; tidak ada fitting model atau target test D4-D10.

[Dokumentasi laporan Vista](https://help.vista.co/hc/en-nz/articles/13519958299033-Marketing-report) membedakan `Paid Admits` dari `Box Office Admits` yang mencakup tiket gratis dan menghitung okupansi dari admissions terhadap total kursi. Ini hipotesis mekanisme umum, bukan bukti bahwa dataset memakai Vista. Untuk setiap baris satu show dengan tiket 1-400 dan okupansi 10-<100%, script mencari denominator kursi integer 50-400 yang cocok setelah menambah atau mengurangi 0-10 tiket, lalu mengacak okupansi 100 kali di dalam desil jumlah tiket sebagai kontrol aritmetika.

| Periode | Baris | Cocok tanpa selisih | Hanya cocok dengan tambahan | Hanya cocok dengan pengurangan | Selisih arah teramati | Kontrol acak, rerata [95%] |
|---|---:|---:|---:|---:|---:|---:|
| Train | 4.259 | 227 | 1.017 | 561 | +10,71 poin | +10,31 [9,10;11,60] poin |
| Test history | 555 | 30 | 177 | 61 | +20,90 poin | +20,48 [17,74;24,15] poin |

Lebih banyak kecocokan pada arah tambahan ternyata hampir sepenuhnya muncul juga pada kontrol acak. Hasil ini **tidak mendukung inferensi spesifik** bahwa tiket gratis adalah penyebab ketidakcocokan denominator integer; hipotesis itu juga belum terbantahkan karena jumlah tiket gratis tidak tersedia. Audit ini mengingatkan bahwa pencarian denominator dan numerator fleksibel mudah menghasilkan pola arah yang semu. Output `summary.json` dan baris audit tersedia di `output/`.
