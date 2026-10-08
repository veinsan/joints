# Ekor pasangan lemah test berasal dari permintaan D2 dan retensi

**Satu kalimat:** porsi pasangan bertiket/show D2 <=8 yang masih aktif D3 naik dari 5,67% train ke 24,05% test; campuran D2 test dengan aturan retensi train akan menghasilkan 17,13%.

## Pertanyaan

Mengapa semakin banyak pasangan kecil masuk sampel test D4-D10: apakah karena permintaan awal lebih lemah, atau karena pasangan lemah lebih sering tetap tayang di D3?

## Metode

Jalankan `python eda/dekomposisi_ekor_kecil/small_tail.py` dari root. D1 train memakai rekonstruksi v2 dan pembatasan resmi proyek; D1 test memakai awal test_history per judul dasar. Pada pasangan aktif D2, hitung campuran bucket tiket/show dan peluang bertransaksi D3. Bentuk counterfactual aritmetik dengan mengalikan jumlah pasangan test pada tiap bucket D2 dengan retensi train pada bucket itu. Bandingkan juga porsi skala MASE <=20 di pasangan terpilih D3. Tidak ada fitting model.

## Hasil

| Ukuran | Train | Test |
|---|---:|---:|
| Porsi tiket/show D2 <=8 di antara pasangan aktif D2 | 10,90% | 28,64% |
| Retensi D3 pada bucket D2 <=8 | 46,11% | 76,86% |
| Porsi bucket D2 <=8 di antara pasangan aktif D2 dan D3 | 5,67% | 24,05% |
| Porsi D2 <=8 di seluruh pasangan terpilih D3 | 5,56% | 23,50% |
| Porsi skala MASE <=20 di seluruh pasangan terpilih | 3,20% | 12,96% |
| Di dalam skala <=20, porsi dengan D2 <=8 | 47,66% | 73,29% |

Bila distribusi D2 test dipakai dengan retensi train per bucket, porsi D2 lemah dalam pasangan D2+D3 aktif menjadi **17,13%**. Dalam urutan dekomposisi ini, kenaikan dari 5,67% ke 17,13% adalah 11,46 poin karena campuran D2, lalu kenaikan ke 24,05% adalah 6,92 poin terkait retensi. Angka kontribusi tergantung urutan counterfactual dan pemilihan bucket; ini bukan efek kausal. Rincian ada di `output/d2_mix_retention.csv`, `selected_cross_tab.csv`, dan `summary.json`.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Validasi | Laporkan OOF menurut skala MASE dan tiket/show D2, selain rerata tunggal | tinggi |
| Validasi | Bandingkan retensi D2->D3 train/test_history secara terpisah; CV train tidak mereplikasi retensi test | tinggi |
| Fitur | Gabungkan sinyal permintaan per show D1-D3 dengan ukuran klaster/show; jangan mengartikan skala rendah semata sebagai film pasti dicabut | tinggi |

Hasil ini konsisten dengan EDA 072, 074, dan 081. Target D4-D10 test tidak terlihat, sehingga kelanjutan setelah D3 tetap tidak diketahui. Tidak ada eksperimen modelling.
