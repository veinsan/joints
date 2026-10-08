# Komposisi skala test mengubah pembacaan CV

**Satu kalimat**: menimbang ulang galat OOF lama menurut distribusi skala test menaikkan ringkasan MASE XGB dari 0,3164 ke 0,4048 dan TabPFN e9 dari 0,2985 ke 0,3770, tanpa fitting model baru.

## Pertanyaan

Berapa bagian kesenjangan CV ke evaluasi test yang dapat dijelaskan oleh campuran skala MASE yang berbeda?

## Metode

Jalankan `python eda/transport_komposisi_cv/transport.py` dari root. Script membaca OOF lama XGB terkunci dan TabPFN e9, menghitung galat absolut tereskala per bucket skala D1-D3, lalu menimbang rata-ratanya dengan distribusi skala pasangan test dari `eda/pergeseran_skala_mase/`. Bootstrap 1.000 kali per judul dasar memberi rentang selisih. Script tidak memanggil proses training, prediksi, atau tuning. Output di `output/summary.csv` dan `output/by_scale.csv`.

## Hasil

| Artefak OOF lama | MASE CV biasa | Dengan campuran skala test | Selisih | CI 95% selisih bootstrap film |
|---|---:|---:|---:|---:|
| XGB terkunci | 0,3164 | 0,4048 | +0,0884 | +0,0283 sampai +0,2060 |
| TabPFN e9 | 0,2985 | 0,3770 | +0,0784 | +0,0207 sampai +0,1928 |

Bucket skala 3-10 menyumbang 0,94% baris CV train tetapi 4,38% pasangan test; galat OOF bucket itu 1,88 (XGB) dan 1,78 (TabPFN). Hanya 75 pasangan train berada di bucket tersebut, sehingga ketidakpastiannya tinggi.

## Interpretasi

Rata-rata CV pada campuran train mengecilkan peran pasangan kecil yang lebih umum di test. Angka tertimbang mengukur sensitivitas ringkasan CV terhadap komposisi, bukan estimasi skor test yang tervalidasi. Hubungan galat dengan skala dapat berubah antara periode, sebagaimana retensi D3 juga berubah. Artefak TabPFN e9 hanya digunakan sebagai pemeriksaan kedua atas pola, bukan rekomendasi model.

## Saran validasi

| Saran | Alasan | Prioritas |
|---|---|---|
| Laporkan MASE OOF per skala dan ringkasan tertimbang komposisi test | Memperlihatkan risiko yang tertutup CV rata-rata | tinggi |
| Bootstrap per judul dasar | Error baris satu film tidak independen | tinggi |
| Pertahankan validasi group film dan analisis temporal | Penimbangan skala tidak mengoreksi pergeseran retensi D3 | tinggi |

## Tingkat keyakinan dan status

Keyakinan tinggi bahwa komposisi skala berbeda dan berdampak pada ringkasan OOF lama; rendah bila angka tertimbang ditafsirkan sebagai skor target test. Analisis ini tidak menjalankan eksperimen modelling.
