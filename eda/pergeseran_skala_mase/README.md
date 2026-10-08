# Pergeseran skala MASE ke pasangan kecil

**Satu kalimat**: pasangan dengan skala MASE <=20 meningkat dari 3,2% train menjadi 13,0% test, bahkan 22,6% pada Februari dan 25,5% pada Maret 2026.

## Pertanyaan

Apakah campuran pasangan yang dinilai oleh MASE sama dengan campuran sampel latih?

## Metode

Jalankan `python eda/pergeseran_skala_mase/scale_composition.py` dari root. Script membangun window v2, menghitung skala resmi dari D1-D3, membandingkan train/test per bucket dan bulan D1, serta mendeskripsikan frekuensi target nol train per bucket. Output CSV dan `output/scale_composition.png` di `output/`.

## Hasil

| Ukuran | Train | Test |
|---|---:|---:|
| Pasangan | 7.988 | 10.373 |
| Median skala | 181,67 | 91,67 |
| Skala <=20 | 3,2% | 13,0% |

Pada train, 75,1% baris target D4-D10 untuk skala 10-20 bernilai nol. Frekuensi itu tidak diketahui untuk target test. Komposisi skala test paling berbeda pada Februari dan Maret.

## Interpretasi

Rata-rata CV pada train menimbang lebih sedikit pasangan kecil daripada evaluasi test. MASE membagi galat dengan skala pasangan, sehingga kelompok kecil perlu dilaporkan tersendiri. EDA retensi D3 menunjukkan perilaku nol train tidak dapat dipindahkan begitu saja ke test.

## Saran FE / validasi

| Saran | Alasan | Prioritas |
|---|---|---|
| Laporkan MASE per bucket skala dan per periode | Memperlihatkan gap yang tertutup rata-rata CV | tinggi |
| Diagnostik sensitivitas CV terhadap komposisi skala test | Mengukur risiko perubahan campuran tanpa memakai label test | tinggi |
| Perhatikan fitur pasar relatif pada pasangan kecil | Skala absolut dapat menandakan pasar sepi, bukan kegagalan film | tinggi |

## Tingkat keyakinan dan status

Keyakinan tinggi pada pergeseran komposisi; efeknya terhadap skor target test tidak diketahui. Tidak ada eksperimen modelling.
