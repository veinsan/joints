# Retensi D3 berubah dalam klaster bioskop yang sama

**Satu kalimat:** 81 dari 86 klaster dengan observasi cukup memiliki retensi D3 lebih tinggi di test untuk pasangan D2 dengan tiket/show <=8; gap terstandardisasi per klaster dan bucket permintaan adalah +15,35 poin.

## Pertanyaan

Apakah pergeseran retensi D2 ke D3 yang ditemukan pada EDA 072 hanya disebabkan perubahan campuran bioskop?

## Metode

Jalankan `python eda/retensi_per_klaster/cluster_retention.py` dari root. D1 train memakai rekonstruksi v2 dengan syarat batas data dan pengecualian outage yang sama. D1 test adalah tanggal pertama test_history per judul dasar. Pasangan yang aktif D2 dibandingkan dengan ada/tidaknya transaksi D3. Klaster dianalisis berpasangan lintas periode; strata klaster x bucket tiket/show D2 dihitung bila memiliki minimal 3 observasi masing-masing periode. Bootstrap 1.000 kali mengambil ulang judul dasar agar ketergantungan antar klaster dalam film ikut diperhitungkan.

## Hasil

| Ukuran | Hasil |
|---|---:|
| Klaster sama di train dan test | 117 |
| Cakupan pasangan aktif D2 test di klaster sama | 99,04% |
| Retensi D3 pada tiket/show D2 <=8, train | 46,11% (n=963) |
| Retensi D3 pada tiket/show D2 <=8, test | 77,12% (n=3.121) |
| Klaster dengan kenaikan / penurunan / sama, min. 5 contoh masing-masing | 81 / 4 / 1 |
| Uji tanda satu sisi | p=5,5e-20 |
| Gap setelah standardisasi klaster x tiket/show D2 | +15,35 poin |
| Cakupan test pada standardisasi | 96,96% |
| CI 95% bootstrap per judul dasar | +11,34 sampai +19,34 poin |

Bila syarat minimum per strata dinaikkan menjadi 5, gap +15,11 poin dengan cakupan test 92,33%; untuk 10, +12,52 poin dengan cakupan 69,71%. Rincian di `output/summary.json`, `by_cinema_low_tps.csv`, dan `threshold_sensitivity.csv`.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Validasi | Laporkan transisi D1-D3 pada test_history sebagai diagnostik rezim, terpisah dari OOF train | tinggi |
| Validasi | Periksa segmen tiket/show rendah pada klaster yang sama dan per film | tinggi |
| Fitur | Gunakan intensitas dan permintaan lokal pada D1-D3, bukan identitas klaster saja, untuk menggambarkan alokasi show | sedang |

Hasil membantah penjelasan sederhana bahwa perbedaan timbul hanya dari klaster bioskop baru atau campuran lokasi. Analisis ini tidak membuktikan sebab operasional maupun nilai target D4-D10 test. D1 train direkonstruksi, sehingga ketidakpastian batas fase rilis masih berlaku.

## Status

Keyakinan tinggi pada perbedaan retensi D2-D3 dalam klaster yang sama; keyakinan rendah untuk ekstrapolasi ke target test tersembunyi. Tidak ada fitting model.
