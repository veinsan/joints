# Pergeseran train vs test adalah level musiman; okupansi relatif pasar tahan pergeseran

**Satu kalimat**: adversarial AUC window train vs test 0,68 dan turun ke 0,51 bila hanya fitur bentuk kurva; okupansi D1-D3 test bermedian 11,0 vs train 20,4 (Feb 2026 6,5, Mar 7,9), tetapi okupansi relatif terhadap film yang rilis 28 hari sebelumnya bermedian sama (1,103 vs 1,108) dengan daya prediksi nol yang setara (spearman -0,47 sampai -0,50).

## Pertanyaan

Apakah model yang belajar dari Apr sampai Sep 2025 akan salah di Okt 2025 sampai Mar 2026 (termasuk Ramadan dan Lebaran)?

## Metode

- `python eda/okupansi_relatif_pasar/adversarial.py`: LGBM classifier train vs test, GroupKFold per film, tiga set fitur (semua, bentuk + okupansi, bentuk saja); statistik per bulan D1; kapasitas kursi per show tersirat per klaster.
- `python eda/okupansi_relatif_pasar/relatif_pasar.py`: indeks pasar dari film lain dengan D1 <= D1 film ini dan > D1 - 28 hari (hanya data masa lalu), okupansi relatif global dan per klaster, stabilitas korelasi per bulan, daftar film uji yang window-nya melintasi libur.

## Hasil

| Bulan D1 test | Okt | Nov | Des | Jan | Feb | Mar |
|---|---|---|---|---|---|---|
| Okupansi D3 median | 9,8 | 16,8 | 30,3 | 15,6 | 6,5 | 7,9 |
| Tiket per show D3 median | 14,0 | 25,5 | 43,2 | 23,0 | 10,0 | 12,4 |

Train per bulan: okupansi D3 18,3 sampai 27,6.

- Adversarial AUC: semua fitur 0,681 (top gain nat_tps_d3 0,40, nat_tix_d3 0,27, n_pairs 0,24); bentuk + okupansi 0,693; bentuk saja 0,511.
- Kapasitas kursi per show per klaster konsisten train-test (spearman 0,96), median 149 vs 150.
- Film test lebih luas: film dengan < 10 pasangan 3% di test vs 23% di train.
- 44 film uji punya libur di D1-D10. 7 film dengan D1 18 Mar 2026 (DANUR: THE LAST CHAPTER dan varian IMAX, SUZZANNA: SANTET DOSA DI ATAS DOSA, TUNGGU AKU SUKSES NANTI, PELANGI DI MARS, SENIN HARGA NAIK, NA WILLA) melintasi Idulfitri 21 sampai 22 Mar di D4-D10. Train tidak punya window yang melintasi Lebaran.

## Interpretasi

Bentuk kurva (tren, pola hari) tidak bergeser; yang bergeser adalah level permintaan pasar (low season Okt, Ramadan Feb sampai Mar, puncak Desember). Eksibitor tidak bisa mencabut semua film saat pasar sepi, jadi ambang pemangkasan kemungkinan relatif terhadap film lain. Fitur okupansi absolut berisiko membuat model memprediksi nol berlebihan pada Okt, Feb, dan Mar.

## Saran FE / modelling

| Saran | Alasan | Prioritas |
|---|---|---|
| Fitur relatif pasar: okupansi, tiket per show, scale dibagi median film yang rilis 28 hari terakhir (global dan per klaster) | Median sama train-test | tinggi |
| Batasi atau buang fitur level absolut film (nat_tps_d3, okupansi absolut) bila validasi pada film pasar rendah memburuk | Sumber utama adversarial AUC | tinggi |
| Evaluasi tambahan: skor terpisah untuk film dengan indeks pasar terendah (proxy Okt/Feb) | Test punya regime yang lebih sepi | sedang |
| Penanganan khusus window Lebaran (7 film, sekitar 5.000 baris test): fitur libur, prior uplift, atau data eksternal terbit sebelum 30 Sep 2025 | Tidak ada padanan di train | tinggi (butuh keputusan user) |
| Kapasitas kursi per show klaster (tiket / okupansi / show) | Identitas klaster stabil | rendah |

## Tingkat keyakinan dan status

- Keyakinan: tinggi untuk adanya pergeseran level; sedang untuk hipotesis ambang relatif (di train rentang pasar sempit sehingga versi absolut dan relatif belum bisa dibedakan tegas).
- Sudah divalidasi dengan eksperimen: belum.
