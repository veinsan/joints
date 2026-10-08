# Pasangan lemah test mempertahankan lebih banyak show tercatat sampai D3

**Satu kalimat:** di antara pasangan D2 dengan tiket/show <=8 yang masih bertiket D3, median show D3 adalah 3 pada test vs 2 pada train (median D2 sama-sama 5); pemangkasan show setelah penyesuaian jumlah show D2 adalah 65,87% vs 73,66%.

## Pertanyaan

Apakah pergeseran retensi D3 pada pasangan lemah disertai perubahan alokasi show, atau sekadar show sisa yang sangat sedikit?

## Metode

Jalankan `python eda/alokasi_show_d3/show_transition.py` dari root. D1 train dan test dibangun seperti EDA 072. Bandingkan pasangan D2 tiket/show <=8 yang masih memiliki tiket D3, lalu timbang tingkat pemangkasan show per bucket jumlah show D2 menurut komposisi test. Analisis tambahan memasukkan semua pasangan D2 aktif dengan show D3 tercatat nol bila tidak ada baris D3.

## Hasil

| Ukuran, D2 tiket/show <=8 dan D3 bertiket | Train (n=444) | Test (n=2.438) |
|---|---:|---:|
| Median show D2 | 5 | 5 |
| Median show D3 | 2 | 3 |
| Median rasio show D3/D2 | 0,591 | 0,667 |
| Porsi jumlah show turun | 71,85% | 65,87% |
| Porsi D3 tinggal satu show | 27,93% | 20,02% |

Setelah menimbang menurut jumlah show D2 test (1, 2, 3-5, >5), proporsi pemangkasan show train menjadi 73,66% dan test tetap 65,87%, selisih -7,78 poin. Pada semua pasangan D2 lemah, termasuk yang tidak punya transaksi D3, median rasio show tercatat D3/D2 ialah 0 di train dan 0,536 di test. Rincian ada di `output/transition_summary.csv`, `weak_by_d2_show_bucket.csv`, `unconditional_d2_transition.csv`, dan `adjusted_cut.json`.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Fitur | Gunakan tren jumlah show D1-D3 bersama tiket/show untuk memisahkan permintaan dan pasokan pertunjukan | tinggi |
| Validasi | Bandingkan transisi show D2-D3 menurut permintaan dan bulan agar CV train tidak menyembunyikan rezim alokasi yang berbeda | tinggi |

Baris D3 yang tidak ada berarti tidak ada transaksi tercatat; tidak diketahui apakah show tetap dijadwalkan tanpa tiket. Perbandingan yang membatasi pada pasangan D3 bertiket terkena seleksi, dan D1 train direkonstruksi. Ini EDA observasional, bukan bukti sebab operasional atau performa model.
