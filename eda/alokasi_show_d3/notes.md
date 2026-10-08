# EDA 086: alokasi show pada pasangan D2 lemah yang bertahan D3

Pertanyaan: apakah retensi D3 test yang tinggi untuk tiket/show D2 <=8 disertai pemangkasan jumlah show?

Metode: bangun panel D2-D3 menurut D1 yang sama dengan EDA 072. Batasi pasangan yang memiliki tiket positif pada D2 dan D3, lalu bandingkan show serta tiket/show D3 di train dan test, termasuk menurut jumlah show D2.

Hasil: untuk D2 tiket/show <=8 dan D3 masih bertiket, n=444 train vs 2.438 test. Median show D2=5 keduanya; median show D3=2 train vs 3 test; median show D3/D2=0,591 vs 0,667. Proporsi pemangkasan show D2->D3 71,85% train vs 65,87% test. Setelah menyamakan campuran jumlah show D2 (bucket 1,2,3-5,>5) ke test, tingkat pemangkasan train 73,66% vs test 65,87%, selisih -7,78 poin. Dengan seluruh pasangan D2 aktif, termasuk tanpa baris D3, median rasio show tercatat D3/D2 adalah 0 train vs 0,536 test; mean 0,303 vs 0,565.

Kesimpulan: pasangan lemah test lebih sering masih bertiket pada D3 dan show tercatat mereka cenderung kurang agresif dipangkas. Tetapi nilai nol bagi pasangan tanpa transaksi D3 tidak membuktikan tidak ada show yang dijadwalkan; baris dengan nol tiket tidak disediakan. Hasil bersyarat pada D3 bertiket juga mengandung bias seleksi. Implikasi FE: tren `total_show` D1-D3 dan tiket/show tampak langsung menggambarkan pengelolaan kapasitas. Tidak ada model dilatih.
