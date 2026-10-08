# EDA 092: satu show D1-D3 dalam pasangan terpilih

Pertanyaan: apakah rezim preview berokupansi persis 100% terbawa ke D1-D3 test_history, dan bagaimana perubahan ekor satu show?

Metode: gunakan `build_windows` v2 pada train dan `test_windows` pada test_history, jadi hanya pasangan D3 aktif yang masuk. Stack D1-D3 dan hanya hari dengan show positif. Tidak fitting model.

Temuan: pada D3, proporsi satu show di pasangan terpilih 469/7988=5.87% train vs 855/10373=8.24% test. Median tiket satu show D3 19 vs9, porsi <=8 tiket 23.24% vs46.78%. Okupansi persis 100% di antara satu show D3 hanya 2.77% train dan1.40% test; sangat berbeda dari preview train 69.03%. D1 satu show median71 train vs10 test pada sampel kecil 85 dan79 baris.

Interpretasi: pola preview penuh tidak terbawa secara umum ke tiga hari input test. Ekor satu show test lebih sering dan lebih lemah, sesuai pergeseran skala dan retensi yang ditemukan EDA lain. Tidak ada informasi D4-D10 test; nilai ini bukan estimasi skor.
