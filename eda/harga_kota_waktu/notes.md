# EDA 098: harga statis dan pergeseran D3

Pertanyaan: apakah campuran kota/kategori harga tiket menjelaskan kenaikan pasangan D3 dengan tiket/show <=8 pada test?

Metode: baca pasangan D3 EDA094. Join ticket_prices.csv pada city_name dan price_day yang diturunkan dari hari kalender. Tabel harga tidak memiliki dimensi tanggal selain kategori Weekday/Friday/Weekend. Bagi nilai ceil menjadi empat bucket berbasis kuantil, bandingkan weak rate, standardisasi deskriptif ke campuran harga test, dan cek kota yang sama dengan >=30 pasangan per periode. Tanpa fitting model.

Hasil: 69 kota masing-masing punya tiga kategori harga, tanpa duplikasi; join 0 missing untuk 7988 train dan10373 test D3. Pada empat bucket harga, weak rate test18.09-23.81% vs train3.50-6.73%. Campuran harga test x tingkat train menghasilkan5.71%, hampir sama dengan train agregat5.71%, jauh dari test21.32%. Dari 66 kota dengan>=30 pasangan pada tiap periode (98.91% cakupan pasangan test), 66/66 punya weak rate lebih tinggi di test.

Interpretasi: komposisi kota dan harga yang tercatat tidak menjelaskan pergeseran D3. Kolom ceil hanya tiga harga kategori per kota dan tidak mengungkap perubahan harga aktual sepanjang waktu, diskon, format layar, atau harga transaksi. Tidak ada klaim efek kausal harga.
