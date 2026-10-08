# EDA 076: sensitivitas tanggal D1 train

Pertanyaan: apakah pergeseran retensi D2 ke D3 hanya akibat salah penentuan D1 di train?

Hasil: pada bucket tiket/show D2 <=8, tingkat berhenti train dengan D1 v2 adalah 53,8% (956 pasangan setelah filter outage ketat), test 23,1% (3.172). Jika D1 train dimajukan satu hari, angka train turun menjadi 29,2% (1.648 pasangan); jika dimundurkan satu hari, 48,9% (454). Untuk film tanpa transaksi sebelum D1 v2, train 72,4% (123). Pada bucket 8-15, train D1 dimajukan satu hari 9,1% vs test 5,0%.
Pada subset 137 judul yang aman untuk pergeseran satu hari, 111 punya transaksi sebelum D1 v2 (median lead 5 hari). D1 v2 cocok persis dengan lima tanggal rilis luas independen: JALAN PULANG, SORE ISTRI DARI MASA DEPAN, BELIEVE, PANGGIL AKU AYAH, JADI TUH BARANG. Transaksi pertama lima film tersebut justru 7 sampai 33 hari lebih awal.

Kesimpulan: D1 v2 mendapat dukungan eksternal kuat untuk lima film yang diuji, sementara transaksi pertama adalah preview. Memajukan D1 satu hari mengubah fase rilis sehingga tidak otomatis merupakan definisi training yang benar. Gap deskriptif tetap ada, tetapi sebab operasionalnya belum terbukti.
