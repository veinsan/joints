# EDA 089: retensi D3 pasangan D2 lemah menurut genre

Pertanyaan: apakah pergeseran retensi lemah hanya karena komposisi genre film test berbeda?

Metode: gunakan agregat per judul dasar dari EDA 088, join metadata `movies.csv`, lalu hitung retensi berbobot pasangan dan median per-film menurut genre utama serta rating usia. Genre utama adalah token sebelum koma; join yang tak cocok dicatat sebagai unknown. Laporkan genre dengan >=50 pasangan D2 lemah pada kedua periode.

Hasil: metadata tersedia untuk 214 baris film-periode (100%). Lima genre utama punya >=50 pasangan D2 lemah per periode dan semuanya menunjukkan kenaikan retensi: Action +34,2 poin, Animation +38,4, Drama +30,8, Horror +37,8, Thriller +24,9. Kelimanya mencakup 74,6% pasangan D2 lemah test dan 71,2% train. Rating Dewasa dan Remaja juga menunjukkan kenaikan, sedangkan Semua Umur lebih kecil dan ukuran sampel lebih terbatas.

Kesimpulan: pergeseran agregat tidak dapat dijelaskan semata oleh kenaikan porsi genre tertentu. Genre utama adalah token pertama metadata multigenre dan bukan kategori kausal; genre minor punya sampel sedikit. Implikasi validasi: diagnostik rezim tetap diperlukan di tiap segmen genre utama. Tidak ada model dilatih.
