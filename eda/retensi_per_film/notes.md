# EDA 088: apakah pergeseran retensi lemah didorong segelintir film?

Pertanyaan: apakah retensi D3 yang lebih tinggi pada pasangan D2 tiket/show <=8 tersebar di banyak film, atau hanya beberapa blockbuster/outlier?

Metode: gunakan D1 train v2 dan D1 test_history per judul dasar. Hitung retensi D3 per judul dasar untuk yang memiliki >=10 pasangan D2 lemah. Bandingkan distribusi rate film dan konsentrasi pasangan terpilih pada top 5/10 judul, termasuk per bulan.

Hasil: 38 film train dan 90 film test memiliki >=10 pasangan D2 tiket/show <=8. Median retensi per film 45,43% train vs 81,39% test, selisih +35,96 poin, CI bootstrap per film +16,52 sampai +48,78 poin. Sebanyak 74,44% film test melampaui kuartil atas retensi film train (67,57%). Share 10 film teratas dari pasangan lemah yang terpilih D3 adalah 41,89% train vs 26,50% test. Seluruh enam bulan test punya median per-film lebih tinggi daripada tiap bulan train kecuali mungkin dibanding September khusus (September train 64,3% vs Desember test 66,7%).

Kesimpulan: kenaikan retensi pasangan lemah tersebar lintas banyak judul, bukan akibat beberapa film dominan. Namun distribusi genre/film dan D1 train yang direkonstruksi tetap dapat berkontribusi; hasil ini tidak membuktikan perubahan kebijakan secara kausal. Implikasi validasi tinggi: bootstrap per judul dasar, segmen permintaan lemah, dan diagnostik test_history wajib berdampingan dengan CV train. Tidak ada model dilatih.
