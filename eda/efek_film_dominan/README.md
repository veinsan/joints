# Decay ditentukan film (R2 0,57), bukan klaster (0,02)

**Satu kalimat**: fixed effect film menjelaskan 57% variansi log rata-rata y/scale D4-D10 sedangkan klaster hanya 2%, sehingga agregat nasional D1-D3 per film (tiket D3 nasional spearman -0,62 dengan jumlah hari nol) adalah fitur terkuat, dan baseline median per (h, desil tiket nasional D3) menurunkan MASE OOF dari 0,451 ke 0,415.

## Pertanyaan

Seberapa besar variasi decay dijelaskan film vs lokasi, dan apa artinya bagi fitur dan validasi?

## Metode

- `python eda/efek_film_dominan/film_klaster.py`: R2 fixed effect (film, klaster, kota, keduanya), korelasi spearman fitur pasangan vs agregat nasional, efek klaster leave-one-film-out, metadata film (genre, rating usia), jumlah rilis pesaing, stabilitas per bulan.
- `python eda/efek_film_dominan/baseline_cv.py`: baseline tanpa model dengan StratifiedGroupKFold(5), group = judul dasar (varian IMAX/3D satu grup), strata = bulan D1, seed 2026.

## Hasil

| Ukuran | Nilai |
|---|---|
| R2 FE film / klaster / kota / film+klaster | 0,572 / 0,020 / 0,013 / 0,591 |
| Spearman jumlah nol vs scale, nat_tix_d3, n_pairs, nat_show_trend | -0,66, -0,62, -0,51, -0,49 |
| Residual klaster LOO lintas film vs residual pasangan | 0,16 |
| Rata-rata hari nol horor vs non-horor | 2,6 vs 3,3 |
| Median y/scale per rating usia | Dewasa 21: 0,09, Semua Umur 0,24, Remaja 0,36, Dewasa 0,36 |
| Jumlah rilis pesaing di D4-D10 vs rasio film | spearman -0,02 (tidak berguna) |

Baseline OOF MASE (std antar fold sekitar 0,10):

| Baseline | OOF |
|---|---|
| prediksi 0 | 0,594 |
| scale x median(h) | 0,451 |
| scale x median(h, hari D1) | 0,436 |
| scale x median(h, desil nat_tix_d3) | **0,415** |
| scale x median(h, desil occ_rel) | 0,431 |

Fold ke-3 selalu terburuk (0,59 sampai 0,78): sedikit film dominan per fold, jadi skor CV sangat tergantung komposisi film.

## Interpretasi

Keputusan tayang dibuat per film secara nasional (jumlah layar dari performa awal), lokasi hanya memodulasi sedikit. Informasi pasangan lain dari film yang sama (agregat D1-D3 di semua klaster) sangat berharga dan sah dipakai di test karena semua D1-D3 tersedia di test_history.

## Saran FE / modelling

| Saran | Alasan | Prioritas |
|---|---|---|
| Agregat nasional film D1-D3: total tiket per hari, total show per hari, tren nasional, tiket per show nasional, jumlah klaster per hari, n_pairs | R2 film 0,57 | tinggi |
| Posisi pasangan relatif film: share tiket, rasio tren pasangan vs tren nasional, peringkat klaster dalam film | Memisahkan efek lokal dari efek film | tinggi |
| Gabungkan varian format ke judul dasar untuk agregat dan grouping CV | Varian IMAX/3D berbagi nasib film | sedang |
| Target encoding klaster lintas film (out-of-fold, residual setelah efek film) | Efek klaster kecil tapi nyata (0,16) | sedang |
| Metadata: horor, rating usia (Dewasa 21), produksi lokal vs impor | Efek sedang | rendah |
| CV: StratifiedGroupKFold group judul dasar, strata bulan D1; ulang beberapa seed dan bandingkan berpasangan per fold | Std antar fold 0,10 | tinggi |

## Tingkat keyakinan dan status

- Keyakinan: tinggi untuk dominasi film. Efek metadata rendah sampai sedang karena hanya 172 film.
- Sudah divalidasi dengan eksperimen: baseline EXP-011 (0,4149).
