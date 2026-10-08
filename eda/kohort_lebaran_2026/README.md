# Kohort rilis Lebaran 18 Maret berbeda dari film Maret lain

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/kohort_lebaran_2026/cohort.py` dari root. Script memakai `test_history.csv`, `holidays.csv`, dan [audit cuti bersama](../cuti_bersama_tidak_terkode/README.md). Tanggal cuti 2026 berasal dari [SKB yang ditetapkan 19 September 2025](https://jdih.kemnaker.go.id/peraturan/detail/2723/keputusan-bersama-menteri-agama-menteri-ketenagakerjaan-dan-menteri-pendayagunaan-aparatur-negara-dan-reformasi-birokrasi-republik-indonesia-nomor-2-tahun-2025), sebelum cutoff data eksternal.

| Input D1-D3 | Enam film D1 18 Maret | Sembilan film D1 Maret lain |
|---|---:|---:|
| Pasangan | 631 | 638 |
| Median skala MASE | 156,67 | 22,00 |
| Skala <=20 | 3,96% | 46,71% |
| Median tiket/show D3 | 22,6 | 5,8 |
| Tiket/show D3 <=8 | 6,97% | 66,77% |
| Median okupansi D3 | 13,39% | 3,45% |

Enam film tersebut ialah Danur: The Last Chapter, Tunggu Aku Sukses Nanti, Senin Harga Naik, Suzzanna: Santet Dosa di Atas Dosa, Na Willa, dan Pelangi di Mars. Kohort mencakup 6,08% seluruh pasangan test dan 49,72% pasangan dengan D1 pada Maret. Target D4-D10 mereka jatuh pada 21–27 Maret, sebanyak 4.417 baris. Dua tanggal pertama adalah libur Idulfitri yang sudah ditandai di `holidays.csv`; tanggal 23–24 Maret adalah cuti bersama resmi tetapi berlabel `normal`. Dua hari cuti itu memuat 1.262 baris target kohort, yaitu 52,76% dari seluruh baris target test pada tanggal cuti bersama dan 77,71% dari baris target cuti Maret.

Implikasi: diagnosis validasi Maret sebaiknya memisahkan kohort rilis 18 Maret dari film lain, karena skala MASE dan kekuatan D3 berbeda tajam. Kategori cuti bersama berpotensi berguna sebagai fitur kalender yang diketahui sebelum cutoff. Nilai target D4-D10 test tersembunyi; EDA ini tidak mengestimasi respons selama libur atau skor. Output ada di `output/cohort_summary.csv`, `output/cohort_by_film.csv`, dan `output/target_dates.csv`.
