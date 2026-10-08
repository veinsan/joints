# EDA 097: kohort rilis 18 Maret 2026

Pertanyaan: berapa besar kohort enam film rilis 18 Maret tepat sebelum Idulfitri, dan apakah profilnya sama dengan film Maret lain?

Metode: builder test_windows pada test_history, D1 per judul dasar. Bandingkan pasangan D1 18 Maret vs D1 Maret lain dan Feb-Mar lain. Hitung tanggal D4-D10 dan gabungkan holidays.csv serta cuti bersama resmi dari EDA083. Semua D4-D10 adalah tanggal target, bukan nilai target. Tidak fitting model.

Hasil: kohort enam film 631 pasangan (6.08% seluruh test; 49.72% pasangan D1 Maret), 4417 baris target. Median MASE scale156.67 vs22 untuk sembilan film D1 Maret lain; porsi scale<=20 3.96% vs46.71%. Median tiket/show D3 22.6 vs5.8; porsi<=8 6.97% vs66.77%. Data 21-22 Maret ditandai libur Idulfitri, tetapi 23-24 Maret cuti bersama resmi ditandai normal pada holidays.csv. Kohort punya 1262 baris target pada dua cuti tersebut: 52.76% seluruh baris target di cuti resmi test dan77.71% baris target pada cuti Maret.

Interpretasi: evaluasi Maret keseluruhan mencampur kohort Lebaran besar dan film Maret lain yang jauh lebih kecil/lemah. Untuk validasi dan diagnosis nanti, laporkan kohort D1 18 Maret tersendiri; fitur kalender cuti bersama tersedia dari SKB 19 September 2025 sebelum cutoff. Tidak mengklaim target D4-D10 atau efek kausal libur, karena nilai target test tidak diketahui.
