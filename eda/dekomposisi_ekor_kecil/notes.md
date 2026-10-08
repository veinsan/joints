# EDA 085: mengapa ekor pasangan kecil bertambah?

Pertanyaan: apakah kenaikan pasangan berpermintaan lemah pada test D3 lebih banyak berasal dari lebih banyak kasus lemah di D2, atau dari retensi D2->D3 yang berubah?

Metode: pada pasangan aktif D2, bagi menurut tiket/show D2 lalu hitung peluang D3 aktif. Hitung campuran pasangan terpilih jika distribusi D2 test digabung dengan tingkat retensi train per bucket. Hubungkan dengan skala MASE <=20. Counterfactual ini aritmetika deskriptif, bukan model terlatih.

Hasil: di antara pasangan D2 aktif, bucket tiket/show <=8 naik 10,90% train menjadi 28,64% test. Retensi bucket itu 46,11% train vs 76,86% test. Di antara pasangan D2+D3 aktif yang terpilih, porsi bucket lemah naik 5,67% menjadi 24,05%. Menggunakan campuran D2 test tetapi peluang retensi train per bucket menghasilkan 17,13% terpilih lemah: 11,46 poin dari selisih terpilih disebabkan perubahan campuran D2 dalam urutan dekomposisi ini, dan 6,92 poin tambahan terkait perubahan retensi. Semua pasangan terpilih D3: porsi D2 lemah 5,56% vs 23,50%; porsi skala MASE <=20 3,20% vs 12,96%; di dalam skala kecil, 47,66% vs 73,29% juga punya D2 lemah.

Kesimpulan: ekor kecil test lebih banyak karena kedua hal, bukan hanya level permintaan yang turun. Angka kontribusi tergantung urutan counterfactual dan bucket, tidak kausal. Implikasi validasi: pelaporan menurut skala dan tiket/show D2, serta diagnostik retensi D3 test_history, perlu berdampingan. Tidak ada model dilatih.
