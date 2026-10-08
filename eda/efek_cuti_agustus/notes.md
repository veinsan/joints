# EDA 084: cuti bersama 18 Agustus 2025

Pertanyaan: adakah sinyal permintaan pada cuti bersama yang tidak diberi label khusus di `holidays.csv`?

Metode: bandingkan total tiket/show 18 Agustus 2025 dengan Senin sebelum dan sesudah. Periksa pula pasangan film x klaster yang muncul pada ketiga Senin; rasio hari cuti dibanding rata-rata geometrik dua Senin tetangga. Ini perbandingan deskriptif, bukan estimasi kausal atau eksperimen modelling.

Hasil: 18 Agustus mencatat 397.281 tiket vs 162.697 (11 Agustus) dan 148.197 (25 Agustus), rasio ke rata-rata geometrik 2,56; show harian justru 0,948 kali rata-rata tetangga. Pada 209 pasangan film x klaster yang hadir ketiga Senin, rasio median tiket 2,51, tiket/show 3,03, show 0,82. Sepuluh dari 11 judul dasar punya median tiket/show >1; enam judul dengan >=10 pasangan semuanya >1. Pada panel enam hari Senin/Selasa tiga minggu (189 pasangan), rasio tiket/show Senin/Selasa di minggu cuti relatif terhadap dua minggu tetangga median 2,28.

Kesimpulan: cuti 18 Agustus 2025 tampak sebagai lonjakan intensitas penonton, bukan tambahan jumlah show. Ini mendukung pemeriksaan indikator cuti terpisah dari `holiday_tipe`, tetapi hanya satu peristiwa dengan konteks HUT RI dan pemilihan pasangan aktif; tidak mengidentifikasi efek kausal atau menjamin transfer ke Maret 2026. Tidak ada model dilatih.
