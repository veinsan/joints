# EDA 106: anomali kapasitas 1-5 Juni lolos dari audit outage lama

## Temuan utama

`train.csv` mencatat **6.272 show pada 31 Mei 2025**, lalu hanya **2.044 pada 1 Juni**. Penurunan itu terjadi serempak: 112 dari 114 ID yang aktif pada kedua hari memiliki total show 1 Juni **paling banyak setengah** dari 31 Mei (median rasio ID 0,321). Jumlah baris hanya turun 728→598 dan ID 116→114, jadi audit outage lama yang mencari klaster/baris hilang tidak menangkap sebagian besar kejadian. Tiket turun 311.321→87.403. Total show tetap 1.941-2.044 pada 1-5 Juni, naik ke 3.965 pada 6 Juni, lalu periode 7-16 Juni yang sudah dikenal mengalami kehilangan baris/ID tidak teratur. Pada 17 Juni total show kembali 6.020 dengan 116 ID.

Pada 586 pasangan film-ID yang ada di 31 Mei **dan** 1 Juni, rasio jumlah show gabungan 0,349 dan rasio tiket 0,297; median rasio show per baris 0,355. Ini bukan hasil sekadar hilangnya judul lama. Sebagai kontrol Sabtu→Minggu, total show 24→25 Mei berubah 5.983→5.974 dan 21→22 Juni 6.422→6.369; median rasio ID keduanya 1,0 dan tidak ada ID yang turun ke <=0,5. Penurunan 31 Mei→1 Juni tidak tampak sebagai pola akhir pekan biasa.

Setelah **1-16 Juni dikeluarkan hanya untuk audit stabilitas**, 116 ID dengan >=150 hari teramati mempunyai median koefisien variasi total show harian 0,0576; median per ID dari proporsi perubahan harian <=2 show adalah 96,36%. Struktur jadwal secara umum stabil. Perubahan pembukaan [Basko Padang](../padang_basko_dan_skala_id/README.md) dan penutupan [Pluit Junction](../penutupan_pluit_junction/README.md) tetap muncul di luar periode anomalous.

## Dampak pada window train v2

Builder `temp/exp_020_data_fix/build_windows_v2.py` hanya membuang window yang menyentuh **7-16 Juni**. Tiga film dengan D1 28 Mei masih masuk: DENDAM MALAM KELAM (56 pasangan), KARATE KID: LEGENDS (108), dan WAKTU MAGHRIB 2 (103). Sebanyak **267/7.988 pasangan (3,34%)** dan **1.335/55.916 baris target (2,39%)** berada pada tanggal 1-5 Juni; 930 dari 1.335 baris itu masih bertiket positif. Input D1-D3 ketiga film berada pada 28-30 Mei, sebelum anomali, sedangkan target D5-D9 melintasinya. Pada masing-masing film, jumlah show dari 31 Mei ke 1 Juni turun tajam, misalnya KARATE KID 999→335 dan WAKTU MAGHRIB 2 1.056→327. Window tersebut bisa membuat penurunan pasca-D3 tampak lebih kuat daripada jadwal normal.

**Rekomendasi riset:** perlakukan 1-16 Juni sebagai rentang kualitas data yang perlu dikeluarkan atau diuji sensitivitasnya pada train/validasi mendatang. Jika seluruh window yang menyentuh 1-16 Juni dibuang dari builder v2 saat ini, tambahan yang terdampak ialah tiga judul/267 pasangan di atas; tidak ada perubahan builder atau eksperimen modelling dalam EDA ini. Jangan memberi label pasti bahwa transaksi hilang: tidak ada sumber publik yang menjelaskan penutupan bioskop serentak saat itu, dan perubahan jadwal nyata tetap mungkin. Namun keserempakan lintas 112 ID dan kontrol pekan sekitar membuat interpretasi permintaan biasa lemah. [Laporan XXI per 30 Juni](https://www.cinema21.co.id/id/newsroom/kinerja-stabil-cinema-xxi-bukukan-pendapatan-rp28-triliun-di-semester-i-2025) masih mencatat 259 bioskop/1.360 layar; angka akhir bulan itu tidak membuktikan operasi harian 1-5 Juni.

Jalankan dari root:

```powershell
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/stabilitas_show_dan_anomali_juni/stability.py
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/stabilitas_show_dan_anomali_juni/transition.py
E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/stabilitas_show_dan_anomali_juni/window_impact.py
```

Output ada di `output/daily_network.csv`, `id_profiles.csv`, `transition_summary.json`, `window_impact.json`, `affected_titles.csv`, dan tabel transisi rinci. Tidak ada target test, leaderboard, atau fitting model yang dipakai.
