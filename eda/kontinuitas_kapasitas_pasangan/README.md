# EDA 102: kontinuitas kapasitas tersirat D1-D3 pada pasangan film–klaster

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/kontinuitas_kapasitas_pasangan/capacity_continuity.py` dari root. Train window dibangun dengan `build_windows_v2.py`; test memakai D1-D3 `test_history.csv`. Tidak ada target test D4-D10 atau fitting model.

Pada D1 dan D3, filter tiket>=20, show>=2, okupansi5-<100%, dan kapasitas tersirat `100*tiket/(show*okupansi)` 50-400. Ini menyisakan 6.028/7.988 pasangan train (75,46%) dan 5.720/10.373 test (55,14%). Bandingkan median `abs(log(kapasitas D3 / kapasitas D1))` dengan 100 pengacakan nilai D3 dalam klaster yang sama, serta kontrol lebih ketat dalam klaster dan bucket jumlah show D3 yang sama.

| Periode | Teramati | Acak dalam klaster | Acak dalam klaster + bucket show D3 |
|---|---:|---:|---:|
| Train | 0,0679 | 0,1754 | 0,1480 (95%: 0,1421-0,1530) |
| Test history | 0,0786 | 0,1766 | 0,1532 (95%: 0,1483-0,1579) |

Sekitar 58,59% pasangan train dan 55,30% test pada subset ini punya perubahan kapasitas tersirat absolut <=10%. Ketika jumlah show D1-D3 relatif stabil (±10%), median perubahan kapasitas tersirat -0,25% train dan +0,53% test; saat show dipangkas >=50%, median -4,56% dan -3,89%. Ini konsisten dengan alokasi ruang tayang yang relatif berlanjut selama pembukaan, tetapi juga bisa berasal dari aturan agregasi/generasi yang mulus. [Audit aritmetika EDA100](../aritmetika_okupansi_kursi/README.md) menunjukkan rasio ini tidak dapat langsung dianggap kursi fisik integer.

Implikasi: perubahan relatif kapasitas tersirat pada D1-D3 dapat dipelajari sebagai sinyal diagnostik tentang penjadwalan, dengan flag kualitas okupansi dan filter nilai ekstrem. Hasil tidak mengidentifikasi mekanisme kausal maupun asli/sintetis. Output ada di `output/summary.json` dan `by_show_*.csv`.
