# `cinema_ids` adalah agregat operasional berskala sangat berbeda

**Satu kalimat:** satu `cinema_ids` mencatat sampai 540 pertunjukan semua film per hari dan satu film mencatat sampai 285 pertunjukan per hari pada satu ID, sehingga ID ini tidak aman ditafsirkan sebagai satu gedung bioskop fisik.

## Pertanyaan

Apa unit sebenarnya di balik `cinema_ids`, dan apakah cakupan dataset cocok persis dengan jaringan publik yang diduga?

## Metode

Jalankan `python eda/skala_klaster_bioskop/footprint.py` dari root. Hitung jumlah ID, kota, pertunjukan, dan tiket per bulan serta maksimum per film x ID x hari dan per ID x hari. Bandingkan besaran September 2025 dengan [laporan Cinema XXI hingga Q3 2025](https://www.cinema21.co.id/id/newsroom/cinema-xxi-catat-pendapatan-rp43-triliun-hingga-kuartal-iii-2025-dan-umumkan-pembagian-dividen-interim), yang menyebut 261 bioskop, 1.369 layar, dan **67** kota/kabupaten per 30 September pada bagian utama. Versi Inggris memuat angka 61 yang usang di boilerplate `About`.

Sebagai tolok ukur operasional, [Eliashberg dkk. (2009)](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Demand-driven-scheduling-of-movies-in-a-multiplex.pdf) menjelaskan sekitar 3-5 show per layar per hari pada multiplex yang mereka teliti; ukuran layar, kontrak distributor, dan permintaan menentukan jadwal. Pada 5 show per layar, 540 show/hari membutuhkan sekitar 108 layar bila seluruhnya berasal dari satu lokasi. Ini pembanding lintas negara, bukan aturan keras untuk Indonesia.

## Hasil

| Ukuran | Train | Test history |
|---|---:|---:|
| `cinema_ids` unik | 117 | 121 |
| `city_name` unik | 67 | 69 |
| ID yang muncul pada lebih dari satu kota | 0 | 0 |
| Show maksimum satu film x ID x hari | 285 | 278 |
| Show maksimum satu ID x hari, semua film | 540 | 477 |
| Baris satu film x ID x hari dengan >100 show | 274 | 54 |

Pada September 2025, train mencatat median 6.188 show per hari di 117 ID dan 67 kota. Sebagai pemeriksaan besaran saja, 6.188/1.369 layar publik sekitar 4,52 show per layar per hari. Jumlah kota sama dengan rilis resmi, dan [audit pembukaan](../jejak_pembukaan_xxi/README.md) menemukan kecocokan tanggal empat ID baru dengan pembukaan XXI. Namun 117 ID vs 261 lokasi resmi serta beberapa ID dengan ratusan show/hari tetap mencegah pemetaan `cinema_ids` langsung ke gedung XXI. Algoritme pembentukan unit tidak diketahui.

## Interpretasi dan saran

| Area | Saran | Prioritas |
|---|---|---|
| Fitur | Tafsirkan tiket dan show mentah bersama ukuran klaster; tiket/show dan rasio ke level historis ID/kota dapat membedakan permintaan dari kapasitas | tinggi |
| Validasi | Jangan menganggap satu ID sebagai satu bioskop independen saat menghitung ketidakpastian; bootstrap per film tetap perlu | tinggi |
| Provenance | Hindari menyamakan agregat internal dengan box office nasional atau angka operator tertentu | tinggi |

Status asal asli/sintetis dan operator tidak terverifikasi. Besaran operasional memperkuat interpretasi agregat, tetapi tidak mengungkap aturan agregasi tepat. Tidak ada eksperimen modelling.
