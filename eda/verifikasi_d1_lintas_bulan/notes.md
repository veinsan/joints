# EDA 090: verifikasi D1 eksternal yang lebih luas

Pertanyaan: apakah lima tanggal rilis publik yang cocok dengan D1 v2 cukup representatif? Uji film tambahan lintas bulan dan panjang preview.

Metode: buat daftar judul dasar dan D1 v2 dari EDA 076, cari tanggal rilis luas yang dapat dikutip dari sumber publik independen. Catat semua kandidat yang diperiksa, termasuk yang ambigu/tidak ditemukan. Tanpa modelling.

Hasil: 14 judul tambahan pada enam bulan, 12 dengan semua klaim tanggal eksternal cocok dengan D1. Dua konflik sumber dicatat. Pada Tabayyun, kolom terstruktur LSF 30 April, tetapi narasi halaman sama dan Indonesian Film Center 8 Mei. Data transaksi memperlihatkan 30 April 12 show pada 2 klaster, 8 Mei 600 show pada 76 klaster. Artikel tentang gala premiere menyebut acara 30 April dan rilis luas 8 Mei. Pada Jodoh 3 Bujang, LSF bahasa Indonesia menulis 26 Juni 2025 sedangkan terjemahan Inggrisnya 2023 meski tahun produksi dan STLS 2025. Kedua kasus mendukung pemeriksaan silang alih-alih mengambil satu field mentah.

Klaim sumber dan URL tersimpan di verified_sources.csv. Audit dapat direproduksi dengan candidates.py, audit_sources.py, dan tabayyun_timeline.py. Pencarian sumber tidak acak, sehingga tingkat cocok bukan estimasi akurasi semua judul. Ini penelitian retrospektif atas definisi D1, bukan fitur eksternal siap pakai untuk inference.
