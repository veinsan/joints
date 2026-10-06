# Handoff: JOINTS x INSPIRE UGM 2026 (prediksi tiket bioskop D4-D10)

Folder ini merangkum **seluruh** percakapan dan pekerjaan (2-7 Oktober 2026). Tujuannya agar chat baru bisa
langsung melanjutkan tanpa membaca ulang riwayat. Notebook dan hasil run tidak disalin ke sini; lokasinya
dijelaskan di [`03_data_dan_pipeline/peta_repository.md`](03_data_dan_pipeline/peta_repository.md).

## TL;DR status per 7 Oktober 2026

- **Skor public terbaik: v8 = 0,39991.** Notebook: `notebooks/v8.ipynb`.
  - Memakai TabPFN-3.5 dengan checkpoint 876 MB, melewati batas bobot 200 MB, sehingga kemungkinan **tidak
    patuh**.
- **Versi patuh 200 MB terbaru: v12, public 0,41060.** Komponennya LightGBM 0,5, TabPFN-2.5 Quantiles 0,4,
  dan TabM 0,1. Bobot 72 MB.
  - Validasi v12 setara v8: TW-MASE 0,3257 vs 0,3251.
  - Validasi v12 lebih baik dari v11 (0,3319), dan perbedaan ini kokoh pada bootstrap per film.
- **Top 1 public: 0,34456.** Selisihnya dengan kita sekitar 0,055.
  - Tidak ada tuas sah yang kita temukan sebesar itu. Metode mereka tidak diketahui.
- Tren public **memburuk** sejak v8: v10 0,40823, lalu v12 0,41060. Validasi lokal datar di sekitar 0,325.
  - Hipotesis utama: TabPFN-3.5 (876 MB) generalisasi ke periode uji lebih baik daripada pengganti yang muat
    200 MB. Lihat `04_temuan/temuan_utama.md` bagian 9.
- **Pertanyaan ke panitia yang belum terjawab, dan menentukan versi final:**
  1. Apakah batas tanggal 30 Sep 2025 berlaku untuk checkpoint pretrained?
  2. Apakah checkpoint yang diunduh saat runtime ikut dihitung ke batas 200 MB?
  3. Apa cakupan larangan "AI Agent"?
- **Batas waktu:**
  - Kaggle (penyisihan): 12 Oktober 2026.
  - Pengumpulan notebook dan bobot (ZIP): 13 Oktober 2026, 23:59 WIB.

## Isi folder

| Folder | Isi |
| :--- | :--- |
| [`01_kompetisi_dan_aturan/`](01_kompetisi_dan_aturan/) | Aturan lomba, metrik, batasan, dan keputusan atau preferensi user yang wajib dipatuhi |
| [`02_progres/`](02_progres/) | Timeline v1-v12 (perubahan, validasi, skor public) dan status saat ini |
| [`03_data_dan_pipeline/`](03_data_dan_pipeline/) | Rekonstruksi data latih, fitur, model, validasi, dan peta repository |
| [`04_temuan/`](04_temuan/) | Temuan EDA terpenting dan indeks 60 script EDA beserta hasilnya |
| [`05_eksperimen/`](05_eksperimen/) | Semua model, preprocessing, fitur, dan data eksternal yang dicoba, dengan angka dan keputusan |
| [`06_pelajaran/`](06_pelajaran/) | Apa yang salah, apa yang tidak efektif, apa yang berhasil |
| [`07_langkah_berikutnya/`](07_langkah_berikutnya/) | Sisa pekerjaan, rekomendasi, dan checklist submisi final |

## Cara melanjutkan di chat baru

1. Baca README ini, lalu `02_progres/status_sekarang.md` dan `07_langkah_berikutnya/sisa_pekerjaan.md`.
2. Patuhi `01_kompetisi_dan_aturan/keputusan_dan_preferensi_user.md`. Ringkasnya:
   - tidak boleh probing leaderboard;
   - model tidak dijalankan di laptop;
   - notebook dibuat dengan skill `ref/notebook/`.
3. Notebook berikutnya dibuat sebagai `notebooks/build_v13.py`, yang menghasilkan `notebooks/v13.ipynb`, disalin
   dari `notebooks/build_v12.py`. **Jangan pernah menimpa notebook yang sudah di-run user.**
4. Detail angka per script ada di `docs/eda_findings.md`. Dua audit eksternal (dari "Astra", asisten lain yang
   dipakai user untuk review) ada di `docs/experiment_audit_2026-10-05.md` dan
   `docs/deep_analysis_2026-10-06.md`.
