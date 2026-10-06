# Keputusan dan preferensi user (WAJIB dipatuhi)

## Cara kerja

1. **Tidak boleh probing leaderboard.**
   - Kutipan: "Gua ga ingin lu melakukan probing terhadap leaderboard. Gua ingin setiap submission benar
     benar dipakai dengan baik yang dipakai untuk hasil run dari setiap version notebook".
   - Setiap submisi = hasil run satu versi notebook, bukan alat untuk mengekstrak label.
   - Keputusan diambil dari **val MASE**: "cek selalu val MASE nya agar tidak jomplang antara public dan
     private".
   - Memakai skor LB nyata dari versi yang sudah disubmit untuk inferensi (misalnya `eda/39`) diperbolehkan.
     Membuat submisi khusus untuk menguji level segmen tidak diperbolehkan.
2. **Model tidak dijalankan di laptop.**
   - Laptop user: Intel Core Ultra 7 155H, Intel Arc iGPU, sering dipakai saat travel dengan baterai.
   - Yang boleh lokal hanya EDA dan preprocessing (detik sampai menit).
   - Evaluasi model (TabPFN, EXAONE, LightGBM grid panjang, fine-tune, dry run penuh) dilakukan di notebook
     Kaggle T4 x2, sebagai komponen dengan bobot otomatis.
   - Pernah dilanggar (run EXAONE dan TabPFN CPU berjam-jam), lalu user menegur.
   - Smoke test beberapa detik pada data kecil masih diterima.
3. **Notebook dibuat dengan skill `ref/notebook/`.** Formatnya builder script (`md()`, `code()`,
   `section()`) yang menghasilkan `.ipynb`. Aturan pentingnya:
   - 3 cell banner;
   - import hanya di cell Libraries;
   - tanpa docstring di cell kode;
   - tanpa komentar pemisah `# --- x ---`;
   - setiap subsection diawali paragraf penghubung;
   - setiap analisis diakhiri blok `#### Insights` dengan `>`;
   - narasi notebook berbahasa **Indonesia**.
4. **Notebook di-run user di Kaggle (T4 x2), bukan lokal.** User kemudian menaruh hasil di `results/vN/` dan
   notebook dengan output di `notebooks/vN.ipynb`.
5. **Setiap putaran user meminta:**
   - "DEEP ANALYSIS sampai mentok";
   - satu file `.py` per analisis yang mengeluarkan angka dan gambar;
   - analisis outputnya;
   - cek bahwa preprocessing benar secara teori **dan** tidak rusak di evaluasi;
   - penanganan duplikasi dan leakage;
   - pencarian model SOTA (HuggingFace) dan data eksternal yang tidak menambah noise;
   - notebook baru siap T4 x2.
6. **Setiap versi selesai, user meminta keputusan "submit atau tidak"** berdasarkan `results/vN/` dan
   `notebooks/vN.ipynb`.
7. **Jangan menimpa notebook yang sudah di-run user.**
   - Builder baru = versi baru (`build_v13.py` menghasilkan `v13.ipynb`).
   - Saat user meminta "tetap namanya v11", builder v11 ditimpa hanya karena v11 belum di-run.

## Keputusan aturan

8. **Batas tanggal 30 Sep 2025 berlaku untuk data eksternal, bukan pretrained model.**
   - Ini tafsiran user berdasarkan broadcast WhatsApp panitia: "Mekanisme Penggunaan Data: Data eksternal
     hanya boleh digunakan jika informasi atau versinya telah tersedia untuk publik PALING LAMBAT 30
     September 2025...".
   - Tafsiran ini **bukan konfirmasi panitia**. Astra mengingatkan agar tidak dianggap izin resmi.
9. **Batas bobot 200 MB berlaku.**
   - User sendiri yang mengangkatnya: "TabPFN-3.5 dan 3 itu diatas 200MB ga sih?".
   - Sejak v9 semua versi memakai checkpoint di bawah 200 MB, dan checkpoint ditanam di berkas bobot.
10. **HF token** tersedia di Kaggle Secrets dengan nama `HF_TOKEN`. Repo Prior Labs gated memerlukannya.

## Preferensi lain

- User sering meneruskan review dari **"Astra"** (asisten AI lain). Polanya: Astra mengkritik, user meminta
  revisi notebook sesuai kritik itu. Kritik Astra sejauh ini hampir selalu valid. Cek ulang klaimnya di kode
  sebelum menerapkan.
- User menyukai jawaban yang jujur dan berbasis angka, termasuk saat hasil jelek.
- Gunakan ringkasan singkat bila diminta ("jawab singkat").
