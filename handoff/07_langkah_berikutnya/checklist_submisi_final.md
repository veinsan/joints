# Checklist submisi final

## Kaggle (sebelum 12 Okt 2026)

- [ ] Pilih 2 submisi final berdasarkan jawaban panitia (lihat `sisa_pekerjaan.md`).
- [ ] Pastikan versi yang dipilih punya notebook dengan output lengkap di `notebooks/vN.ipynb` dan
      `results/vN/`.

## ZIP `[Nama Tim].zip` (13 Okt 2026, 23:59 WIB, dikirim ketua tim lewat Google Form)

- [ ] `[Nama Tim].ipynb`: salin notebook versi final, lalu ganti placeholder `[Nama Tim]` dan
      `[Nama Anggota 1..3]` di cell banner kedua.
- [ ] `[Nama Tim].pkl`: salin `results/vN/model_weights.pkl`, **≤ 200 MB**.
  - v12: 72 MB, dengan TabPFN-2.5 dan TabM tertanam.
  - v11: 25 MB.
  - v9: 109,6 MB.
  - v8: 63 MB **tanpa** TabPFN-3.5. Checkpoint 876 MB-nya diunduh saat runtime, dan tidak bisa masuk ZIP.
- [ ] Jangan sertakan folder checkpoint (`fm/`, `tabpfn/`). Checkpoint yang dipakai sudah ada di pkl untuk
      v9-v12.
- [ ] Notebook harus bisa di-re-run panitia:
  - cell pertama `%pip install` dengan versi terkunci dan `-q`;
  - seed tetap;
  - checkpoint dicari di input Kaggle atau di dalam pkl sebelum HF;
  - `HF_TOKEN` hanya diperlukan bila checkpoint tidak tersedia lokal.
- [ ] Narasi notebook lengkap (10% nilai): sumber data eksternal dengan tanggal, sumber dan metode pretrained,
      dan Insights tiap bagian.
- [ ] Bila panitia meminta: sertakan URL, penerbit, dan tanggal sumber eksternal (tabel di Section 7 notebook
      dan `05_eksperimen/data_eksternal.md`).

## Uji reproduksi (finalis Top 10)

- [ ] Re-run notebook final di Kaggle T4 x2, lalu bandingkan `submission.csv` dengan yang disubmit. Harus sama
      persis, atau sangat dekat bila ada nondeterminisme GPU.
- [ ] Untuk babak final (presentasi): materi diambil dari `handoff/` dan `docs/eda_findings.md`.
