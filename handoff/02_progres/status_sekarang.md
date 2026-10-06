# Status saat ini (7 Oktober 2026)

## Sedang di mana

- v12 sudah di-run dan disubmit: **public 0,41060**, lebih buruk dari v8 (0,39991).
- User menduga public dan private akan "jomplang" dan private lebih baik. Itu belum bisa diverifikasi; skor
  private baru keluar setelah lomba.
- Belum ada notebook v13. Tugas terakhir adalah handoff ini dan pembersihan repository.

## Kandidat submisi final

Di Kaggle biasanya bisa memilih 2 submisi final untuk private.

| Kandidat | Public | Validasi TW (bfd) | Kepatuhan | Catatan |
| :--- | :--- | :--- | :--- | :--- |
| v8 | 0,39991 | 0,3251 | Bermasalah: TabPFN-3.5 876 MB > 200 MB; cutoff tanggal checkpoint juga tidak jelas | Tidak bisa dipaketkan dalam ZIP 200 MB kecuali panitia menyatakan checkpoint unduhan tidak dihitung |
| v12 | 0,41060 | 0,3257 (temporal 0,3019) | Bobot 72 MB, checkpoint tertanam; TabPFN-2.5 rilis Nov 2025 (cutoff tidak jelas) | Validasi paling seimbang di antara versi patuh |
| v9 | tidak diketahui | 0,3261 | 109,6 MB; EXAONE rilis Agu 2026 | Cek apakah pernah disubmit |
| v11 | tidak diketahui | 0,3319 | 25 MB, **hanya LightGBM: paling aman** dari sisi aturan pretrained | Validasi terburuk dari kandidat |

Untuk ZIP final yang aman dari semua tafsiran aturan: LightGBM saja (v11) atau TabPFN v2 revisi Juni 2025
(`FM_MODE = 'tabpfn_v2'` di builder v12). Keputusan tergantung jawaban panitia (lihat
`07_langkah_berikutnya/sisa_pekerjaan.md`).

## Yang sudah selesai

- Pipeline lengkap dan teruji: rekonstruksi, fitur, validasi multi-lensa, hurdle, foundation model, analog
  Lebaran, dan paket bobot ≤ 200 MB dengan SHA-256.
- 60 script EDA (`eda/01`-`eda/61`, tanpa 11) dengan angka dan gambar di `outputs/eda/`. Temuan tercatat di
  `docs/eda_findings.md`.
- Notebook v1-v12 beserta builder-nya.

## Yang belum selesai atau tertunda

- Konfirmasi panitia: cutoff checkpoint, 200 MB untuk unduhan runtime, dan cakupan "AI Agent".
- Causilo belum dieksplorasi tuntas: hanya 1 konfigurasi, hasil janggal (TW 0,3954, 52 detik/fold,
  korelasi galat 0,75). Saya menawarkan diagnosis keluaran, perbandingan λ 0 vs 0,5, dan konteks gabungan.
  User belum menjawab.
- Pemilihan 2 submisi final di Kaggle, dan ZIP final (nama tim, notebook, bobot ≤ 200 MB).
- Isi nama tim dan anggota di cell banner notebook: masih placeholder `[Nama Tim]`.
