# Sisa pekerjaan (urut prioritas)

## 1. Tanyakan panitia (paling menentukan versi final)

Kirim lewat Slido atau grup WhatsApp, dalam satu pesan:
1. Apakah batas "versi tersedia publik paling lambat 30 Sep 2025" berlaku juga untuk **checkpoint pretrained**
   (TabPFN-2.5/3.5, EXAONE-Tabular, Causilo), atau hanya untuk data eksternal?
2. Apakah checkpoint pretrained yang **diunduh saat runtime** dari HuggingFace ikut dihitung ke batas bobot
   200 MB, atau hanya berkas bobot yang dikumpulkan di ZIP?
3. Apa cakupan larangan "AI Agent dalam kompetisi"? Apakah termasuk asisten AI untuk menulis kode atau analisis?

Implikasinya:

| Jawaban | Pilihan |
| :--- | :--- |
| Checkpoint **tidak** terkena cutoff dan unduhan runtime **tidak** dihitung 200 MB | v8 (public 0,39991, TabPFN-3.5) bisa jadi final |
| Checkpoint tidak terkena cutoff, tetapi 200 MB dihitung | v12 (0,41060) atau versi baru dengan model ≤ 200 MB |
| Checkpoint terkena cutoff | Hanya LightGBM (v11) atau TabPFN v2 revisi Juni 2025 (`FM_MODE='tabpfn_v2'` di builder v12) |

## 2. Pilih 2 submisi final di Kaggle (sebelum 12 Okt 2026)

- Satu "terbaik bila aturan longgar": v8.
- Satu "aman": v12, atau varian LightGBM / TabPFN v2.
- Ingat bobot penilaian: 90% private.

## 3. Paket ZIP final (13 Okt 2026, 23:59 WIB)

Lihat `checklist_submisi_final.md`.

## 4. Eksperimen yang masih terbuka (bila waktu dan submisi tersisa)

- **Causilo belum dimentokin** (v11: TW 0,3954, janggal). Langkah diagnosis yang ditawarkan ke user dan
  belum dijawab:
  - bandingkan kuantil kita dengan `predict(output_type='median'/'mean')` bawaan;
  - cek porsi nol tersirat vs aktual (0,30);
  - bandingkan λ 0 vs 0,5;
  - konteks gabungan semua horizon;
  - `n_estimators`.
  - Ukuran 148,4 MB masih muat bersama LightGBM, tetapi tidak muat bersama TabPFN-2.5 + TabM sekaligus. Hitung
    totalnya.
- **TabM masih satu konfigurasi:** k = 32, 2 blok x 512, 48 bin, lr 2e-3. Belum dituning, dan belum dicoba
  sebagai pengganti hurdle LightGBM. Korelasi galatnya 0,984, termasuk yang paling beragam.
- **Kalender input (A3)** selalu membaik kecil dan konsisten di temporal (-0,003). Bisa diadopsi bila ambang TW
  dilonggarkan dengan alasan yang jelas, tetapi itu mengubah aturan setelah melihat hasil, jadi harus
  dijelaskan jujur.
- **Biaya kepatuhan 200 MB:** cari model ≤ 200 MB yang meniru TabPFN-3.5 di periode uji.
  - TabPFN-3.5-fast (334 MB) dan TabPFN-3 (233 MB) tetap melewati batas.
  - Kemungkinan: TabPFN-2.5 varian lain (`default`, `real`, `low-skew`), atau TabPFN-2.6 dengan λ = 0. Di v9
    TabPFN-2.6 dinilai dengan λ = 0,5 dan bobotnya 0.
- **Uji isolasi penurunan public v8 ke v10/v12.** Kalender Jabar dan statistik fold-local ikut berubah bersama
  model. Satu-satunya cara tanpa probing adalah membandingkan versi yang memang dibuat untuk dipakai, jadi
  jangan bikin submisi khusus.
