# EDA 081: pergeseran retensi pada klaster yang sama

Pertanyaan: apakah selisih kelanjutan D2 ke D3 hanya akibat komposisi klaster bioskop yang berubah?

Metode: gunakan pasangan aktif D2 dari rekonstruksi D1 v2 dan test_history; pasangkan `cinema_ids` lintas periode. Ukur bucket D2 tiket/show <=8 per klaster, uji tanda antar klaster yang masing-masing punya >=5 observasi, lalu standardisasi langsung di strata klaster x bucket tiket/show.

Hasil: 117 klaster muncul di kedua periode, mencakup 100% pasangan D2 train dan 99,0% test. Di bucket tiket/show D2 <=8, 81 dari 86 klaster dengan >=5 contoh per periode memiliki retensi D3 lebih tinggi di test, 4 lebih rendah, 1 sama (uji tanda satu sisi p=5,5e-20). Median kenaikan per klaster 32,8 poin. Setelah standardisasi per klaster x bucket tiket/show, gap test minus train +15,35 poin pada 96,96% pasangan D2 test di klaster bersama; bootstrap per film CI 95% +11,34 sampai +19,34 poin. Dengan minimal 5 contoh per sel, gap +15,11 poin dan cakupan 92,33%; minimal 10 memberi +12,52 poin dan cakupan 69,71%.

Kesimpulan: perubahan campuran lokasi tidak menjelaskan perubahan retensi. Penyebabnya tetap belum teridentifikasi; tanggal D1 train rekonstruksi dan label D4-D10 test tersembunyi. Implikasi validasi tinggi: lakukan diagnostik transisi D1-D3 test_history secara terpisah dari CV train dan jangan berharap group CV menangkap pergeseran operasional.
