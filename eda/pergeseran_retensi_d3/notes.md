# EDA 072: kelanjutan D1/D2 ke D3

Pertanyaan: apakah aturan berhenti tayang yang terlihat di train sama dengan test_history?

Hasil: dari pasangan aktif D2, D3 aktif 88,61% pada train (8.835 pasangan) dan 91,55% pada test_history (11.074). Test punya median tiket per show D2 lebih rendah, 14,00 vs 27,45. Setelah standardisasi menurut bucket tiket per show D2, ukuran film, dan hari D1 pada 36 strata (cakupan 95,68% pasangan test aktif D2), D3 aktif 80,72% pada train vs 92,22% pada test, selisih 11,50 poin. Bootstrap 1.000 kali per judul dasar menghasilkan CI 95% 7,89 sampai 15,72 poin. Pada tiket per show D2 <= 8, tingkat berhenti 53,9% train vs 23,1% test. Pada D1 dengan tiket per show <= 8, D2 aktif 70,3% train vs 87,9% test. Pasangan test.csv tepat sama dengan pasangan test_history yang aktif D3.

Kesimpulan: ada pergeseran retensi/seleksi pasangan yang kuat, bukan semata perubahan campuran tiket awal. Tidak bisa dibuktikan dari data ini apakah sebabnya kebijakan program, sumber pengumpulan, atau sintesis. Kondisi D4-D10 test tidak berlabel, jadi besar pergeseran pada target belum diketahui. Validasi SGKF5 train perlu diagnostik terpisah pada transisi D1/D2 ke D3 test_history dan analisis komposisi skala.
