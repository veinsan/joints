# Efek kalender pada permintaan nasional

**Satu kalimat**: relatif rata-rata minggu, Sabtu 1,38 dan Minggu 1,27 sedangkan Selasa 0,77; libur di hari kerja menaikkan permintaan 1,6 sampai 1,9 kali (Maulid, Jumat, 2,2) tetapi libur di hari Minggu tidak (Paskah 0,88), dan minggu libur sekolah (23 Jun sampai 13 Jul) membuat hari kerja relatif lebih ramai.

## Pertanyaan

Seberapa besar efek hari dan libur, dan kalender apa yang perlu jadi fitur tanggal target?

## Metode

`python eda/efek_kalender/kalender.py`: log total tiket nasional harian dikurangi rata-rata minggunya, efek hari dari hari normal, residual pada hari libur, deviasi hari kerja per minggu (proxy libur sekolah), total tiket mingguan.

## Hasil

- Multiplier hari normal: Sen 0,85, Sel 0,77, Rab 0,91, Kam 0,99, Jum 0,99, Sab 1,38, Min 1,27.
- Libur: Hari Buruh (Kam) 1,73, Waisak (Sen) 1,61, Kenaikan (Kam) 1,91, Maulid (Jum) 2,19, 17 Agustus (Min) 1,33, Paskah (Min) 0,88. Idul Adha (Jum) 4,0 dan Hari Pancasila (Min) 0,32 berada di sekitar outage Juni, jangan dipercaya.
- Deviasi hari kerja Senin sampai Kamis per minggu: normal -0,15 sampai -0,23; 23 Jun sampai 13 Jul -0,11 sampai -0,13 (libur sekolah).
- Total tiket mingguan: April 3,6 sampai 4,3 juta (pasca Lebaran), Mei sampai Sep 1,0 sampai 2,3 juta.
- holidays.csv tidak memuat cuti bersama maupun libur sekolah.
- Plot: `output/kalender.png`.

## Interpretasi

Tanggal target yang jatuh pada libur hari kerja berperilaku seperti weekend. Libur sekolah Desember 2025 dan pekan Lebaran Maret 2026 ada di test tetapi tidak tercatat di holidays.csv, kecuali tanggal merahnya.

## Saran FE / modelling

| Saran | Alasan | Prioritas |
|---|---|---|
| Fitur tanggal target: hari, weekend, libur, libur di hari kerja, jumlah libur di D1-D3 dan D4-D10 | Multiplier 1,6 sampai 1,9 | tinggi |
| Rasio bobot kalender tanggal target terhadap rata-rata bobot kalender D1-D3 (multiplier hari/libur) | scale dihitung dari D1-D3 yang komposisi harinya berbeda antar film | tinggi |
| Kalender libur sekolah dan cuti bersama (data eksternal) | Desember dan Lebaran di test | sedang (butuh keputusan user) |

## Tingkat keyakinan dan status

- Keyakinan: tinggi untuk efek hari; sedang untuk libur (satu kejadian per libur).
- Sudah divalidasi dengan eksperimen: belum.
