# Pergeseran D3 terjadi di semua tingkat harga dan kota besar sampel

Jalankan `E:\datsci\conda_envs\kaggle-py311-d\python.exe eda/harga_kota_waktu/price_audit.py` dari root. Script memakai [pasangan D3 EDA 094](../persaingan_film_baru/README.md) dan `ticket_prices.csv`; tidak memakai target D4-D10 atau fitting model.

Tabel harga berisi 69 kota x tiga kategori hari (`Weekday`, `Friday`, `Weekend`) tanpa duplikasi. Semua 7.988 pasangan train dan 10.373 pasangan test D3 mendapat nilai harga setelah join kota-kategori hari.

| Harga `ceil` | Weak D3 train (tiket/show <=8) | Weak D3 test |
|---|---:|---:|
| <=Rp45.000 | 6,73% | 23,81% |
| Rp45.001–50.000 | 6,44% | 21,11% |
| Rp50.001–55.000 | 4,50% | 18,42% |
| >Rp55.000 | 3,50% | 18,09% |

Dengan campuran kelompok harga test dan tingkat weak train, hasilnya **5,71%**, dibanding **21,32%** test. Pada 66 kota yang punya sedikitnya 30 pasangan di tiap periode, **66/66** menunjukkan tingkat weak lebih tinggi di test; kota tersebut mencakup 98,91% pasangan test. Ini menunjukkan pergeseran tidak hanya disebabkan perubahan komposisi kota/harga tabel.

`ticket_prices.csv` tidak menyimpan harga transaksi atau riwayat perubahan harga. Oleh karena itu, audit ini tidak dapat menguji perubahan harga aktual, promosi, dan perbedaan tarif format layar. Hubungan harga dan tiket/show di sini deskriptif; film dan musim tetap berbeda antarperiode. Output ada di `output/by_price_quartile.csv` dan `output/shared_cities.csv`.
