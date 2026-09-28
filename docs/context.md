# Data Science Competition JOINTS X INSPIRE UGM 2026

Seberapa jauh tiga hari pertama penayangan sebuah film dapat memberi gambaran tentang minggu berikutnya?

## Overview

Selamat datang di Data Science Competition JOINTS 2026! Sebagai salah satu rangkaian acara JOINTS 2026 yang diselenggarakan oleh Departemen Ilmu Komputer dan Elektronika UGM, kompetisi ini bertujuan untuk memberikan pengalaman bagi peserta dalam menyelesaikan permasalahan berbasis data menggunakan pendekatan machine learning dan analisis data. Peserta akan ditantang untuk menguasai siklus data science secara end-to-end, mulai dari pemrosesan data, eksplorasi data (EDA), rekayasa fitur, hingga pemodelan dan evaluasi. Peserta juga diminta untuk berpikir secara analitis dalam mengekstrak wawasan berharga dari data serta menafsirkannya dengan tepat dan terstruktur.

### Tujuan Kompetisi

## Problem Statement

Operator bioskop perlu memperkirakan permintaan tiket beberapa hari ke depan untuk membantu menentukan alokasi layar, jumlah pertunjukan, serta kebutuhan operasional di tiap lokasi. Keputusan ini harus dibuat ketika film baru memiliki sedikit riwayat penjualan, sementara pola permintaan dapat berbeda antarfilm dan antarklaster bioskop.

Film dengan pembukaan yang kuat belum tentu mempertahankan pola yang sama pada hari-hari berikutnya. Sebaliknya, film dengan penjualan awal yang rendah masih dapat mengalami perubahan permintaan karena akhir pekan, kalender, atau perkembangan minat penonton. Perbedaan skala dan karakter tiap klaster bioskop membuat perkiraan tingkat nasional saja tidak cukup untuk mendukung keputusan di tingkat lokasi.

## Objektif

Dalam kompetisi ini, peserta akan menggunakan data penjualan tiket dari tiga hari pertama data penjualan tiket selama periode observasi pada hari ke-1 hingga hari ke-3 untuk memprediksi penjualan harian pada hari ke-4 hingga hari ke-10. Setiap prediksi dibuat untuk kombinasi film dan klaster bioskop, sehingga pola penjualan film dan perbedaan karakter tiap lokasi sama-sama berperan.

### Evaluation

Prediksi dinilai menggunakan **Mean Absolute Scaled Error (MASE)**.

Misalkan p adalah satu pasangan film dan klaster bioskop. Skala untuk pasangan tersebut dihitung dari rata-rata penjualan tiket pada tiga hari pertama:

s_p = \max\left( \frac{1}{3} \sum\_{d=1}^{3} y\_{p,d}, 1 \right)\\

dengan y\_{p,d} sebagai jumlah tiket aktual pasangan p pada hari ke-d.

MASE dihitung sebagai rata-rata kesalahan absolut yang dibagi dengan skala pasangan masing-masing:

\mathrm{MASE} = \frac{1}{N} \sum\_{i=1}^{N} \frac{ \vert y_i-\hat{y}\_i \vert }{ s\_{p(i)} }

dengan:

- N adalah jumlah seluruh baris pada data evaluasi.
- y_i adalah jumlah tiket aktual pada baris ke-i.
- \hat{y}\_i adalah jumlah tiket hasil prediksi.
- p(i) adalah pasangan film dan klaster bioskop pada baris ke-i.
- s\_{p(i)} adalah skala pasangan tersebut, yang sama untuk seluruh tujuh hari prediksinya.
```python
from sklearn.metrics import mean_absolute_error

def hitung_skala(history):
    """ 
    history: tiga hari observasi (D1-D3) untuk tiap pasangan klaster-film
    """
    total = history.groupby(["movie_title", "cinema_ids"]).total_ticket.sum()
    return (total / 3).clip(lower=1).rename("scale")

def mean_absolute_scaled_error(y_true, y_pred, scale):
    """
    y_true: jumlah tiket terjual
    y_pred: forecast tiket terjual
    scale: rata-rata tiket terjual di tiga hari pertama
    """
    return mean_absolute_error(y_true / scale, y_pred / scale)

scale = target.join(hitung_skala(history), on=["movie_title", "cinema_ids"]).scale
skor = mean_absolute_scaled_error(y_true, y_pred, scale)

```

content_copy

## Submission File

File submission harus memiliki kolom berikut:
```node-repl
id,total_ticket
1,100
2,101
3,102
...

```

### Extras

## Pastikan Submisi Bersifat Reproducible

Dalam melakukan validasi hasil submisi tiap peserta, panitia akan mengecek apakah hasil prediksi dapat dibuat ulang. Oleh karena itu, peserta diwajibkan untuk menetapkan konstanta random_state yang akan digunakan pada bagian yang membutuhkan randomness number pada pengerjaan. Kejanggalan (disparitas nilai yang besar antara hasil prediksi di Kaggle dengan hasil produksi ulang oleh panitia dapat mengakibatkan peserta yang bersangkutan didiskualifikasi. Berikut adalah potongan kode contoh penetapan random_state untuk mendapatkan hasil yang mampu dibuat ulang (reproducible).
```freedesktop
SEED = 2026
LABEL = "label"

train = pd.read_csv(". ./input/train.csv")
test = pd.read_csv(". ./input/test.csv")

X, y = train.drop(LABEL, axis=1), train[LABEL]

## Penetapan seed pada `random_state` untuk memastikan konsistensi hasil prediksi model
model = lightgbm.LGBMClassifier(random_state = SEED)
model.fit(X, y)

y_pred = model.predict(test)
```

## Dataset Description

## Data

| **Berkas**              | **Baris** | **Keterangan**                                                   |
| :---------------------- | :-------- | :--------------------------------------------------------------- |
| `train.csv`             | 138.959   | Riwayat transaksi harian dari 1 April sampai 30 September 2025   |
| `test_history.csv`      | 32.323    | Transaksi yang tercatat selama D1–D3 film uji                    |
| `test.csv`              | 72.611    | Kombinasi film, klaster, kota, dan tanggal yang perlu diprediksi |
| `sample_submission.csv` | 72.611    | Template file pengumpulan                                        |
| `movies.csv`            | 397       | Informasi metadata film                                          |
| `holidays.csv`          | 366       | Kalender hari dan hari libur                                     |
| `ticket_prices.csv`     | 207       | Daftar harga tiket per kota                                      |

Untuk setiap film uji, D1–D3 adalah tiga tanggal kalender berturut-turut yang berlaku sama di seluruh klaster bioskop. D1 tidak selalu merupakan tanggal pertama film tersebut memiliki transaksi di suatu klaster. D4–D10 adalah tujuh tanggal kalender berikutnya.

Setiap pasangan film dan klaster yang muncul di `test.csv` memiliki tujuh baris prediksi, yaitu D4–D10. Jika pada salah satu tanggal target tidak ada transaksi, jumlah tiket aktualnya adalah nol.

`train.csv` merupakan riwayat transaksi, bukan tabel contoh latih yang sudah diberi penanda D1–D10. Peserta dapat mengolah riwayat tersebut untuk membangun pendekatan dan validasi masing-masing.

## Kolom data

### `train.csv` dan `test_history.csv`

| **Kolom**         | **Keterangan**                                    |
| :---------------- | :------------------------------------------------ |
| `date_show`       | Tanggal penayangan dalam format `YYYY-MM-DD`      |
| `cinema_ids`      | ID anonim klaster bioskop                         |
| `city_name`       | Kota lokasi klaster                               |
| `movie_title`     | Judul film, termasuk penanda format jika tercatat |
| `total_ticket`    | Jumlah tiket yang terjual pada tanggal tersebut   |
| `occupation_rate` | Persentase kursi yang terisi, dari 0 sampai 100   |
| `total_show`      | Jumlah pertunjukan yang terlaksana                |

### `test.csv`

Berisi `id`, `movie_title`, `cinema_ids`, `city_name`, dan `date_show`. Kolom `date_show` menunjukkan tanggal D4–D10 yang perlu diprediksi.

### `sample_submission.csv`

Berisi `id` dan `total_ticket`. Nilai `total_ticket` di berkas ini hanya contoh pengisian, bukan prediksi acuan.

### `movies.csv`

Berisi `original_title`, `age_rating`, `genre`, `producer`, `director`, `writer`, dan `casts`.

### `holidays.csv`

Berisi `date`, `day_tipe` (`weekday`, `friday`, atau `weekend`), `holiday_tipe` (`normal` atau `holiday`), dan `holiday_name`.

### `ticket_prices.csv`

Berisi `city_name`, `ceil` sebagai harga tiket dalam rupiah, dan `price_day` untuk kategori `Weekday`, `Friday`, atau `Weekend`.

## Competition Rules

## Aturan Data Eksternal

Sumber eksternal hanya boleh digunakan jika informasi atau versinya telah tersedia untuk publik paling lambat **30 September 2025**. Batas waktu ini berlaku untuk seluruh film uji.

Data yang baru diterbitkan, diperbarui, atau baru dapat diakses setelah tanggal tersebut tidak boleh dijadikan fitur, walaupun membahas kejadian yang lebih lama. Jika tanggal ketersediaannya tidak dapat dibuktikan, sumber tersebut tidak boleh digunakan.

Berkas yang sudah disertakan dalam paket kompetisi merupakan input resmi dan tidak terkena batas publikasi ini.

Panitia dapat meminta finalis mencantumkan sumber atau URL, penerbit, tanggal publikasi atau arsip, waktu akses, versi data, serta cara penggunaan setiap sumber dalam notebook. Informasi tersebut akan diverifikasi.

Untuk aturan lainnya, silakan referensi ke link salindia TM berikut:

[https://drive.google.com/file/d/1klMZDtXvuNU3gRTku3DuHJI5EgSLTn5y/view?usp=sharing](https://drive.google.com/file/d/1klMZDtXvuNU3gRTku3DuHJI5EgSLTn5y/view?usp=sharing)

---

# JOINTS X INSPIRE 2026

## Data Science Technical Meeting

## Deskripsi Umum

- Kompetisi menguji kemampuan peserta dalam menyelesaikan masalah nyata berbasis data.
- Menggunakan pendekatan machine learning, deep learning, dan analisis data komprehensif.
- Peserta akan diberikan dataset dan objective tertentu untuk menghasilkan model prediksi terbaik.
- Evaluasi mencakup performa akurasi, kualitas dokumentasi, serta interpretasi hasil.

## Materi Umum

- **Data Preprocessing & Cleaning:** Penanganan missing values, outliers, dan pembersihan data.
- **Exploratory Data Analysis (EDA):** Visualisasi data, analisis distribusi, dan pemahaman insight awal.
- **Feature Engineering:** Ekstraksi fitur, seleksi fitur, dan transformasi variabel.
- **Machine Learning & Deep Learning:** Pembangunan model, pemilihan algoritma, serta efisiensi komputasi.
- **Model Evaluation & Validation:** Metrik evaluasi, k-fold cross-validation, dan pencegahan overfitting.

## Timeline

| Tahapan Kegiatan | Jadwal Pelaksanaan |
| --- | --- |
| Pendaftaran Peserta | 18 Agustus—16 September 2026 |
| Technical Meeting (Penyisihan) | 22 September 2026 |
| Babak Penyisihan Kaggle (Online) | 28 September—12 Oktober 2026 |
| Pengumpulan File Notebook dan Model | 13 Oktober 2026 |
| Pengumuman Finalis (Top 10) | 23 Oktober 2026 |
| Technical Meeting (Final) | 25 Oktober 2026 |
| Batas Pengumpulan File Presentasi | 31 Oktober 2026 |
| Babak Final (Offline di FMIPA UGM) | 1 November 2026 |

## PENYISIHAN - Mekanisme

- **Platform:** link Kaggle Competition akan dibagikan di group WhatsApp (28 September 09:00 WIB)
- **Ketentuan Tim:** Ketua tim membuat tim Kaggle dan mengundang seluruh anggota tim
- **Nama Tim:** Sesuai dengan nama tim yang didaftarkan (nama tim yang tidak terdapat di daftar peserta tim didiskualifikasi)
- **Akses Data:** Dataset dan metrik evaluasi disediakan langsung via platform Kaggle.
- **Batas Submission:** Maksimal 3 kali submission per hari 📌

## PENYISIHAN - Ketentuan Tools & Pemodelan

### PERMITTED ✅

- Teknik Transfer Learning dengan Pre-trained Model publik (contoh: HuggingFace, ImageNet weights, dsb.)
- Penggunaan Data Eksternal untuk memperkaya informasi dataset. 📌
- Bebas menggunakan bahasa pemrograman, library, framework, maupun tools open-source apapun.

### STRICTLY FORBIDDEN 🚫

- Sangat DILARANG menggunakan framework Automated Machine Learning (AutoML).

## PENYISIHAN - Ketentuan Notebook

### Aturan Pengkodean Notebook:

- **Instalasi Dependency:** perintah pip install seluruh dependency beserta versi spesifiknya dalam mode quiet (-q). 📌
- **Random State:** mencantumkan seed / random_state pada setiap proses acak demi reprodusibilitas. 📌
- **Alur Runtut:** Notebook memuat seluruh tahapan secara runtut (Data Acquisition, EDA, Preprocessing, Data Cleaning, Feature Engineering, Modeling, Evaluation).
- **Penjelasan & Visualisasi:** Disertai penjelasan naratif dan visualisasi yang jelas pada tiap langkah.
- **Pretrained Model:** Jika memakai Pretrained Model, jelaskan sumber dan metode penerapannya secara detail.

### Contoh Kode Cell Pertama Notebook:

```python
# 1. Instalasi library dengan versi spesifik & mode ringkas (quiet)
!pip install -q transformers==4.38.0 ultralytics==8.1.0 pandas==2.2.0 torch==2.2.0
```

## PENYISIHAN - Submission

### Ketentuan Umum

- **Perwakilan Pengirim:** Dilakukan oleh Ketua Tim melalui google forms yang diberikan panitia.
- **Format Berkas:** Seluruh berkas (notebook dan model weights) wajib digabungkan ke dalam satu file .zip.
- **Batas Ukuran Model Weights:** Ukuran file bobot model (model weights) maksimal 200 MB. 📌
- **Batas Pengumpulan:** 13 Oktober 2026 (23:59 WIB)📌

### Struktur & Penamaan File Submisi

📁 File ZIP Utama: [Nama Tim].zip

- File Notebook: [Nama Tim].ipynb
- File Model Weights: [Nama Tim](.pkl, .pt, .pth, .hf, .tf, atau format relevan lainnya)

### Contoh Penamaan File

📁 DataVictory.zip

- DataVictory.ipynb
- DataVictory.pt (ukuran ≤ 200 MB)

## PENYISIHAN - Sistem Penilaian

### Bobot Asesmen Penyisihan

- Score Private Leaderboard Kaggle: 90%
- Penilaian Kualitas Dokumentasi Notebook: 10%

### Prosedur Uji Reprodusibilitas (Code Verification)

- 10 kandidat finalis wajib lolos Uji Reprodusibilitas.
- Panitia akan melakukan re-run terhadap notebook dan file model yang dikumpulkan.
- Hasil running ulang harus menghasilkan skor yang konsisten dengan Private Leaderboard.
- Jika ada selisih signifikan akibat kelalaian reproduksi, tim akan didiskualifikasi dan digantikan peringkat di bawahnya.

## PENYISIHAN - Ketentuan Diskualifikasi

### Daftar Pelanggaran Fatal:

1. Terdaftar di lebih dari satu tim.
2. Melakukan plagiarisme, berbagi kode/solusi, atau bekerja sama antar tim.
3. Penggunaan framework Automated Machine Learning (AutoML).
4. Segala bentuk kecurangan yang mempengaruhi proses dan hasil lomba.
5. Keterlambatan dalam mengumpulkan file submission penyisihan maupun file presentasi final.
6. Tidak hadir secara offline pada saat Babak Final di FMIPA UGM.

ℹ Panitia akan mengontak tim terduga (wajib direspon maks. 1x24 jam)

## BABAK FINAL- Ketentuan

- **Peserta:** 10 tim terbaik yang lolos verifikasi penyisihan.
- **Format Acara:** OFFLINE pada 1 November 2026 di FMIPA Universitas Gadjah Mada, Yogyakarta. 📌
- **Batas Submit Slide Presentasi:** Maksimal 31 Oktober 2026 (Format .pdf atau .pptx). 📌
- **Mekanisme Presentasi:**
  - Durasi total: 30 Menit (Presentasi di depan juri + Sesi Tanya Jawab).
  - Bahan presentasi didasarkan pada hasil analisis dan pemodelan babak penyisihan.
- Detail rubrik penilaian final akan disampaikan saat Technical Meeting Final (25 Oktober 2026).

## QnA

https://app.sli.do/event/tQVrF7frqAipg3ncbi474Y

## Ketentuan Tambahan / Reminder Kompetisi

### 🤖 1. Penggunaan AI

- LLM diperbolehkan untuk membantu dalam development code.
- Namun, dilarang keras menggunakan LLM sebagai inference secara langsung dalam proses prediksi.
- Dilarang menggunakan AI Agent dalam kompetisi.
- Dilarang menggunakan AI API calling untuk melakukan inference.

### 🏆 2. Sistem Penilaian

- 90% skor akhir sepenuhnya berasal dari Private Leaderboard.
- Ketentuan ini berlaku sebagai bagian utama dalam penentuan hasil kompetisi.

### 📊 3. Data Eksternal

- Penggunaan data eksternal diperbolehkan, baik dalam bentuk labelled maupun unlabelled data.
- Format data eksternal dibebaskan, selama penggunaannya sesuai dengan ketentuan kompetisi.
- Sumber data eksternal wajib dapat diakses secara publik dan dapat diverifikasi.
- Data eksternal hanya boleh digunakan apabila informasi atau versi data tersebut telah tersedia untuk publik paling lambat **30 September 2025**.
- Batas waktu tersebut berlaku untuk seluruh film uji.

## Thank you
