# Semua model yang dicoba

## Ringkasan

Angka TW memakai bobot saat itu; yang bertanda (bfd) memakai bobot bucket x hari pertama.

| Model | Ukuran checkpoint | Hasil | Status |
| :--- | :--- | :--- | :--- |
| LightGBM L1 (5 seed) | - | TW (bfd) 0,3464 | Komponen L1 di mix |
| **LightGBM hurdle + L1 mix** | ~25 MB terkompresi | TW (bfd) **0,3319** (B1), 0,3322 (v8 config) | **Kontrol, dipakai semua versi** |
| CatBoost MAE / XGBoost MAE (v1, `eda/09`) | - | 0,372 / 0,379 vs LightGBM 0,358 | Ditolak |
| CatBoost (hurdle, MultiQuantile) v11 | - | 0,3367; korelasi galat 0,993 | Ditolak |
| XGBoost (hurdle, reg:quantileerror) v11 | - | 0,3342; korelasi 0,997 | Ditolak |
| TabPFN v2 (rev Juni 2025) | 44,4 MB | TW (bucket) 0,3876 λ0 / 0,3951 λ0,5 | Dipakai v1, v3-v5; opsi konservatif `FM_MODE='tabpfn_v2'` |
| **TabPFN-3.5** | **876 MB** (fast 334 MB) | TW (bucket) 0,3815 (Kaggle, 8 anggota); terbaik di public | v2, v6-v8; **melanggar 200 MB** |
| TabPFN-3.5 fine-tuned (CRPS) | 876 MB | 0,3968 vs ICL 0,3840; korelasi 0,994 | Ditolak (v7) |
| TabPFN-3.5 konteks gabungan (`eda/47`) | - | Rata-rata 0,3904 vs per horizon 0,3866 (2 horizon, 1 fold) | Ditolak (17x lebih lambat) |
| TabPFN-3 | 233 MB | - | > 200 MB, tidak diuji |
| TabPFN-2.6 | 51,6 MB | TW (bfd) 0,3263 λ0 / 0,3298 λ | Bobot 0 di v9 |
| **TabPFN-2.5 Quantiles** | 40,8 MB | TW (bfd) **0,3274** λ0 (model tunggal terbaik v12) | Dipakai v12 (bobot 0,4) |
| TabICL v2 | 114 MB | 0,4564 (bucket-κ); massa nol buruk | Ditolak (`eda/35-36`) |
| TabDPT 1.1 | 252-308 MB | Gagal jalan (faiss) | Gugur |
| LimiX-16M | 66 MB | Butuh flash-attn (T4 tidak mendukung); keluaran titik | Gugur |
| Mitra (AutoGluon) | 303 MB | - | > 200 MB |
| Google TabFM 1.0 | 6,6 GB | - | > 200 MB |
| TabH2O | - | Hanya lewat hosted API | Dilarang (AI API inference) |
| **EXAONE-Tabular** (LG AI, Agu 2026) | 84,5 MB | TW (bfd) 0,3258 λ / 0,3259 λ0 | v9 (0,4), v10 (0,7) |
| **Causilo** (nums-ai, Sep 2026) | 148,4 MB | TW 0,3954; s ≤ 20 1,09; korelasi 0,75 | v11; jelek atau integrasi bermasalah |
| **TabM** (yandex-research) | beberapa MB | TW (bfd) 0,3354 λ0 | Dipakai v12 (bobot 0,1) |
| Seed bagging hurdle 3 seed | - | 0,3435 vs seed tunggal 0,3432-0,3440 | Noise, ditolak |
| Foundation model deret waktu (Chronos, TimesFM, Moirai) | - | Tidak relevan: tiap seri hanya 3 titik | Tidak dicoba |
| LLM (EXAONE LLM, dsb.) | - | Dilarang untuk inference | - |

## Catatan teknis per model

### TabPFN (`tabpfn==9.0.0`)

- Panggil `TabPFNRegressor.create_default_for_version(ModelVersion.Vx, model_path=..., n_estimators=...,
  ignore_pretraining_limits=True)`.
- `predict(..., output_type='quantiles', quantiles=[...])` mengembalikan list per kuantil, jadi perlu `.T`.
- Unduh dengan `hf_hub_download(..., local_dir=...)`. Loader memilih format dari akhiran `.safetensors`, dan blob
  cache HF tidak punya akhiran.
- Repo Prior Labs gated, sehingga butuh `HF_TOKEN`.
- Satu model per horizon (konteks sekitar 6,4-8 ribu baris) lebih baik daripada konteks gabungan.
- Di v9 (TabPFN-2.6, 16 anggota) OOF memakan 2.381 detik di T4. Di v12 (TabPFN-2.5, 8 anggota) CV + temporal
  985 detik.

### EXAONE-Tabular (`exaonetabular`, commit `8638e07d`, bobot rev `093ad1c2`, SHA `24891e14…`)

- Paket hanya mengembalikan trimmed mean. Kepalanya mengeluarkan 999 kuantil per anggota, jadi kelas turunan
  (`src/exaone_q.py`) membaca kuantil dan merata-ratakan antar anggota.
- Default memakai FlashAttention, yang tidak ada di T4. Patch `_select_sdpa_backend` mengarahkan ke kernel
  EFFICIENT atau MATH.
- Install `--no-deps`, karena syarat numpy ≥ 2.3.5 dan sklearn ≥ 1.7.2 bentrok dengan pin.
- CPU float32: 2.303 detik per fold x horizon. Hanya praktis di GPU (v9: sekitar 11 menit per fold).

### Causilo (`causilo==1.0.3`, rev rilis `94f2bd91`, regressor 148,4 MB)

- Butuh `torch>=2.13`. Kaggle meng-upgrade torch ke 2.14.1+cu130, dan GPU tetap terlihat.
- Kuantil native: `predict(X, output_type='quantiles', quantiles=...)` mengembalikan (baris, level).
- Bobot dimuat dari salinan lokal ber-SHA dengan menimpa `causilo.checkpoints.load_pretrained_model`.
- **Hasilnya janggal:**
  - TW 0,3954;
  - hanya 52 detik per fold (jauh lebih cepat dari TabPFN);
  - galat pasangan kecil 1,09;
  - korelasi galat dengan LightGBM hanya 0,75, padahal model lain 0,98-0,99.
- Belum dicek:
  - apakah kuantil dan massa nolnya terbaca benar (bandingkan dengan `output_type='median'`/`'mean'` bawaan);
  - perbandingan λ 0 vs 0,5;
  - `n_estimators`;
  - konteks gabungan.
- **Belum dimentokin.**

### TabM (`tabm==0.0.3`)

- `TabM.make(n_num_features, d_out=49, k=32, num_embeddings=PiecewiseLinearEmbeddings(bins, 16,
  activation=False, version='B'))`. Bin dari `compute_bins`.
- Keluaran berbentuk (batch, k, d_out). Loss dirata-rata atas k submodel (sesuai docs). Prediksi = rata-rata
  atas k.
- `compute_bins` **menolak kolom konstan**: `t_ramadan` dan `cin_new` konstan di train-sim, jadi harus
  disaring di inner-training.
- Pertama kali dicoba di v12 dengan satu konfigurasi (tidak dituning). Berhenti di sekitar 9 epoch, 193 detik
  untuk CV + temporal.

### CatBoost dan XGBoost (v11)

- CatBoost `MultiQuantile:alpha=...` dan XGBoost `reg:quantileerror` + `quantile_alpha` menghasilkan 19
  kuantil dalam satu model.
- Struktur identik dengan hurdle LightGBM. Korelasi galat sangat tinggi, jadi tidak menambah keragaman.
