# Peta repository (`/home/rian/Projects/joints`)

```
data/                     berkas resmi lomba (train, test_history, test, sample_submission, movies, holidays, ticket_prices)
docs/
  context.md              deskripsi dan aturan lomba (sumber kebenaran aturan)
  eda_findings.md         catatan temuan semua script EDA, per versi (angka lengkap)
  experiment_audit_2026-10-05.md   audit eksperimen v1-v9 (dari Astra) dan rencana v10
  deep_analysis_2026-10-06.md      deep EDA eda/54-61 (dari Astra) dan dasar v11/v12
eda/                      01-61 (tanpa 11): satu script = satu analisis, mencetak angka dan menyimpan gambar ke outputs/eda/<nama>/
src/
  common.py               load, base_title, fmt, release_dates (D1), simulate, scale, mase, style, fig_dir
  features.py             calendar, build, visible_windows, film_table, make_ctx, dataset() (cache outputs/cache/X*.parquet)
  evaluate.py             folds, test_weights (bucket), test_weights_fd (bucket x first-day), kappa_worlds, report
  model.py                LightGBM bersama (era v1-v3)
  exaone_q.py             kelas EXAONE-Tabular yang mengembalikan kuantil (+ self-test)
notebooks/
  build_notebook.py       builder v1; build_v3.py ... build_v12.py (builder tiap versi)
  v1.ipynb ... v12.ipynb  notebook yang SUDAH DI-RUN user di Kaggle (berisi output) - jangan ditimpa
  check_v10_selection.py  self-check aturan seleksi v10
results/vN/               output Kaggle per versi: submission.csv, oof.csv, model_weights.pkl, figures/
outputs/
  eda/                    gambar dan tabel tiap script EDA
  cache/                  Xtr/Xte/Xlim.parquet, proxy_A/B, hurdle_oof.npz, lgb_parts_v6.npz, tab35_oofQ*.npy,
                          tabicl_oofQ.npy, w_adv.npy, cmi.parquet, film_resid.csv, dll.
ref/notebook/             skill pembuatan notebook (SKILL.md, build_template.py, references/)
handoff/                  folder ini
```

## Environment Python

| Venv | Isi | Dipakai untuk |
| :--- | :--- | :--- |
| `.venv` | Python 3.12, pandas 3.0, LightGBM 4.7, catboost 1.2.10, xgboost 3.4.1 | EDA dan preprocessing; build notebook |
| `.venv-tp9` | tabpfn 9.0.0, torch 2.8 CPU, tabicl, exaonetabular, tabm 0.0.3, rtdl_num_embeddings 0.0.12, pyarrow | Smoke test model (hanya kecil) dan uji preprocessing notebook |

`.venv-xpu` (untuk Intel Arc) sudah dihapus karena tidak bisa dipakai (butuh driver via sudo).

## Cara membuat notebook versi baru

1. Salin `notebooks/build_v12.py` menjadi `build_v13.py`.
2. Ubah `OUT` (v13.ipynb), banner `(v13)`, dan `OUTPUT_DIR` (`../outputs/v13`).
3. Edit lewat helper `md()`, `code()`, `section()`, lalu jalankan `.venv/bin/python notebooks/build_v13.py`.
4. Cek:
   - semua cell kode bisa di-parse (`ast.parse`);
   - `uvx ruff check --select F` atas sel-sel yang digabung;
   - tidak ada docstring di cell kode.
5. Uji preprocessing saja (Section 1-7 dan cell pandas Section 8) di `.venv-tp9`, dengan cell model dilewati.
6. Jangan menjalankan training penuh di laptop.

## Pembersihan 7 Okt 2026 (sudah dihapus, total sekitar 11,6 GB)

- Salinan checkpoint di `results/v6-v8/tabpfn/` (TabPFN-3.5, 3 x 836 MB) dan `results/v9-v12/fm/`. Semuanya
  bisa diunduh ulang dari HF. Checkpoint yang dipakai v9-v12 sudah tertanam di `model_weights.pkl`.
- `outputs/cache/tabpfn35/` dan `outputs/cache/tabpfn_v2.6/` (unduhan checkpoint lokal).
- Output dry-run lokal `outputs/v3*`, `outputs/v4`-`v11`, dan `outputs/notebook`.
- Log run di `outputs/*.log`, serta parquet lama di root `outputs/` (output `eda/08-10`, bisa dibuat ulang).
- `outputs/probes`, `outputs/probes_v3`: berkas submisi probing leaderboard dari awal proyek. **Tidak pernah
  dipakai**, dan bertentangan dengan aturan user.
- `__pycache__` dan `.venv-xpu`.
