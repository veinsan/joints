import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "v12.ipynb"
HR  = "---"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": uuid4().hex[:8], "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code", "execution_count": None, "id": uuid4().hex[:8],
        "metadata": {}, "outputs": [], "source": source,
    }


def section(title: str, anchor: str) -> dict:
    return md(
        f"{HR}\n\n"
        f"# {title} <a name=\"{anchor}\"></a>\n\n"
        f"{HR}"
    )


cells = []

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------
cells.append(md(
    f"{HR}\n\n"
    "# Data Science Competition JOINTS x INSPIRE UGM 2026 (v12)\n\n"
    "*Memprediksi penjualan tiket harian D4 sampai D10 per pasangan film dan klaster bioskop dari tiga hari "
    "pertama, dengan model hurdle yang memisahkan keputusan bioskop mencopot film dari jumlah penonton, "
    "dikoreksi untuk kebijakan pencopotan periode uji yang terukur dari data berlabel periode uji.*"
))

cells.append(md(
    f"{HR}\n\n"
    "## [Nama Tim]\n\n"
    "- [Nama Anggota 1] (Ketua)\n"
    "- [Nama Anggota 2]\n"
    "- [Nama Anggota 3]"
))

cells.append(md(
    f"{HR}\n\n"
    "## Table of Contents\n\n"
    "1. [**Introduction**](#1)\n"
    "2. [**Initialization**](#2)\n"
    "3. [**Data Acquisition & Audit**](#3)\n"
    "4. [**Exploratory Data Analysis**](#4)\n"
    "5. [**Data Cleaning**](#5)\n"
    "6. [**Preprocessing**](#6)\n"
    "7. [**Feature Engineering**](#7)\n"
    "8. [**Modeling**](#8)\n"
    "9. [**Evaluation**](#9)\n"
    "10. [**Inference & Submission**](#10)\n"
))

# ---------------------------------------------------------------------------
# Section 1: Introduction
# ---------------------------------------------------------------------------
cells.append(section("Introduction", "1"))

cells.append(md(
    "## Overview\n\n"
    "*Operator bioskop harus menentukan alokasi layar ketika film baru hanya memiliki tiga hari riwayat "
    "penjualan. Data latih mencakup April sampai September 2025, sedangkan film uji dirilis Oktober 2025 "
    "sampai Maret 2026: musim sepi, libur Natal, Ramadan, dan Lebaran.*\n\n"
    "*Versi sebelumnya memiliki MASE validasi yang dibobot ke komposisi uji (TW-MASE) 0,377 sampai 0,388 dengan "
    "peringkat yang sama seperti leaderboard, tetapi dengan offset tetap sekitar 0,07. Offset yang sama pada "
    "model yang berbeda menandakan kesalahan sistematis yang diwarisi dari data latih. v4 menemukan salah satu "
    "sumbernya (di periode uji bioskop jauh lebih jarang mencopot film pada tingkat penonton yang sama) dan turun "
    "ke 0,4481. v5 menambah dua hal: fitur kompetisi per klaster pada klasifier pencopotan, dan prediksi minggu "
    "Lebaran yang diturunkan dari total klaster Lebaran 2025 di data latih, dan turun ke 0,4087. Dengan itu, "
    "TW-MASE di dunia dengan separuh pergeseran pencopotan (v5: 0,4066) praktis sama dengan leaderboard, sehingga "
    "metrik ini menjadi metrik keputusan. v6 memakai TabPFN-3.5 dan mencapai 0,4012. v7 menguji TabPFN-3.5 yang "
    "di-fine-tune (loss CRPS): lebih buruk dari versi in-context (TW-MASE 0,3968 vs 0,3840), bobot blend nol, dan "
    "dihapus. v8 memperbaiki metrik validasi: pasangan yang baru mulai laku di D3 (pembukaan terlambat) 2,3 kali "
    "lebih banyak di bobot lama daripada di data uji, sehingga bobot uji kini mencocokkan komposisi **skala x hari "
    "penjualan pertama**. TabPFN-3.5 memakai 16 anggota ensemble (bagging 3 seed pada hurdle LightGBM diuji, tidak di atas noise) dan mencapai 0,39991. "
    "**v9 mematuhi batas bobot model 200 MB**: TabPFN-3.5 (876 MB) diganti dua tabular foundation model yang muat, "
    "TabPFN-2.6 (51,6 MB) dan EXAONE-Tabular dari LG AI Research (84,5 MB). Keduanya dinilai out-of-fold dan bobot "
    "blend dipilih otomatis; checkpoint yang dipakai disimpan bersama model LightGBM dalam satu berkas bobot "
    "di bawah 200 MB sehingga notebook dapat dijalankan ulang tanpa internet. **v10 memperbaiki alat ukur sebelum mencari "
    "skor**: statistik klaster dihitung fold-local (sebelumnya melihat transaksi film validasi), backtest temporal "
    "bulanan ditambahkan, dan kandidat perbaikan (perubahan jumlah show dan tiket per show dengan nested cross-fitting; "
    "metadata resmi) diuji satu per satu terhadap baseline bersih dengan aturan adopsi yang ditulis sebelum hasil. "
    "Kalender libur sekolah Jawa Barat (22% baris uji) dikoreksi dari sumber resmi Disdik Jabar. v10 mencapai 0,40823 "
    "(lebih buruk dari v8): kedua kandidatnya gagal aturan adopsi, dan bobot EXAONE 0,7 dipilih dengan label asli "
    "walaupun dunia $\\kappa$ memilih sekitar 0,4. **v11 mengikuti analisis 6 Oktober**: satu kandidat saja, fitur input "
    "D1 sampai D3 yang dinormalisasi kalender (satu-satunya preprocessing yang membaik di grouped CV, split kedua, "
    "backtest temporal, dan pipeline hurdle), dan bobot blend dipilih dengan minimax regret atas label asli, backtest "
    "temporal, dan dunia $\\kappa$. Komponen model v11 adalah keluarga GBDT (LightGBM, CatBoost, XGBoost, masing-masing dengan "
    "struktur L1 + hurdle yang sama) dan Causilo (nums-ai, peringkat 1 TabArena menurut pembuatnya, 148,4 MB) sebagai "
    "foundation model. v11 di Kaggle: A3 membaik (TW 0,3319 ke 0,3310, temporal 0,3058 ke 0,3027) tetapi belum lolos ambang, "
    "CatBoost (0,3367) dan XGBoost (0,3342) lebih buruk dan berkorelasi galat 0,993 sampai 0,997 dengan LightGBM, Causilo jauh "
    "lebih buruk (0,3954), dan tidak ada blend yang lolos, sehingga submisi v11 = LightGBM saja. **v12** mempertahankan fondasi "
    "v11 dan menguji model secara berurutan dengan fitur baseline dibekukan: LightGBM sebagai kontrol, TabPFN-2.5 Quantiles "
    "(40,8 MB), lalu TabM (model tabular dilatih sendiri dengan loss kuantil); koreksi pencopotan dipilih per model, dan fitur "
    "kalender baru diuji sebagai A/B pada kombinasi terbaik.*"
))

cells.append(md(
    "## Aim\n\n"
    "*Memprediksi `total_ticket` D4 sampai D10 untuk 72.611 baris uji dengan MASE sekecil mungkin pada "
    "private leaderboard, memakai metrik validasi yang terbukti sejalan dengan leaderboard.*"
))

cells.append(md(
    "## Metric\n\n"
    "***Mean Absolute Scaled Error (MASE)** membagi galat absolut setiap baris dengan skala pasangan, yaitu "
    "rata-rata D1 sampai D3 yang dibatasi minimal 1. MASE setara MAE pada rasio $r = y / s_p$, sehingga "
    "prediksi optimal adalah median bersyarat. Karena MASE adalah rata-rata biasa per baris, komposisi baris "
    "uji (misalnya porsi pasangan kecil) langsung menentukan skor.*\n\n"
    "$$s_p = \\max\\left(\\frac{1}{3}\\sum_{d=1}^{3} y_{p,d},\\ 1\\right) \\qquad "
    "\\text{MASE} = \\frac{1}{N}\\sum_{i=1}^{N}\\frac{|y_i - \\hat{y}_i|}{s_{p(i)}}$$"
))

cells.append(md(
    "## Dataset\n\n"
    "*Data bersumber dari Kaggle Competition Data Science JOINTS x INSPIRE UGM 2026.*\n\n"
    "*`train.csv` berisi 138.959 transaksi harian (1 April sampai 30 September 2025). `test_history.csv` berisi "
    "32.323 transaksi D1 sampai D3 dari 163 judul uji, dan `test.csv` berisi 72.611 baris target "
    "(10.373 pasangan x 7 hari). Hari tanpa transaksi tidak tercatat dan bernilai nol.*\n\n"
    "*Data digunakan untuk merekonstruksi sampel latih dengan aturan panitia, lalu memprediksi D4 sampai D10.*\n\n"
    "**Attributes**\n\n"
    "1.  `date_show`       : Tanggal penayangan\n"
    "2.  `cinema_ids`      : ID anonim klaster bioskop\n"
    "3.  `city_name`       : Kota lokasi klaster\n"
    "4.  `movie_title`     : Judul film, termasuk penanda format (3D, IMAX 2D, IMAX 3D)\n"
    "5.  `occupation_rate` : Persentase kursi terisi (0 sampai 100)\n"
    "6.  `total_show`      : Jumlah pertunjukan\n\n"
    "**Target**\n"
    " `total_ticket`       : Jumlah tiket terjual per pasangan per tanggal (0 bila tidak ada transaksi)"
))

cells.append(md(
    "## Approach: Hurdle + L1 dengan Koreksi Kebijakan Pencopotan\n\n"
    "*Temuan utama v3: galat per ukuran pasangan stabil antara periode latih dan periode uji, tetapi data uji "
    "memuat empat kali lebih banyak pasangan kecil (skala di bawah 20) yang galatnya paling besar. MASE "
    "validasi biasa karena itu terlalu optimistis. Seluruh keputusan di notebook ini diambil dengan "
    "**TW-MASE**, yaitu galat out-of-fold yang dibobot ke komposisi data uji menurut bucket skala x hari penjualan "
    "pertama (D1, D2, atau D3).*\n\n"
    "*Target $r = y / (s_p c_{mult})$ adalah campuran: nol bila bioskop mencopot film, positif bila tidak. "
    "Prediksi optimal MASE adalah median campuran: $0$ bila $p_0 \\ge 0{,}5$, selain itu kuantil ke-"
    "$(0{,}5 - p_0)/(1 - p_0)$ dari bagian positif. Model hurdle LightGBM (klasifier $p_0$ dan 19 regresi kuantil) "
    "serta TabPFN-2.5 Quantiles dan TabM (kuantil prediktif) memakai rumus ini, dengan koreksi pencopotan dipilih per model. Di periode uji, $p_0$ digeser "
    "$\\text{logit}\\,p_0' = \\text{logit}\\,p_0 + \\lambda a$, dengan $a$ diukur dari data berlabel periode uji "
    "dan $\\lambda = 0{,}5$ dipilih dengan minimax regret dari tiga sumber bukti. Model L1 ikut di-blend sebagai "
    "lindung nilai karena lebih tahan pada pasangan yang ternyata dicopot.*\n\n"
    "```\n"
    "train.csv -> D1 rilis resmi (segmen kontinu) -> simulasi aturan panitia -> + film rilis terbatas (latih saja)\n"
    "                                                   |\n"
    "test_history.csv ---------------------------------+--> build fitur (pasangan, film, kalender, jadwal, klaster)\n"
    "                                                   |\n"
    "      LightGBM L1 + hurdle (kontrol)  +  TabPFN-2.5 Quantiles per horizon  +  TabM (kuantil)\n"
    "                                                   |  p0 digeser lambda x a (a dari proxy periode uji)\n"
    "                  y_hat = r_hat x c_mult x s_p, bobot blend dipilih dengan TW-MASE -> submission.csv\n"
    "```\n\n"
    "*Validasi: 5 fold dikelompokkan per judul dasar film (versi 2D, 3D, IMAX satu film di fold yang sama).*"
))

# ---------------------------------------------------------------------------
# Section 2: Initialization
# ---------------------------------------------------------------------------
cells.append(section("Initialization", "2"))

cells.append(md(
    "## Environment Setup\n\n"
    "Notebook dirancang untuk Kaggle GPU T4 x2 dengan internet aktif (unduh bobot TabPFN dari Hugging Face). "
    "TabPFN memakai GPU pertama; LightGBM (L1, klasifier, dan 19 regresi kuantil) berjalan di CPU dengan "
    "`deterministic=True`. Recorded runtime v3 di T4: TabPFN OOF 18 menit dan fit akhir 21 menit; v4 menambah "
    "sekitar 15 sampai 25 menit untuk hurdle LightGBM."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code(
    "%pip install -q lightgbm==4.6.0 tabpfn==9.0.0 tabm==0.0.3 rtdl_num_embeddings==0.0.12 huggingface_hub==0.34.4 scikit-learn==1.6.1 pandas==2.2.3 "
    "scipy==1.15.2 matplotlib==3.10.0 seaborn==0.13.2"
))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import os\n"
    "import random\n"
    "import time\n"
    "import pickle\n"
    "import io\n"
    "import zlib\n"
    "import itertools\n"
    "import hashlib\n"
    "import warnings\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "import lightgbm as lgb\n"

    "import torch\n"
    "from IPython.display import display\n"
    "from huggingface_hub import hf_hub_download\n"
    "from sklearn.metrics import roc_auc_score\n"
    "from sklearn.model_selection import GroupKFold, StratifiedGroupKFold\n"
    "from sklearn.preprocessing import QuantileTransformer\n"
    "from tabm import TabM\n"
    "from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins\n"
    "from tabpfn import TabPFNRegressor\n"
    "from tabpfn.constants import ModelVersion\n"
    "try:\n"
    "    from kaggle_secrets import UserSecretsClient\n"
    "except ImportError:\n"
    "    UserSecretsClient = None\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "pd.set_option('display.max_columns', 60)\n"
    "PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']\n"
    "sns.set_theme(style='whitegrid', palette=PAL, rc={'axes.spines.top': False, 'axes.spines.right': False})"
))

cells.append(md("## Seed Everything"))

cells.append(code(
    "def seed_everything(seed: int = 2026):\n"
    "    random.seed(seed)\n"
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n"
    "    np.random.seed(seed)\n"
    "    torch.manual_seed(seed)\n"
    "    torch.cuda.manual_seed_all(seed)\n"
    "    torch.backends.cudnn.deterministic = True\n"
    "    torch.backends.cudnn.benchmark = False\n"
    "\n"
    "seed_everything(2026)"
))

cells.append(md(
    "## Settings\n\n"
    "Seluruh path, konstanta hasil EDA, hyperparameter, dan flag dipusatkan di sini. `HURDLE_W` adalah bobot "
    "hurdle dalam komponen LightGBM dan `LAMBDA` porsi pergeseran kebijakan pencopotan yang diterapkan; keduanya "
    "dipilih dengan analisis keputusan di Bagian 8. `LEB_RHO` adalah porsi total pasar minggu Lebaran 2025 yang "
    "diasumsikan dicapai film Lebaran 2026 (Bagian 10). Tidak ada nilai yang disetel pada leaderboard."
))

cells.append(code(
    "class Settings:\n"
    "    SEED       = 2026\n"
    "    _ON_KAGGLE = Path('/kaggle/input').exists()\n"
    "\n"
    "    DATA_DIR   = (\n"
    "        next(Path('/kaggle/input').rglob('train.csv')).parent\n"
    "        if _ON_KAGGLE else Path('../data')\n"
    "    )\n"
    "    OUTPUT_DIR = Path('/kaggle/working') if _ON_KAGGLE else Path('../outputs/v12')\n"
    "    FIG_DIR    = OUTPUT_DIR / 'figures'\n"
    "\n"
    "    TRAIN_END  = pd.Timestamp('2025-09-30')\n"
    "    OUTAGE     = pd.to_datetime(['2025-06-07', '2025-06-09', '2025-06-10', '2025-06-13', '2025-06-16'])\n"
    "    D1_FRAC    = 0.5\n"
    "    MIN_NC     = 25\n"
    "    USE_LIMITED = True\n"
    "\n"
    "    DOW_PROF   = np.array([0.811, 0.789, 0.811, 0.783, 0.848, 1.290, 1.282])\n"
    "    HOL_LEVEL  = 1.29\n"
    "    SCHOOL_WD  = 1.15\n"
    "    QS         = np.round(np.arange(0.05, 1.0, 0.05), 2)\n"
    "    TAB_QS     = np.round(np.arange(0.02, 1.0, 0.02), 2)\n"
    "    ZERO_EPS   = 0.02\n"
    "    HURDLE_W   = 0.75\n"
    "    LAMBDA     = 0.5\n"
    "    LEB_RHO    = 0.5\n"
    "    LEB_DAY1   = 0.75\n"
    "    TW_BINS    = [0, 5, 20, 50, 100, 200, 500, 1e9]\n"
    "\n"
    "    N_FOLDS    = 5\n"
    "    SEEDS      = [2026, 2027, 2028, 2029, 2030]\n"
    "    LGB_PARAMS = dict(objective='l1', learning_rate=0.03, num_leaves=63, min_child_samples=100,\n"
    "                      feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,\n"
    "                      n_estimators=800, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)\n"
    "\n"
    "    FM_MODE      = 'tabpfn25q'  # 'tabpfn25q' | 'tabpfn_v2' (revision 11 Jun 2025, before the cutoff) | 'none'\n"
    "    FM_EST       = 8\n"
    "    TP25_REPO    = 'Prior-Labs/tabpfn_2_5'                      # 40.8 MB\n"
    "    TP25_FILE    = 'tabpfn-v2.5-regressor-v2.5_quantiles.ckpt'\n"
    "    TP25_REV     = '6c45f3a6d0d07c6c5f62572e04a0c2929de91b8b'\n"
    "    TP25_SHA256  = '6dd4dbcdd5b991fde435ba5d59e4f6d170ec52110f5bb313ae8e972b88898044'\n"
    "    USE_TABM     = True\n"
    "    TABM_K       = 32\n"
    "    TABM_BINS    = 48\n"
    "    TABM_DEMB    = 16\n"
    "    TABM_LR      = 2e-3\n"
    "    TABM_WD      = 3e-4\n"
    "    TABM_BATCH   = 256\n"
    "    TABM_EPOCHS  = 40\n"
    "    TABM_PATIENCE = 5\n"
    "    TABPFN_REPO  = 'Prior-Labs/TabPFN-v2-reg'                   # 44.4 MB\n"
    "    TABPFN_FILE  = 'tabpfn-v2-regressor.ckpt'\n"
    "    TABPFN_REV   = '213f8e38ec399a2a385fa46cab6f22b95cd90de8'\n"
    "    TABPFN_SHA256 = '2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736'\n"
    "    STRICT       = True\n"
    "    MAX_WEIGHTS_MB = 200\n"
    "    KAPPA        = 0.5\n"
    "    KAPPA_SEEDS  = 3\n"
    "    ABL_MIN_GAIN  = 0.0015\n"
    "    ABL_MONTH_TOL = 0.003\n"
    "    ABL_MIN_FOLDS = 3\n"
    "    ABL_KAPPA_TOL = 0.002\n"
    "    BLEND_MIN_GAIN = 0.001\n"
    "    REGIONAL_CAL  = True\n"
    "    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "print(CFG.DATA_DIR, CFG.DEVICE, '| torch', torch.__version__)\n"
    "if CFG._ON_KAGGLE and CFG.DEVICE != 'cuda':\n"
    "    raise RuntimeError('GPU not visible to torch (TabPFN and TabM run on the GPU)')"
))

cells.append(md(
    "## Load Dataset\n\n"
    "Semua berkas resmi dibaca dengan kolom tanggal langsung di-parse. Tidak ada visualisasi di sini."
))

cells.append(code(
    "read = lambda f, **k: pd.read_csv(CFG.DATA_DIR / f, **k)\n"
    "train   = read('train.csv', parse_dates=['date_show'])\n"
    "hist    = read('test_history.csv', parse_dates=['date_show'])\n"
    "test    = read('test.csv', parse_dates=['date_show'])\n"
    "sub     = read('sample_submission.csv')\n"
    "movies  = read('movies.csv')\n"
    "hol     = read('holidays.csv', parse_dates=['date'])\n"
    "price   = read('ticket_prices.csv')\n"
    "KEY     = ['movie_title', 'cinema_ids']\n"
    "\n"
    "for n, x in [('train', train), ('test_history', hist), ('test', test), ('movies', movies),\n"
    "             ('holidays', hol), ('ticket_prices', price)]:\n"
    "    print(f'{n:14s} {x.shape}')"
))

# ---------------------------------------------------------------------------
# Section 3: Data Acquisition & Audit
# ---------------------------------------------------------------------------
cells.append(section("Data Acquisition & Audit", "3"))

cells.append(md(
    "Integritas data diperiksa sebelum analisis: rentang tanggal, nilai hilang, duplikat kunci, konsistensi "
    "klaster ke kota, dan irisan film serta klaster antara data latih dan data uji."
))

cells.append(code(
    "rows = []\n"
    "for n, x in [('train', train), ('test_history', hist), ('test', test)]:\n"
    "    rows.append(dict(file=n, rows=len(x), start=x.date_show.min().date(), end=x.date_show.max().date(),\n"
    "                     films=x.movie_title.nunique(), cinemas=x.cinema_ids.nunique(), n_missing=int(x.isna().sum().sum())))\n"
    "display(pd.DataFrame(rows))\n"
    "for n, x in [('train', train), ('test_history', hist)]:\n"
    "    print(f\"{n}: duplicate (date, cinema, film) = {x.duplicated(['date_show'] + KEY).sum()}, zero-ticket rows = {(x.total_ticket == 0).sum()}\")\n"
    "print('clusters mapped to >1 city:', (pd.concat([train, hist]).groupby('cinema_ids').city_name.nunique() > 1).sum())\n"
    "print('test ids aligned with sample_submission:', (sub.id.values == test.id.values).all())\n"
    "print('test films also in train (previews):', len(set(test.movie_title) & set(train.movie_title)))\n"
    "print('test clusters unseen in train:', len(set(test.cinema_ids) - set(train.cinema_ids)))"
))

cells.append(md(
    "#### Insights\n\n"
    "> Tidak ada nilai hilang, duplikat kunci, maupun baris bertiket nol: hari tanpa transaksi memang tidak "
    "tercatat, sehingga target harus diisi nol secara eksplisit. Sepuluh film uji muncul di train hanya "
    "sebagai pratinjau di beberapa klaster, dan empat klaster uji tidak pernah muncul di train."
))

# ---------------------------------------------------------------------------
# Section 4: EDA
# ---------------------------------------------------------------------------
cells.append(section("Exploratory Data Analysis", "4"))

cells.append(md(
    "EDA menjawab tiga pertanyaan yang langsung mengubah pipeline: bagaimana panitia membentuk data uji, "
    "periode kalender apa yang tidak pernah dilihat data latih, dan apakah level pasar periode uji berbeda."
))

cells.append(md(
    "## Aturan Seleksi Pasangan dan Filter Rilis Luas\n\n"
    "`test_history.csv` memuat 11.823 pasangan, tetapi `test.csv` hanya 10.373. Pasangan dikelompokkan menurut "
    "hari terakhir yang masih bertransaksi, lalu cakupan klaster D1 film uji dibandingkan dengan judul di "
    "`movies.csv` yang tidak dipakai."
))

cells.append(code(
    "base_title = lambda t: t.str.replace(r'\\s*\\((IMAX 2D|IMAX 3D|3D)\\)\\s*$', '', regex=True).str.strip()\n"
    "fmt = lambda t: t.str.extract(r'\\((IMAX 2D|IMAX 3D|3D)\\)\\s*$')[0].fillna('2D')\n"
    "\n"
    "d1_test = hist.groupby('movie_title').date_show.min()\n"
    "h = hist.join(d1_test.rename('d1'), on='movie_title')\n"
    "h['d'] = (h.date_show - h.d1).dt.days + 1\n"
    "ph = h.groupby(KEY).d.max().rename('last_day').reset_index()\n"
    "ph = ph.merge(test[KEY].drop_duplicates().assign(in_test=1), how='left').fillna({'in_test': 0})\n"
    "display(pd.crosstab(ph.last_day, ph.in_test, margins=True))\n"
    "nc1 = h[h.d == 1].groupby('movie_title').cinema_ids.nunique()\n"
    "print('smallest D1 coverage of 2D test films:', nc1[fmt(pd.Series(nc1.index, index=nc1.index)) == '2D'].nsmallest(4).to_dict())\n"
    "known = set(base_title(pd.Series(train.movie_title.unique()))) | set(base_title(pd.Series(hist.movie_title.unique())))\n"
    "print('titles in movies.csv used nowhere:', len(set(movies.original_title) - known))\n"
    "print('D1 weekday of test films:', d1_test.dt.day_name().value_counts().to_dict())"
))

cells.append(md(
    "#### Insights\n\n"
    "> Pasangan masuk data uji **jika dan hanya jika** bertransaksi pada D3. Film 2D terkecil di data uji dibuka "
    "di 28 klaster, sedangkan 56 judul lain (Bollywood rilis terbatas, konser, nobar) tidak dipakai: data uji "
    "hanya berisi rilis luas. D1 hampir selalu Rabu atau Kamis, yaitu tanggal rilis resmi."
))

cells.append(md(
    "## Efek Kalender dan Periode yang Tidak Ada di Data Latih\n\n"
    "Efek kalender diisolasi dengan membagi penjualan setiap pasangan-hari dengan rata-rata bergerak 7 hari "
    "terpusat milik pasangan itu sendiri, sehingga umur film dan ukuran klaster terhapus."
))

cells.append(code(
    "DOW = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']\n"
    "x = train.set_index('date_show').groupby(KEY).total_ticket.apply(lambda s: s.asfreq('D', fill_value=0)).reset_index()\n"
    "x['m7'] = x.groupby(KEY).total_ticket.transform(lambda s: s.rolling(7, center=True).mean())\n"
    "x = x[(x.m7 >= 20) & ~x.date_show.isin(CFG.OUTAGE)].merge(hol, left_on='date_show', right_on='date', how='left')\n"
    "x['mult'] = x.total_ticket / x.m7\n"
    "day = x.groupby('date_show').agg(mult=('mult', 'median'), hol=('holiday_tipe', 'first'), name=('holiday_name', 'first'))\n"
    "day['dow'] = day.index.dayofweek\n"
    "prof = day[day.hol == 'normal'].groupby('dow').mult.median()\n"
    "day['excess'] = day.mult / day.dow.map(prof)\n"
    "print('weekday multiplier:', dict(zip(DOW, prof.round(3))))\n"
    "display(day[day.hol == 'holiday'][['name', 'dow', 'excess']].round(2))\n"
    "\n"
    "SEGMENTS = {'xmas': ('2025-12-20', '2026-01-04'), 'ramadan': ('2026-02-19', '2026-03-20'), 'lebaran': ('2026-03-21', '2026-03-29')}\n"
    "for n, (a, b) in SEGMENTS.items():\n"
    "    m = test.date_show.between(a, b)\n"
    "    print(f'{n:8s} test rows {m.sum():6d} ({m.mean():.1%}), films {test[m].movie_title.nunique()}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))\n"
    "ax[0].bar(DOW, prof.values, color=PAL[0]); ax[0].set(title='Pengali hari (train, non-libur)')\n"
    "cnt = test.groupby('date_show').size()\n"
    "ax[1].bar(cnt.index, cnt.values, width=1, color=PAL[0])\n"
    "for (n, (a, b)), c in zip(SEGMENTS.items(), [PAL[3], PAL[6], PAL[7]]):\n"
    "    ax[1].axvspan(pd.Timestamp(a), pd.Timestamp(b), color=c, alpha=.2, label=n)\n"
    "ax[1].set(title='Baris target test per tanggal'); ax[1].legend()\n"
    "fig.autofmt_xdate(); plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'calendar.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Sabtu dan Minggu sekitar 1,6 kali hari kerja, libur di hari kerja setara akhir pekan, libur di akhir "
    "pekan hampir tidak menambah. Sekitar 30% baris uji berada di periode yang tidak pernah ada di train: "
    "Ramadan (17,1%), libur Natal dan Tahun Baru (7,2%), dan Lebaran (6,1%). Efek kalender karena itu dimasukkan "
    "secara struktural sebagai pengali target, bukan dipelajari pohon keputusan yang tidak bisa ekstrapolasi."
))

cells.append(md(
    "## Level Pasar dan Ukuran Pasangan\n\n"
    "Okupansi median per bulan dan distribusi skala pasangan D1 sampai D3 dibandingkan antara train dan data uji. "
    "Skala dihitung dengan rumus resmi pada seluruh pasangan yang laku di D3."
))

cells.append(code(
    "occ = pd.concat([train.assign(src='train'), hist.assign(src='test_history')])\n"
    "occ_m = occ.groupby([occ.date_show.dt.to_period('M'), 'src']).occupation_rate.median().unstack()\n"
    "s_te = (hist.groupby(KEY).total_ticket.sum() / 3).clip(lower=1).reindex(pd.MultiIndex.from_frame(test[KEY].drop_duplicates()))\n"
    "bins = CFG.TW_BINS\n"
    "print('test pairs by scale bucket:', pd.cut(s_te, bins).value_counts(normalize=True).sort_index().round(3).to_dict())\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(14, 3.6))\n"
    "occ_m.plot.bar(ax=ax[0], rot=45, color=[PAL[1], PAL[0]]); ax[0].set(title='Okupansi median per bulan', ylabel='%')\n"
    "ax[1].hist(np.log10(s_te), bins=50, color=PAL[0]); ax[1].set(title='log10 skala pasangan uji')\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'market.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Okupansi median Oktober 2025 (7,6%), Februari (5,5%), dan Maret 2026 (6,4%) berada di bawah semua bulan "
    "train (13,6% sampai 41,8%). Pasar periode uji jauh lebih sepi, sehingga pasangan uji jauh lebih kecil: "
    "13% baris uji memiliki skala di bawah 20. Konsekuensinya diukur di Bagian 6 dan Bagian 8."
))

# ---------------------------------------------------------------------------
# Section 5: Data Cleaning
# ---------------------------------------------------------------------------
cells.append(section("Data Cleaning", "5"))

cells.append(md(
    "Target dibentuk dengan zero-fill, sehingga hari ketika sistem pelaporan mati akan menjadi nol palsu. "
    "Selain itu, akhir pekan pratinjau berbayar sebelum rilis resmi dapat terbaca sebagai D1."
))

cells.append(code(
    "rep = train.groupby('date_show').cinema_ids.nunique().reindex(pd.date_range('2025-04-01', CFG.TRAIN_END))\n"
    "print('dates with no data:', rep[rep.isna()].index.date.tolist())\n"
    "print('dates with < 70% of clusters reporting:', rep[rep < 0.7 * rep.median()].to_dict())\n"
    "mc = train[train.movie_title == 'A MINECRAFT MOVIE'].groupby('date_show').cinema_ids.nunique()\n"
    "print('A MINECRAFT MOVIE coverage 1-10 Apr:', mc.loc[:'2025-04-10'].to_dict())\n"
    "fig, ax = plt.subplots(figsize=(12, 3))\n"
    "ax.plot(rep.index, rep.fillna(0), lw=1.2)\n"
    "ax.scatter(CFG.OUTAGE, rep.reindex(CFG.OUTAGE).fillna(0), color=PAL[7], zorder=3, label='outage')\n"
    "ax.set(title='Klaster yang melapor per hari (train)'); ax.legend()\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'outage.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Lima tanggal Juni 2025 adalah outage pelaporan (13 Juni kosong, 9 Juni hanya 1 klaster). Baris target "
    "pada tanggal tersebut dibuang dan film yang D1 sampai D3-nya menyentuh outage tidak dijadikan sampel. "
    "A MINECRAFT MOVIE memiliki pratinjau 4 sampai 6 April lalu jeda sebelum rilis resmi 9 April: D1 dicari "
    "hanya di dalam segmen tayang kontinu yang memuat puncak cakupan."
))

# ---------------------------------------------------------------------------
# Section 6: Preprocessing
# ---------------------------------------------------------------------------
cells.append(section("Preprocessing", "6"))

cells.append(md(
    "`train.csv` tidak memiliki penanda D1 sampai D10. Aturan panitia direplikasi: D1 rilis resmi, rilis luas, "
    "pasangan laku di D3, target nol diisi eksplisit. Hasilnya diverifikasi terhadap data uji, termasuk dengan "
    "tugas proxy yang memiliki label di dalam periode uji."
))

cells.append(md(
    "## Rekonstruksi D1 dan Simulasi Sampel\n\n"
    "D1 adalah hari pertama cakupan klaster judul dasar mencapai 50% puncak, dicari di dalam segmen tayang "
    "kontinu yang memuat puncak. Judul dengan cakupan D1 di bawah 25 klaster dipisahkan sebagai film rilis "
    "terbatas: tidak pernah dievaluasi, tetapi dipakai sebagai data latih tambahan untuk pasangan kecil."
))

cells.append(code(
    "def release_dates(tx, frac=CFG.D1_FRAC, min_nc=CFG.MIN_NC):\n"
    "    x = tx.assign(base=base_title(tx.movie_title))\n"
    "    nc = x.groupby(['base', 'date_show']).cinema_ids.nunique()\n"
    "    out = {}\n"
    "    for b, s in nc.groupby(level=0):\n"
    "        s = s.droplevel(0)\n"
    "        s = s.reindex(pd.date_range(s.index.min(), s.index.max()), fill_value=0)\n"
    "        pk = s.idxmax()\n"
    "        z = s[:pk][s[:pk] == 0]\n"
    "        seg = s[z.index.max() + pd.Timedelta(days=1):] if len(z) else s\n"
    "        c = seg[seg >= frac * s.max()]\n"
    "        if len(c) and c.iloc[0] >= min_nc:\n"
    "            out[b] = c.index[0]\n"
    "    t = x.drop_duplicates('movie_title').set_index('movie_title').base\n"
    "    return t.map(pd.Series(out, dtype='datetime64[ns]')).dropna().rename('d1')\n"
    "\n"
    "\n"
    "def simulate(tx, d1, last_date=CFG.TRAIN_END, bad=CFG.OUTAGE):\n"
    "    d1 = d1[d1 + pd.Timedelta(days=9) <= last_date]\n"
    "    if len(bad):\n"
    "        d1 = d1[~np.any([(d1 <= b) & (d1 + pd.Timedelta(days=2) >= b) for b in bad], axis=0)]\n"
    "    x = tx.merge(d1, left_on='movie_title', right_index=True)\n"
    "    x['d'] = (x.date_show - x.d1).dt.days + 1\n"
    "    hs = x[x.d.between(1, 3)].drop(columns=['d1', 'd']).reset_index(drop=True)\n"
    "    pairs = x.loc[x.d == 3, KEY].drop_duplicates().merge(d1, left_on='movie_title', right_index=True)\n"
    "    tg = pairs.loc[pairs.index.repeat(7)].copy()\n"
    "    tg['date_show'] = tg.d1 + pd.to_timedelta(np.tile(np.arange(3, 10), len(pairs)), unit='D')\n"
    "    y = x[x.d.between(4, 10)][KEY + ['date_show', 'total_ticket']]\n"
    "    tg = tg.merge(y, on=KEY + ['date_show'], how='left').fillna({'total_ticket': 0})\n"
    "    tg = tg[~tg.date_show.isin(bad)]\n"
    "    tg['city_name'] = tg.cinema_ids.map(tx.drop_duplicates('cinema_ids').set_index('cinema_ids').city_name)\n"
    "    return hs, tg.drop(columns='d1').reset_index(drop=True)\n"
    "\n"
    "\n"
    "def pair_scale(h):\n"
    "    return (h.groupby(KEY).total_ticket.sum() / 3).clip(lower=1).rename('scale')\n"
    "\n"
    "\n"
    "def mase(y, p, s):\n"
    "    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p)) / np.asarray(s)))\n"
    "\n"
    "\n"
    "first = train.groupby('movie_title').date_show.min()\n"
    "running = first[first == first.min()].index\n"
    "D_wide = release_dates(train)\n"
    "D_lim = release_dates(train, CFG.D1_FRAC, 0).drop(D_wide.index, errors='ignore').drop(running, errors='ignore')\n"
    "hs, ts = simulate(train, D_wide.drop(running, errors='ignore'))\n"
    "hl, tl = simulate(train, D_lim)\n"
    "print(f'wide releases: {ts.movie_title.nunique()} films, {len(ts)} target rows | limited (train only): {tl.movie_title.nunique()} films, {len(tl)} rows')\n"
    "print('A MINECRAFT MOVIE D1:', D_wide['A MINECRAFT MOVIE'].date())"
))

cells.append(md(
    "Fungsi `simulate` diuji pada film sintetis: klaster B tidak laku di D3 sehingga harus dibuang, dan hari "
    "tanpa transaksi harus menjadi nol."
))

cells.append(code(
    "dd = pd.date_range('2025-05-01', periods=12)\n"
    "toy = pd.DataFrame({'date_show': list(dd) + [dd[0], dd[1]], 'cinema_ids': ['A'] * 12 + ['B', 'B'], 'city_name': 'X',\n"
    "                    'movie_title': 'M', 'total_ticket': list(range(1, 13)) + [5, 5], 'occupation_rate': 1.0,\n"
    "                    'total_show': 1}).drop(index=5)\n"
    "th_, tg_ = simulate(toy, pd.Series({'M': dd[0]}, name='d1'), bad=pd.DatetimeIndex([]))\n"
    "assert set(tg_.cinema_ids) == {'A'} and tg_.total_ticket.tolist() == [4, 5, 0, 7, 8, 9, 10]\n"
    "assert np.isclose(pair_scale(th_)[('M', 'B')], 10 / 3)\n"
    "print('simulate() self-check passed')"
))

cells.append(md(
    "#### Insights\n\n"
    "> Aturan D1 memakai puncak cakupan sepanjang riwayat film, yaitu informasi masa depan. Audit lokal (`eda/53`) "
    "membandingkannya dengan aturan yang benar-benar kausal: D1 = hari pertama yang jendela D1 sampai D3-nya sendiri "
    "memiliki minimal 25 klaster setiap hari, tanpa melihat puncak. Aturan itu memindahkan 6 film (A MINECRAFT MOVIE "
    "ke pratayang berbayar 4 April, BELIEVE 20 hari lebih awal), mengubah skala 191 pasangan bersama (varian yang masih "
    "memakai segmen puncak: 125), dan membuang 15 film yang dibuka di minimal 25 klaster lalu runtuh di D2 sampai D3 "
    "(rasio cakupan D3/D1 median 0,34, misalnya ARTI CINTA 52 menjadi 2 klaster). Jarak KS rasio cakupan ke data uji "
    "memang turun (0,196 menjadi 0,106), tetapi hanya karena ekor bawah itu dibuang.\n\n"
    "> Data uji justru memuat pembukaan yang runtuh seperti itu: 8 film uji punya hari dengan kurang dari 25 klaster "
    "(SEND HELP 52, 49, lalu 1 klaster; G-DRAGON IN CINEMA 55, 30, 3) dan 30 film uji memiliki rasio cakupan D3/D1 di "
    "bawah 0,8. Seleksi panitia karena itu konsisten dengan aturan cakupan D1 (aturan sekarang), bukan dengan syarat "
    "jendela lebar tiga hari. Aturan D1 dipertahankan dengan dasar ini, bukan karena perubahannya kecil."
))

cells.append(md(
    "## Verifikasi: Komposisi dan Periode Uji\n\n"
    "Sampel simulasi dibandingkan dengan data uji. Karena target uji tidak tersedia, dibuat tugas proxy yang "
    "memiliki label di **dalam periode uji**: memprediksi D3 dari D1 dan D2. Model proxy dilatih pada train, "
    "lalu galatnya per bucket skala dibandingkan antara validasi silang train dan periode uji."
))

cells.append(code(
    "def proxy(hh):\n"
    "    d1 = hh.groupby('movie_title').date_show.min().rename('d1')\n"
    "    x = hh.join(d1, on='movie_title')\n"
    "    x['d'] = (x.date_show - x.d1).dt.days + 1\n"
    "    P = x.pivot_table(index=KEY, columns='d', values='total_ticket', fill_value=0).reindex(columns=[1, 2, 3], fill_value=0)\n"
    "    S = x.pivot_table(index=KEY, columns='d', values='total_show', fill_value=0).reindex(columns=[1, 2], fill_value=0)\n"
    "    O = x.pivot_table(index=KEY, columns='d', values='occupation_rate', fill_value=0).reindex(columns=[1, 2], fill_value=0)\n"
    "    P = P[P[2] > 0]\n"
    "    f = pd.DataFrame({'s': ((P[1] + P[2]) / 2).clip(lower=1), 'y': P[3]}, index=P.index)\n"
    "    f['q1'], f['q2'], f['log_s'] = P[1] / f.s, P[2] / f.s, np.log1p(f.s)\n"
    "    f['sh1'], f['sh2'] = S.reindex(P.index)[1], S.reindex(P.index)[2]\n"
    "    f['sh_tr'] = (f.sh2 + 1) / (f.sh1 + 1)\n"
    "    f['occ2'] = O.reindex(P.index)[2]\n"
    "    f = f.reset_index().join(d1, on='movie_title')\n"
    "    g = x[x.d <= 2].groupby(['movie_title', 'd']).total_ticket.sum().unstack().reindex(columns=[1, 2], fill_value=0)\n"
    "    f['f_tr'] = f.movie_title.map((g[2] + 1) / (g[1] + 1))\n"
    "    f['f_log'] = f.movie_title.map(np.log1p(g.mean(axis=1)))\n"
    "    f['f_nc'] = f.movie_title.map(x[x.d == 2].groupby('movie_title').cinema_ids.nunique())\n"
    "    f['d1_dow'] = f.d1.dt.dayofweek\n"
    "    return f\n"
    "\n"
    "\n"
    "PA, PB = proxy(hs), proxy(hist)\n"
    "pc = ['q1', 'q2', 'log_s', 'sh_tr', 'f_tr', 'f_log', 'd1_dow']\n"
    "pm = dict(objective='l1', n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100, verbose=-1, deterministic=True, force_row_wise=True)\n"
    "oofA = np.zeros(len(PA))\n"
    "for a, b in GroupKFold(5).split(PA, groups=base_title(PA.movie_title)):\n"
    "    m = lgb.LGBMRegressor(**pm, random_state=CFG.SEED).fit(PA.iloc[a][pc], PA.y.iloc[a] / PA.s.iloc[a])\n"
    "    oofA[b] = np.clip(m.predict(PA.iloc[b][pc]), 0, None) * PA.s.values[b]\n"
    "m = lgb.LGBMRegressor(**pm, random_state=CFG.SEED).fit(PA[pc], PA.y / PA.s)\n"
    "pB = np.clip(m.predict(PB[pc]), 0, None) * PB.s.values\n"
    "eA, eB = np.abs(PA.y - oofA) / PA.s, np.abs(PB.y - pB) / PB.s\n"
    "cutA, cutB = pd.cut(PA.s, bins), pd.cut(PB.s, bins)\n"
    "tab = pd.DataFrame({'train_share': cutA.value_counts(normalize=True), 'test_share': cutB.value_counts(normalize=True),\n"
    "                    'train_cv_mase': eA.groupby(cutA, observed=False).mean(), 'test_period_mase': eB.groupby(cutB, observed=False).mean(),\n"
    "                    'train_zero_D3': (PA.y == 0).groupby(cutA, observed=False).mean(), 'test_zero_D3': (PB.y == 0).groupby(cutB, observed=False).mean()}).sort_index()\n"
    "display(tab.round(3))\n"
    "print(f'proxy MASE: train CV {eA.mean():.4f} | re-weighted to test scale mix {(tab.test_share * tab.train_cv_mase).sum():.4f} | real test period {eB.mean():.4f}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))\n"
    "xx = np.arange(len(tab))\n"
    "ax[0].bar(xx - .2, tab.train_cv_mase, .4, label='train CV'); ax[0].bar(xx + .2, tab.test_period_mase, .4, label='periode uji')\n"
    "ax[0].set_xticks(xx, [str(i) for i in tab.index], rotation=30); ax[0].set(title='Proxy MASE per bucket skala'); ax[0].legend()\n"
    "ax[1].bar(xx - .2, tab.train_zero_D3, .4, label='train'); ax[1].bar(xx + .2, tab.test_zero_D3, .4, label='periode uji')\n"
    "ax[1].set_xticks(xx, [str(i) for i in tab.index], rotation=30); ax[1].set(title='Porsi D3 = 0 pada skala yang sama'); ax[1].legend()\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'proxy.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Galat per bucket skala hampir sama antara validasi silang train dan periode uji, tetapi data uji memuat "
    "jauh lebih banyak pasangan kecil yang galatnya terbesar. Karena itu metrik keputusan adalah **TW-MASE**. "
    "TW-MASE v2 (0,377) dan v3 (0,388) memberi peringkat yang sama dengan leaderboard (0,4506 dan 0,4573), "
    "tetapi keduanya berjarak tetap sekitar 0,07.\n\n"
    "> Pada skala absolut yang sama, porsi pasangan yang berhenti di D3 jauh lebih kecil di periode uji "
    "(skala 20 sampai 50: 0,39 di train vs 0,11 di uji). Bagian berikut menguji apakah ini efek musim atau "
    "perubahan perilaku bioskop."
))

cells.append(md(
    "## Kebijakan Pencopotan di Periode Uji\n\n"
    "Porsi pasangan yang berhenti laku di D3 dibandingkan pada **permintaan per pertunjukan yang sama** "
    "(tiket per show D1 sampai D2), sehingga perbedaan ukuran pasar tidak ikut terhitung. Besarnya pergeseran "
    "diestimasi di Bagian 8 setelah kalender tersedia."
))

cells.append(code(
    "DA = PA.assign(z=PA.y == 0, tps=PA.s * 2 / (PA.sh1 + PA.sh2).clip(lower=1))\n"
    "DB = PB.assign(z=PB.y == 0, tps=PB.s * 2 / (PB.sh1 + PB.sh2).clip(lower=1))\n"
    "q = pd.qcut(pd.concat([DA.tps, DB.tps]), 5)\n"
    "qa, qb = q.iloc[:len(DA)].values, q.iloc[len(DA):].values\n"
    "tq = pd.DataFrame({'train_drop_D3': DA.z.groupby(qa, observed=False).mean(), 'test_drop_D3': DB.z.groupby(qb, observed=False).mean(),\n"
    "                   'train_n': pd.Series(qa).value_counts(), 'test_n': pd.Series(qb).value_counts()})\n"
    "display(tq.round(3))\n"
    "zm = pd.concat([DA.assign(m=DA.d1.dt.to_period('M')), DB.assign(m=DB.d1.dt.to_period('M'))]).groupby('m').z.mean()\n"
    "print('D3 stop rate by release month:', zm.round(3).to_dict())\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))\n"
    "xx = np.arange(len(tq))\n"
    "ax[0].bar(xx - .2, tq.train_drop_D3, .4, label='train'); ax[0].bar(xx + .2, tq.test_drop_D3, .4, label='periode uji')\n"
    "ax[0].set_xticks(xx, [f'q{i + 1}' for i in xx]); ax[0].set(title='Porsi berhenti di D3 per kuintil tiket/show'); ax[0].legend()\n"
    "ax[1].bar(zm.index.astype(str), zm.values, color=[PAL[0] if str(m) < '2025-10' else PAL[7] for m in zm.index])\n"
    "ax[1].set(title='Porsi berhenti di D3 per bulan D1 (biru train, merah uji)'); ax[1].tick_params(axis='x', rotation=45)\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'pull_policy.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Pada tiket per show yang sama, porsi pasangan yang berhenti di D3 di periode uji hanya sekitar setengah "
    "sampai seperempat porsi train, di setiap bulan uji, termasuk November dan Desember yang level pasarnya "
    "setara bulan-bulan train. Ini bukan efek musim dan bukan data bolong (pola lubang D1 sampai D3 sama), "
    "melainkan bioskop di periode uji mempertahankan film pada tingkat penonton yang lebih rendah. Odds "
    "pencopotan di D3 sekitar 0,27 kali train, stabil lintas bulan.\n\n"
    "> Semua model yang dilatih di train mewarisi kebijakan lama yang lebih cepat mencopot, kandidat kuat "
    "penyebab offset leaderboard yang sama di v2 dan v3. Namun di train, kecenderungan mencopot di D3 hampir "
    "tidak memprediksi pencopotan D4 sampai D10 (korelasi 0,13 lintas klaster dan bulan), sehingga besarnya "
    "pergeseran di horizon target tidak bisa dipastikan. Porsi yang diterapkan dipilih dengan analisis "
    "keputusan di Bagian 8."
))

# ---------------------------------------------------------------------------
# Section 7: Feature Engineering
# ---------------------------------------------------------------------------
cells.append(section("Feature Engineering", "7"))

cells.append(md(
    "Satu fungsi `build` dipanggil identik untuk sampel simulasi dan data uji. Fitur hanya memakai informasi "
    "yang tersedia pada D3: riwayat D1 sampai D3, kalender resmi, jadwal rilis film lain yang terlihat di "
    "jendela D1 sampai D3 masing-masing, statistik klaster dari masa lalu, dan metadata film."
))

cells.append(md(
    "## Data Eksternal: Kalender Resmi\n\n"
    "Hanya sumber yang publik paling lambat 30 September 2025 yang dipakai, seluruhnya fakta kalender pemerintah.\n\n"
    "| Sumber | Penerbit | Tanggal terbit | Dipakai untuk |\n"
    "| :--- | :--- | :--- | :--- |\n"
    "| [SKB 3 Menteri Libur Nasional dan Cuti Bersama 2025](https://www.kemenkopmk.go.id/skb-3-menteri-libur-nasional-dan-cuti-bersama-tahun-2025) "
    "| Kemenko PMK | Oktober 2024 (revisi Agustus 2025) | cuti bersama 2025 |\n"
    "| [SKB 3 Menteri Libur Nasional dan Cuti Bersama 2026](https://setneg.go.id/baca/index/inilah_skb_3_menteri_libur_nasional_dan_cuti_bersama_2026) "
    "| Kemensetneg | 19 September 2025 | cuti bersama 2026; 1 Syawal 1447 H = 21 Maret 2026 |\n"
    "| [Kalender Pendidikan DKI Jakarta 2024/2025](https://tirto.id/kalender-pendidikan-dki-jakarta-tahun-2024-2025-link-unduh-pdf-g1vR) "
    "| Disdik DKI via Tirto | 2024 | libur 28 Juni sampai 12 Juli 2025 |\n"
    "| [Kalender Pendidikan DKI Jakarta 2025/2026 (Kepdis 89/2025)](https://news.detik.com/berita/d-8049756/kalender-pendidikan-semester-ganjil-2025-2026-jakarta-cek-tanggal-pentingnya) "
    "| Disdik DKI via Detik | 7 Agustus 2025 | libur 22 sampai 31 Desember 2025 |\n"
    "| [SE Kadisdik Jabar 21808/PK.02.01.05/Sekre, Pedoman Kalender Pendidikan 2024/2025](https://mmc.tirto.id/documents/2024/06/24/2479-surat-plh-kadisdik-jabar-kalender-pendidikan-2024-2025-1-14062024-025316-signed.pdf) "
    "| Disdik Jawa Barat | 10 Juni 2024 | Jawa Barat: libur akhir tahun pelajaran 30 Juni sampai 12 Juli 2025 |\n"
    "| [SE Kadisdik Jabar 14995/TU.03/PSMA, Pedoman Kalender Pendidikan 2025/2026](https://www.komunitasbelajar.id/2025/06/kalender-pendidikan-provinsi-jawa-barat.html) "
    "([turunan Disdik Ciamis, Juli 2025](https://disdik.ciamiskab.go.id/storage/2025/07/Kaldik_2025-2026_SD-SMP_DISDIK_FIX.pdf)) "
    "| Disdik Jawa Barat | 20 Juni 2025 | Jawa Barat: libur semester ganjil 29 Desember 2025 sampai 10 Januari 2026 |\n\n"
    "Ramadan 1447 H (19 Februari sampai 20 Maret 2026) diturunkan dari 1 Syawal pada SKB 2026 dikurangi 30 hari."
))

cells.append(md(
    "## Pengali Kalender Struktural\n\n"
    "Nilai kalender $c(t)$ = profil hari, dinaikkan ke level akhir pekan pada libur nasional atau cuti bersama "
    "di luar Ramadan (cuti bersama akhir Ramadan adalah masa mudik), dan dikali 1,15 pada hari kerja libur "
    "sekolah. Pengali target $c_{mult} = c(t) / \\overline{c(D1..D3)}$.\n\n"
    "Kalender sekolah DKI dipakai untuk semua klaster kecuali Jawa Barat (22% baris uji), yang memakai kalender Disdik "
    "Jabar. Provinsi besar lain yang dicek (Banten, Jawa Tengah, Jawa Timur, DIY, Bali, Lampung, Sulawesi Selatan) libur "
    "20 atau 22 Desember sampai 1 sampai 4 Januari: jendela hari kerja (Senin sampai Kamis) libur sekolahnya sama dengan "
    "DKI, karena 25 dan 26 Desember serta 1 Januari sudah libur nasional atau cuti bersama. Jawa Barat berbeda: 22 sampai "
    "24 Desember masih hari sekolah, sedangkan 5 sampai 9 Januari libur. Di periode latih, jendela hari kerja Jawa Barat "
    "(30 Juni sampai 12 Juli) identik dengan DKI: $c_{mult}$ latih tidak berubah, hanya flag `t_school` pada akhir pekan "
    "28 sampai 29 Juni (202 baris latih) yang berbeda."
))

cells.append(code(
    "SCHOOL_BREAKS = [('2025-06-28', '2025-07-12'), ('2025-12-22', '2025-12-31')]\n"
    "JABAR_BREAKS = [('2025-06-30', '2025-07-12'), ('2025-12-29', '2026-01-10')]\n"
    "JABAR_CITIES = ['BOGOR', 'BEKASI', 'BANDUNG', 'DEPOK', 'CIKARANG', 'KARAWANG', 'CIREBON', 'GARUT', 'TASIKMALAYA',\n"
    "                'SUMEDANG', 'CIANJUR', 'INDRAMAYU']\n"
    "CUTI_BERSAMA = pd.to_datetime(['2025-04-02', '2025-04-03', '2025-04-04', '2025-04-07', '2025-05-13', '2025-05-30',\n"
    "                               '2025-06-09', '2025-08-18', '2025-12-26', '2026-02-16', '2026-03-18', '2026-03-20',\n"
    "                               '2026-03-23', '2026-03-24'])\n"
    "RAMADAN = ('2026-02-19', '2026-03-20')\n"
    "\n"
    "\n"
    "def calendar(hol, breaks=SCHOOL_BREAKS):\n"
    "    c = hol.rename(columns={'date': 'date_show'})[['date_show', 'holiday_tipe']].copy()\n"
    "    c['dow'] = c.date_show.dt.dayofweek\n"
    "    c['is_hol'] = ((c.holiday_tipe == 'holiday') | c.date_show.isin(CUTI_BERSAMA)).astype(int)\n"
    "    c['school'] = 0\n"
    "    for a, b in breaks:\n"
    "        c.loc[c.date_show.between(a, b), 'school'] = 1\n"
    "    c['ramadan'] = c.date_show.between(*RAMADAN).astype(int)\n"
    "    base = CFG.DOW_PROF[c.dow]\n"
    "    base = np.where((c.school == 1) & (c.dow < 4), base * CFG.SCHOOL_WD, base)\n"
    "    c['cal'] = np.where((c.is_hol == 1) & (c.ramadan == 0), np.maximum(base, CFG.HOL_LEVEL), base)\n"
    "    return c.drop(columns='holiday_tipe').set_index('date_show')\n"
    "\n"
    "\n"
    "CAL = calendar(hol)\n"
    "CAL_JB = calendar(hol, JABAR_BREAKS)\n"
    "fig, ax = plt.subplots(figsize=(14, 3))\n"
    "ax.plot(CAL.index, CAL.cal, lw=1)\n"
    "ax.axvline(CFG.TRAIN_END, color=PAL[7], ls='--', label='akhir train')\n"
    "ax.set(title='Nilai kalender c(t)'); ax.legend()\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'cal_value.png'); plt.show()"
))

cells.append(md(
    "## Fungsi Build Fitur\n\n"
    "Kelompok fitur: (1) bentuk kurva pasangan D1 sampai D3, (2) agregat film nasional dan pangsa format, "
    "(3) kalender target dan riwayat, (4) jadwal rilis film lain, termasuk film baru yang dibuka di klaster yang "
    "sama pada tanggal target (terlihat di jendela D1 sampai D3 film tersebut), (5) ukuran klaster dan harga kota, "
    "(6) metadata film, (7) level relatif pasar (median film yang rilis 28 hari sebelumnya, gabungan jendela "
    "D1 sampai D3 train dan uji)."
))

cells.append(code(
    "GENRES = ['Horror', 'Drama', 'Action', 'Comedy', 'Romance', 'Animation', 'Family', 'Thriller', 'Mystery']\n"
    "RATING = {'Semua Umur': 0, 'Remaja': 1, 'Dewasa': 2, 'Dewasa 21': 3}\n"
    "\n"
    "\n"
    "def visible_windows(tx, d1):\n"
    "    x = tx.merge(d1.rename('d1'), left_on='movie_title', right_index=True)\n"
    "    return x[(x.date_show >= x.d1) & (x.date_show <= x.d1 + pd.Timedelta(days=2))].drop(columns='d1')\n"
    "\n"
    "\n"
    "def film_table(vis):\n"
    "    g = vis.groupby('movie_title')\n"
    "    return pd.DataFrame({'d1': g.date_show.min(), 'occ': g.occupation_rate.mean(),\n"
    "                         'tps': g.total_ticket.sum() / g.total_show.sum().clip(lower=1),\n"
    "                         'logT': np.log1p(g.total_ticket.sum() / 3),\n"
    "                         'lpc': np.log1p(g.total_ticket.sum() / 3 / g.cinema_ids.nunique().clip(lower=1))})\n"
    "\n"
    "\n"
    "def make_ctx(visible, size_tx, market):\n"
    "    rel = visible.assign(base=base_title(visible.movie_title)).groupby('base').agg(\n"
    "        d1=('date_show', 'min'), tix=('total_ticket', 'sum')).reset_index()\n"
    "    rel['tix'] /= 3\n"
    "    cs = size_tx.groupby('cinema_ids').total_ticket.sum() / size_tx.groupby('cinema_ids').date_show.nunique()\n"
    "    nf = size_tx.groupby(['cinema_ids', 'date_show']).movie_title.nunique().groupby('cinema_ids').mean()\n"
    "    csh = size_tx.groupby(['cinema_ids', 'date_show']).total_show.sum().groupby('cinema_ids').median()\n"
    "    return dict(releases=rel, market=market, visible=visible, cin_size=np.log10(cs), cin_nfilms=nf, cin_shows=csh)\n"
    "\n"
    "\n"
    "def cinema_comp(X, vis, cin_shows):\n"
    "    v = vis.assign(base=base_title(vis.movie_title))\n"
    "    agg = v.groupby(['cinema_ids', 'date_show', 'base']).agg(sh=('total_show', 'sum'), tx=('total_ticket', 'sum')).reset_index()\n"
    "    m = X[['cinema_ids', 'date_show']].assign(base=base_title(X.movie_title)).reset_index().merge(\n"
    "        agg, on=['cinema_ids', 'date_show'], how='left', suffixes=('', '_o'))\n"
    "    m = m[m.base != m.base_o]\n"
    "    g = m.groupby('index').agg(c_new_n=('base_o', 'nunique'), c_new_sh=('sh', 'sum'), c_new_tx=('tx', 'sum'))\n"
    "    X = X.join(g).fillna({'c_new_n': 0, 'c_new_sh': 0, 'c_new_tx': 0})\n"
    "    X['c_new_sh_rel'] = X.c_new_sh / X.cinema_ids.map(cin_shows).fillna(X.c_new_sh.median() + 1)\n"
    "    X['c_new_sh_vs_own'] = X.c_new_sh / X.sh3.clip(lower=1)\n"
    "    X['c_new_tx_vs_own'] = np.log1p(X.c_new_tx) - np.log1p(X.y3)\n"
    "    return X\n"
    "\n"
    "\n"
    "def open_count(X, rel):\n"
    "    d = np.sort(rel.d1.values.astype('datetime64[D]'))\n"
    "    a = np.searchsorted(d, X.d1.values.astype('datetime64[D]'), side='right')\n"
    "    b = np.searchsorted(d, X.date_show.values.astype('datetime64[D]'), side='right')\n"
    "    return b - a\n"
    "\n"
    "\n"
    "def build(hh, tgt, ctx):\n"
    "    mv = movies.set_index('original_title')\n"
    "    d1 = hh.groupby('movie_title').date_show.min().rename('d1')\n"
    "    hh = hh.join(d1, on='movie_title')\n"
    "    hh['d'] = (hh.date_show - hh.d1).dt.days + 1\n"
    "    piv = lambda v, p: hh.pivot_table(index=KEY, columns='d', values=v, fill_value=0).reindex(columns=[1, 2, 3], fill_value=0).add_prefix(p)\n"
    "    P = pd.concat([piv('total_ticket', 'y'), piv('total_show', 'sh')], axis=1)\n"
    "    P['mean3'] = P[['y1', 'y2', 'y3']].mean(axis=1)\n"
    "    P['scale'] = P.mean3.clip(lower=1)\n"
    "    P['log_s'] = np.log1p(P.mean3)\n"
    "    for i in (1, 2, 3):\n"
    "        P[f'p{i}'] = P[f'y{i}'] / P.scale\n"
    "    P['n_hist'] = (P[['y1', 'y2', 'y3']] > 0).sum(axis=1)\n"
    "    P['sh_trend'] = P.sh3 / P[['sh1', 'sh2']].max(axis=1).clip(lower=1)\n"
    "    P = P.reset_index()\n"
    "\n"
    "    Fd = hh.groupby(['movie_title', 'd']).agg(T=('total_ticket', 'sum'), nc=('cinema_ids', 'nunique')).unstack().fillna(0)\n"
    "    Fd.columns = [f'f{a}{b}' for a, b in Fd.columns]\n"
    "    Fd = Fd.reindex(columns=[f'f{a}{b}' for a in ['T', 'nc'] for b in (1, 2, 3)], fill_value=0)\n"
    "    Fm = Fd[['fT1', 'fT2', 'fT3']].mean(axis=1).clip(lower=1)\n"
    "    for i in (1, 2, 3):\n"
    "        Fd[f'fp{i}'] = Fd[f'fT{i}'] / Fm\n"
    "    ncmax = Fd[['fnc1', 'fnc2', 'fnc3']].max(axis=1).clip(lower=1)\n"
    "    Fd['f_logT'] = np.log1p(Fm)\n"
    "    Fd['f_nc_trend'] = Fd.fnc3 / Fd.fnc1.clip(lower=1)\n"
    "    Fd['f_occ'] = hh.groupby('movie_title').occupation_rate.mean()\n"
    "    h3 = hh[hh.d == 3].groupby('movie_title')\n"
    "    Fd['f_tps3'] = h3.total_ticket.sum() / h3.total_show.sum()\n"
    "    Fd = Fd.join(d1)\n"
    "    Fd['base'] = base_title(pd.Series(Fd.index, index=Fd.index))\n"
    "    Fd['fmt'] = fmt(pd.Series(Fd.index, index=Fd.index)).map({'2D': 0, '3D': 1, 'IMAX 2D': 2, 'IMAX 3D': 3})\n"
    "    bT = hh.assign(base=base_title(hh.movie_title)).groupby('base').total_ticket.sum() / 3\n"
    "    Fd['fmt_share'] = np.log1p(Fm) - np.log1p(Fd.base.map(bT))\n"
    "    Fd['d1_dow'] = Fd.d1.dt.dayofweek\n"
    "    m = mv.reindex(Fd.base)\n"
    "    Fd['rating'] = m.age_rating.map(RATING).values\n"
    "    for g_ in GENRES:\n"
    "        Fd[f'g_{g_}'] = m.genre.fillna('').str.contains(g_).astype(int).values\n"
    "    Fd['n_genre'] = m.genre.fillna('').str.count(',').values + 1\n"
    "    Fd['n_cast'] = m.casts.fillna('').str.count(',').values + 1\n"
    "    rel = ctx['releases']\n"
    "    Fd['comp_n'] = [((rel.base != r.base) & (rel.d1 > r.d1) & (rel.d1 <= r.d1 + pd.Timedelta(days=9))).sum() for _, r in Fd.iterrows()]\n"
    "    Fd['f_share_cohort'] = np.log1p(Fd.base.map(bT)) - np.log1p(Fd.d1.map(rel.groupby('d1').tix.sum()))\n"
    "    mk = ctx['market']\n"
    "    win = lambda t: mk[(mk.d1 < t) & (mk.d1 >= t - pd.Timedelta(days=28))]\n"
    "    Fd[['mk_occ', 'mk_tps', 'mk_logT', 'mk_lpc']] = [win(t)[['occ', 'tps', 'logT', 'lpc']].median().tolist() for t in Fd.d1]\n"
    "    Fd['occ_rel'] = Fd.f_occ / Fd.mk_occ\n"
    "    Fd['tps_rel'] = Fd.f_tps3 / Fd.mk_tps\n"
    "    Fd['logT_rel'] = Fd.f_logT - Fd.mk_logT\n"
    "    Fd['film_pc_vs_mkt'] = np.log1p(Fm / ncmax) - Fd.mk_lpc\n"
    "\n"
    "    X = tgt.merge(P, on=KEY, how='left').merge(Fd.drop(columns=['base']), left_on='movie_title', right_index=True, how='left')\n"
    "    X['h'] = (X.date_show - X.d1).dt.days + 1\n"
    "    jb = (X.city_name.isin(JABAR_CITIES) & CFG.REGIONAL_CAL).values\n"
    "    cal_at = lambda dates, col: np.where(jb, CAL_JB[col].reindex(dates).values, CAL[col].reindex(dates).values)\n"
    "    X['dow'] = X.date_show.dt.dayofweek.values\n"
    "    X['t_hol'], X['t_school'], X['t_ramadan'] = cal_at(X.date_show, 'is_hol'), cal_at(X.date_show, 'school'), cal_at(X.date_show, 'ramadan')\n"
    "    ch = np.stack([cal_at(X.d1 + pd.Timedelta(days=k), 'cal') for k in range(3)], 1)\n"
    "    X['cal_mult'] = cal_at(X.date_show, 'cal') / ch.mean(1)\n"
    "    X['hol_in_hist'] = np.stack([cal_at(X.d1 + pd.Timedelta(days=k), 'is_hol') for k in range(3)], 1).sum(1)\n"
    "    # D1-D3 tickets divided by the calendar value of their own day; labels and the MASE scale are untouched\n"
    "    q = np.stack([X[f'y{i}'].values / cal_at(X.d1 + pd.Timedelta(days=i - 1), 'cal') for i in (1, 2, 3)], 1)\n"
    "    for i in (1, 2, 3):\n"
    "        X[f'cal_p{i}'] = q[:, i - 1] / np.maximum(q.mean(1), 1)\n"
    "    X['cal_log31'] = np.log1p(q[:, 2]) - np.log1p(q[:, 0])\n"
    "    X['n_comp_open'] = open_count(X, rel)\n"
    "    X = cinema_comp(X, ctx['visible'], ctx['cin_shows'])\n"
    "    X['share'] = X.log_s - np.log1p(X[['fT1', 'fT2', 'fT3']].mean(axis=1) / X.fnc3.clip(lower=1))\n"
    "    X['pair_vs_mkt'] = X.log_s - X.mk_lpc\n"
    "    X['p3_rel'] = X.p3 - X.fp3\n"
    "    X['p1_rel'] = X.p1 - X.fp1\n"
    "    X['cin_size'] = X.cinema_ids.map(ctx['cin_size'])\n"
    "    X['cin_nfilms'] = X.cinema_ids.map(ctx['cin_nfilms'])\n"
    "    X['cin_new'] = X.cin_size.isna().astype(int)\n"
    "    pr = price.pivot(index='city_name', columns='price_day', values='ceil')\n"
    "    X['price_wkd'] = X.city_name.map(pr.Weekday)\n"
    "    X['price_prem'] = X.city_name.map(pr.Weekend / pr.Weekday)\n"
    "    return X"
))

cells.append(md(
    "Fitur dibangun untuk film rilis luas, film rilis terbatas (hanya latih), dan data uji. Tiga pemeriksaan "
    "wajib: jumlah baris tetap, urutan `id` uji tetap, dan skala sama persis dengan fungsi `hitung_skala` resmi."
))

cells.append(code(
    "vis_tr = visible_windows(train, D_wide)\n"
    "market = pd.concat([film_table(vis_tr), film_table(hist)])\n"
    "ctx_tr = make_ctx(vis_tr, train, market)\n"
    "Xtr = build(hs, ts, ctx_tr)\n"
    "Xlim = build(hl, tl, ctx_tr)\n"
    "Xte = build(hist, test, make_ctx(hist, train, market))\n"
    "\n"
    "assert len(Xtr) == len(ts) and len(Xte) == len(test) and (Xte.id.values == test.id.values).all()\n"
    "ref = Xte.join(pair_scale(hist), on=KEY, rsuffix='_ref')\n"
    "assert np.allclose(ref.scale, ref.scale_ref)\n"
    "\n"
    "BASE_FEATS = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',\n"
    "              'f_nc_trend', 'fmt', 'fmt_share', 'd1_dow', 'rating'] + [f'g_{g_}' for g_ in GENRES] + [\n"
    "              'n_genre', 'n_cast', 'comp_n', 'f_share_cohort', 'h', 'dow', 't_hol', 't_school', 't_ramadan', 'cal_mult',\n"
    "              'hol_in_hist', 'n_comp_open', 'share', 'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'cin_new',\n"
    "              'price_wkd', 'price_prem']\n"
    "COMP_FEATS = ['c_new_n', 'c_new_sh', 'c_new_sh_rel', 'c_new_sh_vs_own', 'c_new_tx_vs_own']\n"
    "CAL_FEATS = ['cal_p1', 'cal_p2', 'cal_p3', 'cal_log31']\n"
    "FEATS = BASE_FEATS\n"
    "ZFEATS = FEATS + COMP_FEATS\n"
    "jb_changed = Xte.t_school.values != CAL.school.reindex(Xte.date_show).values\n"
    "print(f'West Java calendar: {jb_changed.sum()} test rows change school flag ({jb_changed.mean():.2%}); train rows changed: '\n"
    "      f'{int((Xtr.t_school.values != CAL.school.reindex(Xtr.date_show).values).sum())}')\n"
    "for X_ in (Xtr, Xlim, Xte):\n"
    "    assert np.isfinite(X_[CAL_FEATS].values).all()\n"
    "print('calendar-normalised inputs (median train | test):', {c: (round(float(Xtr[c].median()), 3), round(float(Xte[c].median()), 3)) for c in CAL_FEATS})\n"
    "print(f'train {len(Xtr)} rows | limited extra {len(Xlim)} | test {len(Xte)} | features {len(FEATS)}')\n"
    "display(pd.DataFrame({'train_median': Xtr[['log_s', 'f_logT', 'pair_vs_mkt', 'film_pc_vs_mkt', 'share']].median(),\n"
    "                      'test_median': Xte[['log_s', 'f_logT', 'pair_vs_mkt', 'film_pc_vs_mkt', 'share']].median()}).round(3))"
))

cells.append(md(
    "## Verifikasi Drift: Adversarial Validation\n\n"
    "Classifier dilatih membedakan baris simulasi dari baris uji dengan fold dikelompokkan per film (split acak "
    "memberi AUC palsu 1,0 karena fitur level film konstan dalam satu film)."
))

cells.append(code(
    "def adv_auc(cols):\n"
    "    A = pd.concat([Xtr[cols].assign(t=0, g=Xtr.movie_title), Xte[cols].assign(t=1, g=Xte.movie_title)], ignore_index=True)\n"
    "    oof = np.zeros(len(A))\n"
    "    for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=CFG.SEED).split(A, A.t, A.g):\n"
    "        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, random_state=CFG.SEED, verbose=-1)\n"
    "        oof[b] = m.fit(A.iloc[a][cols], A.t.iloc[a]).predict_proba(A.iloc[b][cols])[:, 1]\n"
    "    return roc_auc_score(A.t, oof)\n"
    "\n"
    "\n"
    "shape_cols = ['p1', 'p2', 'p3', 'fp1', 'fp3', 'share', 'sh_trend', 'd1_dow', 'n_hist']\n"
    "print(f\"AUC shape features only:         {adv_auc(shape_cols):.3f}\")\n"
    "print(f\"AUC + absolute level (log_s, f_logT): {adv_auc(shape_cols + ['log_s', 'f_logT']):.3f}\")\n"
    "print(f\"AUC + market-relative level:     {adv_auc(shape_cols + ['pair_vs_mkt', 'film_pc_vs_mkt']):.3f}\")"
))

cells.append(md(
    "#### Insights\n\n"
    "> Fitur bentuk kurva tidak bisa membedakan train dan uji (AUC sekitar 0,51). Level absolut menaikkan AUC "
    "(sekitar 0,54) karena pasar uji lebih sepi, sedangkan level relatif pasar tidak menaikkannya (sekitar 0,48): "
    "median `log_s` bergeser 0,67 tetapi `pair_vs_mkt` hanya 0,14. Varian `both` memanfaatkan hal ini."
))

# ---------------------------------------------------------------------------
# Section 8: Modeling
# ---------------------------------------------------------------------------
cells.append(section("Modeling", "8"))

cells.append(md(
    "Bagian ini disusun sebagai **tangga ablation dengan aturan adopsi yang ditulis sebelum hasil terlihat**, mengikuti audit eksperimen 5 Oktober 2026. Fitur baseline v11 (B1, statistik klaster fold-local) **dibekukan** selama membandingkan model, sehingga gain yang terlihat berasal dari model, bukan fitur. Urutannya: LightGBM sebagai kontrol, TabPFN-2.5 Quantiles, lalu TabM. Setelah kombinasi model terpilih, fitur kalender (A3) diuji sebagai A/B pada kombinasi itu dengan aturan adopsi yang sama. Empat lensa dihitung untuk setiap konfigurasi: (1) **grouped CV label asli** dengan bobot komposisi uji (bucket skala x hari pertama laku), lensa utama; (2) **backtest temporal** untuk film rilis Juli, Agustus, dan September dengan data latih yang labelnya sudah lengkap sebelum cutoff (kalender dan fitur jadwal tetap dibangun sekali dari seluruh periode, jadi ini bukan pipeline kausal ketat); (3) **stabilitas per fold**; (4) **dunia $\\kappa$** sebagai uji stres pergeseran kebijakan pencopotan, dengan generator tetap. Ketiga lensa terakhir dihitung dari train periode April sampai September, sehingga tidak dianggap bukti independen. $\\lambda = 0{,}5$ dan bobot hurdle 0,75 dibekukan sebagai baseline; sensitivitas $\\lambda$ hanya dilaporkan."
))

cells.append(md(
    "## Ringkasan Analisis 6 Oktober dan Hasil v11\n\n"
    "| Temuan | Angka | Konsekuensi di v11 |\n"
    "| :--- | :--- | :--- |\n"
    "| v10 vs v8 pada baris OOF yang sama | TW-MASE 0,325126 vs 0,324683 (selisih 0,00044); bootstrap per film [-0,0054; +0,0045] | Keunggulan lokal v10 tidak terbukti; public v10 0,40823 konsisten dengan itu |\n"
    "| Fitur panel bioskop (tren D1 ke D3 pada bioskop yang sama) | grouped TW -0,00539, tetapi temporal +0,00700 (September +0,0154); versi shrinkage juga gagal temporal (+0,0034) | Ditolak, tidak dipakai |\n"
    "| Show, tiket per show, okupansi D1 sampai D3 | grouped -0,00097, temporal +0,00027, satu bulan +0,0052 | Ditolak, tidak dipakai |\n"
    "| Fitur input dinormalisasi kalender | grouped TW -0,00236 (split kedua -0,00266); temporal -0,00163; di hurdle TW-MASE 0,333985 menjadi 0,332800 dan temporal MASE 0,315517 menjadi 0,313997 | Satu-satunya kandidat v11 (A3) |\n"
    "| Sumber galat | 54,5% galat berbobot ada pada target positif yang **naik** dibanding D3 | Pertumbuhan setelah D3 masih belum tertangkap; tidak dikoreksi dengan menaikkan prediksi global |\n"
    "| v11 di Kaggle | A3: TW 0,3319 ke 0,3310, temporal 0,3058 ke 0,3027 (gagal ambang 0,0015); CatBoost 0,3367, XGBoost 0,3342 (korelasi galat dengan LightGBM 0,993 sampai 0,997); Causilo 0,3954 | CatBoost, XGBoost, dan Causilo tidak dipakai lagi; kalender diuji ulang pada kombinasi model terbaik |\n"
    "| Duplikasi dan kebocoran split | signature identik hanya 26 dari 8.113 pasangan; split yang membocorkan horizon pasangan sama menurunkan MASE semu 0,358 menjadi 0,252 | Fold tetap per judul dasar; statistik klaster fold-local |\n\n"
    "Semua angka di atas berasal dari script lokal `eda/54` sampai `eda/61` dengan model LightGBM satu seed, bukan ensemble final, sehingga besar gain di notebook ini bisa berbeda. Fitur kalender dipertahankan berdampingan dengan fitur mentah agar model tidak dipaksa mempercayai faktor kalender, dan label serta skala MASE tidak diubah."
))

cells.append(md(
    "## Lensa Validasi\n"
    "\n"
    "Fold dikelompokkan per judul dasar film. Bobot `TW` menyamakan porsi setiap sel (bucket skala x hari pertama laku) dengan data uji; bobot lama per bucket skala (`TW_OLD`) tetap dilaporkan sebagai pembanding."
))

cells.append(code(
    "groups = base_title(Xtr.movie_title).values\n"
    "ug = np.array(sorted(set(groups)))\n"
    "fold_of = dict(zip(ug, np.random.RandomState(CFG.SEED).permutation(len(ug)) % CFG.N_FOLDS))\n"
    "FOLD = np.array([fold_of[g_] for g_ in groups])\n"
    "y, s, cm, hh_ = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values, Xtr.h.values\n"
    "\n"
    "bt = pd.cut(Xtr.scale, CFG.TW_BINS)\n"
    "tw = bt.map(pd.cut(Xte.scale, CFG.TW_BINS).value_counts(normalize=True) / bt.value_counts(normalize=True)).astype(float).values\n"
    "TW_OLD = tw / tw.mean()\n"
    "fday = lambda D: pd.Series(np.where(D.y1 > 0, 1, np.where(D.y2 > 0, 2, 3)), index=D.index)\n"
    "cell = lambda D: pd.cut(D.scale, CFG.TW_BINS).astype(str) + '|' + fday(D).astype(str)\n"
    "tw = cell(Xtr).map(cell(Xte).value_counts(normalize=True) / cell(Xtr).value_counts(normalize=True)).fillna(0).astype(float).values\n"
    "TW = tw / tw.mean()\n"
    "print('late starters (first sale on D3): old weights', round(float(TW_OLD[fday(Xtr) == 3].sum() / len(TW)), 3),\n"
    "      '| new weights', round(float(TW[fday(Xtr) == 3].sum() / len(TW)), 3), '| test', round(float((fday(Xte) == 3).mean()), 3))\n"
    "\n"
    "\n"
    "def score(p, name, yy=None):\n"
    "    yy = y if yy is None else yy\n"
    "    e = np.abs(yy - p) / s\n"
    "    r = dict(model=name, MASE=e.mean(), TW_MASE=np.average(e, weights=TW), TW_OLD=np.average(e, weights=TW_OLD),\n"
    "             small=e[s <= 20].mean(), big=e[s > 200].mean())\n"
    "    print(f\"{name:40s} MASE {r['MASE']:.4f} | TW-MASE {r['TW_MASE']:.4f} (old {r['TW_OLD']:.4f}) | s<=20 {r['small']:.4f} | s>200 {r['big']:.4f}\")\n"
    "    return r\n"
    "\n"
    "\n"
    "def train_part(k):\n"
    "    Xa = Xtr[FOLD != k]\n"
    "    return pd.concat([Xa, Xlim], ignore_index=True) if CFG.USE_LIMITED else Xa\n"
    "\n"
    "\n"
    "results = []\n"
    "rc = y / s / cm\n"
    "oof_base = np.zeros(len(Xtr))\n"
    "for k in range(CFG.N_FOLDS):\n"
    "    a, b = FOLD != k, FOLD == k\n"
    "    med = pd.Series(rc[a]).groupby([hh_[a], Xtr.d1_dow.values[a]]).median()\n"
    "    oof_base[b] = med.reindex(pd.MultiIndex.from_arrays([hh_[b], Xtr.d1_dow.values[b]])).fillna(np.median(rc[a])).values * cm[b] * s[b]\n"
    "results.append(score(np.zeros(len(Xtr)), 'zero'))\n"
    "results.append(score(oof_base, 'median r/c by (h, D1 dow)'))"
))

cells.append(code(
    "for Pp in (PA, PB):\n"
    "    c3 = np.stack([CAL.cal.reindex(Pp.d1 + pd.Timedelta(days=k)).values for k in range(3)], 1)\n"
    "    Pp['cm'] = c3[:, 2] / c3[:, :2].mean(1)\n"
    "PCOLS = ['q1', 'q2', 'log_s', 'sh1', 'sh2', 'sh_tr', 'occ2', 'f_tr', 'f_log', 'f_nc', 'cm', 'd1_dow']\n"
    "clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=100, verbose=-1,\n"
    "                         deterministic=True, force_row_wise=True, random_state=CFG.SEED).fit(PA[PCOLS], PA.y == 0)\n"
    "logit = lambda p: np.log(p / (1 - p))\n"
    "sigm = lambda t: 1 / (1 + np.exp(-t))\n"
    "pB0 = np.clip(clf.predict_proba(PB[PCOLS])[:, 1], 1e-4, 1 - 1e-4)\n"
    "zB = (PB.y == 0).values\n"
    "grid = np.linspace(-4, 2, 601)\n"
    "nll = lambda p, z, a: -np.mean(z * np.log(sigm(logit(p) + a)) + (1 - z) * np.log(1 - sigm(logit(p) + a)))\n"
    "PULL_A = float(grid[np.argmin([nll(pB0, zB, a) for a in grid])])\n"
    "mon = PB.d1.dt.to_period('M').astype(str).values\n"
    "per_month = {m: round(float(grid[np.argmin([nll(pB0[mon == m], zB[mon == m], a) for a in grid])]), 2) for m in sorted(set(mon))}\n"
    "print(f'pull-odds shift a = {PULL_A:.3f} (odds x {np.exp(PULL_A):.2f}); mean p0 train-model {pB0.mean():.3f} -> '\n"
    "      f'shifted {sigm(logit(pB0) + PULL_A).mean():.3f} | observed {zB.mean():.3f}')\n"
    "print('per test month:', per_month)"
))

cells.append(md(
    "## Statistik Klaster Fold-Local\n"
    "\n"
    "`make_ctx` menghitung ukuran klaster, jumlah film per hari, dan jumlah show dari seluruh train, termasuk transaksi film yang sedang divalidasi. Di bawah, statistik dihitung ulang per fold tanpa seluruh transaksi film fold validasi, dan per cutoff temporal hanya dari transaksi sebelum cutoff. Data uji tetap memakai seluruh train, yang memang masa lalu bagi film uji. Di v10, perbandingan statistik global dan fold-local mengukur kebocoran ini: TW +0,0007, temporal -0,0002."
))

cells.append(code(
    "def cin_stats(tx):\n"
    "    cs = tx.groupby('cinema_ids').total_ticket.sum() / tx.groupby('cinema_ids').date_show.nunique()\n"
    "    nf = tx.groupby(['cinema_ids', 'date_show']).movie_title.nunique().groupby('cinema_ids').mean()\n"
    "    csh = tx.groupby(['cinema_ids', 'date_show']).total_show.sum().groupby('cinema_ids').median()\n"
    "    return np.log10(cs), nf, csh\n"
    "\n"
    "\n"
    "def with_cin(X, stats):\n"
    "    cs, nf, csh = stats\n"
    "    X = X.copy()\n"
    "    X['cin_size'] = X.cinema_ids.map(cs)\n"
    "    X['cin_nfilms'] = X.cinema_ids.map(nf)\n"
    "    X['cin_new'] = X.cin_size.isna().astype(int)\n"
    "    X['c_new_sh_rel'] = X.c_new_sh / X.cinema_ids.map(csh).fillna(X.c_new_sh.median() + 1)\n"
    "    return X\n"
    "\n"
    "\n"
    "TRAIN_BASE = base_title(train.movie_title)\n"
    "STATS_ALL = cin_stats(train)\n"
    "assert np.allclose(with_cin(Xtr, STATS_ALL).cin_size.fillna(-9), Xtr.cin_size.fillna(-9))\n"
    "STATS_FOLD = {k: cin_stats(train[~TRAIN_BASE.isin(set(groups[FOLD == k]))]) for k in range(CFG.N_FOLDS)}\n"
    "chg = np.concatenate([(with_cin(Xtr[FOLD == k], STATS_FOLD[k]).cin_size - Xtr.cin_size[FOLD == k]).abs().values for k in range(CFG.N_FOLDS)])\n"
    "rk = np.mean([pd.Series(with_cin(Xtr[FOLD == k], STATS_FOLD[k]).cin_size.values).corr(pd.Series(Xtr.cin_size[FOLD == k].values), method='spearman') for k in range(CFG.N_FOLDS)])\n"
    "print(f'|change| of cin_size on validation rows: median {np.nanmedian(chg):.4f}, p95 {np.nanpercentile(chg, 95):.4f} (log10 tickets per day; '\n"
    "      f'mostly a level shift from removing ~20% of films) | rank correlation global vs fold-local {rk:.4f}')"
))

cells.append(md(
    "## Mesin Evaluasi\n"
    "\n"
    "`make_parts` menyiapkan bagian latih dan validasi untuk satu konfigurasi (statistik klaster fold-local, fitur kalender), `run_cv` menjalankan grouped CV, dan `temporal` menjalankan backtest bulanan. Model yang dipakai identik dengan baseline v8: LightGBM L1 (5 seed) ditambah hurdle (klasifier $p_0$ dengan fitur kompetisi klaster dan 19 regresi kuantil bagian positif), dicampur dengan bobot hurdle 0,75 pada $\\lambda = 0{,}5$."
))

cells.append(code(
    "LGB_BASE = {k: v for k, v in CFG.LGB_PARAMS.items() if k not in ('objective', 'n_estimators')}\n"
    "LGB_FAST = {**LGB_BASE, 'n_estimators': 300, 'learning_rate': 0.05}\n"
    "MONTHS = ['2025-07', '2025-08', '2025-09']\n"
    "\n"
    "\n"
    "def r_target(Xa):\n"
    "    return (Xa.total_ticket / Xa.scale / Xa.cal_mult).values\n"
    "\n"
    "\n"
    "def fit_lgb_all(Xa, feats, zfeats):\n"
    "    l1 = [lgb.LGBMRegressor(objective='l1', n_estimators=CFG.LGB_PARAMS['n_estimators'], **LGB_BASE, random_state=sd)\n"
    "          .fit(Xa[feats], r_target(Xa), sample_weight=Xa.cal_mult.values) for sd in CFG.SEEDS]\n"
    "    zc = lgb.LGBMClassifier(objective='binary', **{**LGB_FAST, 'num_leaves': 31}, random_state=CFG.SEED).fit(Xa[zfeats], Xa.total_ticket == 0)\n"
    "    pos = Xa[Xa.total_ticket > 0]\n"
    "    qm = [lgb.LGBMRegressor(objective='quantile', alpha=float(q), **LGB_FAST, random_state=CFG.SEED).fit(pos[feats], r_target(pos))\n"
    "          for q in CFG.QS]\n"
    "    return dict(l1=l1, zc=zc, qm=qm)\n"
    "\n"
    "\n"
    "def predict_lgb_all(m, Xb, feats, zfeats):\n"
    "    l1 = np.clip(np.mean([mm.predict(Xb[feats]) for mm in m['l1']], 0), 0, None)\n"
    "    p0 = m['zc'].predict_proba(Xb[zfeats])[:, 1]\n"
    "    Q = np.sort(np.column_stack([mm.predict(Xb[feats]) for mm in m['qm']]), axis=1)\n"
    "    return l1, p0, Q\n"
    "\n"
    "\n"
    "def mixture_median(p0, Q, qs, lam):\n"
    "    p = sigm(logit(np.clip(p0, 1e-4, 1 - 1e-4)) + lam * PULL_A)\n"
    "    t = np.clip((0.5 - p) / (1 - p), qs[0], qs[-1])\n"
    "    val = np.array([np.interp(ti, qs, qi) for ti, qi in zip(t, Q)])\n"
    "    return np.where(p >= 0.5, 0.0, np.clip(val, 0, None))\n"
    "\n"
    "\n"
    "def lgb_mix(l1, p0, Q, lam=None):\n"
    "    lam = CFG.LAMBDA if lam is None else lam\n"
    "    return (1 - CFG.HURDLE_W) * l1 + CFG.HURDLE_W * mixture_median(p0, Q, CFG.QS, lam)\n"
    "\n"
    "\n"
    "def make_parts(Xa, Xb, cfg, stats):\n"
    "    if cfg['cin'] == 'local':\n"
    "        Xa, Xb = with_cin(Xa, stats), with_cin(Xb, stats)\n"
    "    feats = BASE_FEATS + (CAL_FEATS if cfg['calin'] else [])\n"
    "    return Xa, Xb, feats, feats + COMP_FEATS\n"
    "\n"
    "\n"
    "def run_cv(cfg):\n"
    "    oof = dict(l1=np.zeros(len(Xtr)), p0=np.zeros(len(Xtr)), Q=np.zeros((len(Xtr), len(CFG.QS))))\n"
    "    frames = {}\n"
    "    for k in range(CFG.N_FOLDS):\n"
    "        v = FOLD == k\n"
    "        Xa, Xb, feats, zfeats = make_parts(train_part(k), Xtr[v], cfg, STATS_FOLD[k])\n"
    "        oof['l1'][v], oof['p0'][v], oof['Q'][v] = predict_lgb_all(fit_lgb_all(Xa, feats, zfeats), Xb, feats, zfeats)\n"
    "        frames[k] = (Xa, Xb, feats)\n"
    "    return oof, frames\n"
    "\n"
    "\n"
    "def temporal(cfg):\n"
    "    out = {}\n"
    "    for mth in MONTHS:\n"
    "        cut = pd.Timestamp(mth + '-01')\n"
    "        va = (Xtr.d1.dt.strftime('%Y-%m') == mth).values\n"
    "        ready = lambda X: X[(X.d1 + pd.Timedelta(days=9) < cut).values]\n"
    "        Xa = pd.concat([ready(Xtr), ready(Xlim)], ignore_index=True) if CFG.USE_LIMITED else ready(Xtr)\n"
    "        stats = cin_stats(train[train.date_show < cut]) if cfg['cin'] == 'local' else STATS_ALL\n"
    "        Xa_, Xb_, feats, zfeats = make_parts(Xa, Xtr[va], cfg, stats)\n"
    "        out[mth] = dict(idx=np.where(va)[0], parts=predict_lgb_all(fit_lgb_all(Xa_, feats, zfeats), Xb_, feats, zfeats),\n"
    "                        frames=(Xa_, Xb_, feats), n_train_films=base_title(Xa.movie_title).nunique())\n"
    "    return out"
))

cells.append(md(
    "## Aturan Adopsi (Ditulis Sebelum Hasil)\n"
    "\n"
    "- **B1** (statistik klaster fold-local, konfigurasi LightGBM v10) menjadi baseline bersih. Kebocoran statistik klaster global sudah diukur di v10: TW +0,0007, temporal -0,0002.\n"
    "- Kandidat diadopsi hanya bila **semua** syarat terhadap B1 terpenuhi: (1) TW-MASE grouped CV label asli turun minimal `ABL_MIN_GAIN` (0,0015); (2) rata-rata backtest temporal turun dan tidak ada bulan yang naik lebih dari `ABL_MONTH_TOL` (0,003); (3) membaik di minimal `ABL_MIN_FOLDS` (3) dari 5 fold; (4) TW-MASE dunia $\\kappa$ tidak naik lebih dari `ABL_KAPPA_TOL` (0,002).\n"
    "- Di sini A3 hanya dijalankan untuk LightGBM sebagai informasi; keputusan kalender diambil nanti pada kombinasi model terpilih. Perbandingan model memakai B1.\n"
    "- Dunia $\\kappa$ dibangun sekali dari OOF B1 (generator tetap untuk semua konfigurasi): target nol dibatalkan dengan peluang $1 - p_0'/p_0$, $\\text{logit}\\,p_0' = \\text{logit}\\,p_0 + \\kappa a$, lalu diganti sampel dari distribusi positif B1, dirata-rata atas tiga seed. Dunia ini sintetis, sehingga hanya menjadi uji stres."
))

cells.append(code(
    "CONFIGS = {'B1 fold-local stats': dict(cin='local', calin=False),\n"
    "           'A3 + calendar input': dict(cin='local', calin=True)}\n"
    "RUNS = {}\n"
    "\n"
    "\n"
    "def run_config(name, cfgx):\n"
    "    t0 = time.time()\n"
    "    oof, frames = run_cv(cfgx)\n"
    "    RUNS[name] = dict(cfg=cfgx, oof=oof, frames=frames, temporal=temporal(cfgx), pred=lgb_mix(oof['l1'], oof['p0'], oof['Q']) * cm * s)\n"
    "    print(f'{name:28s} done ({time.time() - t0:.0f}s)')\n"
    "\n"
    "\n"
    "for name, cfgx in CONFIGS.items():\n"
    "    run_config(name, cfgx)\n"
    "\n"
    "REF = 'B1 fold-local stats'\n"
    "WORLDS = []\n"
    "for sd in range(CFG.KAPPA_SEEDS):\n"
    "    rng = np.random.default_rng(CFG.SEED + sd)\n"
    "    p0b, Qb = RUNS[REF]['oof']['p0'], RUNS[REF]['oof']['Q']\n"
    "    p_sh = sigm(logit(np.clip(p0b, 1e-4, 1 - 1e-4)) + CFG.KAPPA * PULL_A)\n"
    "    unpull = (y == 0) & (rng.random(len(y)) < 1 - p_sh / np.clip(p0b, 1e-6, None))\n"
    "    draw = np.array([np.interp(u, CFG.QS, q) for u, q in zip(rng.random(len(y)), Qb)])\n"
    "    WORLDS.append(np.where(unpull, np.clip(draw, 0, None) * cm * s, y))\n"
    "print(f'zero rate: real {np.mean(y == 0):.3f} | kappa worlds {np.mean([np.mean(w_ == 0) for w_ in WORLDS]):.3f}')\n"
    "\n"
    "\n"
    "def temporal_pred(R, lam=None):\n"
    "    return {m: (t['idx'], lgb_mix(*t['parts'], lam=lam) * cm[t['idx']] * s[t['idx']]) for m, t in R['temporal'].items()}\n"
    "\n"
    "\n"
    "def lens(p, tpred):\n"
    "    e = np.abs(y - p) / s\n"
    "    r = dict(TW=np.average(e, weights=TW), TW_OLD=np.average(e, weights=TW_OLD), MASE=e.mean())\n"
    "    r.update({f'fold{k}': np.average(e[FOLD == k], weights=TW[FOLD == k]) for k in range(CFG.N_FOLDS)})\n"
    "    tm = {m: np.average(np.abs(y[i] - pr) / s[i], weights=TW[i]) for m, (i, pr) in tpred.items()}\n"
    "    r.update({f'temp_{m}': v for m, v in tm.items()})\n"
    "    r['temporal_mean'] = np.mean(list(tm.values()))\n"
    "    r['kappa'] = np.mean([np.average(np.abs(w_ - p) / s, weights=TW) for w_ in WORLDS])\n"
    "    return r\n"
    "\n"
    "\n"
    "def passes(L, cand, ref=REF):\n"
    "    a, b = L.loc[cand], L.loc[ref]\n"
    "    checks = {'tw_gain': b.TW - a.TW >= CFG.ABL_MIN_GAIN,\n"
    "              'temporal_mean': a.temporal_mean < b.temporal_mean,\n"
    "              'no_month_worse': max(a[f'temp_{m}'] - b[f'temp_{m}'] for m in MONTHS) <= CFG.ABL_MONTH_TOL,\n"
    "              'folds': sum(a[f'fold{k}'] < b[f'fold{k}'] for k in range(CFG.N_FOLDS)) >= CFG.ABL_MIN_FOLDS,\n"
    "              'kappa': a.kappa - b.kappa <= CFG.ABL_KAPPA_TOL}\n"
    "    return all(checks.values()), checks\n"
    "\n"
    "\n"
    "L = pd.DataFrame({n: lens(R['pred'], temporal_pred(R)) for n, R in RUNS.items()}).T\n"
    "display(L.round(4))\n"
    "print('temporal training films per cutoff:', {m: t['n_train_films'] for m, t in RUNS[REF]['temporal'].items()})"
))

cells.append(md(
    "Keputusan adopsi diterapkan otomatis. Selisih per film (bootstrap 2.000 kali per film, bobot tetap) ikut dicetak agar terlihat apakah perbaikan tersebar atau hanya dari beberapa film."
))

cells.append(code(
    "def film_delta_ci(p_new, p_ref):\n"
    "    d = (np.abs(y - p_new) - np.abs(y - p_ref)) / s * TW\n"
    "    G = pd.DataFrame({'g': groups, 'd': d, 'w': TW}).groupby('g').sum()\n"
    "    ix = np.random.default_rng(CFG.SEED).integers(0, len(G), (2000, len(G)))\n"
    "    b = G.d.values[ix].sum(1) / G.w.values[ix].sum(1)\n"
    "    return np.quantile(b, [0.025, 0.5, 0.975]), int((G.d < 0).sum()), len(G)\n"
    "\n"
    "\n"
    "rows = []\n"
    "for cand in ['A3 + calendar input']:\n"
    "    ok, checks = passes(L, cand)\n"
    "    ci, better, nf = film_delta_ci(RUNS[cand]['pred'], RUNS[REF]['pred'])\n"
    "    rows.append(dict(candidate=cand, adopted=ok, **checks, film_delta_ci95=f'[{ci[0]:+.4f}, {ci[2]:+.4f}]', films_better=f'{better}/{nf}'))\n"
    "D = pd.DataFrame(rows).set_index('candidate')\n"
    "display(D)\n"
    "SELECTED = REF\n"
    "SEL = RUNS[SELECTED]\n"
    "SEL_FEATS = SEL['frames'][0][2]\n"
    "print(f'SELECTED configuration: {SELECTED} | features {len(SEL_FEATS)}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))\n"
    "names_ = list(L.index)\n"
    "ax[0].bar(range(len(names_)), L.TW, color=[PAL[1] if n_ == SELECTED else PAL[0] for n_ in names_])\n"
    "ax[0].errorbar(range(len(names_)), L.TW, yerr=L[[f'fold{k}' for k in range(CFG.N_FOLDS)]].std(axis=1), fmt='none', color='k', capsize=3)\n"
    "ax[0].set_xticks(range(len(names_)), names_, rotation=20); ax[0].set(title='Grouped CV TW-MASE (label asli), sd antar-fold', ylim=(L.TW.min() - .01, L.TW.max() + .01))\n"
    "for n_ in names_:\n"
    "    ax[1].plot(MONTHS, L.loc[n_, [f'temp_{m}' for m in MONTHS]].values, marker='o', label=n_)\n"
    "ax[1].set(title='Backtest temporal TW-MASE per bulan rilis'); ax[1].legend(fontsize=7)\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'ablation.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n"
    "\n"
    "> Tabel ini mengulang uji A3 v11 untuk LightGBM saja (di v11: gagal tipis karena gain TW 0,0009 di bawah ambang 0,0015, padahal temporal, fold, dan dunia $\\kappa$ lolos). Perbandingan model di bawah memakai B1; A3 diputuskan ulang setelah kombinasi model terpilih. Kandidat yang lolos harus menang di grouped CV, backtest temporal, mayoritas fold, dan tidak memburuk di dunia $\\kappa$. Interval bootstrap per film memperlihatkan seberapa bergantung hasilnya pada komposisi film; interval yang melewati nol berarti perbaikannya belum kokoh.\n"
    "\n"
    "> Kalender wilayah (Jawa Barat) tidak masuk tangga ablation karena hampir tidak mengubah train: di periode latih jendela libur hari kerjanya identik dengan DKI, sehingga $c_{mult}$ latih sama persis dan hanya flag `t_school` pada akhir pekan 28 sampai 29 Juni (202 baris) yang berbeda. Koreksi itu terutama mengubah baris uji (Bagian 7, 1,6% baris) dan diterapkan sebagai koreksi fakta bersumber, bukan hasil validasi."
))

cells.append(md(
    "## Sensitivitas $\\lambda$ (Dibekukan 0,5)\n"
    "\n"
    "$\\lambda$ tidak dioptimalkan ulang. Tabel berikut hanya memperlihatkan arah sensitivitas pada konfigurasi terpilih: label asli, backtest temporal, dan dunia $\\kappa$."
))

cells.append(code(
    "lam_rows = []\n"
    "for lam in (0.0, 0.25, 0.5, 0.75, 1.0):\n"
    "    p = lgb_mix(SEL['oof']['l1'], SEL['oof']['p0'], SEL['oof']['Q'], lam=lam) * cm * s\n"
    "    r = lens(p, temporal_pred(SEL, lam=lam))\n"
    "    lam_rows.append(dict(lam=lam, TW=r['TW'], temporal_mean=r['temporal_mean'], kappa=r['kappa']))\n"
    "display(pd.DataFrame(lam_rows).set_index('lam').round(4))\n"
    "oof_l1, oof_lgbmix = SEL['oof']['l1'], SEL['pred']\n"
    "results.append(score(oof_lgbmix, f'LightGBM mix [{SELECTED}]'))"
))

cells.append(md(
    "## Kandidat 1: TabPFN-2.5 Quantiles\n"
    "\n"
    "TabPFN-2.5 ([Prior-Labs/tabpfn_2_5](https://huggingface.co/Prior-Labs/tabpfn_2_5), checkpoint `tabpfn-v2.5-regressor-v2.5_quantiles.ckpt`, 40,8 MB, revisi `6c45f3a6`) adalah varian checkpoint regresi yang dikhususkan untuk keluaran kuantil, relevan karena median campuran hurdle bergantung pada bentuk distribusi prediktif, bukan hanya titik tengahnya. Model in-context dipakai per horizon (tanpa `h`) dengan konteks bagian latih per fold dan per cutoff yang sama dengan LightGBM. Varian ini dipilih sebagai kandidat berikutnya, bukan karena pasti lebih akurat: TabPFN-2.6 (51,6 MB) sudah diuji di v9 (TW 0,3263 tanpa koreksi pencopotan, 0,3298 dengan koreksi) dan bobot blendnya nol.\n"
    "\n"
    "**Status kepatuhan.** Rilis TabPFN-2.5 November 2025, setelah batas 30 September 2025; seperti EXAONE dan Causilo, status checkpoint menunggu konfirmasi panitia. `FM_MODE = 'tabpfn_v2'` (revisi 11 Juni 2025, sebelum batas, 44,4 MB) atau `'none'` tersedia bila panitia menafsirkan ketat. Checkpoint diverifikasi dengan SHA-256 revisi terkunci dari input Kaggle, berkas bobot, atau unduhan."
))

cells.append(code(
    "TAB_QS = CFG.TAB_QS\n"
    "\n"
    "\n"
    "def hf_token():\n"
    "    if os.environ.get('HF_TOKEN'):\n"
    "        return os.environ['HF_TOKEN']\n"
    "    if UserSecretsClient is not None:\n"
    "        try:\n"
    "            return UserSecretsClient().get_secret('HF_TOKEN')\n"
    "        except Exception:\n"
    "            return None\n"
    "    return None\n"
    "\n"
    "\n"
    "def sha256(path):\n"
    "    h = hashlib.sha256()\n"
    "    with open(path, 'rb') as f:\n"
    "        for chunk in iter(lambda: f.read(1 << 20), b''):\n"
    "            h.update(chunk)\n"
    "    return h.hexdigest()\n"
    "\n"
    "\n"
    "def find_checkpoint(repo, fname, rev):\n"
    "    roots = [Path('/kaggle/input')] if CFG._ON_KAGGLE else []\n"
    "    for root in roots:\n"
    "        hit = [p for p in root.rglob(Path(fname).name) if str(p).endswith(fname)]\n"
    "        if hit:\n"
    "            return str(hit[0])\n"
    "        for pk in root.rglob('*.pkl'):\n"
    "            try:\n"
    "                with open(pk, 'rb') as f:\n"
    "                    blob = pickle.load(f).get('fm_files', {})\n"
    "            except Exception:\n"
    "                continue\n"
    "            if fname in blob:\n"
    "                out = CFG.OUTPUT_DIR / 'fm' / fname\n"
    "                out.parent.mkdir(parents=True, exist_ok=True)\n"
    "                out.write_bytes(blob[fname])\n"
    "                return str(out)\n"
    "    # local_dir keeps the real file name: loaders pick the format from the suffix\n"
    "    return hf_hub_download(repo, fname, revision=rev, token=hf_token(), local_dir=str(CFG.OUTPUT_DIR / 'fm'))\n"
    "\n"
    "\n"
    "def checkpoint_path(repo, fname, rev, digest):\n"
    "    path = find_checkpoint(repo, fname, rev)\n"
    "    got = sha256(path)\n"
    "    if got != digest:\n"
    "        raise RuntimeError(f'{fname}: SHA-256 {got} does not match the pinned revision ({digest})')\n"
    "    print(f'{fname}: SHA-256 verified ({path})')\n"
    "    return path\n"
    "\n"
    "\n"
    "FM_SPEC = {'tabpfn25q': (ModelVersion.V2_5, CFG.TP25_REPO, CFG.TP25_FILE, CFG.TP25_REV, CFG.TP25_SHA256),\n"
    "           'tabpfn_v2': (ModelVersion.V2, CFG.TABPFN_REPO, CFG.TABPFN_FILE, CFG.TABPFN_REV, CFG.TABPFN_SHA256)}\n"
    "\n"
    "\n"
    "def tabpfn_quantiles(Xa, Xb, feats):\n"
    "    tf = [c for c in feats if c != 'h']\n"
    "    out = np.zeros((len(Xb), len(TAB_QS)))\n"
    "    for hz in range(4, 11):\n"
    "        ia, ib = (Xa.h == hz).values, (Xb.h == hz).values\n"
    "        if ib.sum() == 0:\n"
    "            continue\n"
    "        m = TabPFNRegressor.create_default_for_version(FM_SPEC[CFG.FM_MODE][0], model_path=FM_CKPT, device=CFG.DEVICE,\n"
    "                                                       n_estimators=CFG.FM_EST, random_state=CFG.SEED, ignore_pretraining_limits=True)\n"
    "        m.fit(Xa[ia][tf].values.astype(np.float32), r_target(Xa)[ia])\n"
    "        Qb = Xb[ib][tf].values.astype(np.float32)\n"
    "        out[ib] = np.concatenate([np.asarray(m.predict(Qb[i:i + 3000], output_type='quantiles', quantiles=list(TAB_QS))).T\n"
    "                                  for i in range(0, len(Qb), 3000)])\n"
    "        del m\n"
    "        torch.cuda.empty_cache()\n"
    "    return np.sort(out, axis=1)\n"
    "\n"
    "\n"
    "def quant_median(Q, lam):\n"
    "    p0 = np.array([np.interp(CFG.ZERO_EPS, q, TAB_QS, left=0.0, right=1.0) for q in Q])\n"
    "    p = sigm(logit(np.clip(p0, 1e-4, 1 - 1e-4)) + lam * PULL_A)\n"
    "    u = np.clip(p0 + (1 - p0) * (0.5 - p) / (1 - p), TAB_QS[0], TAB_QS[-1])\n"
    "    val = np.array([np.interp(ui, TAB_QS, qi) for ui, qi in zip(u, Q)])\n"
    "    return np.where(p >= 0.5, 0.0, np.clip(val, 0, None))\n"
    "\n"
    "\n"
    "def run_quant(fn, R):\n"
    "    oof = np.zeros((len(Xtr), len(TAB_QS)))\n"
    "    for k in range(CFG.N_FOLDS):\n"
    "        Xa_, Xb_, f_ = R['frames'][k]\n"
    "        oof[FOLD == k] = fn(Xa_, Xb_, f_)\n"
    "    return oof, {m: fn(*t['frames']) for m, t in R['temporal'].items()}\n"
    "\n"
    "\n"
    "QUANT, QFN, FM_CKPT = {}, {}, None\n"
    "if CFG.FM_MODE != 'none':\n"
    "    t0 = time.time()\n"
    "    try:\n"
    "        FM_CKPT = checkpoint_path(*FM_SPEC[CFG.FM_MODE][1:])\n"
    "        QUANT['tabpfn'], QFN['tabpfn'] = run_quant(tabpfn_quantiles, SEL), tabpfn_quantiles\n"
    "        print(f'{CFG.FM_MODE}: CV + temporal {time.time() - t0:.0f}s')\n"
    "    except Exception as e:\n"
    "        if CFG.STRICT:\n"
    "            raise\n"
    "        print(f'{CFG.FM_MODE} skipped:', type(e).__name__, str(e)[:300])"
))

cells.append(md(
    "## Kandidat 2: TabM\n"
    "\n"
    "TabM ([yandex-research/tabm](https://github.com/yandex-research/tabm), paket `tabm==0.0.3`, lisensi Apache-2.0) bukan foundation model pretrained: arsitekturnya satu MLP yang secara efisien mewakili ensemble $k = 32$ submodel (BatchEnsemble), dilatih dari nol pada data latih kita. Paper dan benchmark-nya mencakup TabReD, yang berisi pergeseran distribusi temporal, situasi yang mirip dengan train April sampai September versus uji Oktober sampai Maret. Bobotnya kecil (beberapa MB), sehingga tidak menekan batas 200 MB.\n"
    "\n"
    "Penerapan di sini: satu model untuk semua horizon (`h` sebagai fitur). Pembagian early stopping (10% film bagian latih) ditentukan **lebih dulu**; fitur yang konstan di inner-training dibuang (misalnya `t_ramadan` dan `cin_new` konstan di seluruh train-sim, dan *piecewise-linear embeddings* menolak kolom konstan), lalu median imputasi, `QuantileTransformer`, dan 48 bin *piecewise-linear embeddings* (versi `B` seperti rekomendasi paper TabM) di-fit hanya pada inner-training. Daftar fitur yang dipakai ikut disimpan di artefak TabM. Keluarannya 49 kuantil $r$ yang sama dengan grid TabPFN, dilatih dengan **pinball loss** berbobot $c_{mult}$ (setara MASE pada median), dan loss dirata-rata atas ke-32 submodel, bukan loss prediksi rata-ratanya, sesuai dokumentasi TabM. Optimizer AdamW (lr 0,002, weight decay 0,0003, batch 256). Early stopping memakai 10% film dari bagian latih (dikelompokkan per film), sehingga fold validasi luar tidak pernah dilihat."
))

cells.append(code(
    "TABM_LAST = {}\n"
    "\n"
    "\n"
    "def pinball(pred, target, w, taus):\n"
    "    d = target[:, None, None] - pred\n"
    "    loss = torch.maximum(taus * d, (taus - 1) * d).mean(dim=(1, 2))\n"
    "    return (loss * w).sum() / w.sum()\n"
    "\n"
    "\n"
    "def tabm_quantiles(Xa, Xb, feats):\n"
    "    torch.manual_seed(CFG.SEED)\n"
    "    np.random.seed(CFG.SEED)\n"
    "    gA = base_title(Xa.movie_title).values\n"
    "    ugA = np.array(sorted(set(gA)))\n"
    "    iv = np.isin(gA, np.random.RandomState(CFG.SEED).permutation(ugA)[: max(1, len(ugA) // 10)])\n"
    "    # every input statistic is fitted on the inner-training films only (early-stopping films stay unseen)\n"
    "    fit_part = Xa[~iv]\n"
    "    use = [f for f in feats if fit_part[f].nunique(dropna=True) > 1]\n"
    "    med = fit_part[use].median()\n"
    "    qt = QuantileTransformer(output_distribution='normal', n_quantiles=1000, subsample=10 ** 9, random_state=CFG.SEED)\n"
    "    qt.fit(fit_part[use].fillna(med).values)\n"
    "    prep = lambda X: qt.transform(X[use].fillna(med).values).astype(np.float32)\n"
    "    A, B = prep(Xa), prep(Xb)\n"
    "    yA, wA = r_target(Xa).astype(np.float32), Xa.cal_mult.values.astype(np.float32)\n"
    "    dev = torch.device(CFG.DEVICE)\n"
    "    T = lambda a: torch.as_tensor(a, device=dev)\n"
    "    Xt, yt, wt, Xv, yv, wv = T(A[~iv]), T(yA[~iv]), T(wA[~iv]), T(A[iv]), T(yA[iv]), T(wA[iv])\n"
    "    bins = compute_bins(torch.as_tensor(A[~iv]), n_bins=CFG.TABM_BINS)\n"
    "    model = TabM.make(n_num_features=A.shape[1], d_out=len(TAB_QS), k=CFG.TABM_K,\n"
    "                      num_embeddings=PiecewiseLinearEmbeddings(bins, CFG.TABM_DEMB, activation=False, version='B')).to(dev)\n"
    "    opt = torch.optim.AdamW(model.parameters(), lr=CFG.TABM_LR, weight_decay=CFG.TABM_WD)\n"
    "    taus = T(np.asarray(TAB_QS, dtype=np.float32))\n"
    "    gen = torch.Generator().manual_seed(CFG.SEED)\n"
    "    best, best_state, bad = np.inf, None, 0\n"
    "    for epoch in range(CFG.TABM_EPOCHS):\n"
    "        model.train()\n"
    "        perm = torch.randperm(len(Xt), generator=gen).to(dev)\n"
    "        for i in range(0, len(Xt), CFG.TABM_BATCH):\n"
    "            b = perm[i:i + CFG.TABM_BATCH]\n"
    "            loss = pinball(model(Xt[b]), yt[b], wt[b], taus)\n"
    "            opt.zero_grad()\n"
    "            loss.backward()\n"
    "            opt.step()\n"
    "        model.eval()\n"
    "        with torch.no_grad():\n"
    "            vl = pinball(torch.cat([model(Xv[i:i + 4096]) for i in range(0, len(Xv), 4096)]), yv, wv, taus).item()\n"
    "        if vl < best - 1e-6:\n"
    "            best, bad = vl, 0\n"
    "            best_state = {k_: v_.detach().clone() for k_, v_ in model.state_dict().items()}\n"
    "        else:\n"
    "            bad += 1\n"
    "            if bad >= CFG.TABM_PATIENCE:\n"
    "                break\n"
    "    model.load_state_dict(best_state)\n"
    "    model.eval()\n"
    "    with torch.no_grad():\n"
    "        P = torch.cat([model(T(B[i:i + 4096])).mean(dim=1) for i in range(0, len(B), 4096)]).cpu().numpy()\n"
    "    TABM_LAST.update(state=best_state, qt=qt, median=med, feats=use, dropped_constant=[f for f in feats if f not in use],\n"
    "                     bins=bins, epochs=epoch + 1, val_pinball=best)\n"
    "    return np.sort(P, axis=1)\n"
    "\n"
    "\n"
    "if CFG.USE_TABM:\n"
    "    t0 = time.time()\n"
    "    QUANT['tabm'], QFN['tabm'] = run_quant(tabm_quantiles, SEL), tabm_quantiles\n"
    "    print(f'TabM: CV + temporal {time.time() - t0:.0f}s | last fit stopped after {TABM_LAST[\"epochs\"]} epochs | '\n"
    "          f'constant features dropped in the last fit: {TABM_LAST[\"dropped_constant\"]}')"
))

cells.append(md(
    "## Koreksi Pencopotan per Model\n"
    "\n"
    "Koreksi pencopotan ($\\lambda a$ pada logit $p_0$) membantu LightGBM, tetapi v9 memperlihatkan koreksi yang sama dapat memperburuk model lain (TabPFN-2.6: TW 0,3263 tanpa koreksi, 0,3298 dengan koreksi). Karena itu setiap model kuantil dinilai dua kali, **median langsung** ($\\lambda = 0$) dan **dengan koreksi** ($\\lambda = 0{,}5$), pada tiga lensa (TW-MASE label asli, rata-rata backtest temporal, dunia $\\kappa$). Varian dengan regret maksimum terkecil dipakai untuk model tersebut; kedua varian selalu dicetak. **Keterbatasan:** dunia $\\kappa$ dibangun dari $p_0$ dan kuantil LightGBM B1, sehingga bukan pembanding netral bagi TabPFN dan TabM (cenderung menguntungkan prediksi yang mirip LightGBM). Karena itu pilihan tanpa lensa $\\kappa$ ikut dicetak; bila keduanya berbeda, keputusan bergantung pada asumsi $\\kappa$ dan perlu dibaca hati-hati."
))

cells.append(code(
    "lenses_ = ['TW', 'temporal_mean', 'kappa']\n"
    "LAM_CHOICE = {}\n"
    "COMP = {'lgb': (oof_lgbmix, temporal_pred(SEL))}\n"
    "\n"
    "\n"
    "def quant_component(oq, tq, lam, R):\n"
    "    return (quant_median(oq, lam) * cm * s,\n"
    "            {m: (t['idx'], quant_median(tq[m], lam) * cm[t['idx']] * s[t['idx']]) for m, t in R['temporal'].items()})\n"
    "\n"
    "\n"
    "rows = []\n"
    "for n, (oq, tq) in QUANT.items():\n"
    "    for lam in (0.0, CFG.LAMBDA):\n"
    "        rows.append(dict(model=n, lam=lam, **lens(*quant_component(oq, tq, lam, SEL))))\n"
    "VAR = pd.DataFrame(rows)\n"
    "if len(VAR):\n"
    "    display(VAR[['model', 'lam'] + lenses_ + [f'temp_{m}' for m in MONTHS]].round(4))\n"
    "    for n, g in VAR.groupby('model'):\n"
    "        reg = (g[lenses_] - g[lenses_].min()).max(axis=1)\n"
    "        LAM_CHOICE[n] = float(g.loc[reg.idxmin(), 'lam'])\n"
    "        reg2 = (g[['TW', 'temporal_mean']] - g[['TW', 'temporal_mean']].min()).max(axis=1)\n"
    "        print(f'{n}: lambda by 3 lenses {LAM_CHOICE[n]} | without the kappa lens {float(g.loc[reg2.idxmin(), \"lam\"])}')\n"
    "        COMP[n] = quant_component(*QUANT[n], LAM_CHOICE[n], SEL)\n"
    "        results.append(score(COMP[n][0], f'{n} median (lambda {LAM_CHOICE[n]})'))\n"
    "print('pull-shift choice per model:', LAM_CHOICE)\n"
    "display(pd.DataFrame({n: lens(*COMP[n]) for n in COMP}).T[lenses_ + ['TW_OLD', 'MASE']].round(4))"
))

cells.append(md(
    "## Blending: Kontrol LightGBM dan Kandidat\n"
    "\n"
    "Bobot komponen (LightGBM dan kandidat yang berhasil dijalankan) dicari pada grid simpleks berjarak 0,1. Kombinasi hanya boleh dipakai bila, terhadap LightGBM saja: (1) TW-MASE label asli membaik minimal `BLEND_MIN_GAIN` (0,001), (2) rata-rata backtest temporal tidak memburuk, dan (3) dunia $\\kappa$ tidak memburuk lebih dari `ABL_KAPPA_TOL`. Di antara yang lolos dipilih kombinasi dengan **regret maksimum terkecil** atas tiga lensa; kombinasi pilihan label asli saja dicetak sebagai pembanding. Bila tidak ada yang lolos, LightGBM saja dipertahankan, seperti v11."
))

cells.append(code(
    "names = list(COMP)\n"
    "base_l = lens(*COMP['lgb'])\n"
    "grid = np.round(np.arange(0, 1.01, 0.1), 1)\n"
    "rows = []\n"
    "for w in itertools.product(grid, repeat=len(names)):\n"
    "    if abs(sum(w) - 1) > 1e-9:\n"
    "        continue\n"
    "    p = sum(wi * COMP[n][0] for wi, n in zip(w, names))\n"
    "    tp = {m: (i, sum(wi * COMP[n][1][m][1] for wi, n in zip(w, names))) for m, (i, _) in COMP['lgb'][1].items()}\n"
    "    rows.append(dict(zip([f'w_{n}' for n in names], w), **lens(p, tp)))\n"
    "BW = pd.DataFrame(rows)\n"
    "BW['max_regret'] = (BW[lenses_] - BW[lenses_].min()).max(axis=1)\n"
    "BW['pass'] = ((BW['w_lgb'] < 1) & (base_l['TW'] - BW.TW >= CFG.BLEND_MIN_GAIN) & (BW.temporal_mean <= base_l['temporal_mean'])\n"
    "              & (BW.kappa - base_l['kappa'] <= CFG.ABL_KAPPA_TOL))\n"
    "wcols = [f'w_{n}' for n in names]\n"
    "print(f'{len(BW)} weight combinations, {int(BW[\"pass\"].sum())} pass every condition')\n"
    "display(BW.sort_values('max_regret').head(10)[wcols + lenses_ + ['max_regret', 'pass']].round(4))\n"
    "WEIGHTS = {'lgb': 1.0}\n"
    "if BW['pass'].any():\n"
    "    ok_ = BW[BW['pass']]\n"
    "    best_real = ok_.loc[ok_.TW.idxmin(), wcols].to_dict()\n"
    "    pick = ok_.loc[ok_.max_regret.idxmin()]\n"
    "    WEIGHTS = {n: float(pick[f'w_{n}']) for n in names if pick[f'w_{n}'] > 0}\n"
    "    print('best by real labels only:', {k[2:]: v for k, v in best_real.items() if v > 0})\n"

    "# kappa-free reference: filter and minimax use only real labels and the temporal backtest (printed, not used to decide)\n"
    "nk = BW[(BW['w_lgb'] < 1) & (base_l['TW'] - BW.TW >= CFG.BLEND_MIN_GAIN) & (BW.temporal_mean <= base_l['temporal_mean'])]\n"
    "if len(nk):\n"
    "    reg2 = (nk[['TW', 'temporal_mean']] - BW[['TW', 'temporal_mean']].min()).max(axis=1)\n"
    "    print(f'kappa-free reference ({len(nk)} combinations pass without the kappa condition):',\n"
    "          {k[2:]: v for k, v in nk.loc[reg2.idxmin(), wcols].to_dict().items() if v > 0})\n"
    "else:\n"
    "    print('kappa-free reference: no combination beats LightGBM on real labels and temporal')\n"
    "print('signed-error correlation:')\n"
    "display(pd.DataFrame({n: (y - COMP[n][0]) / s for n in names}).corr().round(3))\n"
    "oof_final = sum(WEIGHTS[n] * COMP[n][0] for n in WEIGHTS)\n"
    "tp_final = {m: (i, sum(WEIGHTS[n] * COMP[n][1][m][1] for n in WEIGHTS)) for m, (i, _) in COMP['lgb'][1].items()}\n"
    "print('chosen weights:', WEIGHTS)"
))

cells.append(md(
    "## A/B Fitur Kalender pada Kombinasi Terpilih\n"
    "\n"
    "Setelah kombinasi model dan bobotnya terkunci, fitur input kalender (A3) diuji ulang pada **kombinasi itu**, bukan pada LightGBM saja. Komponen kandidat dengan bobot di atas nol dilatih ulang memakai bagian latih A3 (fold dan cutoff sama), LightGBM memakai run A3 yang sudah ada, dan blend dengan bobot yang sama dibandingkan dengan blend B1 memakai aturan adopsi tertulis (`passes`): TW turun minimal 0,0015, temporal turun tanpa bulan yang naik lebih dari 0,003, membaik di minimal 3 dari 5 fold, dan dunia $\\kappa$ tidak naik lebih dari 0,002. Bila gagal, fitur baseline dipertahankan."
))

cells.append(code(
    "CAL = 'A3 + calendar input'\n"
    "comp_cal = {'lgb': (RUNS[CAL]['pred'], temporal_pred(RUNS[CAL]))}\n"
    "for n in WEIGHTS:\n"
    "    if n != 'lgb':\n"
    "        t0 = time.time()\n"
    "        comp_cal[n] = quant_component(*run_quant(QFN[n], RUNS[CAL]), LAM_CHOICE[n], RUNS[CAL])\n"
    "        print(f'{n} on calendar features: {time.time() - t0:.0f}s')\n"
    "p_cal = sum(WEIGHTS[n] * comp_cal[n][0] for n in WEIGHTS)\n"
    "tp_cal = {m: (i, sum(WEIGHTS[n] * comp_cal[n][1][m][1] for n in WEIGHTS)) for m, (i, _) in comp_cal['lgb'][1].items()}\n"
    "LC = pd.DataFrame({'blend B1': lens(oof_final, tp_final), 'blend A3': lens(p_cal, tp_cal)}).T\n"
    "display(LC.round(4))\n"
    "cal_ok, cal_checks = passes(LC, 'blend A3', ref='blend B1')\n"
    "ci, better, nf = film_delta_ci(p_cal, oof_final)\n"
    "print(f'calendar A/B: {cal_checks} -> adopted {cal_ok} | film bootstrap 95% CI [{ci[0]:+.4f}, {ci[2]:+.4f}], films better {better}/{nf}')\n"
    "FEATURE_CFG = CAL if cal_ok else REF\n"
    "if cal_ok:\n"
    "    oof_final, tp_final = p_cal, tp_cal\n"
    "print('feature configuration for the final model:', FEATURE_CFG)\n"
    "results.append(score(oof_final, 'final blend'))\n"
    "results.append(score(oof_final, 'final blend, kappa world 0', yy=WORLDS[0]))\n"
    "display(pd.DataFrame(results).set_index('model').round(4))"
))

cells.append(md(
    "#### Insights\n"
    "\n"
    "> Model dibandingkan pada fitur yang sama (B1), sehingga perbedaan antar baris tabel adalah perbedaan model. TabPFN-2.5 Quantiles dan TabM masing-masing memilih sendiri apakah koreksi pencopotan dipakai, dan bobot blend hanya berubah dari LightGBM saja bila lolos ketiga syarat. Korelasi galat bertanda memperlihatkan apakah kandidat membawa informasi yang berbeda: CatBoost dan XGBoost di v11 berkorelasi 0,993 sampai 0,997 dengan LightGBM, sehingga tidak menambah apa-apa.\n"
    "\n"
    "> Fitur kalender diputuskan paling akhir, pada kombinasi terpilih. Di v11 kalender membantu LightGBM di semua lensa kecuali ambang TW; uji di sini menjawab apakah gain itu bertahan atau hilang ketika model lain ikut di blend."
))

# ---------------------------------------------------------------------------
# Section 9: Evaluation
# ---------------------------------------------------------------------------
cells.append(section("Evaluation", "9"))

cells.append(md(
    "Galat out-of-fold model final dibedah per bucket skala, per horizon, per bulan rilis, dan per film. Kontribusi per bucket dihitung dengan porsi uji, dan sensitivitas skor terhadap komposisi film diukur dengan bootstrap per film."
))

cells.append(code(
    "E = Xtr[['movie_title', 'h', 'scale', 'total_ticket', 'd1']].assign(pred=oof_final, w=TW)\n"
    "E['ae'] = np.abs(E.total_ticket - E.pred) / E.scale\n"
    "cut = pd.cut(E.scale, CFG.TW_BINS)\n"
    "bk = pd.DataFrame({'test_share': pd.cut(Xte.scale, CFG.TW_BINS).value_counts(normalize=True), 'train_share': cut.value_counts(normalize=True),\n"
    "                   'mase': E.groupby(cut, observed=False).ae.mean(), 'zero_rate': (E.total_ticket == 0).groupby(cut, observed=False).mean()}).sort_index()\n"
    "bk['contribution_to_test'] = bk.test_share * bk.mase\n"
    "display(bk.round(3))\n"
    "wm = lambda g: np.average(g.ae, weights=g.w)\n"
    "print('TW-MASE by horizon:', E.groupby('h').apply(wm, include_groups=False).round(3).to_dict())\n"
    "print('TW-MASE by release month:', E.groupby(E.d1.dt.strftime('%Y-%m')).apply(wm, include_groups=False).round(3).to_dict())\n"
    "fe = E.assign(c=E.ae * E.w / E.w.sum()).groupby(base_title(E.movie_title)).c.sum()\n"
    "print(f'top-5 films carry {fe.nlargest(5).sum() / fe.sum():.1%} of the weighted error:', fe.nlargest(5).index.tolist())\n"
    "ci, better, nf = film_delta_ci(oof_final, RUNS[REF]['pred'])\n"
    "print(f'final vs B1 LightGBM film bootstrap 95% CI of TW change: [{ci[0]:+.4f}, {ci[2]:+.4f}] | films better {better}/{nf}')\n"
    "\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))\n"
    "xx = np.arange(len(bk))\n"
    "ax[0].bar(xx - .2, bk.train_share, .4, label='porsi train'); ax[0].bar(xx + .2, bk.test_share, .4, label='porsi uji')\n"
    "ax[0].set_xticks(xx, [str(i) for i in bk.index], rotation=30); ax[0].legend(); ax[0].set_title('Komposisi skala')\n"
    "ax[1].bar(xx, bk.contribution_to_test, color=PAL[7]); ax[1].set_xticks(xx, [str(i) for i in bk.index], rotation=30)\n"
    "ax[1].set_title('Kontribusi bucket ke MASE uji yang diharapkan')\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'evaluation.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n"
    "\n"
    "> Pasangan dengan skala di bawah 50 hanya sekitar 13% baris train tetapi 33% baris uji dan menyumbang porsi terbesar MASE uji yang diharapkan. Lima film menyumbang kira-kira seperempat sampai sepertiga galat berbobot (`eda/51`), sehingga perbedaan versi di bawah lebar interval bootstrap per film tidak boleh dibaca sebagai perbaikan pasti."
))

# ---------------------------------------------------------------------------
# Section 10: Inference & Submission
# ---------------------------------------------------------------------------
cells.append(section("Inference & Submission", "10"))

cells.append(md(
    "Model final dilatih pada seluruh sampel (rilis luas dan rilis terbatas) dengan konfigurasi terpilih. Statistik klaster untuk data uji memakai seluruh train (masa lalu film uji). Tidak ada parameter yang disetel pada leaderboard."
))

cells.append(md(
    "## Minggu Lebaran: Analog Top-Down dari Lebaran 2025\n"
    "\n"
    "Tujuh judul slate Lebaran 2026 dirilis 18 Maret, sehingga D4 sampai D10 mereka adalah Lebaran hari 1 sampai 7 (6,1% baris uji), periode yang tidak memiliki padanan di sampel latih. Namun `train.csv` dimulai 1 April 2025, yaitu Lebaran 2025 hari ke-2, sehingga total tiket per klaster selama minggu Lebaran 2025 teramati. Prediksi top-down: total slate per klaster pada Lebaran hari ke-$k$ = $\\rho$ x total klaster pada Lebaran 2025 hari ke-$k$, dibagi ke pasangan menurut pangsa skala D1 sampai D3. Rasio klaster dirata-rata geometrik dengan rasio nasional agar tidak liar pada klaster kecil."
))

cells.append(code(
    "LEB25, LEB26 = pd.Timestamp('2025-03-31'), pd.Timestamp('2026-03-21')\n"
    "leb_mask = Xte.date_show.between(LEB26, LEB26 + pd.Timedelta(days=6)).values\n"
    "slate = sorted(Xte.movie_title[leb_mask].unique())\n"
    "hsl = hist[hist.movie_title.isin(slate)]\n"
    "day25 = train.groupby('date_show').total_ticket.sum()\n"
    "normal_day = day25.loc['2025-05-01':].median()\n"
    "m25 = pd.Series({k: day25[LEB25 + pd.Timedelta(days=k - 1)] for k in range(2, 8)})\n"
    "m25.loc[1] = CFG.LEB_DAY1 * m25.loc[2]\n"
    "m25 = m25.sort_index()\n"
    "S26 = hsl.groupby('date_show').total_ticket.sum().mean()\n"
    "seats26 = (hsl.total_ticket / (hsl.occupation_rate / 100).clip(lower=.005)).groupby(hsl.date_show).sum().mean()\n"
    "top25 = train[train.date_show.between('2025-04-01', '2025-04-07')].groupby('movie_title').total_ticket.sum().nlargest(5).index\n"
    "a25 = train[train.movie_title.isin(top25) & train.date_show.between('2025-04-01', '2025-04-07')]\n"
    "print('slate:', slate)\n"
    "print(f'normal 2025 day {normal_day:.0f} | 2025 Lebaran week mean {m25.loc[2:7].mean():.0f} | 2026 slate mean D1-D3 {S26:.0f}')\n"
    "print(f'2026 slate: {seats26:.0f} seats/day at occupancy {100 * S26 / seats26:.0f}% | 2025 slate occupancy median {a25.occupation_rate.median():.0f}%')\n"
    "imp_r = pd.DataFrame({f'rho={r_}': r_ * m25 / S26 for r_ in (0.4, 0.5, 0.7, 1.0)})\n"
    "imp_r.index = [f'D{k + 3} (Lebaran day {k})' for k in imp_r.index]\n"
    "imp_r['occupancy_needed_rho=0.5'] = (0.5 * m25.values / seats26 * 100).round(0)\n"
    "display(imp_r.round(2))\n"
    "\n"
    "c25 = {k: train[train.date_show == LEB25 + pd.Timedelta(days=k - 1)].groupby('cinema_ids').total_ticket.sum() for k in range(2, 8)}\n"
    "c25[1] = CFG.LEB_DAY1 * c25[2]\n"
    "c26 = hsl.groupby('cinema_ids').total_ticket.sum() / 3\n"
    "kday = (Xte.date_show - LEB26).dt.days.values + 1\n"
    "leb_pred = np.zeros(len(Xte))\n"
    "for k in range(1, 8):\n"
    "    m = leb_mask & (kday == k)\n"
    "    r_nat = m25.loc[k] / S26\n"
    "    r_cl = (Xte.cinema_ids[m].map(c25[k]) / Xte.cinema_ids[m].map(c26)).values\n"
    "    r_cl = np.where(np.isfinite(r_cl) & (r_cl > 0), r_cl, r_nat)\n"
    "    # geometric mean of cluster and national ratio\n"
    "    leb_pred[m] = Xte.scale.values[m] * CFG.LEB_RHO * np.sqrt(r_cl * r_nat)\n"
    "print('analog mean r by Lebaran day:', {int(k): round(float((leb_pred / Xte.scale.values)[leb_mask & (kday == k)].mean()), 2) for k in range(1, 8)})\n"
    "\n"
    "dd = day25.loc['2025-04-01':'2025-04-21']\n"
    "fig, ax = plt.subplots(1, 2, figsize=(13, 3.6))\n"
    "ax[0].bar(dd.index, dd.values, color=PAL[0]); ax[0].axhline(normal_day, color=PAL[7], ls='--', label='hari normal (median Mei-Sep)')\n"
    "ax[0].set(title='Train: tiket nasional per hari setelah Lebaran 2025'); ax[0].legend(); ax[0].tick_params(axis='x', rotation=45)\n"
    "imp_r[['rho=0.4', 'rho=0.5', 'rho=0.7', 'rho=1.0']].plot(ax=ax[1], marker='o'); ax[1].tick_params(axis='x', rotation=30)\n"
    "ax[1].set(title='Rasio slate y / skala yang tersirat dari Lebaran 2025')\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'lebaran.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n"
    "\n"
    "> Minggu Lebaran 2025 berjalan di sekitar 608 ribu tiket per hari pada klaster yang sama (2,8 kali hari normal) dan tetap 2,3 sampai 3 kali normal pada hari kerja setelah cuti bersama. Slate 2026 sudah menawarkan sekitar satu juta kursi per hari pada D1 sampai D3 dengan okupansi hanya 15 sampai 23%, sedangkan slate Lebaran 2025 mencapai okupansi median 73%. Model statistik memprediksi slate turun ke sekitar 129 ribu tiket per hari (rasio 0,79, anjlok ke 0,25 pada 25 sampai 27 Maret), setara okupansi 13% pada minggu puncak tahunan, yang tidak masuk akal secara fisik.\n"
    "\n"
    "> Analog memakai $\\rho = 0{,}5$: slate diasumsikan hanya mencapai separuh total pasar Lebaran 2025 (rasio rata-rata sekitar 1,7, okupansi tersirat di bawah 35%). Bahkan skenario pesimis (pasar 60% dari 2025 dan pangsa slate 70%) memberi rasio 1,4. Nilai ini ditetapkan dari data 2025 dan kapasitas kursi, bukan dari leaderboard. Sumber eksternal pendukung yang terbit sebelum batas: pemberitaan April 2025 tentang 2 juta penonton dalam 3 hari libur Lebaran dan 5 juta penonton selama libur Lebaran."
))

cells.append(code(
    "t0 = time.time()\n"
    "Xall = pd.concat([Xtr, Xlim], ignore_index=True) if CFG.USE_LIMITED else Xtr\n"
    "cfg_sel = RUNS[FEATURE_CFG]['cfg']\n"
    "XA, XT, feats_f, zfeats_f = make_parts(Xall, Xte, cfg_sel, STATS_ALL)\n"
    "cm_te, s_te_ = Xte.cal_mult.values, Xte.scale.values\n"
    "te_comp, final_lgb, lgb_mb = {}, None, 0.0\n"
    "if WEIGHTS.get('lgb', 0) > 0:\n"
    "    final_lgb = fit_lgb_all(XA, feats_f, zfeats_f)\n"
    "    te_comp['lgb'] = lgb_mix(*predict_lgb_all(final_lgb, XT, feats_f, zfeats_f)) * cm_te * s_te_\n"
    "    lgb_mb = sum(len(zlib.compress(m.booster_.model_to_string().encode(), 9)) for m in final_lgb['l1'] + [final_lgb['zc']] + final_lgb['qm']) / 1e6\n"
    "fm_mb = Path(FM_CKPT).stat().st_size / 1e6 if WEIGHTS.get('tabpfn', 0) > 0 else 0.0\n"
    "print(f'weights before TabM: LightGBM {lgb_mb:.1f} MB + TabPFN checkpoint {fm_mb:.1f} MB')\n"
    "assert lgb_mb + fm_mb <= CFG.MAX_WEIGHTS_MB - 10, 'weights would exceed the 200 MB rule'\n"
    "for n in WEIGHTS:\n"
    "    if n != 'lgb':\n"
    "        te_comp[n] = quant_median(QFN[n](XA, XT, feats_f), LAM_CHOICE[n]) * cm_te * s_te_\n"
    "assert set(te_comp) == set(WEIGHTS), f'component mismatch {set(te_comp)} vs {set(WEIGHTS)}'\n"
    "pred = np.clip(sum(WEIGHTS[n] * te_comp[n] for n in te_comp), 0, None)\n"
    "print('model mean r on Lebaran rows:', round(float((pred / s_te_)[leb_mask].mean()), 3), '-> analog', round(float((leb_pred / s_te_)[leb_mask].mean()), 3))\n"
    "pred = np.where(leb_mask, leb_pred, pred)\n"
    "print(f'final fit + inference: {time.time() - t0:.0f}s')\n"
    "\n"
    "seg = pd.Series('normal', index=Xte.index)\n"
    "for n, (a, b) in SEGMENTS.items():\n"
    "    seg[Xte.date_show.between(a, b)] = n\n"
    "rr = pred / s_te_\n"
    "display(pd.DataFrame({'rows': seg.value_counts(), 'mean_r_hat': pd.Series(rr).groupby(seg.values).mean(),\n"
    "                      'pred_zero_share': pd.Series(rr < .05).groupby(seg.values).mean()}).round(3))"
))

cells.append(md(
    "Submisi disimpan dengan format `id,total_ticket` dan urutan `id` yang sama dengan `sample_submission.csv`. Seluruh bobot (booster LightGBM terkompresi, state TabM beserta transformasi inputnya bila dipakai, dan checkpoint TabPFN bila dipakai) disimpan dalam satu berkas `.pkl`, dengan pemeriksaan batas 200 MB yang menghentikan notebook bila terlampaui."
))

cells.append(code(
    "submission = pd.DataFrame({'id': test.id.values, 'total_ticket': pred})\n"
    "assert len(submission) == len(sub) and (submission.id.values == sub.id.values).all()\n"
    "assert submission.total_ticket.notna().all() and (submission.total_ticket >= 0).all()\n"
    "submission.to_csv(CFG.OUTPUT_DIR / 'submission.csv', index=False)\n"
    "pd.DataFrame({'movie_title': Xtr.movie_title, 'cinema_ids': Xtr.cinema_ids, 'h': Xtr.h, 'fold': FOLD, 'y': y, 'scale': s,\n"
    "              'oof_b1': RUNS[REF]['pred'], 'oof_lgbmix': oof_lgbmix, 'oof_final': oof_final}).to_csv(CFG.OUTPUT_DIR / 'oof.csv', index=False)\n"
    "zs = lambda m: zlib.compress(m.booster_.model_to_string().encode(), 9)\n"
    "tabm_blob = None\n"
    "if 'tabm' in te_comp:\n"
    "    buf = io.BytesIO()\n"
    "    torch.save({k_: v_.cpu() for k_, v_ in TABM_LAST['state'].items()}, buf)\n"
    "    tabm_blob = {'state': zlib.compress(buf.getvalue(), 9), 'quantile_transformer': pickle.dumps(TABM_LAST['qt']),\n"
    "                 'fill_median': TABM_LAST['median'].to_dict(), 'features': TABM_LAST['feats'],\n"
    "                 'bins': [b_.cpu().numpy() for b_ in TABM_LAST['bins']], 'k': CFG.TABM_K, 'd_embedding': CFG.TABM_DEMB}\n"
    "fm_files = {FM_SPEC[CFG.FM_MODE][2]: Path(FM_CKPT).read_bytes()} if 'tabpfn' in te_comp else {}\n"
    "lgb_blob = None if final_lgb is None else {'l1': [zs(m) for m in final_lgb['l1']], 'zero_clf': zs(final_lgb['zc']),\n"
    "                                           'quantiles': [zs(m) for m in final_lgb['qm']]}\n"
    "payload = {'lightgbm': lgb_blob,\n"
    "           'tabm': tabm_blob, 'compression': 'zlib', 'features': feats_f, 'zero_features': zfeats_f, 'feature_config': FEATURE_CFG,\n"
    "           'pull_a': PULL_A, 'blend_weights': WEIGHTS, 'pull_shift_per_model': LAM_CHOICE, 'fm_mode': CFG.FM_MODE, 'fm_files': fm_files,\n"
    "           'fm_revision': FM_SPEC.get(CFG.FM_MODE, (None,) * 5)[3], 'fm_sha256': FM_SPEC.get(CFG.FM_MODE, (None,) * 5)[4],\n"
    "           'settings': {k: v for k, v in vars(Settings).items() if k.isupper()}}\n"
    "blob = pickle.dumps(payload)\n"
    "(CFG.OUTPUT_DIR / 'model_weights.pkl').write_bytes(blob)\n"
    "print(f'weights: {len(blob) / 1e6:.1f} MB (cap {CFG.MAX_WEIGHTS_MB} MB) | TabPFN checkpoint inside: {list(fm_files)} | TabM inside: {tabm_blob is not None}')\n"
    "assert len(blob) / 1e6 <= CFG.MAX_WEIGHTS_MB, 'model weights exceed the 200 MB rule'\n"
    "display(submission.head())"
))

cells.append(md(
    "#### Insights\n"
    "\n"
    "> v12 mempertahankan fondasi v11 (split per judul dasar, statistik klaster fold-local, target dan skala resmi, backtest temporal) dan hanya mengganti kandidat model: LightGBM sebagai kontrol, TabPFN-2.5 Quantiles, dan TabM, dengan koreksi pencopotan dipilih per model. Fitur kalender diputuskan paling akhir pada kombinasi terpilih. CatBoost, XGBoost, dan Causilo dihapus karena di v11 tidak menambah informasi atau jauh lebih buruk. Rekonstruksi D1 dan override Lebaran tetap menjadi batas validasi yang belum terselesaikan. Fitur panel bioskop, perubahan show dan tiket per show, serta metadata resmi tidak dipakai karena gagal di analisis lokal atau di aturan adopsi v10. Kalender wilayah Jawa Barat diterapkan sebagai koreksi fakta bersumber pada baris uji. Level minggu Lebaran tetap dari analog Lebaran 2025, yang tidak tervalidasi oleh OOF.\n"
    "\n"
    "> Untuk reproduksi, semua komponen yang dipakai disimpan di satu berkas bobot di bawah 200 MB, checkpoint dicari dulu di input Kaggle atau di dalam berkas bobot sebelum ke Hugging Face, dan kegagalan komponen menghentikan notebook (`STRICT = True`) alih-alih mengubah prediksi diam-diam."
))

# ---------------------------------------------------------------------------
# Notebook writer
# ---------------------------------------------------------------------------
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
    },
    "cells": cells,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(cells)} cells -> {OUT}")
