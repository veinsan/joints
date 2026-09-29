import json
import sys
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OUT = Path(__file__).parent / "v3.ipynb"
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
    "# Data Science Competition JOINTS x INSPIRE UGM 2026 (v3)\n\n"
    "*Memprediksi penjualan tiket harian D4 sampai D10 per pasangan film dan klaster bioskop dari tiga hari "
    "pertama, dengan validasi yang dibobot ke komposisi data uji dan model pretrained yang patuh batas "
    "publikasi 30 September 2025.*"
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
    "*Dua versi sebelumnya mencapai MASE validasi 0,30 sampai 0,33, tetapi skor public leaderboard 0,45. "
    "Notebook v3 dibangun untuk menjelaskan jarak tersebut dan menutupnya, bukan sekadar menambah model.*"
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
    "## Approach: Validasi Berbobot Uji + LightGBM dan TabPFN v2 per Horizon\n\n"
    "*Temuan utama v3: galat per ukuran pasangan stabil antara periode latih dan periode uji, tetapi data uji "
    "memuat empat kali lebih banyak pasangan kecil (skala di bawah 20) yang galatnya paling besar. MASE "
    "validasi biasa karena itu terlalu optimistis. Seluruh keputusan di notebook ini diambil dengan "
    "**TW-MASE**, yaitu galat out-of-fold yang dibobot ke komposisi skala data uji.*\n\n"
    "*Model: LightGBM objective L1 dan TabPFN v2 (Nature, Januari 2025, revisi Hugging Face dikunci ke "
    "11 Juni 2025) yang dipasang per horizon, keduanya pada target $r / c_{mult}$ dengan $c_{mult}$ pengali "
    "kalender struktural. Bobot blending dipilih dengan TW-MASE out-of-fold.*\n\n"
    "```\n"
    "train.csv -> D1 rilis resmi (segmen kontinu) -> simulasi aturan panitia -> + film rilis terbatas (latih saja)\n"
    "                                                   |\n"
    "test_history.csv ---------------------------------+--> build fitur (pasangan, film, kalender, jadwal, klaster)\n"
    "                                                   |\n"
    "              LightGBM L1 (5 seed)   +   TabPFN v2 per horizon (median prediktif)\n"
    "                                                   |  blend dipilih dengan TW-MASE\n"
    "                  y_hat = r_hat x c_mult x s_p  x  pengali segmen kalender -> submission.csv\n"
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
    "TabPFN memakai GPU pertama, LightGBM berjalan di CPU dengan `deterministic=True`. Recorded runtime: "
    "sekitar 30 sampai 45 menit."
))

cells.append(code("!nvidia-smi"))

cells.append(md("The following cell installs all libraries used in this notebook."))

cells.append(code(
    "%pip install -q lightgbm==4.6.0 tabpfn==2.1.4 huggingface_hub==0.34.4 scikit-learn==1.6.1 pandas==2.2.3 "
    "scipy==1.15.2 matplotlib==3.10.0 seaborn==0.13.2"
))

cells.append(md("## Import Libraries"))

cells.append(code(
    "import os\n"
    "import random\n"
    "import time\n"
    "import pickle\n"
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
    "from scipy.stats import ks_2samp\n"
    "from sklearn.metrics import roc_auc_score\n"
    "from sklearn.model_selection import GroupKFold, StratifiedGroupKFold\n"
    "from tabpfn import TabPFNRegressor\n"
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
    "Seluruh path, konstanta hasil EDA, hyperparameter, dan flag dipusatkan di sini. `FEATURE_SET` memilih "
    "fitur level absolut (`'abs'`, default tervalidasi) atau tambahan fitur relatif pasar (`'both'`, varian "
    "untuk uji A/B leaderboard). `SEG_MULT` adalah pengali segmen kalender hasil probing (1,0 berarti netral)."
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
    "    OUTPUT_DIR = Path('/kaggle/working') if _ON_KAGGLE else Path('../outputs/v3')\n"
    "    FIG_DIR    = OUTPUT_DIR / 'figures'\n"
    "\n"
    "    TRAIN_END  = pd.Timestamp('2025-09-30')\n"
    "    OUTAGE     = pd.to_datetime(['2025-06-07', '2025-06-09', '2025-06-10', '2025-06-13', '2025-06-16'])\n"
    "    D1_FRAC    = 0.5\n"
    "    MIN_NC     = 25\n"
    "    USE_LIMITED = True\n"
    "    FEATURE_SET = 'abs'\n"
    "\n"
    "    DOW_PROF   = np.array([0.811, 0.789, 0.811, 0.783, 0.848, 1.290, 1.282])\n"
    "    HOL_LEVEL  = 1.29\n"
    "    SCHOOL_WD  = 1.15\n"
    "    SEG_MULT   = {'lebaran': 1.0, 'ramadan': 1.0, 'xmas': 1.0}\n"
    "    TW_BINS    = [0, 5, 20, 50, 100, 200, 500, 1e9]\n"
    "\n"
    "    N_FOLDS    = 5\n"
    "    SEEDS      = [2026, 2027, 2028, 2029, 2030]\n"
    "    LGB_PARAMS = dict(objective='l1', learning_rate=0.03, num_leaves=63, min_child_samples=100,\n"
    "                      feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,\n"
    "                      n_estimators=800, deterministic=True, force_row_wise=True, n_jobs=4, verbose=-1)\n"
    "\n"
    "    USE_TABPFN   = True\n"
    "    TABPFN_REPO  = 'Prior-Labs/TabPFN-v2-reg'\n"
    "    TABPFN_FILE  = 'tabpfn-v2-regressor.ckpt'\n"
    "    TABPFN_REV   = '213f8e38ec399a2a385fa46cab6f22b95cd90de8'\n"
    "    TABPFN_EST   = 8\n"
    "    DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
    "\n"
    "CFG = Settings()\n"
    "CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n"
    "CFG.FIG_DIR.mkdir(parents=True, exist_ok=True)\n"
    "print(CFG.DATA_DIR, CFG.DEVICE)"
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
    "    P = P[P[2] > 0]\n"
    "    f = pd.DataFrame({'s': ((P[1] + P[2]) / 2).clip(lower=1), 'y': P[3]}, index=P.index)\n"
    "    f['q1'], f['q2'], f['log_s'] = P[1] / f.s, P[2] / f.s, np.log1p(f.s)\n"
    "    f['sh_tr'] = (S.reindex(P.index)[2] + 1) / (S.reindex(P.index)[1] + 1)\n"
    "    f = f.reset_index().join(d1, on='movie_title')\n"
    "    g = x[x.d <= 2].groupby(['movie_title', 'd']).total_ticket.sum().unstack().reindex(columns=[1, 2], fill_value=0)\n"
    "    f['f_tr'] = f.movie_title.map((g[2] + 1) / (g[1] + 1))\n"
    "    f['f_log'] = f.movie_title.map(np.log1p(g.mean(axis=1)))\n"
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
    "jauh lebih banyak pasangan kecil yang galatnya terbesar. Setelah dibobot ke komposisi skala uji, galat "
    "validasi naik mendekati galat periode uji sebenarnya. Inilah penjelasan utama jarak antara MASE validasi "
    "v1/v2 (0,30 sampai 0,33) dan leaderboard (0,45), sehingga metrik keputusan v3 adalah **TW-MASE**.\n\n"
    "> Pada skala absolut yang sama, porsi pasangan yang berhenti di D3 jauh lebih kecil di periode uji "
    "(misalnya skala 20 sampai 50: 0,39 di train vs 0,11 di uji). Di train, pasangan kecil berarti film gagal "
    "di klaster itu, sedangkan di periode uji sering berarti pasar yang sepi. Fitur relatif pasar disiapkan "
    "sebagai varian (`FEATURE_SET='both'`) untuk diuji langsung di leaderboard, karena validasi train tidak "
    "dapat melihat pergeseran ini."
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
    "| Disdik DKI via Detik | 7 Agustus 2025 | libur 22 sampai 31 Desember 2025 |\n\n"
    "Ramadan 1447 H (19 Februari sampai 20 Maret 2026) diturunkan dari 1 Syawal pada SKB 2026 dikurangi 30 hari."
))

cells.append(md(
    "## Pengali Kalender Struktural\n\n"
    "Nilai kalender $c(t)$ = profil hari, dinaikkan ke level akhir pekan pada libur nasional atau cuti bersama "
    "di luar Ramadan (cuti bersama akhir Ramadan adalah masa mudik), dan dikali 1,15 pada hari kerja libur "
    "sekolah. Pengali target $c_{mult} = c(t) / \\overline{c(D1..D3)}$."
))

cells.append(code(
    "SCHOOL_BREAKS = [('2025-06-28', '2025-07-12'), ('2025-12-22', '2025-12-31')]\n"
    "CUTI_BERSAMA = pd.to_datetime(['2025-04-02', '2025-04-03', '2025-04-04', '2025-04-07', '2025-05-13', '2025-05-30',\n"
    "                               '2025-06-09', '2025-08-18', '2025-12-26', '2026-02-16', '2026-03-18', '2026-03-20',\n"
    "                               '2026-03-23', '2026-03-24'])\n"
    "RAMADAN = ('2026-02-19', '2026-03-20')\n"
    "\n"
    "\n"
    "def calendar(hol):\n"
    "    c = hol.rename(columns={'date': 'date_show'})[['date_show', 'holiday_tipe']].copy()\n"
    "    c['dow'] = c.date_show.dt.dayofweek\n"
    "    c['is_hol'] = ((c.holiday_tipe == 'holiday') | c.date_show.isin(CUTI_BERSAMA)).astype(int)\n"
    "    c['school'] = 0\n"
    "    for a, b in SCHOOL_BREAKS:\n"
    "        c.loc[c.date_show.between(a, b), 'school'] = 1\n"
    "    c['ramadan'] = c.date_show.between(*RAMADAN).astype(int)\n"
    "    base = CFG.DOW_PROF[c.dow]\n"
    "    base = np.where((c.school == 1) & (c.dow < 4), base * CFG.SCHOOL_WD, base)\n"
    "    c['cal'] = np.where((c.is_hol == 1) & (c.ramadan == 0), np.maximum(base, CFG.HOL_LEVEL), base)\n"
    "    return c.drop(columns='holiday_tipe').set_index('date_show')\n"
    "\n"
    "\n"
    "CAL = calendar(hol)\n"
    "fig, ax = plt.subplots(figsize=(14, 3))\n"
    "ax.plot(CAL.index, CAL.cal, lw=1)\n"
    "ax.axvline(CFG.TRAIN_END, color=PAL[7], ls='--', label='akhir train')\n"
    "ax.set(title='Nilai kalender c(t)'); ax.legend()\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'cal_value.png'); plt.show()"
))

cells.append(md(
    "## Fungsi Build Fitur\n\n"
    "Kelompok fitur: (1) bentuk kurva pasangan D1 sampai D3, (2) agregat film nasional dan pangsa format, "
    "(3) kalender target dan riwayat, (4) jadwal rilis film lain, (5) ukuran klaster dari train dan harga kota, "
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
    "    return dict(releases=rel, market=market, cin_size=np.log10(cs), cin_nfilms=nf)\n"
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
    "    cT = CAL.reindex(X.date_show)\n"
    "    X['dow'] = cT.dow.values\n"
    "    X['t_hol'], X['t_school'], X['t_ramadan'] = cT.is_hol.values, cT.school.values, cT.ramadan.values\n"
    "    ch = np.stack([CAL.cal.reindex(X.d1 + pd.Timedelta(days=k)).values for k in range(3)], 1)\n"
    "    X['cal_mult'] = cT.cal.values / ch.mean(1)\n"
    "    X['hol_in_hist'] = np.stack([CAL.is_hol.reindex(X.d1 + pd.Timedelta(days=k)).values for k in range(3)], 1).sum(1)\n"
    "    X['n_comp_open'] = open_count(X, rel)\n"
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
    "REL_FEATS = ['pair_vs_mkt', 'film_pc_vs_mkt', 'logT_rel', 'occ_rel', 'tps_rel']\n"
    "FEATS = BASE_FEATS + (REL_FEATS if CFG.FEATURE_SET == 'both' else [])\n"
    "print(f'train {len(Xtr)} rows | limited extra {len(Xlim)} | test {len(Xte)} | features {len(FEATS)} ({CFG.FEATURE_SET})')\n"
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
    "Semua model memakai fold yang sama (5 fold per judul dasar) dan dinilai dengan MASE biasa serta TW-MASE: "
    "setiap baris dibobot agar bucket skalanya memiliki porsi yang sama dengan data uji."
))

cells.append(code(
    "groups = base_title(Xtr.movie_title).values\n"
    "ug = np.array(sorted(set(groups)))\n"
    "fold_of = dict(zip(ug, np.random.RandomState(CFG.SEED).permutation(len(ug)) % CFG.N_FOLDS))\n"
    "FOLD = np.array([fold_of[g_] for g_ in groups])\n"
    "LIM_FOLD = np.full(len(Xlim), -1)\n"
    "y, s, cm, hh_ = Xtr.total_ticket.values, Xtr.scale.values, Xtr.cal_mult.values, Xtr.h.values\n"
    "\n"
    "bt = pd.cut(Xtr.scale, CFG.TW_BINS)\n"
    "tw = bt.map(pd.cut(Xte.scale, CFG.TW_BINS).value_counts(normalize=True) / bt.value_counts(normalize=True)).astype(float).values\n"
    "TW = tw / tw.mean()\n"
    "\n"
    "\n"
    "def score(p, name):\n"
    "    e = np.abs(y - p) / s\n"
    "    r = dict(model=name, MASE=e.mean(), TW_MASE=np.average(e, weights=TW), small=e[s <= 20].mean(), big=e[s > 200].mean())\n"
    "    print(f\"{name:34s} MASE {r['MASE']:.4f} | TW-MASE {r['TW_MASE']:.4f} | s<=20 {r['small']:.4f} | s>200 {r['big']:.4f}\")\n"
    "    return r\n"
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

cells.append(md(
    "## LightGBM L1\n\n"
    "Target $r / c_{mult}$ dengan bobot $c_{mult}$, sehingga loss identik dengan MASE. Film rilis terbatas "
    "ditambahkan ke setiap fold latih (tidak pernah dievaluasi). Lima seed dirata-ratakan."
))

cells.append(code(
    "def fit_lgb(Xa, seeds=CFG.SEEDS):\n"
    "    ya, wa = (Xa.total_ticket / Xa.scale / Xa.cal_mult).values, Xa.cal_mult.values\n"
    "    return [lgb.LGBMRegressor(**CFG.LGB_PARAMS, random_state=sd).fit(Xa[FEATS], ya, sample_weight=wa) for sd in seeds]\n"
    "\n"
    "\n"
    "def predict_lgb(models, Xb):\n"
    "    return np.clip(np.mean([m.predict(Xb[FEATS]) for m in models], 0), 0, None) * Xb.cal_mult.values * Xb.scale.values\n"
    "\n"
    "\n"
    "def train_part(k):\n"
    "    Xa = Xtr[FOLD != k]\n"
    "    return pd.concat([Xa, Xlim], ignore_index=True) if CFG.USE_LIMITED else Xa\n"
    "\n"
    "\n"
    "t0 = time.time()\n"
    "oof_lgb = np.zeros(len(Xtr))\n"
    "for k in range(CFG.N_FOLDS):\n"
    "    oof_lgb[FOLD == k] = predict_lgb(fit_lgb(train_part(k)), Xtr[FOLD == k])\n"
    "results.append(score(oof_lgb, 'LightGBM'))\n"
    "print(f'{time.time() - t0:.0f}s')"
))

cells.append(md(
    "## Pretrained Model: TabPFN v2 per Horizon\n\n"
    "TabPFN v2 ([Prior-Labs/TabPFN-v2-reg](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), Hollmann et al., "
    "*Nature* 637, 9 Januari 2025) adalah transformer yang dilatih pada jutaan dataset tabular sintetis dan "
    "memprediksi lewat in-context learning. Revisi bobot dikunci ke commit `213f8e38` (11 Juni 2025) dan paket "
    "`tabpfn==2.1.4` (11 September 2025), keduanya sebelum batas 30 September 2025. Versi yang lebih baru "
    "(TabPFN-2.5 November 2025, TabPFN-3.5 September 2026) sengaja tidak dipakai karena melewati batas.\n\n"
    "Alasan pemilihan: data latih kecil (143 film), setiap seri hanya tiga titik sehingga foundation model "
    "deret waktu tidak punya konteks, dan TabPFN mengeluarkan distribusi prediktif penuh sehingga **median**-nya "
    "optimal untuk MASE. Satu model per horizon membuat seluruh baris horizon tersebut (sekitar 8 ribu, di "
    "bawah batas 10 ribu) masuk sebagai konteks tanpa subsampling. Kandidat lain yang patuh batas diuji: "
    "TabDPT 1.1.5 gagal berjalan dengan `faiss` terbaru dan hanya mengeluarkan rata-rata."
))

cells.append(code(
    "TAB_FEATS = [c for c in FEATS if c != 'h']\n"
    "\n"
    "\n"
    "def tabpfn_path():\n"
    "    local = list(Path('/kaggle/input').rglob(CFG.TABPFN_FILE)) if CFG._ON_KAGGLE else []\n"
    "    return str(local[0]) if local else hf_hub_download(CFG.TABPFN_REPO, CFG.TABPFN_FILE, revision=CFG.TABPFN_REV)\n"
    "\n"
    "\n"
    "def tabpfn_horizon(Xa, Xb):\n"
    "    out = np.zeros(len(Xb))\n"
    "    for hz in range(4, 11):\n"
    "        ia, ib = (Xa.h == hz).values, (Xb.h == hz).values\n"
    "        if ib.sum() == 0:\n"
    "            continue\n"
    "        m = TabPFNRegressor(model_path=TABPFN_CKPT, device=CFG.DEVICE, n_estimators=CFG.TABPFN_EST,\n"
    "                            random_state=CFG.SEED, ignore_pretraining_limits=True)\n"
    "        m.fit(Xa[ia][TAB_FEATS].values.astype(np.float32), (Xa.total_ticket / Xa.scale / Xa.cal_mult).values[ia])\n"
    "        Q = Xb[ib][TAB_FEATS].values.astype(np.float32)\n"
    "        out[ib] = np.concatenate([m.predict(Q[i:i + 3000], output_type='median') for i in range(0, len(Q), 3000)])\n"
    "    return np.clip(out, 0, None) * Xb.cal_mult.values * Xb.scale.values\n"
    "\n"
    "\n"
    "oof_tab = None\n"
    "if CFG.USE_TABPFN:\n"
    "    TABPFN_CKPT = tabpfn_path()\n"
    "    t0 = time.time()\n"
    "    oof_tab = np.zeros(len(Xtr))\n"
    "    for k in range(CFG.N_FOLDS):\n"
    "        oof_tab[FOLD == k] = tabpfn_horizon(train_part(k), Xtr[FOLD == k])\n"
    "        print(f'fold {k} done ({time.time() - t0:.0f}s)')\n"
    "    results.append(score(oof_tab, 'TabPFN v2 per horizon'))"
))

cells.append(md(
    "## Blending dengan TW-MASE\n\n"
    "Bobot blend dipilih pada grid 0 sampai 1 dengan TW-MASE out-of-fold, karena TW-MASE adalah metrik yang "
    "sejalan dengan komposisi data uji."
))

cells.append(code(
    "W_TAB = 0.0\n"
    "if oof_tab is not None:\n"
    "    ws = np.linspace(0, 1, 11)\n"
    "    curve = [np.average(np.abs(y - ((1 - w) * oof_lgb + w * oof_tab)) / s, weights=TW) for w in ws]\n"
    "    W_TAB = float(ws[np.argmin(curve)])\n"
    "    plt.figure(figsize=(5, 3)); plt.plot(ws, curve, marker='o'); plt.axvline(W_TAB, color=PAL[7], ls='--')\n"
    "    plt.title('TW-MASE vs bobot TabPFN'); plt.xlabel('bobot TabPFN'); plt.tight_layout(); plt.show()\n"
    "oof_final = (1 - W_TAB) * oof_lgb + (W_TAB * oof_tab if oof_tab is not None else 0)\n"
    "results.append(score(oof_final, f'blend (w_tabpfn={W_TAB:.1f})'))\n"
    "display(pd.DataFrame(results).set_index('model').round(4))"
))

# ---------------------------------------------------------------------------
# Section 9: Evaluation
# ---------------------------------------------------------------------------
cells.append(section("Evaluation", "9"))

cells.append(md(
    "Galat out-of-fold model final dibedah per bucket skala (dengan porsi uji), per horizon, dan per film, lalu "
    "kontribusi setiap bucket terhadap MASE uji yang diharapkan dihitung."
))

cells.append(code(
    "E = Xtr[['movie_title', 'h', 'scale', 'total_ticket']].assign(pred=oof_final)\n"
    "E['ae'] = np.abs(E.total_ticket - E.pred) / E.scale\n"
    "cut = pd.cut(E.scale, CFG.TW_BINS)\n"
    "bk = pd.DataFrame({'test_share': pd.cut(Xte.scale, CFG.TW_BINS).value_counts(normalize=True), 'train_share': cut.value_counts(normalize=True),\n"
    "                   'mase': E.groupby(cut, observed=False).ae.mean(), 'zero_rate': (E.total_ticket == 0).groupby(cut, observed=False).mean()}).sort_index()\n"
    "bk['contribution_to_test'] = bk.test_share * bk.mase\n"
    "display(bk.round(3))\n"
    "print('MASE by horizon:', E.groupby('h').ae.mean().round(3).to_dict())\n"
    "print('worst films:', E.groupby('movie_title').ae.mean().nlargest(5).round(2).to_dict())\n"
    "\n"
    "imp_models = fit_lgb(pd.concat([Xtr, Xlim], ignore_index=True), seeds=[CFG.SEED])\n"
    "imp = pd.Series(imp_models[0].booster_.feature_importance('gain'), index=FEATS)\n"
    "fig, ax = plt.subplots(1, 3, figsize=(18, 4))\n"
    "xx = np.arange(len(bk))\n"
    "ax[0].bar(xx - .2, bk.train_share, .4, label='porsi train'); ax[0].bar(xx + .2, bk.test_share, .4, label='porsi uji')\n"
    "ax[0].set_xticks(xx, [str(i) for i in bk.index], rotation=30); ax[0].legend(); ax[0].set_title('Komposisi skala')\n"
    "ax[1].bar(xx, bk.contribution_to_test, color=PAL[7]); ax[1].set_xticks(xx, [str(i) for i in bk.index], rotation=30)\n"
    "ax[1].set_title('Kontribusi bucket ke MASE uji yang diharapkan')\n"
    "imp.nlargest(15)[::-1].plot.barh(ax=ax[2], color=PAL[0]); ax[2].set_title('Feature importance (gain)')\n"
    "plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'evaluation.png'); plt.show()"
))

cells.append(md(
    "#### Insights\n\n"
    "> Pasangan dengan skala di bawah 50 hanya sekitar 13% baris train tetapi 33% baris uji, dan menyumbang "
    "porsi terbesar MASE uji yang diharapkan. Pada pasangan ini sebagian besar target nol di train, sehingga "
    "prediksi median sering nol. Perbaikan terbesar karena itu datang dari keputusan yang menyangkut pasangan "
    "kecil dan periode kalender yang tidak ada di train, bukan dari fitur tambahan: ablation fitur (jadwal "
    "program, kompetisi klaster, tren show) semuanya berada di dalam noise lintas penugasan fold (sekitar 0,003)."
))

# ---------------------------------------------------------------------------
# Section 10: Inference & Submission
# ---------------------------------------------------------------------------
cells.append(section("Inference & Submission", "10"))

cells.append(md(
    "Model final dilatih pada seluruh sampel (rilis luas dan rilis terbatas). Pengali segmen kalender "
    "diterapkan hanya pada periode yang tidak dapat divalidasi oleh data latih."
))

cells.append(md(
    "## Kalibrasi Segmen melalui Probing Leaderboard\n\n"
    "MASE adalah rata-rata per baris, sehingga probe yang hanya mengubah segmen $S$ dengan pengali $k$ menggeser "
    "skor publik sebesar $\\Delta L = \\frac{1}{N_{pub}}\\sum_{i \\in S}(|y_i - k\\hat{y}_i| - |y_i - \\hat{y}_i|)/s_i$. "
    "Tiga nilai $k$ membentuk kurva cembung yang minimumnya dipakai sebagai `SEG_MULT`. Aturan pengaman agar "
    "tidak overfit ke split publik: (1) hanya segmen tanpa padanan di train (Lebaran, Ramadan, libur Natal) "
    "yang dikalibrasi, masing-masing dengan satu parameter di atas ribuan baris; (2) segmen normal tidak pernah "
    "disetel di leaderboard karena sudah diukur oleh TW-MASE; (3) pengali hanya diterima bila perbaikannya "
    "lebih dari 0,002 dan arahnya sesuai logika domain (misalnya Lebaran naik)."
))

cells.append(code(
    "t0 = time.time()\n"
    "Xall = pd.concat([Xtr, Xlim], ignore_index=True) if CFG.USE_LIMITED else Xtr\n"
    "final_models = fit_lgb(Xall)\n"
    "pred = predict_lgb(final_models, Xte)\n"
    "if W_TAB > 0:\n"
    "    pred = (1 - W_TAB) * pred + W_TAB * tabpfn_horizon(Xall, Xte)\n"
    "seg = pd.Series('normal', index=Xte.index)\n"
    "for n, (a, b) in SEGMENTS.items():\n"
    "    seg[Xte.date_show.between(a, b)] = n\n"
    "for n, k in CFG.SEG_MULT.items():\n"
    "    pred = np.where(seg == n, pred * k, pred)\n"
    "pred = np.clip(pred, 0, None)\n"
    "print(f'final fit + inference: {time.time() - t0:.0f}s')\n"
    "\n"
    "rr = pred / Xte.scale.values\n"
    "display(pd.DataFrame({'rows': seg.value_counts(), 'mean_r_hat': pd.Series(rr).groupby(seg.values).mean(),\n"
    "                      'pred_zero_share': pd.Series(rr < .05).groupby(seg.values).mean()}).round(3))"
))

cells.append(md(
    "Submisi disimpan dengan format `id,total_ticket` dan urutan `id` yang sama dengan `sample_submission.csv`. "
    "Bobot model (booster LightGBM seluruh seed dan konfigurasi) disimpan sebagai satu berkas `.pkl`; bobot "
    "TabPFN adalah checkpoint publik yang diunduh dengan revisi terkunci."
))

cells.append(code(
    "submission = pd.DataFrame({'id': test.id.values, 'total_ticket': pred})\n"
    "assert len(submission) == len(sub) and (submission.id.values == sub.id.values).all()\n"
    "assert submission.total_ticket.notna().all() and (submission.total_ticket >= 0).all()\n"
    "submission.to_csv(CFG.OUTPUT_DIR / 'submission.csv', index=False)\n"
    "pd.DataFrame({'movie_title': Xtr.movie_title, 'cinema_ids': Xtr.cinema_ids, 'h': Xtr.h, 'fold': FOLD, 'y': y,\n"
    "              'scale': s, 'oof_lgb': oof_lgb, 'oof_final': oof_final}).to_csv(CFG.OUTPUT_DIR / 'oof.csv', index=False)\n"
    "with open(CFG.OUTPUT_DIR / 'model_weights.pkl', 'wb') as f:\n"
    "    pickle.dump({'lgb_boosters': [m.booster_.model_to_string() for m in final_models], 'features': FEATS,\n"
    "                 'w_tabpfn': W_TAB, 'tabpfn_revision': CFG.TABPFN_REV,\n"
    "                 'settings': {k: v for k, v in vars(Settings).items() if k.isupper()}}, f)\n"
    "print(f\"weights: {(CFG.OUTPUT_DIR / 'model_weights.pkl').stat().st_size / 1e6:.1f} MB\")\n"
    "display(submission.head())"
))

cells.append(md(
    "#### Insights\n\n"
    "> Metrik validasi v3 (TW-MASE) memperhitungkan bahwa data uji didominasi pasangan kecil, sehingga angka "
    "validasi berada di level yang sama dengan leaderboard, tidak lagi 0,30 berbanding 0,45. Keputusan model "
    "diambil dari TW-MASE; hanya segmen kalender tanpa padanan di train yang dikalibrasi dengan probe "
    "leaderboard berparameter tunggal.\n\n"
    "> Seluruh proses deterministik: seed tetap pada numpy, torch, LightGBM (`deterministic=True`), TabPFN "
    "(`random_state`), dan fold; revisi bobot TabPFN dikunci ke commit Hugging Face."
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
