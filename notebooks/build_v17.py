"""v17: purged calendar blocks, early backtests, official genres and shared-horizon ablations."""
import ast
import copy
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
nb=copy.deepcopy(json.loads((HERE/'v16.ipynb').read_text()))
cells=nb['cells']
for c in cells:
    c['source']=''.join(c['source'])
    if c['cell_type']=='code':
        c['outputs'],c['execution_count']=[],None


def md(i,text):
    assert cells[i]['cell_type']=='markdown'
    cells[i]['source']=text


def code(i,text):
    assert cells[i]['cell_type']=='code'
    cells[i]['source']=text.strip()


def replace(old,new):
    found=[c for c in cells if c['cell_type']=='code' and old in c['source']]
    assert len(found)==1 and found[0]['source'].count(old)==1,old[:80]
    found[0]['source']=found[0]['source'].replace(old,new)


def function(i,name,new):
    src=cells[i]['source'];tree=ast.parse(src)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    assert len(nodes)==1,name
    node=nodes[0];lines=src.splitlines()
    cells[i]['source']='\n'.join(lines[:node.lineno-1]+new.strip().splitlines()+lines[node.end_lineno:])


md(0,'---\n\n# JOINTS x INSPIRE UGM 2026 (v17)\n\n*Validasi blok waktu dengan purge, metadata resmi lengkap, dan pembelajaran bersama D4 sampai D10.*')
md(4,'## Overview\n\nv16 menolak log1p: gain TW hanya 0,000004 dan ketiga backtest memburuk. '
   'Submisi akhirnya identik dengan v15 (public 0,39918 menurut peserta). v17 menguji dua hipotesis '
   'berbeda: informasi genre resmi yang belum digunakan dan pembelajaran lintas horizon. Tidak ada '
   'target skor yang dijanjikan; 0,33662 memerlukan penurunan error public sekitar 15,7%.')
md(8,'## Approach\n\nKontrol tetap LightGBM 0,1 + TabPFN-3.5 int7 per horizon 0,9. '
   'Kandidat G menambah genre resmi lengkap pada TabPFN per horizon. Kandidat H memakai fitur '
   'baseline termasuk horizon h dalam satu TabPFN, dengan konteks maksimal 8.000 baris yang '
   'disampel seimbang menurut horizon. Keduanya diuji terpisah, tidak digabung otomatis. '
   'Target, lambda 0,5, kalender, dan analog Lebaran dibekukan.')
md(66,'## Fitur Relatif dan Genre Lengkap\n\nFitur relatif lama tetap dihitung untuk audit, '
   'tetapi tidak dipakai kontrol. Token genre lengkap diambil dari `movies.csv` resmi dan '
   'dipetakan ke judul dasar, termasuk format IMAX/3D. Kandidat genre menambah token yang belum '
   'ada dalam sembilan flag baseline dan flag metadata tidak tersedia. Kosakata tidak memakai label target.')
md(68,'#### Insights\n\n> Audit menunjukkan baris dengan genre terbuang membawa 26,1% error '
   'v16. Namun 78,7% kontribusi error Adventure berasal dari satu film. Angka ini alasan untuk '
   'ablation, bukan bukti bahwa genre akan memperbaiki generalisasi.')
md(73,'Kontrol dan dua kandidat dilatih ulang pada split identik. Validasi baru tidak dapat '
   'dihitung dengan sekadar mengelompokkan ulang prediksi OOF lama. Seluruh fitting berjalan di Kaggle.')
md(74,'## Dasar Eksperimen\n\nDua hipotesis dibekukan sebelum run. Genre resmi memiliki 29 '
   'token, sementara baseline hanya memakai sembilan. Model per horizon juga membagi pembelajaran '
   'menjadi tujuh masalah terpisah; kandidat pooled menguji apakah berbagi konteks membantu '
   'lintasan D4 sampai D10. Jumlah baris konteks dibatasi untuk T4, sehingga ini menguji resep '
   'pooling beserta samplingnya, bukan efek pooling murni. Eksperimen 47 terdahulu pada satu fold '
   'dan dua horizon justru menemukan pooled 45 ribu baris lebih buruk dan 17 kali lebih lambat. '
   'Kandidat ini adalah pemeriksaan ulang dengan batas konteks dan validasi lebih luas, bukan '
   'metode yang sudah terbukti unggul.\n\n#### Insights\n\n> '
   'Tidak ada data eksternal baru yang masuk fitur. Pemeriksaan tanggal rilis eksternal pada '
   'beberapa film bererror besar cocok dengan D1 simulator; cakupannya belum membuktikan semua D1 benar.')
md(75,'## Validasi Blok Waktu dan Backtest\n\nMinggu rilis diurutkan lalu dibagi menjadi lima '
   'blok berurutan. Semua jendela D1 sampai D10 train yang menyentuh rentang validasi dikeluarkan, '
   'termasuk limited. Statistik klaster mengeluarkan film validasi dan transaksi dalam rentang '
   'blok tersebut. Ini stress test kohort dengan train sebelum maupun sesudah blok, bukan '
   'forecasting kronologis. Backtest terpisah pada Mei sampai September hanya memakai target '
   'yang selesai sebelum awal bulan. Mei hanya memiliki sekitar sepuluh film latih dan dilaporkan '
   'sebagai cold-start stress; rerata Juni–September dan Juli–September juga disimpan. '
   'Rekonstruksi D1 masih memakai aturan cakupan lama yang melihat seluruh riwayat.')
md(79,'## Statistik Klaster pada Split Baru\n\nStatistik ukuran klaster dan jumlah show '
   'dihitung ulang tanpa film validasi dan tanpa transaksi pada interval validasi. Ini '
   'memperbaiki pemisahan informasi tingkat kalender. Fitur pesaing dari jendela history resmi '
   'tetap tersedia sebagaimana paket kompetisi batch, bukan skenario produksi online.')
md(85,'## Kandidat Foundation Model\n\n`tp35_int7` menjadi kontrol per horizon, `tp35_genre` '
   'menambah genre resmi lengkap, dan `tp35_pool` memakai satu konteks lintas horizon dengan '
   'fitur baseline. Ketiganya memakai satu checkpoint int7 yang identik dan tidak memakai '
   'log1p. Paling banyak satu varian masuk berkas final. Tidak ada AI API untuk inference.')
md(91,'## Koreksi Pencopotan Dibekukan\n\nSemua varian menggunakan lambda 0,5. '
   'Parameter ini tidak dituning ulang pada split baru.')
md(93,'## Aturan Adopsi\n\nSetiap kandidat dibandingkan dengan blend kontrol yang sama: '
   'TW turun minimal 0,0015, MASE tanpa bobot tidak memburuk, rerata temporal Mei–September '
   'serta Juli–September tidak memburuk, tidak ada bulan memburuk lebih dari 0,003, menang '
   'setidaknya tiga blok, dan batas atas bootstrap dua-minggu delta error berada di bawah nol. '
   'Interval 97,5% per kandidat dipakai sebagai penyesuaian sederhana untuk dua kandidat '
   '(bukan koreksi atas seluruh sejarah eksperimen). Kandidat juga harus tidak memperburuk '
   'error growth atau D4–D5 lebih dari 0,003. Jika keduanya lolos, pilih TW terkecil. '
   'Jika tidak ada yang lolos, kontrol tetap dipakai.')
md(95,'#### Insights\n\n> Split lebih ketat dapat membuat MASE naik. Itu tidak berarti model '
   'lebih buruk di test; angka baru harus dibandingkan antar kandidat pada split yang sama. '
   'Semua evaluasi tetap dipengaruhi sejarah riset dataset, bukan holdout baru yang tidak pernah dilihat.')
md(108,'#### Insights\n\n> Keputusan, skor per model, manifest split, konteks sampling, OOF, '
   'prediksi temporal, dan prediksi sebelum override disimpan. Jika semua kandidat gagal dan '
   'submission sama dengan v15/v16, jangan menggunakan slot submission untuk duplikat. '
   'Tidak ada bukti skor 0,33662 sampai evaluasi dan submission nyata menunjukkan demikian.')
replace("Path('../outputs/v16')","Path('../outputs/v17')")
replace("FM_LIST      = ('tp35_int7', 'tp35_log')","FM_LIST      = ('tp35_int7', 'tp35_genre', 'tp35_pool')\n    POOL_CONTEXT = 8000")
replace("MONTHS = ['2025-07', '2025-08', '2025-09']","MONTHS = ['2025-05', '2025-06', '2025-07', '2025-08', '2025-09']")

# Add exact genre-token flags without changing the original feature columns.
cells[67]['source'] += """

META = movies.copy()
META['original_title'] = META.original_title.str.strip()
assert META.original_title.is_unique
META = META.set_index('original_title')
GENRE_TOKENS = META.genre.fillna('').str.split(',').map(lambda z: {v.strip() for v in z if v.strip()})
EXTRA_GENRES = sorted(set().union(*GENRE_TOKENS) - set(GENRES))
EXTRA_GENRE_FEATS = [f'genre_extra_{i}' for i in range(len(EXTRA_GENRES))] + ['metadata_missing']

def complete_genres(X):
    X = X.copy()
    b = base_title(X.movie_title)
    tokens = b.map(GENRE_TOKENS).map(lambda z: z if isinstance(z, set) else set())
    for i, token in enumerate(EXTRA_GENRES):
        X[f'genre_extra_{i}'] = tokens.map(lambda z: int(token in z))
    X['metadata_missing'] = (~b.isin(META.index) | b.map(META.genre).isna()).astype(int)
    assert np.isfinite(X[EXTRA_GENRE_FEATS].values).all()
    return X

Xtr, Xlim, Xte = [complete_genres(X_) for X_ in (Xtr, Xlim, Xte)]
print('Additional official genre tokens:', EXTRA_GENRES)
"""
replace("fold_of_week = dict(zip(weeks_, np.random.RandomState(CFG.SEED).permutation(len(weeks_)) % CFG.N_FOLDS))",
        "fold_of_week = {w: k for k, block in enumerate(np.array_split(weeks_, CFG.N_FOLDS)) for w in block}")
replace("    a, b = FOLD != k, FOLD == k", """    b = FOLD == k
    lo, hi = Xtr.d1[b].min(), Xtr.d1[b].max() + pd.Timedelta(days=9)
    a = ((Xtr.d1 + pd.Timedelta(days=9) < lo) | (Xtr.d1 > hi)).values""")
function(76,'train_part',"""
def train_part(k):
    v = FOLD == k
    lo, hi = Xtr.d1[v].min(), Xtr.d1[v].max() + pd.Timedelta(days=9)
    excluded = set(groups[v])
    ready = lambda X: X[((X.d1 + pd.Timedelta(days=9) < lo) | (X.d1 > hi)) & ~base_title(X.movie_title).isin(excluded)]
    a = pd.concat([ready(Xtr), ready(Xlim)], ignore_index=True) if CFG.USE_LIMITED else ready(Xtr)
    assert ((a.d1 + pd.Timedelta(days=9) < lo) | (a.d1 > hi)).all()
    assert not set(base_title(a.movie_title)) & excluded
    return a
""")
old="""LIMITED_WEEK = ((Xlim.d1 - pd.Timestamp('2025-03-31')).dt.days // 7)
STATS_FOLD = {}
for k in range(CFG.N_FOLDS):
    excluded = set(groups[FOLD == k]) | set(base_title(Xlim.loc[LIMITED_WEEK.isin(WEEK[FOLD == k]), 'movie_title']))
    STATS_FOLD[k] = cin_stats(train[~TRAIN_BASE.isin(excluded)])"""
replace(old,"""STATS_FOLD = {}
split_manifest = []
for k in range(CFG.N_FOLDS):
    v = FOLD == k
    lo, hi = Xtr.d1[v].min(), Xtr.d1[v].max() + pd.Timedelta(days=9)
    excluded = set(groups[v])
    raw_ok = ~TRAIN_BASE.isin(excluded) & ~train.date_show.between(lo, hi)
    STATS_FOLD[k] = cin_stats(train[raw_ok])
    for role, frame in [('train', train_part(k)), ('validation', Xtr[v])]:
        for title in frame.movie_title.unique():
            split_manifest.append(dict(fold=k, role=role, movie_title=title, start=lo, end=hi))
pd.DataFrame(split_manifest).to_csv(CFG.OUTPUT_DIR / 'split_manifest.csv', index=False)""")
replace("r['temporal_mean'] = np.mean(list(tm.values()))", """r['temporal_mean'] = np.mean(list(tm.values()))
    r['temporal_jun_sep'] = np.mean([v for m, v in tm.items() if m >= '2025-06'])
    r['temporal_jul_sep'] = np.mean([v for m, v in tm.items() if m >= '2025-07'])""")
replace("Xa_, Xb_ = with_cin(Xa, stats), with_cin(Xtr[va], stats)","""assert (Xa.d1 + pd.Timedelta(days=9) < cut).all()
        assert not set(base_title(Xa.movie_title)) & set(base_title(Xtr.loc[va, 'movie_title']))
        Xa_, Xb_ = with_cin(Xa, stats), with_cin(Xtr[va], stats)""")

function(86,'make_tabpfn',"""
def make_tabpfn(version, ckpt, n_est, variant='tp35_int7'):
    def fn(Xa, Xb):
        pooled = variant == 'tp35_pool'
        tf = list(FEATS) if pooled else [c for c in FEATS if c != 'h']
        if variant == 'tp35_genre':
            tf += EXTRA_GENRE_FEATS
        out = np.zeros((len(Xb), len(TAB_QS)))
        if pooled:
            pieces = []
            for hz in range(4, 11):
                part = Xa[Xa.h == hz].sort_values(['movie_title', 'cinema_ids', 'h'])
                pieces.append(part.sample(n=min(len(part), CFG.POOL_CONTEXT // 7), random_state=CFG.SEED + hz))
            context = pd.concat(pieces, ignore_index=True)
            assert len(context) <= CFG.POOL_CONTEXT and not context.duplicated(KEY + ['h']).any()
            print(f'pooled context {len(context)}/{len(Xa)} rows | {base_title(context.movie_title).nunique()} base films')
        else:
            context = Xa
        for hz in ([None] if pooled else range(4, 11)):
            a = context if hz is None else context[context.h == hz]
            ib = np.ones(len(Xb), dtype=bool) if hz is None else (Xb.h == hz).values
            if not ib.any():
                continue
            assert len(a) > 0
            m = TabPFNRegressor.create_default_for_version(version, model_path=ckpt, device=CFG.DEVICE,
                    n_estimators=n_est, random_state=CFG.SEED, ignore_pretraining_limits=True)
            m.fit(a[tf].values.astype(np.float32), r_target(a))
            B = Xb.loc[ib, tf].values.astype(np.float32)
            out[ib] = np.concatenate([np.asarray(m.predict(B[i:i + 3000], output_type='quantiles', quantiles=list(TAB_QS))).T
                                     for i in range(0, len(B), 3000)])
            del m
            torch.cuda.empty_cache()
        assert np.isfinite(out).all()
        return np.sort(out, axis=1)
    return fn
""")
replace("if n in ('tp35_int7', 'tp35_log'):","if n in ('tp35_int7', 'tp35_genre', 'tp35_pool'):")
replace("make_tabpfn(ModelVersion.V3_5, ckpt, CFG.TP35_EST, log_target=(n == 'tp35_log'))",
        "make_tabpfn(ModelVersion.V3_5, ckpt, CFG.TP35_EST, variant=n)")
replace("assert set(COMP) == {'lgb', 'tp35_int7', 'tp35_log'}",
        "assert set(COMP) == {'lgb', 'tp35_int7', 'tp35_genre', 'tp35_pool'}")

code(94,"""
CONTROL_WEIGHTS = {'lgb': 0.1, 'tp35_int7': 0.9}
control_pred, control_temp = blend(CONTROL_WEIGHTS, COMP)
control_lens = lens(control_pred, control_temp)
WEIGHTS = CONTROL_WEIGHTS
ablation_rows, gate_rows, candidate_outputs = [], [], {}
best_tw = control_lens['TW']
for name in ['tp35_genre', 'tp35_pool']:
    weights = {'lgb': 0.1, name: 0.9}
    cp, ct = blend(weights, COMP)
    candidate_outputs[name] = (cp, ct)
    cl = lens(cp, ct)
    delta = {k: cl[k] - control_lens[k] for k in cl}
    units = pd.DataFrame({'unit': WEEK // 2, 'w': TW,
        'wd': (np.abs(y - cp) - np.abs(y - control_pred)) / s * TW}).groupby('unit')[['w', 'wd']].sum()
    ix = np.random.default_rng(CFG.SEED).integers(len(units), size=(4000, len(units)))
    ci = np.quantile(units.wd.values[ix].sum(1) / units.w.values[ix].sum(1), [.0125, .9875])
    error_delta = (np.abs(y - cp) - np.abs(y - control_pred)) / s
    growth = y > Xtr.y3.values
    early = Xtr.h.values <= 5
    gates = dict(TW_gain=delta['TW'] <= -CFG.ABL_MIN_GAIN,
                 official_MASE=delta['MASE'] <= 0,
                 temporal_mean=delta['temporal_mean'] <= 0,
                 temporal_jun_sep=delta['temporal_jun_sep'] <= 0,
                 temporal_jul_sep=delta['temporal_jul_sep'] <= 0,
                 each_month=max(delta[f'temp_{m}'] for m in MONTHS) <= CFG.ABL_MONTH_TOL,
                 fold_wins=sum(delta[f'fold{k}'] < 0 for k in range(CFG.N_FOLDS)) >= CFG.ABL_MIN_FOLDS,
                 block_bootstrap=ci[1] < 0,
                 growth_guard=np.average(error_delta[growth], weights=TW[growth]) <= .003,
                 early_guard=np.average(error_delta[early], weights=TW[early]) <= .003,
                 weights_size=sum(SIZE_MB[n] for n in weights) + CFG.SIZE_MARGIN_MB <= CFG.MAX_WEIGHTS_MB)
    passed = all(gates.values())
    gate_rows.append(dict(candidate=name, **gates, passed=passed, delta_ci_low=ci[0], delta_ci_high=ci[1]))
    ablation_rows.extend([dict(candidate=name, kind='score', **cl), dict(candidate=name, kind='delta', **delta)])
    if passed and cl['TW'] < best_tw:
        WEIGHTS, best_tw = weights, cl['TW']
ablation_rows.append(dict(candidate='control', kind='score', **control_lens))
display(pd.DataFrame(ablation_rows)); display(pd.DataFrame(gate_rows))
pd.DataFrame(ablation_rows).to_csv(CFG.OUTPUT_DIR / 'ablation.csv', index=False)
pd.DataFrame(gate_rows).to_csv(CFG.OUTPUT_DIR / 'adoption.csv', index=False)
print('chosen weights:', WEIGHTS)
oof_final, tp_final = blend(WEIGHTS, COMP)
FEATURE_CFG = 'v17 purged contiguous blocks; fixed candidate recipes'
base_l = lens(*COMP['lgb'])
results.append(score(oof_final, 'selected fixed blend'))
fig, ax = plt.subplots(figsize=(9, 4))
pd.DataFrame([dict(name='control', **control_lens)] +
    [dict(name=n, **lens(*candidate_outputs[n])) for n in candidate_outputs]).set_index('name')[['TW', 'temporal_mean', 'temporal_jul_sep']].plot.bar(ax=ax, rot=0)
ax.set(title='New fits on identical purged/temporal splits', ylabel='MASE (lower is better)')
plt.tight_layout(); plt.savefig(CFG.FIG_DIR / 'v17_ablation.png'); plt.show()
""")
replace("oof_df['oof_candidate'] = candidate_pred", """for name, (cp, ct) in candidate_outputs.items():
    oof_df[f'oof_{name}'] = cp""")
replace("frame['control'], frame['candidate'], frame['selected'] = control_temp[m][1], candidate_temp[m][1], pr", """frame['control'], frame['selected'] = control_temp[m][1], pr
    for name, (_, ct) in candidate_outputs.items():
        frame[name] = ct[m][1]""")
replace("payload = {'target_transform': {n: ('log1p' if n == 'tp35_log' else 'identity') for n in WEIGHTS},", """payload = {'target_transform': {n: 'identity' for n in WEIGHTS},
           'extra_genre_tokens': EXTRA_GENRES, 'extra_genre_features': EXTRA_GENRE_FEATS,
           'variant_features': {n: (list(FEATS) if n in ('lgb', 'tp35_pool') else
               [f for f in FEATS if f != 'h'] + (EXTRA_GENRE_FEATS if n == 'tp35_genre' else [])) for n in WEIGHTS},
           'pool_context': CFG.POOL_CONTEXT, 'pool_sampling_seed': CFG.SEED,""")
replace("'fold_scheme': 'release-week cohort'", "'fold_scheme': 'purged contiguous release-week blocks'")

for c in cells:
    if c['cell_type']=='code':
        ast.parse('\n'.join(line for line in c['source'].splitlines() if not line.startswith(('!','%'))))
nb['metadata'].pop('widgets',None)
out=HERE/'v17.ipynb'
out.write_text(json.dumps(nb,indent=1,ensure_ascii=False))
print(f'Wrote {len(cells)} cells -> {out}')
