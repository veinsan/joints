"""Build a standalone Kaggle audit using the executed v13 pipeline; never execute training locally."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks/validation_audit.ipynb'
source=ROOT/'notebooks/v13.ipynb'
nb=json.loads(source.read_text())
if OUT.exists():
    old=json.loads(OUT.read_text())
    assert not any(c.get('outputs') or c.get('execution_count') is not None for c in old['cells']), 'Do not overwrite an executed notebook'
md=lambda s:dict(cell_type='markdown',metadata={},source=s)
code=lambda s:dict(cell_type='code',metadata={},source=s,outputs=[],execution_count=None)
src=lambda i:''.join(nb['cells'][i]['source'])
manifest=ROOT/'outputs/eda/78_target_like_validation/target_like_film_manifest.csv'
import csv
held=[r['base'] for r in csv.DictReader(manifest.open()) if r['role']=='validation']
assert len(held)==51
cells=[md('---\n\n# JOINTS: audit validasi setelah v13\n\n*Model dibekukan; validasi temporal dan pergeseran distribusi diperiksa terpisah.*'),
       md('---\n\n## Tim\n\nNotebook eksperimen internal. Informasi tim mengikuti notebook utama.'),
       md('---\n\n## Daftar Isi\n\n1. [Pendahuluan](#intro)\n2. [Inisialisasi](#init)\n3. [Data dan preprocessing](#data)\n4. [Audit model tetap](#audit)'),
       md('# 1. Pendahuluan <a id="intro"></a>\n\nPipeline B1 v13 digunakan tanpa pencarian hyperparameter. Empat cutoff bulanan memakai film training yang D10-nya selesai sebelum cutoff. Holdout 51 film dengan distribusi fitur mendekati test merupakan stress test non-temporal. Semua model tetap dikelompokkan per judul dasar. D1 rekonstruksi dan kalender tetap seperti v13; fitur lintas-film menggunakan jendela resmi yang tersedia dalam paket. Oleh sebab itu ini label-purged backtest, bukan simulasi real-time sempurna. September pernah dievaluasi sebelumnya dan bukan holdout baru yang belum tersentuh.\n\nTiga kandidat tetap: blend v13 (0.2 LGB, 0.5 Fast, 0.3 TabM), blend tanpa TabM (2/7 LGB, 5/7 Fast), serta CDF mixture dengan bobot v13. Tidak ada pemilihan bobot berdasarkan hasil run ini. Perbaikan CDF pada OOF sebelumnya dibayar dengan kerugian growth, sehingga seluruh segmen wajib dilaporkan.\n\nNotebook ini tidak menulis submission atau memilih model final otomatis.'),
       md('# 2. Inisialisasi <a id="init"></a>\n\nJalankan pada Kaggle T4 dengan data kompetisi dan HF_TOKEN. Instalasi serta implementasi model mengikuti v13.'),code(src(13)),
       code(src(15)+'\nimport json\nassert Path("/kaggle/input").exists(), "Training audit hanya dijalankan di Kaggle"'),code(src(17)),code(src(19)),
       md('# 3. Data dan preprocessing <a id="data"></a>\n\nKode pembentukan sampel dan fitur digunakan kembali dari v13. Skala resmi dipertahankan; fitur relative dihitung untuk kesetaraan pipeline, tetapi model memakai fitur absolute B1.')]
for i in [21,29,44,46,55,61,63,65,67,76,81,83]:
    s=src(i)
    if i==55:s=s.split('\n\nlogit =')[0]
    cells.append(code(s))
cells.append(md('#### Insights\n\n> Jumlah baris, skala, kalender regional, dan pemeriksaan simulasi harus sama dengan v13 sebelum skor model dibandingkan. Memotong OOF lama berdasarkan bulan tidak menggantikan fitting pada cutoff baru.'))
cells.append(md('# 4. Audit model tetap <a id="audit"></a>\n\nDefinisi Fast FP16 dan TabM dipakai kembali. Pergeseran pull dibekukan pada nilai v13; tidak ada grid lambda atau bobot baru. Semua prediksi dan kuantil per split disimpan agar audit berikutnya tidak perlu mengulang fitting.'))
cells.append(code("logit = lambda p: np.log(p / (1-p))\nsigm = lambda x: 1/(1+np.exp(-x))\nPULL_A = -1.32\nWORLDS = []\n"+src(92).split('\n\nQUANT, QFN,')[0]))
cells.append(code(src(94).split('\n\nif CFG.USE_TABM:')[0]))
module=ast.parse((ROOT/'eda/76_distribution_aggregation.py').read_text())
helpers='\n\n'.join(ast.unparse(n) for n in module.body if isinstance(n,ast.FunctionDef) and n.name in ['shifted','positive_cdf','mixture_median'])
cells.append(code(helpers))
cells.append(code('TARGET_LIKE_FILMS = set('+repr(held)+')\n'+'''AUDIT_DIR = CFG.OUTPUT_DIR / 'validation_audit'
AUDIT_DIR.mkdir(parents=True, exist_ok=True)
fast_fn = make_tabpfn(ModelVersion.V3_5, fast16_checkpoint())
cases = []
for month in ['2025-06', '2025-07', '2025-08', '2025-09']:
    cutoff = pd.Timestamp(month + '-01')
    mask = Xtr.d1.dt.strftime('%Y-%m').eq(month).values
    ready = lambda X: X[X.d1 + pd.Timedelta(days=9) < cutoff]
    xa = pd.concat([ready(Xtr), ready(Xlim)], ignore_index=True)
    assert (xa.d1 + pd.Timedelta(days=9) < cutoff).all()
    cases.append((month, xa, np.flatnonzero(mask), cin_stats(train[train.date_show < cutoff])))
hold = np.isin(groups, list(TARGET_LIKE_FILMS))
assert len(set(groups[hold])) == 51
extra = Xlim[~base_title(Xlim.movie_title).isin(TARGET_LIKE_FILMS)]
cases.append(('target_like', pd.concat([Xtr[~hold], extra], ignore_index=True), np.flatnonzero(hold),
              cin_stats(train[~TRAIN_BASE.isin(TARGET_LIKE_FILMS)])))
audit_rows = []
for name, xa, idx, stats in cases:
    started = time.time()
    xb = Xtr.iloc[idx]
    assert not set(base_title(xa.movie_title)) & set(base_title(xb.movie_title))
    xa, xb, feats, zfeats = make_parts(xa, xb, dict(cin='local', level='abs'), stats)
    models = fit_lgb_all(xa, feats, zfeats)
    l1, p0, ql = predict_lgb_all(models, xb, feats, zfeats)
    qf, qt = fast_fn(xa, xb, feats), tabm_quantiles(xa, xb, feats)
    c, s_, yy, ww = xb.cal_mult.values, xb.scale.values, xb.total_ticket.values, TW[idx]
    pl = lgb_mix(l1, p0, ql, lam=.5, a=PULL_A)
    pf, pt = quant_median(qf, .5, PULL_A), quant_median(qt, 0., PULL_A)
    part = [(.15, shifted(p0, .5, PULL_A), np.maximum(ql, 0), np.asarray(CFG.QS))]
    taus = np.asarray(TAB_QS)
    for weight, q, lam in [(.5, qf, .5), (.3, qt, 0.)]:
        z = np.array([np.interp(CFG.ZERO_EPS, qr, taus, left=0., right=1.) for qr in q])
        pos = np.maximum(np.array([np.interp(zi+(1-zi)*taus, taus, qr) for zi, qr in zip(z, q)]), 0)
        part.append((weight, shifted(z, lam, PULL_A), pos, taus))
    preds = {'v13_fixed': (.2*pl+.5*pf+.3*pt)*c*s_,
             'without_tabm_fixed': (2/7*pl+5/7*pf)*c*s_,
             'cdf_fixed': mixture_median(part, [(.05, l1)])*c*s_}
    saved_rows = xb[KEY+['h', 'date_show', 'd1', 'scale', 'total_ticket', 'cal_mult']].copy()
    saved_rows['weight'] = ww
    saved_rows['base'] = base_title(xb.movie_title)
    for candidate, pred in preds.items():
        assert np.isfinite(pred).all() and (pred >= 0).all()
        saved_rows[candidate] = pred
        error = np.abs(yy-pred)/s_
        row = dict(split=name, candidate=candidate, train_films=base_title(xa.movie_title).nunique(),
                   val_films=base_title(xb.movie_title).nunique(), rows=len(idx), MASE=error.mean(), TW=np.average(error, weights=ww))
        for seg, mask in {'zero': yy==0, 'positive': yy>0, 'growth': yy>xb.y3.values, 'D4_D5': xb.h.values<=5}.items():
            row[seg] = np.average(error[mask], weights=ww[mask]) if mask.any() else np.nan
        audit_rows.append(row)
    saved_rows.to_csv(AUDIT_DIR / f'{name}_predictions.csv', index=False)
    np.savez_compressed(AUDIT_DIR / f'{name}_quantiles.npz', row_index=idx, l1=l1, p0=p0, lgb_Q=ql, fast_Q=qf, tabm_Q=qt)
    print(name, 'seconds', round(time.time()-started), 'train films', base_title(xa.movie_title).nunique())
    del models
    torch.cuda.empty_cache()
    pd.DataFrame(audit_rows).to_csv(AUDIT_DIR / 'scores.csv', index=False)
scores = pd.DataFrame(audit_rows)
display(scores.round(5))
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
scores.pivot(index='split', columns='candidate', values='TW').plot.bar(ax=ax[0], rot=15)
scores.pivot(index='split', columns='candidate', values='growth').plot.bar(ax=ax[1], rot=15)
ax[0].set_title('Same split: weighted MASE')
ax[1].set_title('Growth errors: watch the trade-off')
plt.tight_layout()
plt.savefig(AUDIT_DIR / 'validation_comparison.png')
plt.show()
print('No submission generated. Compare candidate deltas by split and bootstrap original release cohorts.')'''))
cells.append(md('#### Insights\n\n> Kesimpulan diisi setelah output Kaggle tersedia. Jangan menyebut penurunan skor sebagai peningkatan bila hanya berasal dari target nol, satu bulan, atau perubahan populasi evaluasi. Test-like stress split tidak membuktikan label kondisional test serupa. Kandidat baru tetap memerlukan pemeriksaan ukuran paket dan reproduksi sebelum menjadi submission.'))
for i,c in enumerate(cells):
    c['id']=f'audit-{i:03d}'
    if c['cell_type']=='code':
        text='\n'.join(l for l in c['source'].splitlines() if not l.startswith(('%','!')))
        ast.parse(text)
result=dict(nbformat=4,nbformat_minor=5,metadata=nb['metadata'],cells=cells)
result['metadata']['audit_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
result['metadata']['split_manifest_sha256']=hashlib.sha256(manifest.read_bytes()).hexdigest()
OUT.write_text(json.dumps(result,indent=1,ensure_ascii=False))
print(f'Wrote {len(cells)} cells -> {OUT}; build only, no training executed')
