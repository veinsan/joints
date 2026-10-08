"""Build a self-contained v16 notebook from the preserved v15 notebook cells."""
import ast
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
nb = copy.deepcopy(json.loads((HERE / 'v15.ipynb').read_text()))
cells = nb['cells']
for c in cells:
    c['source'] = ''.join(c['source'])
    if c['cell_type'] == 'code':
        c['outputs'], c['execution_count'] = [], None


def replace(old, new):
    matches = [c for c in cells if c['cell_type'] == 'code' and old in c['source']]
    assert len(matches) == 1, (old[:70], len(matches))
    assert matches[0]['source'].count(old) == 1
    matches[0]['source'] = matches[0]['source'].replace(old, new)


def md(i, text):
    assert cells[i]['cell_type'] == 'markdown'
    cells[i]['source'] = text


def code(i, text):
    assert cells[i]['cell_type'] == 'code'
    cells[i]['source'] = text.strip()


md(0, '---\n\n# JOINTS x INSPIRE UGM 2026 (v16)\n\n*Audit kohort lengkap dan ablation target log1p pada TabPFN-3.5 int7.*')
md(4, '## Overview\n\nv15 memperoleh public MASE 0,39918 menurut hasil yang dilaporkan peserta. '
   'Target 0,33662 memerlukan penurunan error 15,7%; notebook ini tidak menjanjikan target tersebut. '
   'Audit menemukan 55% error validasi berasal dari penjualan yang naik dibanding D3, sementara '
   'film limited masih berbagi minggu dengan fold validasi. v16 memperbaiki pemisahan limited '
   'untuk semua model dan menguji satu perubahan: target log1p pada TabPFN yang sama.')
md(8, '## Approach\n\nBaseline memakai LightGBM 0,1 dan TabPFN-3.5 int7 0,9. '
   'Kandidat memakai bobot identik dengan target TabPFN log1p(y / scale / cal_mult), '
   'lalu expm1 pada keluaran kuantil. Lambda tetap 0,5. Skala kompetisi, fitur, '
   'kalender, dan analog Lebaran sama. Hanya satu kandidat diuji; kegagalan model menghentikan run.')
md(12, 'Seluruh dependency dipasang dengan versi spesifik untuk menjaga reproduksibilitas.')
md(73, 'Baseline dan kandidat dihitung ulang pada fold yang sama setelah seluruh data limited '
   'dari minggu validasi dikeluarkan. Tidak ada training pada laptop; bagian ini dijalankan di Kaggle GPU.')
md(74, '## Dasar Eksperimen\n\nTarget ternormalisasi memiliki skew rata-rata per horizon 19,77, '
   'dibanding 1,54 setelah log1p. Transformasi monoton mempertahankan urutan kuantil pada distribusi '
   'ideal, tetapi prediksi model tetap dapat memburuk. Karena itu baseline mentah wajib dihitung ulang.\n\n'
   '#### Insights\n\n> TW-MASE v15 0,33149 memiliki interval bootstrap minggu sekitar '
   '[0,26972; 0,40564]. Ini ketidakpastian pada kohort train, bukan interval prediksi leaderboard.')
md(75, '## Lensa Validasi\n\nFold dikelompokkan berdasarkan minggu rilis. Film limited dari minggu '
   'validasi dan seluruh format judul validasi dikeluarkan dari training. Statistik klaster juga '
   'mengecualikan film tersebut. Backtest Juli, Agustus, September hanya melatih pada target yang '
   'sudah selesai sebelum awal bulan. Cohort CV bukan pengganti temporal CV. Seluruh bulan ini '
   'pernah digunakan dalam eksperimen terdahulu, sehingga bukan holdout baru yang belum pernah dilihat.')
md(85, '## Kandidat Foundation Model\n\nDua evaluasi memakai checkpoint TabPFN-3.5 int7 yang sama: '
   '`tp35_int7` dengan target mentah dan `tp35_log` dengan log1p. Tidak ada checkpoint tambahan '
   'pada berkas final karena hanya satu varian dipilih. Pengaturan internal TabPFN tetap sama; '
   'manfaat log1p eksternal harus dibuktikan melalui ablation ini.')
md(89, '## Status TabM\n\nTabM dinonaktifkan dalam ablation ini agar perubahan hanya pada '
   'transformasi target TabPFN. Fungsi lama dipertahankan untuk kompatibilitas format penyimpanan.')
md(91, '## Koreksi Pencopotan Dibekukan\n\nLambda 0,5 dipakai untuk kedua varian. '
   'Tidak ada pemilihan lambda memakai label evaluasi pada eksperimen ini.')
md(93, '## Aturan Adopsi Kandidat\n\nBobot LightGBM 0,1 dan TabPFN 0,9 dibekukan. Kandidat '
   'log1p harus memperbaiki TW-MASE setidaknya 0,0015, tidak memperburuk rata-rata backtest, '
   'tidak memperburuk bulan mana pun lebih dari 0,003, menang pada sedikitnya tiga fold, '
   'dan memiliki batas atas interval bootstrap perubahan MASE per minggu di bawah nol. '
   'Jika salah satu syarat gagal, baseline dipakai. Angka setelah keputusan tetap merupakan '
   'hasil seleksi validasi, bukan estimasi private leaderboard yang tidak bias.')
md(95, '#### Insights\n\n> Keputusan diambil otomatis dari satu perbandingan yang ditentukan '
   'sebelum run. Skor baseline baru dapat berbeda dari v15 karena kebocoran minggu limited diperbaiki.')
md(104, '#### Insights\n\n> Analog Lebaran dipertahankan persis seperti v15 untuk mengisolasi '
   'ablation. Override ini mencakup 6,08% baris test dan belum memiliki validasi lintas Lebaran '
   'yang setara. Perubahan 10% pada prediksinya dapat mengubah MASE total paling besar sekitar '
   '0,01076; arah perubahannya tidak diketahui tanpa label. Kapasitas kursi bukan bukti permintaan aktual.')
md(108, '#### Insights\n\n> Notebook menyimpan baseline dan kandidat OOF, kuantil, hasil temporal '
   'per baris, keputusan adopsi, serta bobot model terpilih. Jalankan di Kaggle dan tinjau hasil '
   'sebelum submisi. Skor 0,33662 belum terbukti dapat dicapai dari eksperimen ini.')
replace("FM_LIST      = ('tp35_int7', 'fast35', 'causilo_h', 'causilo_pool')", "FM_LIST      = ('tp35_int7', 'tp35_log')")
replace('FM_STRICT    = False', 'FM_STRICT    = True')
replace('USE_TABM     = True', 'USE_TABM     = False')
replace("Path('../outputs/v15')", "Path('../outputs/v16')")
replace("def train_part(k):\n    Xa = Xtr[FOLD != k]\n    return pd.concat([Xa, Xlim], ignore_index=True) if CFG.USE_LIMITED else Xa", """def train_part(k):
    Xa = Xtr[FOLD != k]
    val_weeks = set(WEEK[FOLD == k])
    val_bases = set(groups[FOLD == k])
    lw = ((Xlim.d1 - pd.Timestamp('2025-03-31')).dt.days // 7)
    extra = Xlim[~lw.isin(val_weeks) & ~base_title(Xlim.movie_title).isin(val_bases)]
    result = pd.concat([Xa, extra], ignore_index=True) if CFG.USE_LIMITED else Xa
    rw = ((result.d1 - pd.Timestamp('2025-03-31')).dt.days // 7)
    assert not set(rw) & val_weeks
    assert not set(base_title(result.movie_title)) & val_bases
    return result""")
replace("STATS_FOLD = {k: cin_stats(train[~TRAIN_BASE.isin(set(groups[FOLD == k]))]) for k in range(CFG.N_FOLDS)}", """LIMITED_WEEK = ((Xlim.d1 - pd.Timestamp('2025-03-31')).dt.days // 7)
STATS_FOLD = {}
for k in range(CFG.N_FOLDS):
    excluded = set(groups[FOLD == k]) | set(base_title(Xlim.loc[LIMITED_WEEK.isin(WEEK[FOLD == k]), 'movie_title']))
    STATS_FOLD[k] = cin_stats(train[~TRAIN_BASE.isin(excluded)])""")
replace('def make_tabpfn(version, ckpt, n_est):', 'def make_tabpfn(version, ckpt, n_est, log_target=False):')
replace('m.fit(Xa[ia][tf].values.astype(np.float32), r_target(Xa)[ia])', """target = r_target(Xa)[ia]
            m.fit(Xa[ia][tf].values.astype(np.float32), np.log1p(target) if log_target else target)""")
replace("            del m\n            torch.cuda.empty_cache()\n        return np.sort(out, axis=1)", """            del m
            torch.cuda.empty_cache()
        if log_target:
            out = np.expm1(out)
        assert np.isfinite(out).all()
        return np.sort(out, axis=1)""")
replace("if n == 'tp35_int7':", "if n in ('tp35_int7', 'tp35_log'):")
replace('make_tabpfn(ModelVersion.V3_5, ckpt, CFG.TP35_EST), {CFG.TP35_PACK: pack}',
        "make_tabpfn(ModelVersion.V3_5, ckpt, CFG.TP35_EST, log_target=(n == 'tp35_log')), {CFG.TP35_PACK: pack}")
code(92, """
LAM_CHOICE = {n: CFG.LAMBDA for n in QUANT}
COMP = {'lgb': SEL['comp']}
for n, (oq, tq) in QUANT.items():
    COMP[n] = (quant_median(oq, CFG.LAMBDA) * cm * s,
               {m: (t['idx'], quant_median(tq[m], CFG.LAMBDA) * cm[t['idx']] * s[t['idx']])
                for m, t in SEL['temporal'].items()})
    results.append(score(COMP[n][0], n))
assert set(COMP) == {'lgb', 'tp35_int7', 'tp35_log'}
""")
code(94, """
CONTROL_WEIGHTS = {'lgb': 0.1, 'tp35_int7': 0.9}
CANDIDATE_WEIGHTS = {'lgb': 0.1, 'tp35_log': 0.9}
control_pred, control_temp = blend(CONTROL_WEIGHTS, COMP)
candidate_pred, candidate_temp = blend(CANDIDATE_WEIGHTS, COMP)
control_lens = lens(control_pred, control_temp)
candidate_lens = lens(candidate_pred, candidate_temp)
delta_lens = {k: candidate_lens[k] - control_lens[k] for k in control_lens}
week_delta = pd.DataFrame({'week': WEEK, 'w': TW,
    'wd': (np.abs(y - candidate_pred) - np.abs(y - control_pred)) / s * TW}).groupby('week')[['w', 'wd']].sum()
ix = np.random.default_rng(CFG.SEED).integers(len(week_delta), size=(4000, len(week_delta)))
delta_ci = np.quantile(week_delta.wd.values[ix].sum(1) / week_delta.w.values[ix].sum(1), [.025, .975])
adoption = {
    'TW_gain': delta_lens['TW'] <= -CFG.ABL_MIN_GAIN,
    'temporal_mean': delta_lens['temporal_mean'] <= 0,
    'each_month': max(delta_lens[f'temp_{m}'] for m in MONTHS) <= CFG.ABL_MONTH_TOL,
    'fold_wins': sum(delta_lens[f'fold{k}'] < 0 for k in range(CFG.N_FOLDS)) >= CFG.ABL_MIN_FOLDS,
    'week_bootstrap': delta_ci[1] < 0,
    'weights_size': sum(SIZE_MB[n] for n in CANDIDATE_WEIGHTS) + CFG.SIZE_MARGIN_MB <= CFG.MAX_WEIGHTS_MB,
}
WEIGHTS = CANDIDATE_WEIGHTS if all(adoption.values()) else CONTROL_WEIGHTS
display(pd.DataFrame({'control': control_lens, 'log1p_candidate': candidate_lens, 'delta': delta_lens}).T)
print('week bootstrap delta 95%:', delta_ci, '| adoption gates:', adoption)
print('chosen weights:', WEIGHTS)
pd.DataFrame({'passed': adoption}).to_csv(CFG.OUTPUT_DIR / 'adoption.csv')
pd.DataFrame({'control': control_lens, 'candidate': candidate_lens, 'delta': delta_lens}).to_csv(CFG.OUTPUT_DIR / 'ablation.csv')
oof_final, tp_final = blend(WEIGHTS, COMP)
FEATURE_CFG = 'v16 complete cohort exclusion; fixed features'
base_l = lens(*COMP['lgb'])
results.append(score(oof_final, 'selected fixed blend'))
""")
replace("payload = {'lightgbm': lgb_blob,", "payload = {'target_transform': {n: ('log1p' if n == 'tp35_log' else 'identity') for n in WEIGHTS},\n           'lightgbm': lgb_blob,")
replace('pred = np.where(leb_mask, leb_pred, pred)', """pd.DataFrame({'id': Xte.id, 'model_prediction': pred, 'lebaran_override': leb_mask,
              'final_prediction': np.where(leb_mask, leb_pred, pred)}).to_csv(CFG.OUTPUT_DIR / 'inference_audit.csv', index=False)
pred = np.where(leb_mask, leb_pred, pred)""")
replace("oof_df.to_csv(CFG.OUTPUT_DIR / 'oof.csv', index=False)", """oof_df['oof_control'] = control_pred
oof_df['oof_candidate'] = candidate_pred
oof_df.to_csv(CFG.OUTPUT_DIR / 'oof.csv', index=False)
temporal_rows = []
for m, (idx, pr) in tp_final.items():
    frame = Xtr.iloc[idx][KEY + ['h', 'date_show', 'scale', 'total_ticket']].copy()
    frame['month'], frame['TW'] = m, TW[idx]
    frame['control'], frame['candidate'], frame['selected'] = control_temp[m][1], candidate_temp[m][1], pr
    temporal_rows.append(frame)
pd.concat(temporal_rows, ignore_index=True).to_csv(CFG.OUTPUT_DIR / 'temporal.csv', index=False)""")

# Keep the inherited pinned libraries and preprocessing; the emitted notebook needs no v15 file.
for c in cells:
    if c['cell_type'] == 'code':
        src = '\n'.join(line for line in c['source'].splitlines() if not line.startswith(('!', '%')))
        ast.parse(src)
nb['metadata'].pop('widgets', None)
out = HERE / 'v16.ipynb'
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f'Wrote {len(cells)} cells -> {out}')
