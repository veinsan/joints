"""Audit saved v15 predictions and raw preprocessing. No model fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import KEY, OUTAGE, base_title, fig_dir, load, scale, style
from evaluate import test_weights_fd


def main():
    out = fig_dir('96_v15_validation_audit')
    d = load()
    x = pd.read_parquet(ROOT / 'outputs/cache/Xtr.parquet')
    t = pd.read_parquet(ROOT / 'outputs/cache/Xte.parquet')
    limited = pd.read_parquet(ROOT / 'outputs/cache/Xlim.parquet')
    saved = pd.read_csv(ROOT / 'results/v15/oof.csv')
    # Cache supplies only grouping/history fields, never replacement labels or predictions.
    x = x.merge(saved, on=KEY + ['h'], validate='one_to_one', suffixes=('', '_saved'))
    assert len(x) == len(saved) and np.array_equal(x.total_ticket, x.y)
    assert np.allclose(x.scale, x.scale_saved)
    assert x.groupby(base_title(x.movie_title)).fold.nunique().max() == 1
    assert x.groupby('week').fold.nunique().max() == 1
    w = test_weights_fd(x, t)
    x['w'] = w
    x['e'] = (x.y - x.oof_final).abs() / x.scale
    x['delta'] = x.e - (x.y - x.comp_lgb).abs() / x.scale
    x['base'] = base_title(x.movie_title)
    x['contribution'] = x.e * w / w.sum()
    x['scale_bin'] = pd.cut(x.scale, [0, 5, 20, 50, 100, 200, 500, 1e9]).astype(str)
    x['month'] = x.d1.dt.strftime('%Y-%m')
    x['outcome'] = np.where(x.y == 0, 'zero', np.where(x.y > x.y3, 'growth_vs_D3', 'positive_no_growth'))
    summary = {'public_user_reported': .39918, 'target_user_reported': .33662,
               'required_absolute_gain': .39918 - .33662,
               'required_relative_gain': 1 - .33662 / .39918,
               'MASE': float(x.e.mean()), 'TW_MASE': float(np.average(x.e, weights=w))}
    tables = {}
    for col in ('fold', 'week', 'h', 'month', 'scale_bin', 'first_day', 'outcome'):
        rows = []
        for name, g in x.groupby(col, observed=True):
            rows.append(dict(group=name, rows=len(g), films=g.base.nunique(),
                             weight_share=g.w.sum()/w.sum(),
                             MASE=g.e.mean(), TW_MASE=np.average(g.e, weights=g.w),
                             delta_vs_lgb=np.average(g.delta, weights=g.w),
                             contribution=g.contribution.sum()))
        tables[col] = pd.DataFrame(rows).set_index('group')
        tables[col].to_csv(out / f'by_{col}.csv')
        if col != 'week':
            print('\n', col, '\n', tables[col].round(5).to_string())
    for unit in ('base', 'week'):
        g = x.assign(we=x.e*w, wd=x.delta*w).groupby(unit)[['we', 'wd', 'w']].sum()
        ix = np.random.default_rng(2026).integers(len(g), size=(4000, len(g)))
        den = g.w.to_numpy()[ix].sum(1)
        summary[f'{unit}_count'] = len(g)
        summary[f'{unit}_weight_ess'] = float(g.w.sum()**2 / (g.w**2).sum())
        summary[f'{unit}_bootstrap_TW_95'] = np.quantile(g.we.to_numpy()[ix].sum(1)/den, [.025,.975]).tolist()
        summary[f'{unit}_bootstrap_delta_lgb_95'] = np.quantile(g.wd.to_numpy()[ix].sum(1)/den, [.025,.975]).tolist()
    # Label reweighting support: no artificial targets are introduced.
    film_error = x.groupby('base').contribution.sum().sort_values(ascending=False)
    film_error.to_csv(out / 'film_error_contribution.csv')
    summary['top5_error_share'] = float(film_error.head(5).sum()/film_error.sum())
    rows = []
    for name in ('train', 'hist'):
        a = d[name]
        rows.append(dict(split=name, rows=len(a), missing=int(a.isna().sum().sum()),
                         duplicates=int(a.duplicated(KEY+['date_show']).sum()),
                         negative_tickets=int(a.total_ticket.lt(0).sum()),
                         invalid_occupation=int((~a.occupation_rate.between(0,100)).sum()),
                         zero_occupation_positive=int((a.occupation_rate.eq(0)&a.total_ticket.gt(0)).sum())))
    pd.DataFrame(rows).to_csv(out/'raw_quality.csv', index=False)
    ts = t.join(scale(d['hist']), on=KEY, rsuffix='_raw')
    assert np.allclose(ts.scale, ts.scale_raw)
    assert d['test'].groupby(KEY).size().eq(7).all()
    assert not d['test'].duplicated('id').any()
    first_target = d['test'].groupby('movie_title').date_show.min()
    hd = d['hist'].join(first_target.rename('first_target'), on='movie_title')
    summary['history_rows_format_absent_from_test'] = int(hd.first_target.isna().sum())
    base_target = d['test'].assign(base=base_title(d['test'].movie_title)).groupby('base').date_show.min()
    hd['first_target'] = hd.first_target.fillna(base_title(hd.movie_title).map(base_target))
    assert ((hd.first_target-hd.date_show).dt.days.between(1,3)).all()
    pairs_d3 = pd.MultiIndex.from_frame(hd.loc[hd.date_show.eq(hd.first_target-pd.Timedelta(days=1)), KEY].drop_duplicates())
    test_pairs = pd.MultiIndex.from_frame(d['test'][KEY].drop_duplicates())
    summary['test_pairs_missing_D3'] = int((~test_pairs.isin(pairs_d3)).sum())
    summary['D3_pairs_not_in_test'] = int((~pairs_d3.isin(test_pairs)).sum())
    summary['train_scale_median'] = float(x.drop_duplicates(KEY).scale.median())
    summary['test_scale_median'] = float(t.drop_duplicates(KEY).scale.median())
    summary['history_scale_verified'] = True
    # Global missingness is a diagnostic, not proof of outage at an individual cinema.
    active = d['train'].groupby(['cinema_ids','date_show']).size()
    idx = pd.MultiIndex.from_frame(x[['cinema_ids','date_show']])
    x['cinema_unobserved'] = ~idx.isin(active.index)
    x['suspect_zero'] = x.y.eq(0) & x.cinema_unobserved
    daily = x.groupby('date_show').agg(rows=('h','size'), zeros=('y', lambda a: a.eq(0).sum()),
                                               suspect_zero=('suspect_zero','sum'), error=('contribution','sum'))
    daily['reporting_clusters'] = d['train'].groupby('date_show').cinema_ids.nunique()
    daily.sort_values('suspect_zero', ascending=False).to_csv(out/'missingness_by_date.csv')
    summary['zero_without_any_cluster_transactions'] = int(x.suspect_zero.sum())
    summary['fraction_zero_without_cluster_activity'] = float(x.loc[x.y.eq(0),'suspect_zero'].mean())
    summary['suspect_zero_error_share'] = float(x.loc[x.suspect_zero,'contribution'].sum()/x.contribution.sum())
    # v15 appends all limited films to every training fold.
    limited['base'] = base_title(limited.movie_title)
    limited['week'] = ((limited.d1-pd.Timestamp('2025-03-31')).dt.days//7)
    splits = []
    for fold, g in x.groupby('fold'):
        same_week = limited.week.isin(g.week)
        same_base = limited.base.isin(g.base)
        splits.append(dict(fold=fold, limited_rows=len(limited),
                           limited_rows_in_validation_week=int(same_week.sum()),
                           limited_rows_same_base=int(same_base.sum()),
                           validation_weeks=g.week.nunique(),
                           val_weeks_with_limited_overlap=len(set(g.week)&set(limited.week))))
    split = pd.DataFrame(splits)
    split.to_csv(out/'limited_fold_overlap.csv', index=False)
    # Confirm the precise missing-date list used in shared preprocessing.
    summary['excluded_dates'] = OUTAGE.strftime('%Y-%m-%d').tolist()
    summary['june11_rows'] = int(x.date_show.eq('2025-06-11').sum())
    # Bound on how much unvalidated Lebaran override can move total MASE.
    sub = d['test'][['id','date_show']].merge(pd.read_csv(ROOT/'results/v15/submission.csv'),on='id',validate='one_to_one')
    sub = sub.merge(t[['id','scale']], on='id', validate='one_to_one')
    leb = sub.date_show.between('2026-03-21','2026-03-27')
    summary['lebaran_row_share'] = float(leb.mean())
    summary['lebaran_mean_pred_over_scale'] = float((sub.total_ticket/sub.scale)[leb].mean())
    summary['MASE_change_bound_10pct_lebaran_rescale'] = float((sub.total_ticket/sub.scale)[leb].sum()/len(sub)*.1)
    (out/'summary.json').write_text(json.dumps(summary, indent=2))
    print('\nLIMITED OVERLAP\n',split.to_string(index=False))
    print('\nSUMMARY\n',json.dumps(summary,indent=2))
    plt = style()
    fig, ax = plt.subplots(2,2,figsize=(12,8))
    tables['h'].TW_MASE.plot.bar(ax=ax[0,0], title='v15 TW-MASE by horizon')
    tables['month'].TW_MASE.plot.bar(ax=ax[0,1], title='Cohort OOF by release month')
    tables['outcome'].contribution.plot.bar(ax=ax[1,0], title='Error contribution by observed outcome',rot=15)
    film_error.head(8).sort_values().plot.barh(ax=ax[1,1], title='Largest film error contributions')
    fig.tight_layout(); fig.savefig(out/'audit.png'); plt.close(fig)


if __name__ == '__main__':
    main()
