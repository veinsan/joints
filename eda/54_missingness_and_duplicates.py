"""Audit missing transactions, duplicate histories and validation leverage. No training.
Run from repo root: .venv/bin/python eda/54_missingness_and_duplicates.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import KEY, base_title, load, style, fig_dir
from evaluate import test_weights_fd


def main():
    out = fig_dir('54_missingness')
    d = load()
    X = pd.read_parquet(ROOT / 'outputs/cache/Xtr.parquet')
    T = pd.read_parquet(ROOT / 'outputs/cache/Xte.parquet')
    for a in (X,T):
        assert a.h.between(4,10).all()
        assert a.assign(base=base_title(a.movie_title)).groupby('base').d1.nunique().max()==1
    w = test_weights_fd(X, T)
    key = KEY + ['h']
    z = X.copy()
    for v in (8, 10):
        o = pd.read_csv(ROOT / f'results/v{v}/oof.csv')
        col = 'oof_final' if 'oof_final' in o else 'pred'
        z = z.merge(o[key + [col]].rename(columns={col: f'p{v}'}), on=key, validate='one_to_one')
    assert len(z) == len(X) and np.array_equal(z.total_ticket, X.total_ticket)
    z['base'] = base_title(z.movie_title)
    z['weight'] = w
    for v in (8, 10):
        z[f'e{v}'] = (z.total_ticket - z[f'p{v}']).abs() / z.scale
        z[f'we{v}'] = z[f'e{v}'] * w
    g = z.groupby('base').agg(rows=('h','size'), weight=('weight','sum'), e8=('we8','sum'), e10=('we10','sum'))
    g['delta_contribution'] = (g.e10-g.e8)/w.sum()
    g['error_share8'] = g.e8/g.e8.sum()
    g.sort_values('error_share8', ascending=False).to_csv(out/'film_leverage.csv')
    rng = np.random.default_rng(2026)
    ix = rng.integers(len(g), size=(5000,len(g)))
    delta = (g.e10-g.e8).to_numpy()[ix].sum(1)/g.weight.to_numpy()[ix].sum(1)
    # Whole-cinema activity only diagnoses train zeros; never becomes an inference feature.
    active = d['train'].groupby(['cinema_ids','date_show']).size().rename('other_activity')
    z = z.join(active, on=['cinema_ids','date_show'])
    zero = z.total_ticket.eq(0)
    z['zero_with_cinema_activity'] = zero & z.other_activity.notna()
    z['zero_without_cinema_activity'] = zero & z.other_activity.isna()
    states = []
    for k, p in z.groupby(KEY):
        p = p.sort_values('h')
        if len(p) != 7:  # excluded outage dates break run-length interpretation
            continue
        iszero = p.total_ticket.eq(0).to_numpy()
        returns = bool(np.any(iszero & np.maximum.accumulate((~iszero)[::-1])[::-1]))
        states.append(dict(zip(KEY,k), zeros=int(iszero.sum()), returned=returns,
                           first_day=int(p.first_day.iloc[0]), scale=float(p.scale.iloc[0])))
    states = pd.DataFrame(states)
    states.to_csv(out/'zero_sequences.csv', index=False)
    signatures = ['y1','y2','y3','sh1','sh2','sh3']
    duplicates = []
    for name, a in [('train',X), ('test',T)]:
        p = a.drop_duplicates(KEY).copy()
        p['base'] = base_title(p.movie_title)
        nfilms = p.groupby(signatures).base.transform('nunique')
        duplicate = p[nfilms > 1]
        duplicate[KEY+signatures].to_csv(out/f'{name}_repeated_histories.csv', index=False)
        duplicates.append({'split':name, 'pairs':len(p), 'repeated_across_films':len(duplicate),
                           'fraction':len(duplicate)/len(p), 'median_scale_repeated':duplicate.scale.median()})
    pd.DataFrame(duplicates).to_csv(out/'duplicate_summary.csv', index=False)
    raw = []
    for name in ('train','hist'):
        a = d[name]
        raw.append({'split':name, 'rows':len(a), 'duplicate_keys':int(a.duplicated(KEY+['date_show']).sum()),
                    'zero_tickets':int(a.total_ticket.eq(0).sum()),
                    'zero_occupation_positive_tickets':int((a.occupation_rate.eq(0)&a.total_ticket.gt(0)).sum())})
    pd.DataFrame(raw).to_csv(out/'raw_integrity.csv', index=False)
    summary = {'mase8':float(z.e8.mean()), 'mase10':float(z.e10.mean()),
               'tw8':float(np.average(z.e8,weights=w)), 'tw10':float(np.average(z.e10,weights=w)),
               'paired_film_bootstrap_delta10_minus8_95':np.quantile(delta,[.025,.5,.975]).tolist(),
               'row_weight_ess':float(w.sum()**2/(w*w).sum()),
               'film_weight_ess':float(g.weight.sum()**2/(g.weight**2).sum()),
               'films':len(g), 'top5_error_share':float(g.error_share8.nlargest(5).sum()),
               'zero_targets':int(zero.sum()),
               'fraction_zeros_cinema_other_activity':float(z.loc[zero,'zero_with_cinema_activity'].mean()),
               'full_pairs':len(states), 'pairs_with_zero':int(states.zeros.gt(0).sum()),
               'zero_then_return_pairs':int(states.returned.sum())}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    print(pd.DataFrame(raw).to_string(index=False))
    print(pd.DataFrame(duplicates).to_string(index=False))
    plt = style()
    fig, ax = plt.subplots(2,2,figsize=(13,9))
    g.error_share8.nlargest(12).sort_values().plot.barh(ax=ax[0,0], title='Film contribution to weighted v8 error')
    ax[0,1].hist(delta,bins=50); ax[0,1].axvline(0,color='black',ls='--')
    ax[0,1].set_title('v10 minus v8: paired FILM bootstrap (lower better)')
    states.groupby('first_day').returned.mean().plot.bar(ax=ax[1,0],title='P(zero then positive), complete pairs')
    tab = z.assign(kind=np.where(~zero,'positive',np.where(z.other_activity.notna(),'zero: cinema active','zero: cinema unobserved'))).groupby('kind').agg(rows=('h','size'), weighted_error=('we8','sum'))
    tab.to_csv(out/'missingness_error.csv')
    (tab.weighted_error/tab.weighted_error.sum()).plot.bar(ax=ax[1,1],rot=20,title='Error shares; absence does not prove permanent pull')
    fig.tight_layout(); fig.savefig(out/'diagnostics.png'); plt.close(fig)
    assert z.groupby('base').movie_title.nunique().max() >= 1
    assert all(r['duplicate_keys']==0 for r in raw)
    assert summary['zero_then_return_pairs'] <= summary['pairs_with_zero']


if __name__ == '__main__':
    main()
