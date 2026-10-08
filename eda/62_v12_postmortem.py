"""Audit saved predictions only: no fitting, hidden labels, or submissions."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import KEY, base_title, style
from evaluate import BINS, test_weights_fd


def main():
    out = ROOT / 'outputs/eda/62_v12_postmortem'
    out.mkdir(parents=True, exist_ok=True)
    x = pd.read_parquet(ROOT / 'outputs/cache/Xtr.parquet')
    t = pd.read_parquet(ROOT / 'outputs/cache/Xte.parquet')
    keys = KEY + ['h']
    w = test_weights_fd(x, t)
    y, s = x.total_ticket.to_numpy(), x.scale.to_numpy()
    films = base_title(x.movie_title)
    public = {8: .39991, 10: .40823, 12: .41060}
    scores, predictions, submissions, hashes = [], {}, {}, {}
    for v in [8, 10, 11, 12]:
        path = ROOT / f'results/v{v}/oof.csv'
        o = x[keys].merge(pd.read_csv(path), on=keys, validate='one_to_one', how='left')
        assert len(o) == len(x) and o.oof_final.notna().all()
        assert np.allclose(o.y, y) and np.allclose(o.scale, s)
        assert o.assign(base=films).groupby('base').fold.nunique().max() == 1
        predictions[v] = o.oof_final.to_numpy()
        for col in [c for c in o if c.startswith('oof_')]:
            err = np.abs(y - o[col].to_numpy()) / s
            scores.append(dict(version=v, component=col, MASE=err.mean(), TW=np.average(err, weights=w),
                               public=public.get(v) if col == 'oof_final' else None))
        sp = ROOT / f'results/v{v}/submission.csv'
        sub = t[['id']].merge(pd.read_csv(sp), on='id', validate='one_to_one', how='left')
        assert len(sub) == len(t) and np.isfinite(sub.total_ticket).all() and sub.total_ticket.ge(0).all()
        submissions[v] = sub.total_ticket.to_numpy()
        for p in [path, sp]:
            hashes[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    scores = pd.DataFrame(scores)
    scores.to_csv(out / 'scores.csv', index=False)
    e8, e12 = [np.abs(y - predictions[v]) / s for v in [8, 12]]
    e = x.assign(base=films, e8=e8, e12=e12, delta=e12-e8, w=w,
                 contribution8=e8*w/w.sum(), contribution12=e12*w/w.sum(),
                 delta_contribution=(e12-e8)*w/w.sum(),
                 bucket=pd.cut(s, BINS).astype(str), month=x.d1.dt.strftime('%Y-%m'), zero=y == 0)
    tables = {}
    for col in ['base', 'h', 'bucket', 'first_day', 'month', 'zero']:
        g = e.groupby(col).agg(rows=('w', 'size'), weight=('w', 'sum'),
                              contribution8=('contribution8', 'sum'), contribution12=('contribution12', 'sum'),
                              delta=('delta_contribution', 'sum'))
        g['TW8'] = g.contribution8 / (g.weight / w.sum())
        g['TW12'] = g.contribution12 / (g.weight / w.sum())
        g.sort_values('delta', ascending=False).to_csv(out / f'validation_{col}.csv')
        tables[col] = g
    g = tables['base']
    ix = np.random.default_rng(2026).integers(0, len(g), (10000, len(g)))
    boot = g.delta.to_numpy()[ix].sum(1) / (g.weight.to_numpy()[ix].sum(1)/w.sum())
    seg = pd.Series('normal', index=t.index)
    for name, a, b in [('xmas', '2025-12-20', '2026-01-04'), ('ramadan', '2026-02-19', '2026-03-20'),
                       ('lebaran', '2026-03-21', '2026-03-27')]:
        seg.loc[t.date_show.between(a, b)] = name
    dt = (submissions[12]-submissions[8]) / t.scale.to_numpy()
    tt = t.assign(segment=seg, base=base_title(t.movie_title), month=t.d1.dt.strftime('%Y-%m'),
                  bucket=pd.cut(t.scale, BINS).astype(str),
                  r8=submissions[8]/t.scale, r12=submissions[12]/t.scale, delta=dt, abs_delta=np.abs(dt),
                  contribution=np.abs(dt)/len(t))
    for col in ['segment', 'month', 'bucket', 'base', 'h']:
        z = tt.groupby(col).agg(rows=('id', 'size'), r8=('r8', 'mean'), r12=('r12', 'mean'),
                                signed_change=('delta', 'mean'), mean_abs_change=('abs_delta', 'mean'),
                                contribution_to_full_test_bound=('contribution', 'sum'))
        z.sort_values('contribution_to_full_test_bound', ascending=False).to_csv(out / f'test_{col}.csv')
    stats = dict(v12_minus_v8_TW=float(np.average(e12-e8, weights=w)),
                 paired_film_bootstrap_95pct=np.quantile(boot, [.025, .5, .975]).tolist(),
                 films_better=int((g.delta < 0).sum()), films=len(g),
                 signed_error_correlation=float(np.corrcoef((y-predictions[8])/s, (y-predictions[12])/s)[0, 1]),
                 oof_mean_abs_scaled_prediction_change=float(np.average(np.abs(predictions[12]-predictions[8])/s, weights=w)),
                 test_mean_abs_scaled_prediction_change=float(np.abs(dt).mean()),
                 test_lebaran_max_ticket_change=float(np.abs(submissions[12]-submissions[8])[seg.eq('lebaran')].max()),
                 test_change_bound_note='Triangle inequality bounds score change on these same rows with any fixed labels. Public subset is unknown; this is NOT a public bound or error attribution.',
                 bootstrap_note='Fixed post-selected predictions and fixed weights; does not correct for repeated model/feature selection.',
                 hashes=hashes)
    conditional = []
    masks = {'zero': y == 0, 'positive': y > 0, 'growth_vs_D3': y > x.y3.to_numpy(),
             'positive_no_growth': (y > 0) & (y <= x.y3.to_numpy())}
    for name, mask in masks.items():
        for v, pred in predictions.items():
            err = np.abs(y-pred)/s
            conditional.append(dict(segment=name, version=v, weight_share=w[mask].sum()/w.sum(),
                                    TW=np.average(err[mask], weights=w[mask]),
                                    signed_error=np.average((pred[mask]-y[mask])/s[mask],weights=w[mask]),
                                    contribution=np.sum(err[mask]*w[mask])/w.sum()))
    conditional = pd.DataFrame(conditional)
    conditional.to_csv(out/'conditional_errors.csv',index=False)
    cond = conditional.pivot(index='segment',columns='version',values='TW')
    stress = pd.DataFrame({'assumed_weighted_zero_share':np.linspace(0,.6,61)})
    for v in predictions:
        stress[f'v{v}'] = stress.assumed_weighted_zero_share*cond.loc['zero',v] + (1-stress.assumed_weighted_zero_share)*cond.loc['positive',v]
    stress.to_csv(out/'zero_prevalence_stress.csv',index=False)
    stats['zero_stress_assumption'] = 'Conditional absolute errors held fixed while zero mass varies. Sensitivity analysis only, not estimated test MASE; D3 proxy zero rate is NOT a D4-D10 zero-rate estimate.'
    stats['current_weighted_zero_share'] = float(w[y==0].sum()/w.sum())
    (out / 'summary.json').write_text(json.dumps(stats, indent=2))
    plt = style()
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    final = scores[scores.component.eq('oof_final')].set_index('version')
    final[['TW', 'public']].plot(marker='o', ax=ax[0])
    ax[0].set(title='Similar validation, different public scores', ylabel='MASE', xticks=final.index)
    tables['h'][['TW8', 'TW12']].plot(marker='o', ax=ax[1])
    ax[1].set(title='Identical validation rows and weights', ylabel='Weighted MASE')
    tt.groupby('segment')[['r8', 'r12']].mean().plot.bar(ax=ax[2], rot=0)
    ax[2].set(title='Test predictions; actual error is unknown', ylabel='Mean prediction / official scale')
    fig.tight_layout()
    fig.savefig(out / 'comparison.png')
    plt.close(fig)
    fig, ax = plt.subplots(1,2,figsize=(13,4))
    changes = []
    for v in [10,12]:
        row = {'version':f'v{v} - v8'}
        for name in ['zero','growth_vs_D3','positive_no_growth']:
            z = conditional[conditional.segment.eq(name)].set_index('version').contribution
            row[name] = z[v]-z[8]
        changes.append(row)
    pd.DataFrame(changes).set_index('version').plot.bar(ax=ax[0],rot=0)
    ax[0].axhline(0,color='black',lw=.8)
    ax[0].set(title='Validation improvements on zeros conceal growth losses',ylabel='Change in contribution to total TW-MASE')
    for v in [10,12]:
        ax[1].plot(stress.assumed_weighted_zero_share,stress[f'v{v}']-stress.v8,label=f'v{v} - v8')
    ax[1].axhline(0,color='black',lw=.8)
    ax[1].axvline(stats['current_weighted_zero_share'],ls='--',color='gray',label='Current weighted OOF prevalence')
    ax[1].set(title='Fixed-conditional-error stress; NOT test-score estimates',xlabel='Hypothetical weighted zero share',ylabel='Score delta (positive = worse)'); ax[1].legend()
    fig.tight_layout(); fig.savefig(out/'zero_growth_tradeoff.png'); plt.close(fig)
    print(final.round(6).to_string())
    print('Conditional errors (label-based diagnostics only):\n',cond.round(6).to_string())
    print(json.dumps({k:v for k,v in stats.items() if k != 'hashes'}, indent=2))
    print('Test segment changes:\n', pd.read_csv(out / 'test_segment.csv').round(6).to_string(index=False))
    print('Validation buckets:\n', tables['bucket'].round(6).to_string())


if __name__ == '__main__':
    main()
