"""Check unused lifecycle observations before proposing any augmentation; no fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import KEY, OUTAGE, base_title, load, release_dates, simulate, scale, style


def main():
    out = ROOT / 'outputs/eda/63_lifecycle_windows'
    out.mkdir(parents=True, exist_ok=True)
    d = load()
    tr = d['train']
    x = pd.read_parquet(ROOT / 'outputs/cache/Xtr.parquet')
    t = pd.read_parquet(ROOT / 'outputs/cache/Xte.parquet')
    first = tr.groupby('movie_title').date_show.min()
    d1 = release_dates(tr).drop(first[first.eq(tr.date_show.min())].index, errors='ignore')
    used = tr.merge(x[KEY + ['date_show']], on=KEY + ['date_show'], validate='one_to_one')
    rows, blocks = [], []
    for offset in [0, 3, 7, 14]:
        anchors = (d1 + pd.Timedelta(days=offset)).rename('d1')
        hist, target = simulate(tr, anchors)
        target = target.join(scale(hist), on=KEY)
        pp = hist.assign(day=(hist.date_show-hist.movie_title.map(anchors)).dt.days+1)
        p = pp.pivot(index=KEY, columns='day', values='total_ticket').reindex(columns=[1,2,3]).fillna(0)
        p.columns = ['y1', 'y2', 'y3']
        p['occ_mean'] = hist.groupby(KEY).occupation_rate.mean()
        z = target.merge(p.reset_index(), on=KEY, validate='many_to_one')
        z['origin_age'] = offset+1
        z['h'] = (z.date_show-z.movie_title.map(anchors)).dt.days+1
        z['base'] = base_title(z.movie_title)
        z['first_day'] = np.where(z.y1 > 0, 1, np.where(z.y2 > 0, 2, 3))
        z['ratio'] = z.total_ticket/z.scale
        z['d3_ratio'] = z.y3/z.scale
        assert z.h.between(4,10).all() and z.y3.gt(0).all()
        assert not z.date_show.isin(OUTAGE).any() and z.date_show.le('2025-09-30').all()
        if offset == 0:
            chk = x[KEY+['h', 'scale', 'total_ticket']].merge(z, on=KEY+['h'], suffixes=('_x', ''), validate='one_to_one')
            assert len(chk) == len(x) == len(z)
            assert np.allclose(chk.scale_x, chk.scale) and np.allclose(chk.total_ticket_x, chk.total_ticket)
        rows.append(dict(origin_age=offset+1, rows=len(z), films=z.base.nunique(), pairs=len(z[KEY].drop_duplicates()),
                         scale_median=z.scale.median(), small_share=z.scale.le(20).mean(),
                         occ_median=z.occ_mean.median(), late_share=z.first_day.eq(3).mean(),
                         zero_rate=z.total_ticket.eq(0).mean(), median_target_ratio=z.ratio.median(),
                         growth_share=(z.total_ticket > z.y3).mean(),
                         naive_d3_MASE=np.abs(z.ratio-z.d3_ratio).mean()))
        blocks.append(z)
    table = pd.DataFrame(rows).set_index('origin_age')
    table.to_csv(out/'profiles.csv')
    combined = pd.concat(blocks, ignore_index=True)
    # Windows are different forecasting tasks, but may share the very same target observation.
    multiplicity = combined.groupby(KEY+['date_show']).size()
    stats = dict(raw_positive_transactions=len(tr), positive_transactions_used_as_primary_targets=len(used),
                 fraction_used_as_primary_targets=len(used)/len(tr),
                 note='Unused as primary targets does not mean unused everywhere: raw data also feeds history, limited-release samples, calendar, cluster statistics and Lebaran analog.',
                 extra_window_target_rows=len(combined)-len(x), unique_target_keys=len(multiplicity),
                 target_keys_in_multiple_windows=int(multiplicity.gt(1).sum()),
                 max_windows_per_target=int(multiplicity.max()),
                 test_profile=dict(rows=len(t), scale_median=float(t.scale.median()),
                                   small_share=float(t.scale.le(20).mean()), occ_median=float(t.occ_mean.median()),
                                   late_share=float(t.first_day.eq(3).mean())),
                 protocol='EDA only, no accuracy gain established. Different origin ages cannot be relabelled as true D1. Split all origins/formats by original base film; temporal training requires target date before validation origin; normalize aggregate augmentation weight per film. D1 uses existing reconstruction, not verified official release dates.')
    (out/'summary.json').write_text(json.dumps(stats, indent=2))
    # Source rows for a future Kaggle ablation, preserving origin age and target provenance.
    combined[KEY+['base','date_show','origin_age','h','scale','y1','y2','y3','total_ticket']].to_parquet(out/'window_targets.parquet', index=False)
    plt = style()
    fig, ax = plt.subplots(1,3,figsize=(14,4))
    for a,col,title,reference in zip(ax,['small_share','occ_median','zero_rate'],
                                  ['Small-pair share (scale <= 20)','Median observed occupancy (%)','Target zero rate (train only)'],
                                  [t.scale.le(20).mean(),t.occ_mean.median(),None]):
        table[col].plot.bar(ax=a, rot=0)
        a.set(title=title, xlabel='True lifecycle age at window start')
        if reference is not None:
            a.axhline(reference,color='#e34948',ls='--',label='test D1-D3'); a.legend()
    fig.tight_layout(); fig.savefig(out/'window_profiles.png'); plt.close(fig)
    print(table.round(4).to_string()); print(json.dumps(stats,indent=2))


if __name__ == '__main__':
    main()
