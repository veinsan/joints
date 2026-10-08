"""Build stricter split manifests and audit support, without pretending saved OOF is new CV."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import base_title, fig_dir, style


def blocked_parts(x,limited,k):
    weeks=((x.d1-pd.Timestamp('2025-03-31')).dt.days//7)
    chunks=np.array_split(np.sort(weeks.unique()),5)
    val=weeks.isin(chunks[k])
    lo=x.loc[val,'d1'].min();hi=x.loc[val,'d1'].max()+pd.Timedelta(days=9)
    held=set(base_title(x.loc[val,'movie_title']))
    ready=lambda a:a[((a.d1+pd.Timedelta(days=9)<lo)|(a.d1>hi))&~base_title(a.movie_title).isin(held)]
    tr=pd.concat([ready(x),ready(limited)],ignore_index=True)
    assert ((tr.d1+pd.Timedelta(days=9)<lo)|(tr.d1>hi)).all()
    assert not set(base_title(tr.movie_title))&held
    return tr,x[val],lo,hi


def main():
    out=fig_dir('100_purged_blocks')
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    l=pd.read_parquet(ROOT/'outputs/cache/Xlim.parquet')
    rows=[];manifest=[]
    for k in range(5):
        tr,va,lo,hi=blocked_parts(x,l,k)
        rows.append(dict(kind='purged_block',split=str(k),train_rows=len(tr),validation_rows=len(va),
                         train_films=base_title(tr.movie_title).nunique(),val_films=base_title(va.movie_title).nunique(),
                         start=str(lo.date()),end=str(hi.date())))
        for role,a in [('train',tr),('validation',va)]:
            for title in a.movie_title.unique():
                manifest.append(dict(kind='purged_block',split=str(k),role=role,movie_title=title))
    for month in ['2025-05','2025-06','2025-07','2025-08','2025-09']:
        cut=pd.Timestamp(month+'-01')
        allx=pd.concat([x,l],ignore_index=True)
        tr=allx[allx.d1+pd.Timedelta(days=9)<cut]
        va=x[x.d1.dt.strftime('%Y-%m').eq(month)]
        assert not set(base_title(tr.movie_title))&set(base_title(va.movie_title))
        assert tr.date_show.max()<va.d1.min()
        rows.append(dict(kind='expanding_month',split=month,train_rows=len(tr),validation_rows=len(va),
                         train_films=base_title(tr.movie_title).nunique(),val_films=base_title(va.movie_title).nunique(),
                         start=str(va.d1.min().date()),end=str((va.d1.max()+pd.Timedelta(days=9)).date())))
        for role,a in [('train',tr),('validation',va)]:
            for title in a.movie_title.unique():
                manifest.append(dict(kind='expanding_month',split=month,role=role,movie_title=title))
    counts=pd.DataFrame(rows);counts.to_csv(out/'counts.csv',index=False)
    pd.DataFrame(manifest).to_csv(out/'manifest.csv',index=False)
    (out/'summary.json').write_text(json.dumps({'new_scores_computed':False,
        'validation_rows_block_total':int(counts.query('kind=="purged_block"').validation_rows.sum()),
        'expected_rows':len(x),
        'limits':'Purged block CV has training both before and after the block; it is a cohort stress test, not chronological forecasting. Monthly fits are strictly past-label-only conditional on inferred D1. May has little training and must be reported separately, not hidden inside one mean.'},indent=2))
    assert counts.query('kind=="purged_block"').validation_rows.sum()==len(x)
    plt=style();fig,ax=plt.subplots(1,2,figsize=(12,4))
    for a,kind in zip(ax,['purged_block','expanding_month']):
        counts[counts.kind.eq(kind)].set_index('split')[['train_films','val_films']].plot.bar(ax=a,rot=15,title=kind)
    fig.tight_layout();fig.savefig(out/'split_support.png');plt.close(fig)
    print(counts.to_string(index=False));print('PASS: every train window is disjoint from held-out block; all monthly train labels precede origin.')


if __name__=='__main__':
    main()
