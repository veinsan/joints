"""Does unused lifecycle data add independent support for test-like histories? No fitting."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY,base_title,fig_dir,style


def cells(a):
    scale=pd.cut(a.scale,[0,20,100,500,np.inf],include_lowest=True).astype(str)
    fd=np.where(a.y1>0,1,np.where(a.y2>0,2,3))
    dow=a.d1.dt.dayofweek
    return scale+'|D'+pd.Series(fd,index=a.index).astype(str)+'|dow'+dow.astype(str)


def main():
    out=fig_dir('107_lifecycle_support')
    a=pd.read_parquet(ROOT/'outputs/eda/63_lifecycle_windows/window_targets.parquet')
    t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    a['d1']=a.date_show-pd.to_timedelta(a.h-1,unit='D');a['cell']=cells(a)
    t=t.copy();t['base']=base_title(t.movie_title);t['cell']=cells(t)
    assert np.allclose(a.scale,np.maximum(a[['y1','y2','y3']].sum(axis=1)/3,1))
    support=[];counts=[]
    for name,ages in [('opening_only',[1]),('opening_plus_age4',[1,4]),('all_ages',[1,4,8,15])]:
        z=a[a.origin_age.isin(ages)]
        n=z.groupby('cell').base.nunique();pairs=z.drop_duplicates(KEY+['origin_age']).groupby('cell').size()
        nt=t.cell.map(n).fillna(0)
        support.append(dict(pool=name,rows=len(z),independent_films=z.base.nunique(),
            unique_label_keys=len(z.drop_duplicates(KEY+['date_show'])),
            test_share_no_film_support=float((nt==0).mean()),test_share_fewer_than5_films=float((nt<5).mean()),
            test_share_fewer_than10_films=float((nt<10).mean()),median_support_films=float(nt.median())))
        for c,g in t.groupby('cell'):
            counts.append(dict(pool=name,cell=c,test_rows=len(g),train_films=int(n.get(c,0)),train_pairs=int(pairs.get(c,0))))
    pd.DataFrame(support).to_csv(out/'pool_support.csv',index=False);pd.DataFrame(counts).to_csv(out/'support_cells.csv',index=False)
    # Same film labels are repeated across windows, so rows cannot be counted as independent new evidence.
    age=[]
    for (origin,cell),g in a.groupby(['origin_age','cell']):
        age.append(dict(origin_age=origin,cell=cell,rows=len(g),films=g.base.nunique(),zero_rate=g.total_ticket.eq(0).mean(),
            median_normalized_target=(g.total_ticket/g.scale).median(),growth_rate=(g.total_ticket>g.y3).mean()))
    aa=pd.DataFrame(age);aa.to_csv(out/'age_conditional_labels.csv',index=False)
    # Direct standardisation to test cells is descriptive; it does not resolve unmeasured age confounding.
    weights=t.cell.value_counts(normalize=True);std=[]
    for origin,g in aa.groupby('origin_age'):
        ww=g.cell.map(weights).fillna(0);covered=ww.sum()
        std.append(dict(origin_age=origin,covered_test_share=covered,
            zero_rate_test_cell_standardized=np.average(g.zero_rate,weights=ww),
            growth_rate_test_cell_standardized=np.average(g.growth_rate,weights=ww)))
    pd.DataFrame(std).to_csv(out/'standardized_label_shift.csv',index=False)
    original=set(a.loc[a.origin_age==1,'base']);extra=set(a.base)-original
    summary=dict(additional_independent_films=sorted(extra),feature_cells='scale bucket x first observed sale day x origin weekday',
        no_training=True,limits='Support is coarse marginal geometry, not proof of conditional exchangeability. Later windows have different lifecycle age, repeated outcomes and selection. More windows do not equal more independent films. Never change test D1/scale or train on validation-film extra windows.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(figsize=(9,4));pd.DataFrame(support).set_index('pool')[['test_share_no_film_support','test_share_fewer_than5_films','test_share_fewer_than10_films']].plot.bar(ax=ax,rot=0)
    ax.set(ylabel='Fraction of test rows',title='History support from extra lifecycle windows: coarse cells only');fig.tight_layout();fig.savefig(out/'support.png');plt.close(fig)
    print(pd.DataFrame(support).round(4).to_string(index=False));print(pd.DataFrame(std).round(4).to_string(index=False));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
