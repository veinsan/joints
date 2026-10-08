"""Audit discarded, already-observed labels near cutoffs; export candidates without training."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY,OUTAGE,TRAIN_END,base_title,fig_dir,load,release_dates,simulate,scale,style


def observed_partial(tx,anchors,cutoff):
    # Keep complete D1-D3 only, and never export target dates beyond the observation cutoff.
    eligible=anchors[(anchors+pd.Timedelta(days=2)<=cutoff)&(anchors+pd.Timedelta(days=9)>cutoff)]
    history,target=simulate(tx[tx.date_show<=cutoff],eligible,last_date=cutoff+pd.Timedelta(days=9))
    target=target[target.date_show<=cutoff].copy()
    target=target.join(scale(history),on=KEY)
    target['d1']=target.movie_title.map(eligible)
    target['h']=(target.date_show-target.d1).dt.days+1
    target['base']=base_title(target.movie_title)
    assert target.date_show.le(cutoff).all() and target.h.between(4,10).all()
    assert not target.date_show.isin(OUTAGE).any()
    assert not target.duplicated(KEY+['date_show']).any()
    assert (target.d1+pd.Timedelta(days=2)<=cutoff).all()
    return history,target


def main():
    out=fig_dir('108_partial_windows');d=load();raw=d['train'];anchors=release_dates(raw)
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    first=raw.groupby('movie_title').date_show.min();anchors=anchors.drop(first[first.eq(raw.date_show.min())].index,errors='ignore')
    history,z=observed_partial(raw,anchors,TRAIN_END)
    existing=pd.MultiIndex.from_frame(x[KEY+['date_show']]);assert not pd.MultiIndex.from_frame(z[KEY+['date_show']]).isin(existing).any()
    rawy=raw[KEY+['date_show','total_ticket']]
    checked=z.merge(rawy,on=KEY+['date_show'],how='left',validate='one_to_one',suffixes=('','_raw'))
    np.testing.assert_array_equal(checked.total_ticket,checked.total_ticket_raw.fillna(0))
    # Fixed anchors separate this check from the future-dependent D1 reconstruction problem.
    future=raw[raw.date_show>TRAIN_END-pd.Timedelta(days=7)].copy();future.date_show+=pd.Timedelta(days=100)
    future.total_ticket=99999999
    _,poison=observed_partial(pd.concat([raw,future],ignore_index=True),anchors,TRAIN_END)
    pd.testing.assert_frame_equal(z,poison)
    history.to_parquet(out/'candidate_history.parquet',index=False);z.to_parquet(out/'candidate_targets.parquet',index=False)
    films=z.groupby('base').agg(rows=('h','size'),pairs=('cinema_ids','nunique'),d1=('d1','min'),
        h_min=('h','min'),h_max=('h','max'),scale_median=('scale','median'),zero_share=('total_ticket',lambda q:q.eq(0).mean()))
    films.to_csv(out/'new_films.csv')
    byh=z.groupby('h').agg(rows=('h','size'),films=('base','nunique'));byh['existing_rows']=x.groupby('h').size()
    byh['additional_fraction']=byh.rows/byh.existing_rows;byh.to_csv(out/'horizon_support.csv')
    # Cutoff examples are availability counts only, conditional on globally inferred anchors.
    rows=[]
    for c in pd.to_datetime(['2025-04-30','2025-05-31','2025-06-30','2025-07-31','2025-08-31','2025-09-30']):
        _,q=observed_partial(raw,anchors,c)
        rows.append(dict(cutoff=str(c.date()),additional_rows=len(q),additional_films=q.base.nunique(),
            h4_rows=int(q.h.eq(4).sum()),h5_rows=int(q.h.eq(5).sum())))
    pd.DataFrame(rows).to_csv(out/'cutoff_opportunities.csv',index=False)
    # Toy case: last two unobserved target days are absent, never pseudo-zero labels.
    dates=pd.date_range('2025-08-01',periods=8)
    toy=pd.DataFrame(dict(movie_title='toy',cinema_ids='c',city_name='city',date_show=dates,
        total_ticket=np.arange(1,9),occupation_rate=10,total_show=1))
    _,q=observed_partial(toy,pd.Series({'toy':dates[0]},name='d1'),dates[-1])
    assert q.h.tolist()==[4,5,6,7,8] and q.total_ticket.tolist()==[4,5,6,7,8]
    assert np.allclose(q.scale,2)
    summary=dict(additional_rows=len(z),additional_base_films=z.base.nunique(),new_bases_vs_main=sorted(set(z.base)-set(base_title(x.movie_title))),
        new_scale_median=float(z.scale.median()),new_small_share=float(z.scale.le(20).mean()),test_small_share=float(t.scale.le(20).mean()),
        future_record_poison_check=True,toy_right_censoring_check=True,raw_label_lookup_check=True,
        no_training=True,limits='Candidate preprocessing only. Validation remains complete-window. Training extras must obey fold base-film exclusion and purge/cutoff per row. Availability audit still uses existing future-dependent release reconstruction. No new CV/public gain established; extra recent films are not independent evidence of generalization.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(figsize=(8,4));byh.rows.plot.bar(ax=ax,rot=0)
    ax.set(title='Known target labels discarded by requiring complete D10',xlabel='Horizon',ylabel='Additional observed training rows')
    fig.tight_layout();fig.savefig(out/'known_labels.png');plt.close(fig)
    print(films.to_string());print(byh.to_string());print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
