"""Pair influence, activity visibility and candidate sensitivity; no fitting or data deletion."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, fig_dir, load, style
from evaluate import test_weights_fd


def main():
    out=fig_dir('104_tail_influence');d=load()
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    x=x.merge(pd.read_csv(ROOT/'results/v17/oof.csv'),on=KEY+['h'],validate='one_to_one',suffixes=('','_saved'))
    assert np.array_equal(x.y,x.total_ticket) and np.allclose(x.scale,x.scale_saved)
    x['base']=base_title(x.movie_title);x['w']=test_weights_fd(x,t)
    x['err']=abs(x.y-x.oof_control)/x.scale;x['we']=x.w*x.err
    p=x.groupby(KEY).agg(we=('we','sum'),error=('err','sum'),scale=('scale','first'),
        y1=('y1','first'),y2=('y2','first'),y3=('y3','first'),d1=('d1','first'),
        ymean=('y','mean'),predmean=('oof_control','mean'),fold=('fold','first')).reset_index()
    p['error_share']=p.we/x.we.sum();p['plain_error_share']=p.error/x.err.sum()
    p=p.sort_values('error_share',ascending=False)
    active=d['train'].groupby(['cinema_ids','date_show']).agg(other_rows=('movie_title','size'),all_tickets=('total_ticket','sum'))
    for k in [1,2]:
        idx=pd.MultiIndex.from_arrays([p.cinema_ids,p.d1+pd.Timedelta(days=k-1)])
        p[f'cinema_active_D{k}']=idx.isin(active.index)
    p.to_csv(out/'pair_influence.csv',index=False)
    concentration=[]
    for n in [1,3,5,10,50,100]:
        concentration.append(dict(top_pairs=n,TW_error_share=p.error_share.head(n).sum(),plain_error_share=p.plain_error_share.head(n).sum()))
    pd.DataFrame(concentration).to_csv(out/'concentration.csv',index=False)
    # Top-film exclusion is sensitivity analysis selected with labels, never a replacement metric.
    films=x.groupby('base').we.sum().sort_values(ascending=False)
    rows=[]
    for label,mask in [('all',np.ones(len(x),dtype=bool)),('without_top5_error_films',~x.base.isin(films.head(5).index)),
                       ('p3_gt2',x.p3>2),('p3_le2',x.p3<=2),('scale_le20',x.scale<=20),('scale_gt20',x.scale>20)]:
        for name in ['control','tp35_genre','tp35_pool']:
            g=x[mask];e=abs(g.y-g['oof_'+name])/g.scale
            rows.append(dict(subset=label,candidate=name,rows=len(g),films=g.base.nunique(),TW=np.average(e,weights=g.w),
                             delta=np.average(e-g.err,weights=g.w)))
    pd.DataFrame(rows).to_csv(out/'candidate_sensitivity.csv',index=False)
    # Visibility flags use history only and explicitly distinguish absent film from absent cinema.
    visibility=[]
    for name,a,raw in [('train',x,d['train']),('test',t,d['hist'])]:
        a=a.drop_duplicates(KEY).copy(); a['base']=base_title(a.movie_title)
        late=a.y1.eq(0)&a.y2.eq(0)&a.y3.gt(0)
        seen=raw.groupby(['cinema_ids','date_show']).size().index
        for day in [1,2]:
            idx=pd.MultiIndex.from_arrays([a.cinema_ids,a.d1+pd.Timedelta(days=day-1)])
            visibility.append(dict(split=name,day=day,late_pairs=int(late.sum()),
                late_pairs_with_other_visible_films=int((late&idx.isin(seen)).sum()),
                note='Train includes all transactions; test only D1-D3 windows of supplied films. Absence is not comparable evidence of closure.'))
    pd.DataFrame(visibility).to_csv(out/'late_pair_visibility.csv',index=False)
    # Fixed multiplicative diagnostics on temporal predictions do not fit any correction.
    temporal=pd.read_csv(ROOT/'results/v17/temporal.csv');checks=[]
    for month,g in temporal.groupby('month'):
        for factor in [.9,1,1.1]:
            checks.append(dict(month=month,factor=factor,TW=np.average(abs(g.total_ticket-g.control*factor)/g.scale,weights=g.TW)))
    pd.DataFrame(checks).to_csv(out/'temporal_fixed_scaling.csv',index=False)
    summary=dict(total_pairs=len(p),top3_error_share=float(p.error_share.head(3).sum()),
        top3_plain_error_share=float(p.plain_error_share.head(3).sum()),
        top3_all_other_cinema_activity_D1_D2=bool(p.head(3)[['cinema_active_D1','cinema_active_D2']].all().all()),
        no_labels_or_predictions_changed=True,
        caveat='Error-ranked exclusions are retrospective sensitivity checks, not deployable rules. Raw activity rules out a whole-cinema missing day, not film-specific missing records.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(figsize=(7,4));z=pd.DataFrame(concentration)
    ax.plot(z.top_pairs,z.TW_error_share*100,marker='o',label='TW error');ax.plot(z.top_pairs,z.plain_error_share*100,marker='o',label='plain MASE error')
    ax.set(xlabel='Top pairs ranked by TW error',ylabel='Cumulative error (%)',title='Influence of a few film–cinema pairs');ax.legend()
    fig.tight_layout();fig.savefig(out/'concentration.png');plt.close(fig)
    print(json.dumps(summary,indent=2));print(pd.DataFrame(rows).round(6).to_string(index=False));print(pd.DataFrame(checks).round(6).to_string(index=False))


if __name__=='__main__':main()
