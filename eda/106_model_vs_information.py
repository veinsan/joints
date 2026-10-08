"""Model-family gains, common errors and unseen calendar support. No model fitting."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY,base_title,fig_dir,load,style
from evaluate import test_weights_fd


def main():
    out=fig_dir('106_model_vs_information');d=load()
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    key=KEY+['h'];w=test_weights_fd(x,t);base=base_title(x.movie_title)
    versions={v:x[key].merge(pd.read_csv(ROOT/f'results/v{v}/oof.csv'),on=key,validate='one_to_one') for v in [8,12,13,15,16,17]}
    y=x.total_ticket.to_numpy();s=x.scale.to_numpy();rows=[];rng=np.random.default_rng(2026)
    for v in [15,17]:
        o=versions[v];assert np.allclose(y,o.y) and np.allclose(s,o.scale)
        baseline=abs(y-o.comp_lgb.to_numpy())/s
        for c in [a for a in o if a.startswith('comp_')]+['oof_final']:
            err=abs(y-o[c].to_numpy())/s;delta=err-baseline
            a=pd.DataFrame(dict(week=(x.d1-pd.Timestamp('2025-03-31')).dt.days//14,w=w,wd=w*delta)).groupby('week')[['w','wd']].sum()
            ix=rng.integers(len(a),size=(3000,len(a)))
            ci=np.quantile(a.wd.to_numpy()[ix].sum(1)/a.w.to_numpy()[ix].sum(1),[.025,.975])
            rows.append(dict(version=v,component=c,TW=np.average(err,weights=w),delta_vs_same_run_lgb=np.average(delta,weights=w),
                ci_low=ci[0],ci_high=ci[1],fold_wins=sum(np.average(delta[o.fold==k],weights=w[o.fold==k])<0 for k in sorted(o.fold.unique()))))
    pd.DataFrame(rows).to_csv(out/'within_run_model_gains.csv',index=False)
    # Correlations are descriptive: all residuals share y; extreme targets can inflate Pearson.
    correlations=[]
    err17=abs(y-versions[17].oof_final.to_numpy())/s
    top5=pd.Series(err17*w).groupby(base).sum().nlargest(5).index
    for a,b in [(8,12),(8,15),(8,17),(15,17)]:
        ea=abs(y-versions[a].oof_final.to_numpy())/s;eb=abs(y-versions[b].oof_final.to_numpy())/s
        for lab,m in [('all',np.ones(len(x),bool)),('without_top5',~base.isin(top5).to_numpy())]:
            correlations.append(dict(a=a,b=b,subset=lab,pearson=pd.Series(ea[m]).corr(pd.Series(eb[m])),
                spearman=pd.Series(ea[m]).rank().corr(pd.Series(eb[m]).rank()),
                prediction_distance=np.average(abs(versions[a].oof_final.to_numpy()[m]-versions[b].oof_final.to_numpy()[m])/s[m],weights=w[m])))
    pd.DataFrame(correlations).to_csv(out/'robust_correlations.csv',index=False)
    # Same-row comparison for v2, whose validation population differs from later versions.
    old=pd.read_csv(ROOT/'results/v2/oof.csv');z=x.merge(old,on=key,validate='one_to_one',suffixes=('','_old'))
    assert np.allclose(z.total_ticket,z.y) and np.allclose(z.scale,z.scale_old)
    ii=pd.MultiIndex.from_frame(x[key]).get_indexer(pd.MultiIndex.from_frame(z[key]));assert (ii>=0).all()
    compare=[dict(version=2,rows=len(z),MASE=np.mean(abs(z.y-z.pred)/z.scale),TW=np.average(abs(z.y-z.pred)/z.scale,weights=w[ii]))]
    for v,o in versions.items():
        e=abs(y[ii]-o.oof_final.to_numpy()[ii])/s[ii]
        compare.append(dict(version=v,rows=len(ii),MASE=e.mean(),TW=np.average(e,weights=w[ii])))
    pd.DataFrame(compare).to_csv(out/'v2_common_subset.csv',index=False)
    def segments(a):
        return np.select([a.date_show.between('2026-03-21','2026-03-27'),
            a.date_show.between('2026-02-19','2026-03-20'),a.date_show.between('2025-12-20','2026-01-04')],
            ['lebaran','ramadan','year_end'],default='other')
    t=t.copy();t['segment']=segments(t);t['base']=base_title(t.movie_title)
    subs={v:t[['id']].merge(pd.read_csv(ROOT/f'results/v{v}/submission.csv'),on='id',validate='one_to_one').total_ticket.to_numpy() for v in range(1,18)}
    seg=[]
    for name,g in t.groupby('segment'):
        ix=g.index.to_numpy();share=len(ix)/len(t)
        seg.append(dict(segment=name,rows=len(g),films=g.base.nunique(),full_test_share=share,
            conditional_required_local_gain_if_only_segment_improves=(.39918-.33662)/share,
            mean_v15_pred_over_scale=np.mean(subs[15][ix]/g.scale.to_numpy()),
            v8_v15_distance=np.mean(abs(subs[15][ix]-subs[8][ix])/g.scale.to_numpy()),
            v12_v15_distance=np.mean(abs(subs[15][ix]-subs[12][ix])/g.scale.to_numpy())))
    pd.DataFrame(seg).to_csv(out/'test_segments.csv',index=False)
    changes=[]
    for a,b in [(3,4),(4,5),(5,6),(6,8),(8,15),(15,17)]:
        for name,g in t.groupby('segment'):
            ix=g.index.to_numpy();changes.append(dict(a=a,b=b,segment=name,rows_changed=int(np.sum(subs[a][ix]!=subs[b][ix])),
                distance_contribution=np.sum(abs(subs[a][ix]-subs[b][ix])/g.scale.to_numpy())/len(t)))
    pd.DataFrame(changes).to_csv(out/'historical_test_changes.csv',index=False)
    raw=d['train'].assign(base=base_title(d['train'].movie_title))
    overlap=t[t.base.isin(raw.base)].groupby('base').agg(test_rows=('id','size'),test_d1=('d1','min'))
    overlap=overlap.join(raw.groupby('base').agg(raw_rows=('date_show','size'),first_raw=('date_show','min'),last_raw=('date_show','max')))
    overlap.to_csv(out/'test_titles_seen_in_train.csv')
    support=[]
    for name,a in [('train',x),('test',t)]:
        for dow,g in a.groupby('d1_dow'):
            support.append(dict(split=name,d1_dow=dow,rows=len(g),films=base_title(g.movie_title).nunique(),
                first_weekend_outside_history=dow in [0,1,2],share=len(g)/len(a)))
    pd.DataFrame(support).to_csv(out/'release_weekday_support.csv',index=False)
    summary=dict(no_model_fit=True,external_features_added=False,overlap_test_titles=len(overlap),
        overlap_test_row_share=float(overlap.test_rows.sum()/len(t)),
        limits='Block bootstrap conditions on saved selected models; no correction for historical model search. Segment arithmetic assumes public proportions equal full test, which is unknown. Same-film overlap is sparse prerelease screenings, not D4-D10 target labels.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(figsize=(9,4));z=pd.DataFrame(rows).query('version==15')
    ax.barh(z.component,z.delta_vs_same_run_lgb);ax.axvline(0,color='black',lw=1)
    ax.set(xlabel='TW difference vs same-run LightGBM',title='v15: model family matters, but gains are limited here')
    fig.tight_layout();fig.savefig(out/'model_gains.png');plt.close(fig)
    print(pd.DataFrame(rows).round(6).to_string(index=False));print(pd.DataFrame(correlations).round(4).to_string(index=False));print(pd.DataFrame(seg).round(5).to_string(index=False))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
