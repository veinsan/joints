"""Second grouped split, and one predefined repair for the panel feature rejected by 56.
Calendar is confirmed on a different grouping (same data, not independent new evidence).
Panel repair shrinks the common-cinema correction toward the original trend by n/(n+20).
No tuning of 20, no post-hoc weight optimization. Run after 55 and 56.
"""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import base_title, load, fig_dir, style
from evaluate import folds, test_weights_fd
from model import fit_predict


def main():
    spec=importlib.util.spec_from_file_location('ab',Path(__file__).with_name('56_preprocessing_ablation.py'))
    ab=importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
    out=fig_dir('59_confirmation')
    X=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/train.parquet')
    T=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/test.parquet')
    groups=json.loads((ROOT/'outputs/eda/55_preprocessing/feature_groups.json').read_text())
    rawtrend=np.log1p(X.fT3)-np.log1p(X.fT1)
    X['panel_correction_shrunk']=(X.panel_log31-rawtrend)*X.panel_n/(X.panel_n+20)
    assert np.all(X.panel_correction_shrunk.abs()<=(X.panel_log31-rawtrend).abs()+1e-12)
    variants={'baseline':ab.BASE,'calendar':ab.BASE+groups['calendar'],
              'panel_repaired':ab.BASE+['panel_correction_shrunk']}
    (out/'protocol.json').write_text(json.dumps({'seed':2027,'rounds':600,'shrinkage_pseudocount':20,
        'variants':variants,'gate':'negative grouped TW and rolling mean delta; no month worse >0.003; paired film bootstrap reported, no claim independent holdout'},indent=2))
    raw=load()['train']; bg=base_title(X.movie_title); rg=base_title(raw.movie_title)
    w=test_weights_fd(X,T); fold=folds(X,seed=2027)
    splits=[(f'group{k}',fold!=k,fold==k,raw[~rg.isin(bg[fold==k].unique())]) for k in range(5)]
    for month in (7,8,9):
        cut=pd.Timestamp(2025,month,1)
        splits.append((f'month{month}',(X.d1+pd.Timedelta(days=9)<cut).to_numpy(),(X.d1.dt.month==month).to_numpy(),raw[raw.date_show<cut]))
    records=[]; predictions=[]
    # Calendar/baseline temporal runs are deterministic repeats: reuse verified 56 predictions.
    previous=pd.read_parquet(ROOT/'outputs/eda/56_ablation_2026/predictions.parquet')
    for split,tr,va,rr in splits:
        assert not set(bg[tr])&set(bg[va])
        a,b=ab.local_stats(X[tr],X[va],rr)
        for name,cols in variants.items():
            if split.startswith('month') and name in ('baseline','calendar'):
                p=previous[(previous.split==split)&(previous.variant==name)].sort_values('row')
                assert np.array_equal(p.row,np.flatnonzero(va))
                pred=p.prediction.to_numpy()
            else:
                pred,_=fit_predict(a,b,cols,n_estimators=600,params={'n_jobs':4})
            e=np.abs(b.total_ticket.to_numpy()-pred)/b.scale.to_numpy()
            records.append({'split':split,'variant':name,'mase':e.mean(),'tw':np.average(e,weights=w[va])})
            print(records[-1],flush=True)
            predictions.append(pd.DataFrame({'row':np.flatnonzero(va),'split':split,'variant':name,'prediction':pred}))
            pd.DataFrame(records).to_csv(out/'scores.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    P=pd.concat(predictions,ignore_index=True); S=pd.DataFrame(records)
    PP=P[P.split.str.startswith('group')].pivot(index='row',columns='variant',values='prediction')
    assert np.array_equal(PP.index,np.arange(len(X)))
    err=PP.apply(lambda col:np.abs(col-X.total_ticket)/X.scale)
    monthly=S[S.split.str.startswith('month')].pivot(index='variant',columns='split',values='mase')
    rows=[]; rng=np.random.default_rng(2027)
    for name in variants:
        d=(err[name]-err.baseline)*w
        g=pd.DataFrame({'base':bg,'delta':d,'w':w}).groupby('base').sum()
        ix=rng.integers(len(g),size=(3000,len(g)))
        boot=g.delta.to_numpy()[ix].sum(1)/g.w.to_numpy()[ix].sum(1)
        rows.append({'variant':name,'group_mase':err[name].mean(),'group_tw':np.average(err[name],weights=w),
                     'tw_delta':np.average(err[name]-err.baseline,weights=w),
                     'ci_low':np.quantile(boot,.025),'ci_high':np.quantile(boot,.975),
                     'films_better':int((g.delta<0).sum()),
                     'rolling_delta':(monthly.loc[name]-monthly.loc['baseline']).mean(),
                     'worst_month_delta':(monthly.loc[name]-monthly.loc['baseline']).max()})
    summary=pd.DataFrame(rows).set_index('variant'); summary.to_csv(out/'summary.csv')
    print(summary.to_string(),flush=True)
    plt=style(); fig,ax=plt.subplots(1,3,figsize=(14,4))
    summary[['group_mase','group_tw']].plot.bar(ax=ax[0],rot=15,title='Different grouped split (2027)')
    (monthly-monthly.loc['baseline']).T.plot.bar(ax=ax[1],rot=0,title='Temporal delta after repair')
    f=X.drop_duplicates('movie_title'); correction=f.panel_log31-(np.log1p(f.fT3)-np.log1p(f.fT1))
    ax[2].scatter(correction,f.panel_correction_shrunk,c=f.panel_n,cmap='viridis',s=16)
    ax[2].plot([-2,2],[-2,2],'k--'); ax[2].set(xlabel='Raw panel correction',ylabel='Shrunk correction',title='Inspect repaired preprocessing')
    fig.tight_layout(); fig.savefig(out/'confirmation.png'); plt.close(fig)


if __name__=='__main__':
    main()
