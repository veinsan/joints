"""Does the calendar-input improvement survive the hurdle model, not just direct L1?
One fixed model seed, existing 19 quantiles, lambda=.5 and hurdle weight=.75; no tuning.
Reuses direct L1 predictions from 56. Not a full foundation-model ensemble reproduction.
Run after 56: .venv/bin/python eda/60_calendar_hurdle_check.py
"""
import importlib.util
import json
import sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import base_title, load, fig_dir, style
from evaluate import folds, test_weights_fd
from model import PARAMS


def mix(l1,p0,Q,scale,cm):
    qs=np.arange(.05,1,.05)
    p=expit(logit(np.clip(p0,1e-4,1-1e-4))-.5*1.32)
    q=np.clip((.5-p)/(1-p),qs[0],qs[-1])
    positive=np.array([np.interp(v,qs,row) for v,row in zip(q,Q)])
    hurdle=np.where(p>=.5,0,np.maximum(positive,0))*scale*cm
    return .25*l1+.75*hurdle


def uncertainty(X,E,w,out):
    g=pd.DataFrame({'base':base_title(X.movie_title),'delta':(E.calendar-E.baseline)*w,'w':w}).groupby('base').sum()
    ix=np.random.default_rng(2026).integers(len(g),size=(3000,len(g)))
    boot=g.delta.to_numpy()[ix].sum(1)/g.w.to_numpy()[ix].sum(1)
    result={'tw_delta':float(np.average(E.calendar-E.baseline,weights=w)),
            'paired_film_bootstrap_95':np.quantile(boot,[.025,.975]).tolist(),
            'films_better':int((g.delta<0).sum()),'films':len(g)}
    (out/'uncertainty.json').write_text(json.dumps(result,indent=2))
    print(result)


def main():
    # Constant positive quantiles: high zero mass gives zero hurdle; low mass gives 2.
    np.testing.assert_allclose(mix(np.array([4.,4.]),np.array([.9,.1]),
                                  np.full((2,19),2.),np.ones(2),np.ones(2)),[1.,2.5])
    spec=importlib.util.spec_from_file_location('ab',Path(__file__).with_name('56_preprocessing_ablation.py'))
    ab=importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
    out=fig_dir('60_calendar_hurdle')
    X=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/train.parquet')
    T=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/test.parquet')
    groups=json.loads((ROOT/'outputs/eda/55_preprocessing/feature_groups.json').read_text())
    direct=pd.read_parquet(ROOT/'outputs/eda/56_ablation_2026/predictions.parquet')
    variants={'baseline':ab.BASE,'calendar':ab.BASE+groups['calendar']}
    comp=['c_new_n','c_new_sh','c_new_sh_rel','c_new_sh_vs_own','c_new_tx_vs_own']
    raw=load()['train']; bg=base_title(X.movie_title); rg=base_title(raw.movie_title)
    fold=folds(X); w=test_weights_fd(X,T)
    splits=[(f'group{k}',fold!=k,fold==k,raw[~rg.isin(bg[fold==k].unique())]) for k in range(5)]
    for month in (7,8,9):
        cut=pd.Timestamp(2025,month,1)
        splits.append((f'month{month}',(X.d1+pd.Timedelta(days=9)<cut).to_numpy(),(X.d1.dt.month==month).to_numpy(),raw[raw.date_show<cut]))
    params={**PARAMS,'n_estimators':300,'learning_rate':.05,'n_jobs':4,'random_state':2026}
    params.pop('objective')
    (out/'protocol.json').write_text(json.dumps({'params':params,'lambda':.5,'hurdle_weight':.75,'features':variants,
        'limits':'single seed, no limited extra rows; fixed cached calendar and visible-window covariates, baseline vs calendar only'},indent=2))
    rec=[]; preds=[]
    for split,tr,va,rr in splits:
        assert not set(bg[tr])&set(bg[va])
        a,b=ab.local_stats(X[tr],X[va],rr)
        csh=rr.groupby(['cinema_ids','date_show']).total_show.sum().groupby('cinema_ids').median()
        for x in (a,b):
            x['c_new_sh_rel']=x.c_new_sh/x.cinema_ids.map(csh).fillna(x.c_new_sh.median()+1)
        for name,cols in variants.items():
            clf=lgb.LGBMClassifier(objective='binary',**{**params,'num_leaves':31}).fit(a[cols+comp],a.total_ticket.eq(0))
            p0=clf.predict_proba(b[cols+comp])[:,1]
            pos=a[a.total_ticket>0]; target=pos.total_ticket/pos.scale/pos.cal_mult
            Q=[]
            for q in np.arange(.05,1,.05):
                m=lgb.LGBMRegressor(objective='quantile',alpha=float(q),**params).fit(pos[cols],target)
                Q.append(m.predict(b[cols]))
            Q=np.sort(np.column_stack(Q),axis=1)
            ref=direct[(direct.split==split)&(direct.variant==name)].sort_values('row')
            assert np.array_equal(ref.row,np.flatnonzero(va))
            p=mix(ref.prediction.to_numpy(),p0,Q,b.scale.to_numpy(),b.cal_mult.to_numpy())
            np.savez_compressed(out/f'{split}_{name}.npz',row=np.flatnonzero(va),p0=p0,Q=Q,pred=p)
            e=np.abs(b.total_ticket.to_numpy()-p)/b.scale.to_numpy()
            rec.append({'split':split,'variant':name,'mase':e.mean(),'tw':np.average(e,weights=w[va])})
            print(rec[-1],flush=True)
            preds.append(pd.DataFrame({'row':np.flatnonzero(va),'split':split,'variant':name,'prediction':p}))
            pd.DataFrame(rec).to_csv(out/'scores.csv',index=False)
        pd.concat(preds,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    S=pd.DataFrame(rec); P=pd.concat(preds,ignore_index=True)
    G=P[P.split.str.startswith('group')].pivot(index='row',columns='variant',values='prediction')
    E=G.apply(lambda p:np.abs(p-X.total_ticket)/X.scale)
    uncertainty(X,E,w,out)
    summary=pd.DataFrame({'group_mase':E.mean(),'group_tw':E.apply(lambda e:np.average(e,weights=w)),
        'rolling_mase':S[S.split.str.startswith('month')].groupby('variant').mase.mean(),
        'rolling_tw':S[S.split.str.startswith('month')].groupby('variant').tw.mean()})
    summary.to_csv(out/'summary.csv'); print(summary.to_string(),flush=True)
    plt=style(); fig,ax=plt.subplots(1,2,figsize=(11,4))
    summary.plot.bar(ax=ax[0],rot=0,title='Calendar preprocessing inside hurdle model')
    D=S.pivot(index='split',columns='variant',values='tw')
    (D.calendar-D.baseline).plot.bar(ax=ax[1],title='TW delta by split (lower better)'); ax[1].axhline(0,color='black')
    fig.tight_layout(); fig.savefig(out/'hurdle_check.png'); plt.close(fig)
    assert np.isfinite(G.to_numpy()).all() and (G.to_numpy()>=0).all()


if __name__=='__main__':
    main()
