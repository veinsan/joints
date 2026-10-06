"""Predeclared, controlled feature experiments: grouped-film CV + rolling months.
Run 55 first, then: .venv/bin/python eda/56_preprocessing_ablation.py
These are local LGB screening experiments, NOT a rerun of v10's ensemble/hurdle.
Cached calendar constants and official visible release-window features are held fixed;
the rolling test is not a fully historical real-time reconstruction of their availability.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import base_title, load, fig_dir, style
from evaluate import folds, test_weights_fd
from features import GENRES
from model import fit_predict

BASE = ['log_s','p1','p2','p3','n_hist','sh_trend','fnc1','fnc2','fnc3','fp1','fp2','fp3','f_logT',
        'f_nc_trend','fmt','fmt_share','d1_dow','rating']+[f'g_{g}' for g in GENRES]+[
        'n_genre','n_cast','comp_n','f_share_cohort','h','dow','t_hol','t_school','t_ramadan','cal_mult',
        'hol_in_hist','n_comp_open','share','p3_rel','p1_rel','cin_size','cin_nfilms','cin_new','price_wkd','price_prem']


def local_stats(a,b,raw):
    cs = raw.groupby('cinema_ids').total_ticket.sum()/raw.groupby('cinema_ids').date_show.nunique()
    nf = raw.groupby(['cinema_ids','date_show']).movie_title.nunique().groupby('cinema_ids').mean()
    a,b = a.copy(),b.copy()
    for x in (a,b):
        x['cin_size'] = x.cinema_ids.map(np.log10(cs))
        x['cin_nfilms'] = x.cinema_ids.map(nf)
        x['cin_new'] = x.cin_size.isna().astype(int)
    return a,b


def error_slices(X, P, w, out):
    p=P[P.split.str.startswith('group')].pivot(index='row',columns='variant',values='prediction')
    assert np.array_equal(p.index,np.arange(len(X)))
    for variant in p.columns.drop('baseline'):
        delta=((p[variant]-X.total_ticket).abs()-(p.baseline-X.total_ticket).abs())/X.scale
        z=X.assign(delta_contribution=delta*w/w.sum(),base=base_title(X.movie_title))
        for key in ('first_day','h','base'):
            z.groupby(key).delta_contribution.sum().sort_values().to_csv(out/f'{variant}_delta_{key}.csv')


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--rounds',type=int,default=600)
    args=parser.parse_args()
    out=fig_dir(f'56_ablation_{args.seed}')
    pre=ROOT/'outputs/eda/55_preprocessing'
    X=pd.read_parquet(pre/'train.parquet'); T=pd.read_parquet(pre/'test.parquet')
    groups=json.loads((pre/'feature_groups.json').read_text())
    variants={'baseline':BASE, **{k:BASE+v for k,v in groups.items()}}
    # Fixed before seeing results. Each addition separately. No tuning from public scores.
    protocol={'seed':args.seed,'rounds':args.rounds,'features':variants,
              'decision':'Candidate needs negative grouped TW delta, negative mean rolling delta, no rolling month > +0.003; then confirm on second grouped split.',
              'limits':'Single-seed LGB, fixed existing calendar/constants and release-window covariates; not ensemble validation.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    raw=load()['train']; bg=base_title(X.movie_title); rg=base_title(raw.movie_title)
    fold=folds(X,seed=args.seed); w=test_weights_fd(X,T)
    splits=[]
    for f in range(5):
        va=fold==f; tr=~va
        rr=raw[~rg.isin(bg[va].unique())]
        assert not set(bg[tr]) & set(bg[va])
        splits.append((f'group{f}',tr,va,rr))
    for month in (7,8,9):
        cut=pd.Timestamp(2025,month,1)
        tr=(X.d1+pd.Timedelta(days=9)<cut).to_numpy()
        va=(X.d1.dt.month==month).to_numpy()
        assert not set(bg[tr]) & set(bg[va])
        splits.append((f'month{month}',tr,va,raw[raw.date_show<cut]))
    records=[]; predictions=[]; curves=[]
    for name,tr,va,rr in splits:
        a,b=local_stats(X[tr],X[va],rr)
        for variant,cols in variants.items():
            pred,m=fit_predict(a,b,cols,params={'n_jobs':4},n_estimators=args.rounds)
            e=np.abs(b.total_ticket.to_numpy()-pred)/b.scale.to_numpy()
            rec={'split':name,'variant':variant,'rows':len(b),'films':base_title(b.movie_title).nunique(),
                 'mase':e.mean(),'tw':np.average(e,weights=w[va]),
                 'small_mase':e[b.scale.to_numpy()<=20].mean()}
            records.append(rec); print(json.dumps(rec),flush=True)
            p=b[['movie_title','cinema_ids','h','total_ticket','scale']].copy()
            p['row']=np.flatnonzero(va); p['split']=name; p['variant']=variant; p['prediction']=pred
            predictions.append(p)
            if variant=='baseline':
                for it in (100,300,args.rounds):
                    for kind,c in [('train',a),('validation',b)]:
                        pp=np.maximum(m.predict(c[cols],num_iteration=it),0)*c.scale.to_numpy()*c.cal_mult.to_numpy()
                        curves.append({'split':name,'rounds':it,'kind':kind,'mase':np.mean(np.abs(c.total_ticket-pp)/c.scale)})
            pd.DataFrame(records).to_csv(out/'scores.csv',index=False)
        pd.concat(predictions,ignore_index=True).to_parquet(out/'predictions.parquet',index=False)
    scores=pd.DataFrame(records); curves=pd.DataFrame(curves)
    curves.to_csv(out/'learning_curves.csv',index=False)
    P=pd.concat(predictions,ignore_index=True)
    error_slices(X,P,w,out)
    summary=[]
    for variant in variants:
        p=P[(P.variant==variant)&P.split.str.startswith('group')].sort_values('row')
        assert np.array_equal(p.row,np.arange(len(X)))
        e=np.abs(X.total_ticket.to_numpy()-p.prediction.to_numpy())/X.scale.to_numpy()
        months=scores[(scores.variant==variant)&scores.split.str.startswith('month')]
        summary.append({'variant':variant,'group_mase':e.mean(),'group_tw':np.average(e,weights=w),
                        'rolling_macro_mase':months.mase.mean()})
    summary=pd.DataFrame(summary).set_index('variant')
    for c in summary.columns:
        summary[c+'_delta']=summary[c]-summary.loc['baseline',c]
    ms=scores[scores.split.str.startswith('month')].pivot(index='variant',columns='split',values='mase')
    summary['worst_month_delta']=(ms-ms.loc['baseline']).max(axis=1)
    summary['screen_pass']=(summary.group_tw_delta<0)&(summary.rolling_macro_mase_delta<0)&(summary.worst_month_delta<=.003)
    summary.to_csv(out/'summary.csv'); print(summary.to_string(),flush=True)
    plt=style(); fig,ax=plt.subplots(1,3,figsize=(15,4))
    summary[['group_mase','group_tw','rolling_macro_mase']].plot.bar(ax=ax[0],rot=15,title='Controlled feature additions; lower better')
    (ms-ms.loc['baseline']).T.plot.bar(ax=ax[1],rot=0,title='Monthly MASE difference vs baseline'); ax[1].axhline(.003,color='black',ls='--')
    curves.groupby(['rounds','kind']).mase.mean().unstack().plot(ax=ax[2],marker='o',title='Baseline learning curves (all splits)')
    fig.tight_layout(); fig.savefig(out/'ablation.png'); plt.close(fig)


if __name__=='__main__':
    main()
