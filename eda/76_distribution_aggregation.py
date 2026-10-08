"""Fixed-weight CDF-mixture experiment on saved OOF distributions; no model fitting.

This is exploratory screening on reused OOF, not a fresh temporal validation.
"""
import ast
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, style
from evaluate import test_weights_fd


def shifted(p,lam,a):
    p=np.clip(p,1e-4,1-1e-4)
    return 1/(1+np.exp(-(np.log(p/(1-p))+lam*a)))


def positive_cdf(values,q,taus):
    j=np.clip((q<=values[:,None]).sum(1)-1,0,len(taus)-2)
    rr=np.arange(len(q));lo,hi=q[rr,j],q[rr,j+1]
    f=taus[j]+(taus[j+1]-taus[j])*np.divide(values-lo,hi-lo,out=np.zeros_like(values),where=hi>lo)
    return np.where(values<q[:,0],0,np.where(values>=q[:,-1],1,np.clip(f,0,1)))


def mixture_median(parts,atoms):
    n=len(parts[0][1]);lo=np.zeros(n)
    hi=np.maximum.reduce([q[:,-1] for _,_,q,_ in parts]+[v for _,v in atoms])+1
    def cdf(v):
        return sum(w*(p+(1-p)*positive_cdf(v,q,taus)) for w,p,q,taus in parts)+sum(w*(v>=a) for w,a in atoms)
    at_zero=cdf(lo)>=.5
    for _ in range(36):
        mid=(lo+hi)/2;left=cdf(mid)>=.5
        hi=np.where(left,mid,hi);lo=np.where(left,lo,mid)
    return np.where(at_zero,0,hi)


def main():
    out=ROOT/'outputs/eda/76_distribution_aggregation';out.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    o=x[KEY+['h']].merge(pd.read_csv(ROOT/'results/v13/oof.csv'),on=KEY+['h'],validate='one_to_one')
    assert np.allclose(o.y,x.total_ticket) and np.allclose(o.scale,x.scale)
    with (ROOT/'results/v13/model_weights.pkl').open('rb') as f: saved=pickle.load(f)
    cfg=SimpleNamespace(**saved['settings'])
    # Reuse the executed notebook's calendar definitions, without executing fitting or I/O cells.
    nb=json.loads((ROOT/'notebooks/v13.ipynb').read_text())
    source=next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and ''.join(c['source']).startswith('SCHOOL_BREAKS'))
    names={'SCHOOL_BREAKS','JABAR_BREAKS','JABAR_CITIES','CUTI_BERSAMA','RAMADAN'}
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='calendar' or isinstance(n,ast.Assign) and any(isinstance(z,ast.Name) and z.id in names for z in n.targets)]
    env={'np':np,'pd':pd,'CFG':cfg};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<v13 calendar>','exec'),env)
    hol=pd.read_csv(ROOT/'data/holidays.csv',parse_dates=['date']);cal=env['calendar'](hol);jb=env['calendar'](hol,env['JABAR_BREAKS'])
    regional=x.city_name.isin(env['JABAR_CITIES']).to_numpy()&cfg.REGIONAL_CAL
    at=lambda dates:np.where(regional,jb.cal.reindex(dates),cal.cal.reindex(dates))
    cm=at(x.date_show)/np.stack([at(x.d1+pd.Timedelta(days=d)) for d in range(3)],1).mean(1)
    y,s=o.y.to_numpy(),o.scale.to_numpy();w=test_weights_fd(x,t)
    z=np.load(ROOT/'results/v13/oof_quantiles.npz');taus=np.asarray(cfg.TAB_QS);ltaus=np.asarray(cfg.QS)
    a=saved['pull_a_used'];raw_parts={};parts=[]
    for name in ['fast35','tabm','tabpfn25q']:
        q=z[f'{name}_Q'];p0=np.array([np.interp(cfg.ZERO_EPS,qr,taus,left=0,right=1) for qr in q])
        pn=shifted(p0,saved['pull_shift_per_model'][name],a)
        u=np.clip(p0+(1-p0)*(.5-pn)/(1-pn),taus[0],taus[-1])
        median=np.where(pn>=.5,0,np.maximum([np.interp(ui,taus,qr) for ui,qr in zip(u,q)],0))
        assert np.allclose(median*cm*s,o[f'comp_{name}'],rtol=1e-7,atol=1e-6),name
        # Positive conditional quantile grid, avoiding interpolation through the zero atom.
        pos=np.maximum(np.array([np.interp(pi+(1-pi)*taus,taus,qr) for pi,qr in zip(p0,q)]),0)
        raw_parts[name]=(pn,pos,taus)
        if name in saved['blend_weights']:
            parts.append((saved['blend_weights'][name],pn,pos,taus))
    lp=shifted(z['lgb_p0'],cfg.LAMBDA,a);lq=np.maximum(z['lgb_Q'],0)
    u=np.clip((.5-lp)/(1-lp),ltaus[0],ltaus[-1]);hm=np.where(lp>=.5,0,np.array([np.interp(ui,ltaus,qr) for ui,qr in zip(u,lq)]))
    l1=(o.comp_lgb.to_numpy()/(cm*s)-cfg.HURDLE_W*hm)/(1-cfg.HURDLE_W)
    assert l1.min()>-1e-5
    lw=saved['blend_weights']['lgb'];parts.append((lw*cfg.HURDLE_W,lp,lq,ltaus));atoms=[(lw*(1-cfg.HURDLE_W),np.maximum(l1,0))]
    assert abs(sum(p[0] for p in parts)+sum(p[0] for p in atoms)-1)<1e-10
    # Runnable mathematical checks: atom median and a continuous uniform example.
    tq=np.tile([1.,2.,3.],(2,1));tt=np.array([.1,.5,.9])
    assert np.allclose(mixture_median([(1.,np.array([.6,0.]),tq,tt)],[]),[0.,2.],atol=1e-6)
    pm=mixture_median(parts,atoms)*cm*s
    base=o.oof_final.to_numpy();assert np.allclose(base,sum(saved['blend_weights'][n]*o[f'comp_{n}'].to_numpy() for n in saved['blend_weights']))
    unanimous=np.logical_and.reduce([o[f'comp_{n}'].to_numpy()>0 for n in saved['blend_weights']])
    candidates={'v13_saved':base,'cdf_mixture_fixed_weights':pm,'halfway_point_and_cdf':(base+pm)/2,
                'cdf_only_unanimous_positive':np.where(unanimous,pm,base)}
    rows=[];detail=[]
    for name,pred in candidates.items():
        err=np.abs(y-pred)/s
        rows.append(dict(candidate=name,MASE=err.mean(),TW=np.average(err,weights=w),
                         zero_TW=np.average(err[y==0],weights=w[y==0]),positive_TW=np.average(err[y>0],weights=w[y>0]),
                         growth_TW=np.average(err[y>x.y3],weights=w[y>x.y3])))
        for month in sorted(x.d1.dt.strftime('%Y-%m').unique()):
            m=x.d1.dt.strftime('%Y-%m').eq(month);detail.append(dict(candidate=name,month=month,TW=np.average(err[m],weights=w[m])))
    scores=pd.DataFrame(rows).set_index('candidate');scores.to_csv(out/'scores.csv')
    monthly=pd.DataFrame(detail).pivot(index='month',columns='candidate',values='TW');monthly.to_csv(out/'grouped_oof_by_month_NOT_temporal.csv')
    ci=[];rng=np.random.default_rng(2026)
    for name,p in list(candidates.items())[1:]:
        d=(np.abs(y-p)-np.abs(y-base))/s*w
        g=pd.DataFrame({'week':x.d1.dt.to_period('W-TUE').astype(str),'d':d,'w':w}).groupby('week').sum()
        ix=rng.integers(0,len(g),(10000,len(g)));b=g.d.to_numpy()[ix].sum(1)/g.w.to_numpy()[ix].sum(1)
        ci.append(dict(candidate=name,delta=d.sum()/w.sum(),lo=np.quantile(b,.025),hi=np.quantile(b,.975)))
    pd.DataFrame(ci).to_csv(out/'cohort_bootstrap.csv',index=False)
    pd.DataFrame(candidates).to_parquet(out/'oof_candidates.parquet',index=False)
    summary=dict(fixed_weights=saved['blend_weights'],calendar_reproduction_checked=True,
                 limits='Quantile grids approximate distributions. CDF blend is different from averaging quantiles. All scores reuse grouped OOF; month slices are NOT rolling-origin predictions. No test prediction or submission created. No temporal arrays were saved in v13, so a positive screening result still requires Kaggle confirmation.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(1,3,figsize=(15,4))
    scores[['TW','positive_TW']].plot.bar(ax=ax[0],rot=15);ax[0].set(title='Fixed weights: point median vs CDF median')
    (monthly.sub(monthly.v13_saved,axis=0)).drop(columns='v13_saved').plot.bar(ax=ax[1],rot=15);ax[1].axhline(0,color='gray');ax[1].set(title='Existing grouped OOF sliced by month',ylabel='TW change')
    sample=np.random.default_rng(2026).choice(len(x),1200,replace=False);ax[2].scatter(base[sample]/s[sample],pm[sample]/s[sample],s=5,alpha=.3);ax[2].plot([0,4],[0,4],ls='--',color='gray');ax[2].set(xlim=(0,4),ylim=(0,4),xlabel='Saved v13 prediction / scale',ylabel='CDF mixture / scale',title='Inspect transformed predictions')
    fig.tight_layout();fig.savefig(out/'cdf_check.png');plt.close(fig)
    print(scores.round(6).to_string());print(pd.DataFrame(ci).round(6).to_string(index=False));print(monthly.round(6).to_string())


if __name__=='__main__':main()
