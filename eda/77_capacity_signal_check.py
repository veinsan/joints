"""D1-D3 capacity-pressure features: check preprocessing, support, and film-level signal. No fitting."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, style
from evaluate import test_weights_fd


def pressure(x):
    p=x.drop_duplicates(KEY).copy()
    p['base']=base_title(p.movie_title)
    valid=(p.sh3>0)&(p.occ3>0)&(p.y3>0)
    # Zero-filled missing transactions do not represent measured empty auditoriums.
    p['seats_per_show3']=np.where(valid,p.y3/(p.occ3/100).clip(lower=.005)/p.sh3.clip(lower=1),np.nan)
    p['sat3']=np.where(valid,p.occ3.ge(70).astype(float),np.nan)
    p['spare3']=np.where(valid,1-p.occ3.clip(0,100)/100,np.nan)
    p['sat1']=np.where((p.sh1>0)&(p.y1>0),p.occ1.ge(70).astype(float),np.nan)
    # Clip inverse spare capacity; do not fabricate ticket labels or change the official scale.
    p['pressure3']=np.where(valid,1/p.spare3.clip(lower=.1),np.nan)
    film=p.groupby('base').agg(saturated_share3=('sat3','mean'),saturated_share1=('sat1','mean'),
                              pressure_median3=('pressure3','median'),pressure_p903=('pressure3',lambda z:z.quantile(.9)),
                              seats_per_show_median3=('seats_per_show3','median'),occ3_median=('occ3','median'),
                              pairs=('sat3','size'),d1=('d1','min'))
    return p,film


def main():
    out=ROOT/'outputs/eda/77_capacity_signal';out.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    pp,f=pressure(x);pt,ft=pressure(t)
    o=x[KEY+['h']].merge(pd.read_csv(ROOT/'results/v13/oof.csv'),on=KEY+['h'],validate='one_to_one')
    assert np.allclose(o.y,x.total_ticket) and np.allclose(o.scale,x.scale)
    w=test_weights_fd(x,t);base=base_title(x.movie_title)
    er=pd.DataFrame({'base':base,'w':w,'signed':(o.y-o.oof_final)/o.scale*w,'absolute':abs(o.y-o.oof_final)/o.scale*w,
                     'growth':(o.y>x.y3)*w})
    sums=er.groupby('base').sum();f=f.join(sums)
    f['signed_miss']=f.signed/f.w;f['TW']=f.absolute/f.w;f['growth_share']=f.growth/f.w
    poison=x.copy();poison['total_ticket']=999999999.;poison['date_show']=pd.Timestamp('2099-01-01')
    _,fp=pressure(poison);pd.testing.assert_frame_equal(fp,f.drop(columns=list(sums.columns)+['signed_miss','TW','growth_share']))
    assert np.allclose(x.scale,np.maximum(x[['y1','y2','y3']].sum(axis=1)/3,1))
    assert pp.pressure3.dropna().between(1,10).all()
    f.to_csv(out/'train_film_features_and_errors.csv');ft.to_csv(out/'test_film_features.csv')
    cols=['saturated_share3','saturated_share1','pressure_median3','pressure_p903','seats_per_show_median3']
    rows=[];rng=np.random.default_rng(2026)
    for col in cols:
        q=f[[col,'signed_miss']].dropna();r=spearmanr(q[col],q.signed_miss)
        boots=[]
        for _ in range(1500):
            b=q.iloc[rng.integers(0,len(q),len(q))]
            if b[col].nunique()>1:boots.append(spearmanr(b[col],b.signed_miss).statistic)
        rows.append(dict(feature=col,films=len(q),spearman=r.statistic,p_unadjusted=r.pvalue,
                         lo=np.quantile(boots,.025),hi=np.quantile(boots,.975),
                         train_median=f[col].median(),test_median=ft[col].median()))
    corr=pd.DataFrame(rows);corr.to_csv(out/'signal_screen.csv',index=False)
    month=[]
    for m,q in f.groupby(f.d1.dt.strftime('%Y-%m')):
        for col in cols:
            r=spearmanr(q[col],q.signed_miss).statistic if q[col].nunique()>1 else np.nan
            month.append(dict(month=m,feature=col,rho=r,films=len(q)))
    pd.DataFrame(month).to_csv(out/'signal_by_month.csv',index=False)
    # Inspect examples from all pressure levels, including extreme capacities.
    examples=pd.concat([pp.nlargest(5,'pressure3'),pp.nsmallest(5,'pressure3'),pp.nlargest(5,'seats_per_show3')]).drop_duplicates(KEY)
    examples[KEY+['d1','y3','sh3','occ3','seats_per_show3','pressure3','scale']].to_csv(out/'preprocessing_examples.csv',index=False)
    summary=dict(train_pairs=len(pp),test_pairs=len(pt),train_saturated_D3_share=float(pp.sat3.mean()),
                 test_saturated_D3_share=float(pt.sat3.mean()),
                 poison_future_labels_check='passed',official_scale_unchanged=True,
                 limitations='Exploratory feature-residual associations, not incremental predictive value or causal effects. Multiple features screened; p-values unadjusted. Existing OOF models may have seen future films. Reconstructed D1 remains unchanged. No correction is applied to predictions.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(1,3,figsize=(15,4))
    ax[0].hist(pp.pressure3,bins=30,density=True,alpha=.5,label='train');ax[0].hist(pt.pressure3,bins=30,density=True,alpha=.5,label='test');ax[0].legend();ax[0].set(title='Bounded inverse spare capacity',xlabel='D3 pressure proxy')
    ax[1].scatter(f.saturated_share3,f.signed_miss,s=np.sqrt(f.pairs)*3,alpha=.6);ax[1].axhline(0,color='gray');ax[1].set(xlabel='Fraction of D3 pairs with occupancy >=70%',ylabel='Film weighted (actual - forecast) / scale',title='Film-level residual association')
    for i,r in corr.iterrows():ax[2].plot([r.lo,r.hi],[i,i]);ax[2].scatter(r.spearman,i)
    ax[2].axvline(0,color='gray',ls='--');ax[2].set(yticks=range(len(corr)),yticklabels=corr.feature,title='Film bootstrap correlation intervals')
    fig.tight_layout();fig.savefig(out/'capacity_preprocessing.png');plt.close(fig)
    print(corr.round(4).to_string(index=False));print(json.dumps(summary,indent=2));print('Top error films:\n',f.nlargest(5,'absolute')[['saturated_share3','pressure_p903','signed_miss','TW']].round(4).to_string())


if __name__=='__main__':main()
