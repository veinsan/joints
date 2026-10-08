"""Construct an unlabeled target-like stress split and check its coverage, without fitting models."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, style


def film_features(x):
    p=x.drop_duplicates(KEY).copy();p['base']=base_title(p.movie_title)
    p['log_pair_scale']=np.log1p(p.scale);p['log_coverage']=np.log1p(p.fnc3)
    p['coverage_trend']=np.log((p.fnc3+1)/(p.fnc1+1))
    p['late']=p.first_day.eq(3).astype(float)
    return p.groupby('base').agg(log_pair_scale=('log_pair_scale','median'),occupancy=('occ_mean','median'),
                                log_coverage=('log_coverage','median'),coverage_trend=('coverage_trend','median'),
                                late_share=('late','mean'),d1=('d1','min'),pairs=('scale','size'))


def main():
    out=ROOT/'outputs/eda/78_target_like_validation';out.mkdir(parents=True,exist_ok=True)
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    a,b=film_features(x),film_features(t)
    pd.testing.assert_frame_equal(a,film_features(x.assign(total_ticket=999999999.,date_show=pd.Timestamp('2099-01-01'))))
    cols=['log_pair_scale','occupancy','log_coverage','coverage_trend','late_share']
    center=a[cols].median();iqr=(a[cols].quantile(.75)-a[cols].quantile(.25)).replace(0,1)
    za=(a[cols]-center)/iqr;zb=(b[cols]-center)/iqr
    assert np.isfinite(za.to_numpy()).all() and np.isfinite(zb.to_numpy()).all()
    # Five equally weighted dimensions fixed before looking at any labels or model residuals.
    distance=cdist(za,zb)
    a['distance_to_test']=np.sort(distance,axis=1)[:,:5].mean(1)
    n=int(np.ceil(.4*len(a)))
    nearest=set(a.sort_values(['distance_to_test','d1']).head(n).index)
    # Repair: nearest-neighbour proximity over-selects the shared dense mode. Match marginal CDFs instead.
    features=[];desired=[]
    for col in cols:
        edges=np.unique(np.quantile(pd.concat([a[col],b[col]]),np.linspace(.05,.95,19)))
        features.append((a[col].to_numpy()[:,None]<=edges).astype(float))
        desired.append((b[col].to_numpy()[:,None]<=edges).mean(0))
    features=np.concatenate(features,axis=1);desired=np.concatenate(desired)
    selected=a.index.isin(nearest).copy();trace=[]
    for step in range(100):
        inside=np.where(selected)[0];outside_idx=np.where(~selected)[0]
        mean=features[selected].mean(0);current=float(np.mean((mean-desired)**2))
        trial=mean+(features[outside_idx][None,:,:]-features[inside][:,None,:])/n
        loss=np.mean((trial-desired)**2,axis=2);i,j=np.unravel_index(loss.argmin(),loss.shape)
        trace.append(dict(step=step,cdf_discrepancy=current))
        if loss[i,j]>=current-1e-12:break
        selected[inside[i]]=False;selected[outside_idx[j]]=True
    held=set(a.index[selected]);a['nearest_role']=np.where(a.index.isin(nearest),'validation','train')
    pd.DataFrame(trace).to_csv(out/'preprocessing_repair_trace.csv',index=False)
    a['role']=np.where(a.index.isin(held),'validation','train')
    a.to_csv(out/'target_like_film_manifest.csv')
    # This is a stress split, not historical forecasting: the training part may contain later films.
    assert a.role.eq('validation').sum()==n and not set(a.index[a.role.eq('train')])&held
    existing=x[KEY+['h']].merge(pd.read_csv(ROOT/'results/v13/oof.csv'),on=KEY+['h'],validate='one_to_one')
    quiet=set(base_title(existing.loc[existing.quiet_film,'movie_title']))
    rows=[]
    for col in cols:
        for name,mask in [('all_train',np.ones(len(a),dtype=bool)),('quiet_v13',a.index.isin(quiet)),
                          ('nearest_initial',a.index.isin(nearest)),('target_like',a.index.isin(held)),('remaining_training',~a.index.isin(held))]:
            rows.append(dict(feature=col,split=name,films=int(mask.sum()),KS_to_test=ks_2samp(a.loc[mask,col],b[col]).statistic))
    ks=pd.DataFrame(rows).pivot(index='feature',columns='split',values='KS_to_test');ks.to_csv(out/'feature_coverage.csv')
    # Support audit: target observations outside the axis-aligned train range.
    outside=(b[cols]<a[cols].min())|(b[cols]>a[cols].max())
    b.assign(outside_any=outside.any(axis=1)).to_csv(out/'test_support.csv')
    # Compare saturation of all months in the proposed stress holdout.
    counts=pd.crosstab(a.d1.dt.strftime('%Y-%m'),a.role);counts.to_csv(out/'month_counts.csv')
    summary=dict(train_films=len(a),test_films=len(b),held_out_films=n,
                 overlap_with_v13_quiet_films=len(held&quiet),quiet_films=len(quiet),
                 initial_cdf_discrepancy=trace[0]['cdf_discrepancy'],final_cdf_discrepancy=float(np.mean((features[selected].mean(0)-desired)**2)),
                 target_label_poison_check='passed',
                 test_films_outside_any_train_feature_range=int(outside.any(axis=1).sum()),
                 test_rows_from_outside_range_films=int(base_title(t.movie_title).isin(b.index[outside.any(axis=1)]).sum()),
                 limits='Split construction uses D1-D3 covariates only; no y D4-D10 or public scores. This is deliberately challenging covariate extrapolation, not a probability sample or a temporal backtest. Distances are descriptive and do not prove conditional-label similarity. Existing OOF scores cannot evaluate this new split. Fit fresh models on Kaggle.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(1,3,figsize=(15,4))
    ks[['quiet_v13','nearest_initial','target_like']].plot.bar(ax=ax[0],rot=20);ax[0].set(title='Before/after repair: smaller KS is closer')
    ax[1].scatter(a.log_pair_scale,a.occupancy,c=a.role.eq('validation'),cmap='coolwarm',label='train films',s=25)
    ax[1].scatter(b.log_pair_scale,b.occupancy,marker='x',color='gray',s=15,label='test films');ax[1].legend();ax[1].set(xlabel='Median log(1+scale)',ylabel='Median occupancy',title='Red: proposed target-like holdout')
    counts.plot.bar(ax=ax[2],rot=15);ax[2].set(title='Temporal composition of stress split')
    fig.tight_layout();fig.savefig(out/'target_like_split.png');plt.close(fig)
    print(ks.round(4).to_string());print(counts.to_string());print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
