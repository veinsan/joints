"""Audit dependence and produce purged chronological split manifests, without refitting models."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import KEY, base_title, load, style
from evaluate import test_weights_fd


def main():
    out = ROOT/'outputs/eda/75_validation_geometry'
    out.mkdir(parents=True, exist_ok=True)
    x = pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    t = pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    o = x[KEY+['h']].merge(pd.read_csv(ROOT/'results/v13/oof.csv'), on=KEY+['h'], validate='one_to_one')
    assert np.allclose(o.y,x.total_ticket) and np.allclose(o.scale,x.scale)
    x = x.assign(base=base_title(x.movie_title), fold=o.fold, cohort=x.d1.dt.to_period('W-TUE').astype(str))
    w = test_weights_fd(x,t)
    y,s = x.total_ticket.to_numpy(), x.scale.to_numpy()
    pred = {}
    for v in [8,10,12,13]:
        q=x[KEY+['h']].merge(pd.read_csv(ROOT/f'results/v{v}/oof.csv'),on=KEY+['h'],validate='one_to_one')
        assert np.allclose(q.y,y) and np.allclose(q.scale,s)
        pred[v]=q.oof_final.to_numpy()
    films=x.groupby('base').agg(d1=('d1','min'), last_label=('date_show','max'), cohort=('cohort','first'),
                                 n_cohorts=('cohort','nunique'), fold=('fold','first'), n_folds=('fold','nunique'), rows=('h','size'))
    assert films.n_folds.eq(1).all() and films.n_cohorts.eq(1).all()
    co=films.groupby('cohort').agg(films=('fold','size'),folds=('fold','nunique'))
    co.to_csv(out/'cohorts.csv')
    # Does the same release environment occur in another training fold?
    shared=x.cohort.map(co.folds).gt(1)
    bootrows=[]
    rng=np.random.default_rng(2026)
    for unit in ['base','cohort']:
        for ref in [8,12]:
            de=(np.abs(y-pred[13])-np.abs(y-pred[ref]))/s
            g=pd.DataFrame({'unit':x[unit],'de':de*w,'w':w}).groupby('unit').sum()
            ix=rng.integers(0,len(g),(10000,len(g)))
            b=g.de.to_numpy()[ix].sum(1)/g.w.to_numpy()[ix].sum(1)
            bootrows.append(dict(unit=unit,reference=ref,groups=len(g),delta=np.average(de,weights=w),
                                 lo=np.quantile(b,.025),hi=np.quantile(b,.975),fraction_delta_below_zero=np.mean(b<0)))
    boots=pd.DataFrame(bootrows);boots.to_csv(out/'paired_uncertainty.csv',index=False)
    # Actual new splits. Scores require new fitting: old grouped OOF cannot be relabelled temporal CV.
    splitrows=[]; counts=[]
    for month in ['2025-06','2025-07','2025-08','2025-09']:
        cut=pd.Timestamp(month+'-01');end=cut+pd.offsets.MonthBegin(1)
        # Full D1-D10 completion, even when an outage target row was removed.
        train=(films.d1+pd.Timedelta(days=9)<cut)
        val=films.d1.ge(cut)&films.d1.lt(end)
        assert not (train&val).any()
        assert (films.loc[train,'d1']+pd.Timedelta(days=9)<cut).all()
        assert films.loc[val,'d1'].min()>=cut
        for role,mask in [('train',train),('validation',val)]:
            for base in films.index[mask]:splitrows.append(dict(split=month,base=base,role=role,cutoff=str(cut.date())))
        counts.append(dict(split=month,train_films=int(train.sum()),validation_films=int(val.sum()),
                           train_rows=int(films.rows[train].sum()),validation_rows=int(films.rows[val].sum())))
    pd.DataFrame(splitrows).to_csv(out/'purged_monthly_manifest.csv',index=False)
    counts=pd.DataFrame(counts);counts.to_csv(out/'split_counts.csv',index=False)
    # Bootstrap uncertainty in rank, using release-week blocks and the fixed predictions only.
    errors=pd.DataFrame({f'v{v}':np.abs(y-p)/s*w for v,p in pred.items()}).assign(cohort=x.cohort)
    errors=errors.groupby('cohort').sum(); wg=pd.Series(w).groupby(x.cohort).sum().reindex(errors.index)
    ix=rng.integers(0,len(errors),(10000,len(errors)))
    losses=errors.to_numpy()[ix].sum(1)/wg.to_numpy()[ix].sum(1)[:,None]
    rank=pd.DataFrame({'version':errors.columns,'bootstrap_best_share':np.bincount(losses.argmin(1),minlength=4)/len(ix)})
    rank.to_csv(out/'rank_stability.csv',index=False)
    raw=load()
    sig=['y1','y2','y3','sh1','sh2','sh3']
    duplicates={}
    for name,a in [('train',x),('test',t)]:
        pp=a.drop_duplicates(KEY).assign(base=base_title(a.drop_duplicates(KEY).movie_title))
        ng=pp.groupby(sig,dropna=False).base.transform('nunique')
        duplicates[name]=dict(pair_key_duplicates=int(a.duplicated(KEY+['h']).sum()),
                              pairs_sharing_signature_across_films=int(ng.gt(1).sum()))
    summary=dict(base_films=len(films),release_weeks=len(co),weeks_split_across_folds=int(co.folds.gt(1).sum()),
                 validation_rows_with_cohort_in_training=float(shared.mean()),
                 raw_duplicate_keys={n:int(raw[n].duplicated(KEY+['date_show']).sum()) for n in ['train','hist']},
                 duplicate_audit=duplicates,
                 limitations='Shared release week is dependence, not proof of prohibited label leakage. Week bootstrap assumes independent weeks and does not fix prior selection. Manifests are new split definitions, NOT newly evaluated CV. Limited-release extras must also finish D10 before cutoff; all preprocessing and statistics must obey each fold.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(1,3,figsize=(15,4))
    co[['films','folds']].plot.bar(ax=ax[0],legend=True);ax[0].set(title='Release cohorts spread across existing folds',xticklabels=[])
    for i,r in boots.iterrows():
        ax[1].plot([r.lo,r.hi],[i,i]);ax[1].scatter(r.delta,i)
    ax[1].axvline(0,color='gray',ls='--');ax[1].set(yticks=range(len(boots)),yticklabels=[f'{r.unit}: v13-v{r.reference}' for r in boots.itertuples()],title='Fixed OOF paired uncertainty; not new CV')
    counts.set_index('split')[['train_films','validation_films']].plot.bar(ax=ax[2],rot=15);ax[2].set(title='Purged expanding-window manifests')
    fig.tight_layout();fig.savefig(out/'validation_geometry.png');plt.close(fig)
    print(json.dumps(summary,indent=2));print(boots.round(5).to_string(index=False));print(counts.to_string(index=False));print(rank.to_string(index=False))


if __name__=='__main__': main()
