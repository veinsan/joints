"""Separate future growth mechanisms and build history-only breadth features. No model fit."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, fig_dir, load, style
from evaluate import test_weights_fd
from features import calendar


def history_features(x,raw,cal):
    starts=x[['movie_title','d1']].drop_duplicates()
    assert starts.movie_title.is_unique
    a=raw.merge(starts,on='movie_title',validate='many_to_one')
    a['day']=(a.date_show-a.d1).dt.days+1
    a=a[a.day.between(1,3)].copy()
    a['base']=base_title(a.movie_title)
    # Merge formats before comparing the same cinema; omitted transactions remain missing for demand.
    a=a.groupby(['base','cinema_ids','day']).agg(tickets=('total_ticket','sum'),shows=('total_show','sum'),date=('date_show','min')).reset_index()
    a['demand']=a.tickets/a.shows.clip(lower=1)/cal.cal.reindex(a.date).to_numpy()
    p=a.pivot(index=['base','cinema_ids'],columns='day',values='demand').reindex(columns=[1,2,3])
    sh=a.pivot(index=['base','cinema_ids'],columns='day',values='shows').reindex(columns=[1,2,3])
    rows=[]
    for b,g in p.groupby(level=0):
        z=g.dropna(); s=sh.loc[z.index]
        d12=np.log1p(z[2])-np.log1p(z[1]);d23=np.log1p(z[3])-np.log1p(z[2])
        rows.append(dict(base=b,panel_n=len(z),panel_coverage=len(z)/len(g),
            demand_breadth23=float((d23>0).mean()) if len(z) else np.nan,
            demand_sustained=float(((d12>0)&(d23>0)).mean()) if len(z) else np.nan,
            demand_acceleration=float((d23-d12).median()),
            demand_log23=float(d23.median()),
            supply_breadth23=float((s[3]>s[2]).mean()) if len(z) else np.nan))
    return pd.DataFrame(rows).set_index('base')


def main():
    out=fig_dir('103_growth_supply_demand');d=load();cal=calendar(d['hol'])
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    o=pd.read_csv(ROOT/'results/v17/oof.csv')
    x=x.merge(o,on=KEY+['h'],validate='one_to_one',suffixes=('','_saved'))
    assert len(x)==len(o) and np.allclose(x.total_ticket,x.y) and np.allclose(x.scale,x.scale_saved)
    x['base']=base_title(x.movie_title);x['w']=test_weights_fd(x,t)
    x['e']=abs(x.y-x.oof_control)/x.scale;x['we']=x.e*x.w
    E=x.we.sum();W=x.w.sum()
    # Future show/attendance data are diagnostics only, never exported as model inputs.
    x=x.merge(d['train'][KEY+['date_show','total_show','occupation_rate']],on=KEY+['date_show'],how='left',validate='one_to_one')
    x['supply_ratio']=x.total_show/x.sh3
    x['demand_ratio']=(x.y/x.total_show)/(x.y3/x.sh3)
    pos=x.y>0
    np.testing.assert_allclose((x.supply_ratio*x.demand_ratio)[pos],(x.y/x.y3)[pos])
    growth=x.y>x.y3
    x['mechanism']=np.select([~growth,growth&(x.supply_ratio<=1),growth&(x.demand_ratio<=1)],
        ['not_growth','demand_up_without_more_shows','shows_up_without_more_demand'],default='both_shows_and_demand_up')
    rows=[]
    for (kind,h),g in x.groupby(['mechanism','h']):
        ef=g.groupby('base').we.sum()
        rows.append(dict(mechanism=kind,h=h,rows=len(g),films=g.base.nunique(),weight_share=g.w.sum()/W,
            error_share=g.we.sum()/E,largest_film_error_share=ef.max()/ef.sum(),
            median_supply_ratio=g.supply_ratio.median(),median_demand_ratio=g.demand_ratio.median()))
    mechanisms=pd.DataFrame(rows);mechanisms.to_csv(out/'mechanisms_by_horizon.csv',index=False)
    mechanism=mechanisms.groupby('mechanism')[['rows','weight_share','error_share']].sum()
    mechanism.to_csv(out/'mechanisms.csv')
    train=history_features(x,d['train'],cal);test=history_features(t,d['hist'],cal)
    poisoned=d['train'].copy();starts=x.set_index('movie_title').d1.to_dict()
    future=poisoned.date_show>poisoned.movie_title.map(starts)+pd.Timedelta(days=2)
    poisoned.loc[future,['total_ticket','total_show','occupation_rate']]=987654
    pd.testing.assert_frame_equal(train,history_features(x,poisoned.sample(frac=1,random_state=2026),cal))
    train.to_csv(out/'history_features_train.csv');test.to_csv(out/'history_features_test.csv')
    f=x.groupby('base').agg(error=('we','sum'),w=('w','sum'),d1=('d1','min'),fold=('fold','first'))
    f['under_rate']=x.groupby('base').apply(lambda g: np.average(g.y>g.oof_control,weights=g.w),include_groups=False)
    f['growth_rate']=x.groupby('base').apply(lambda g: np.average(g.y>g.y3,weights=g.w),include_groups=False)
    f['signed']=x.groupby('base').apply(lambda g: np.average((g.y-g.oof_control)/g.scale,weights=g.w),include_groups=False)
    f=f.join(train);f['error_share']=f.error/E;f['month']=f.d1.dt.strftime('%Y-%m')
    f.sort_values('error',ascending=False).to_csv(out/'film_diagnostics.csv')
    associations=[]
    for col in train.columns:
        for name,g in [('all',f),('exclude_top5',f.drop(f.nlargest(5,'error').index))]+list(f.groupby('month')):
            a=g[[col,'under_rate','growth_rate']].dropna()
            associations.append(dict(feature=col,subset=name,films=len(a),rho_under=a[col].rank().corr(a.under_rate.rank()),
                rho_growth=a[col].rank().corr(a.growth_rate.rank())))
    pd.DataFrame(associations).to_csv(out/'feature_associations.csv',index=False)
    # Support counts are films, not thousands of correlated cinema-day rows.
    bins=[]
    for name,g in [('train',train),('test',test)]:
        for col in ['demand_breadth23','demand_sustained','panel_coverage']:
            for label,v in g.groupby(pd.cut(g[col],[-.001,.25,.5,.75,1.001]),observed=True):
                bins.append(dict(split=name,feature=col,bin=str(label),films=len(v)))
    pd.DataFrame(bins).to_csv(out/'feature_support.csv',index=False)
    summary=dict(future_poison_check='passed',real_training=False,train_films=len(train),test_films=len(test),
        train_low_panel_films=int((train.panel_n<10).sum()),test_low_panel_films=int((test.panel_n<10).sum()),
        note='Unweighted film-level rank correlations are exploratory, not conditional incremental signal or CV gains. DOW normalization reuses existing fixed calendar. Do not put future supply/demand ratios in model inputs.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(1,2,figsize=(13,4))
    mechanism.error_share.plot.barh(ax=ax[0]);ax[0].set_title('Fraction of total weighted error by mechanism')
    ax[1].scatter(f.demand_breadth23,f.under_rate,alpha=.6,s=20)
    ax[1].set(xlabel='D1-D3 panel: fraction with rising adjusted tickets/show D2→D3',ylabel='Fraction of underpredicted target rows',title='One point per film; association only')
    fig.tight_layout();fig.savefig(out/'growth_mechanisms.png');plt.close(fig)
    print(mechanism.to_string());print(pd.DataFrame(associations).query('subset in ["all","exclude_top5"]').round(4).to_string(index=False))
    print(f.nlargest(8,'error')[['error_share','under_rate','demand_breadth23','demand_sustained','demand_log23','panel_n']].round(4).to_string())
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
