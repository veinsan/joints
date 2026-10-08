"""Residual and preprocessing audit only: no fitting, no changed predictions or labels."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import KEY, OUTAGE, base_title, fig_dir, load, release_dates, style
from evaluate import test_weights_fd


def summarize(g, total_w, total_error):
    return dict(rows=len(g), films=g.base.nunique(), weight_share=g.w.sum()/total_w,
                TW=np.average(g.err, weights=g.w), error_share=(g.err*g.w).sum()/total_error,
                under_rate=np.average(g.y > g.pred, weights=g.w),
                signed=np.average((g.y-g.pred)/g.scale, weights=g.w),
                growth_rate=np.average(g.y > g.y3, weights=g.w))


def main():
    out = fig_dir('102_v17_error_sources')
    d = load(); raw = d['train']
    x = pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    t = pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    o = pd.read_csv(ROOT/'results/v17/oof.csv')
    x = x.merge(o, on=KEY+['h'], validate='one_to_one', suffixes=('', '_saved'))
    assert len(x)==len(o) and np.array_equal(x.total_ticket, x.y)
    assert np.allclose(x.scale, x.scale_saved)
    x['pred']=x.oof_control; x['w']=test_weights_fd(x,t)
    x['base']=base_title(x.movie_title); t['base']=base_title(t.movie_title)
    x['err']=(x.y-x.pred).abs()/x.scale
    W=x.w.sum(); E=(x.err*x.w).sum()
    x['outcome']=np.where(x.y.eq(0),'zero',np.where(x.y>x.y3,'growth','positive_not_growth'))
    x['month']=x.d1.dt.strftime('%Y-%m')
    tables={}
    for key in ['base','outcome','h','fold','month']:
        a=pd.DataFrame([dict(group=k,**summarize(g,W,E)) for k,g in x.groupby(key)])
        a=a.sort_values('error_share',ascending=False)
        a.to_csv(out/f'by_{key}.csv',index=False); tables[key]=a

    # A signed mean is not a reason to increase the forecast: inspect absolute-loss slopes too.
    directional=[]
    for factor in [.9,1.,1.1]:
        p=x.pred*factor
        directional.append(dict(factor=factor,TW=np.average(abs(x.y-p)/x.scale,weights=x.w),
                                MASE=np.mean(abs(x.y-p)/x.scale)))
    pd.DataFrame(directional).to_csv(out/'fixed_scale_diagnostic.csv',index=False)

    # Zeros followed by a later positive transaction are not permanent withdrawals.
    last=raw.groupby(KEY).date_show.max().rename('last_sale')
    x=x.join(last,on=KEY)
    x['zero_kind']=np.where(x.y>0,'positive',np.where(x.last_sale>x.date_show,'zero_then_return','zero_no_later_sale_observed'))
    pd.DataFrame([dict(group=k,**summarize(g,W,E)) for k,g in x.groupby('zero_kind')]).to_csv(out/'zero_kinds.csv',index=False)
    rawkeys=pd.MultiIndex.from_frame(raw[KEY+['date_show']])
    def activity(delta):
        a=x[KEY+['date_show']].copy(); a.date_show+=pd.Timedelta(days=delta)
        return pd.MultiIndex.from_frame(a).isin(rawkeys)
    x['one_day_hole']=x.y.eq(0)&activity(-1)&activity(1)
    x['near_outage']=x.date_show.isin([z+pd.Timedelta(days=i) for z in OUTAGE for i in [-1,1]])

    # The threshold alternatives diagnose window ambiguity, never replace authoritative dates.
    dates=pd.concat({str(f):release_dates(raw,frac=f) for f in [.25,.5,.75]},axis=1)
    dates['shift25']=(dates['0.25']-dates['0.5']).dt.days
    dates['shift75']=(dates['0.75']-dates['0.5']).dt.days
    dates['uncertain']=dates[['0.25','0.75']].isna().any(axis=1)|dates.shift25.ne(0)|dates.shift75.ne(0)
    x['d1_sensitive']=x.movie_title.map(dates.uncertain).fillna(True)
    dates.join(x.groupby('movie_title').apply(lambda z: (z.err*z.w).sum()/E,include_groups=False).rename('error_share')).to_csv(out/'d1_sensitivity.csv')
    for flag in ['one_day_hole','near_outage','d1_sensitive']:
        pd.DataFrame([dict(group=str(k),**summarize(g,W,E)) for k,g in x.groupby(flag)]).to_csv(out/f'{flag}.csv',index=False)

    # Film-age common error versus opposing cluster errors; an anatomy statistic, not an oracle gain.
    x['weighted_signed']=(x.y-x.pred)/x.scale*x.w
    x['weighted_abs']=x.err*x.w
    coherent=[]
    for keys in [['base'],['base','h'],['date_show'],['cinema_ids']]:
        agg=x.groupby(keys)[['weighted_signed','weighted_abs']].sum()
        coherent.append(dict(grouping='+'.join(keys),groups=len(agg),same_direction_error_fraction=agg.weighted_signed.abs().sum()/E))
    pd.DataFrame(coherent).to_csv(out/'error_coherence.csv',index=False)

    # Fixed, label-independent bins; require support across films and folds before considering a feature.
    specs={'p3':[0,1,2,3.01], 'sh_trend':[0,.8,1.2,2,1e9],
           'occ3':[-1,10,30,60,101], 'f_nc_trend':[0,.8,1.2,2,1e9],
           'scale':[0,20,100,500,1e12]}
    screens=[]
    for feature,edges in specs.items():
        bins=pd.cut(x[feature],edges,include_lowest=True); tb=pd.cut(t[feature],edges,include_lowest=True)
        for b in bins.cat.categories:
            mask=bins==b; g=x[mask]
            if not len(g):continue
            ef=g.groupby('base').weighted_abs.sum().sort_values(ascending=False)
            folds=g.groupby('fold').apply(lambda z: np.average(z.y>z.pred,weights=z.w),include_groups=False)
            screens.append(dict(feature=feature,bin=str(b),test_share=float((tb==b).mean()),
                largest_film_error_share=ef.iloc[0]/ef.sum(), folds_majority_under=int((folds>.5).sum()),
                **summarize(g,W,E)))
    pd.DataFrame(screens).to_csv(out/'history_signal_bins.csv',index=False)

    # Check cached day anchors against raw official test target dates, including all formats.
    target_anchor=d['test'].groupby('movie_title').date_show.min()-pd.Timedelta(days=3)
    anchor_compare=t.drop_duplicates('movie_title')[['movie_title','d1']].set_index('movie_title')
    anchor_compare['official_d1']=target_anchor
    anchor_compare['delta_days']=(anchor_compare.d1-anchor_compare.official_d1).dt.days
    anchor_compare.to_csv(out/'test_anchor_parity.csv')
    assert anchor_compare.delta_days.eq(0).all()
    assert np.allclose(t.scale,np.maximum(t[['y1','y2','y3']].sum(axis=1)/3,1))
    # Report only actual positive transactions: absent occupancy is unobserved, not measured zero.
    quality=[]
    for name,r in [('train',raw),('test_history',d['hist'])]:
        quality.append(dict(split=name,rows=len(r),positive_tickets_zero_shows=int(((r.total_ticket>0)&(r.total_show<=0)).sum()),
                            positive_tickets_zero_occ=int(((r.total_ticket>0)&(r.occupation_rate<=0)).sum()),
                            tickets_noninteger=int((r.total_ticket%1!=0).sum()),duplicate_keys=int(r.duplicated(KEY+['date_show']).sum())))
    pd.DataFrame(quality).to_csv(out/'raw_quality.csv',index=False)
    # Raw film trajectories for manual review: do not expose these as future input features.
    top=tables['base'].group.head(10).tolist()
    rr=raw.assign(base=base_title(raw.movie_title)); rr=rr[rr.base.isin(top)]
    rr.groupby(['base','date_show']).agg(tickets=('total_ticket','sum'),cinemas=('cinema_ids','nunique'),shows=('total_show','sum')).to_csv(out/'top_film_raw_trajectories.csv')
    pd.DataFrame([dict(base=b,h=h,**summarize(g,W,E)) for (b,h),g in x[x.base.isin(top)].groupby(['base','h'])]).to_csv(out/'top_film_horizon_errors.csv',index=False)
    summary=dict(MASE=float(x.err.mean()),TW=float(E/W),top5_error_share=float(tables['base'].error_share.head(5).sum()),
        top10_error_share=float(tables['base'].error_share.head(10).sum()),
        growth_error_share=float(tables['outcome'].set_index('group').loc['growth','error_share']),
        test_anchor_mismatches=0,training_performed=False,
        caveat='Residual-selected exploratory analysis. No new CV gain established. Zero-return and D1 sensitivity use future train labels for diagnosis only. Fixed scaling is a diagnostic, not an adopted correction.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(2,2,figsize=(13,8))
    z=tables['base'].head(10).iloc[::-1];ax[0,0].barh(z.group,z.error_share*100);ax[0,0].set_title('Share of weighted error (%)')
    z=tables['outcome'];ax[0,1].bar(z.group,z.error_share*100);ax[0,1].set_title('Error by outcome (diagnostic labels)')
    z=pd.DataFrame(coherent);ax[1,0].bar(z.grouping,z.same_direction_error_fraction);ax[1,0].set_title('Within-group signed error / absolute error')
    z=pd.DataFrame(directional);ax[1,1].plot(z.factor,z.TW,marker='o');ax[1,1].set(xlabel='Fixed multiplier, no fitting',ylabel='TW-MASE',title='Does raising every forecast actually help?')
    fig.tight_layout();fig.savefig(out/'error_sources.png');plt.close(fig)
    print(json.dumps(summary,indent=2));print(tables['base'].head(10).to_string(index=False))
    print(pd.DataFrame(directional).to_string(index=False));print(pd.DataFrame(coherent).to_string(index=False))


if __name__=='__main__':main()
