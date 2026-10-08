"""Quantile reliability and target geometry; saved OOF only, no training or submission."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import KEY, fig_dir, load, style
from evaluate import test_weights_fd
from features import calendar, DOW_PROF, HOL_LEVEL, SCHOOL_WEEKDAY


def quant_median(Q, lam, qs):
    p0 = np.array([np.interp(.02, q, qs, left=0., right=1.) for q in Q])
    z = np.clip(p0, 1e-4, 1-1e-4)
    p = 1 / (1 + np.exp(-(np.log(z/(1-z)) - lam*1.32)))
    u = np.clip(p0+(1-p0)*(.5-p)/(1-p), qs[0], qs[-1])
    val = np.array([np.interp(a, qs, q) for a,q in zip(u,Q)])
    return np.where(p >= .5, 0., np.maximum(val,0))


def main():
    out = fig_dir('97_v15_quantile_diagnostics')
    x = pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    t = pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    o = pd.read_csv(ROOT/'results/v15/oof.csv')
    # Quantile archive has the notebook/CSV order, never assume the cache order.
    x = o[KEY+['h']].merge(x,on=KEY+['h'],validate='one_to_one')
    assert np.array_equal(x.total_ticket,o.y) and np.allclose(x.scale,o.scale)
    cal = calendar(load()['hol'])
    jb = cal.copy()
    jb['school'] = (jb.index.to_series().between('2025-06-30','2025-07-12') |
                    jb.index.to_series().between('2025-12-29','2026-01-10')).astype(int)
    b = DOW_PROF[jb.dow]
    b = np.where((jb.school==1)&(jb.dow<4),b*SCHOOL_WEEKDAY,b)
    jb['cal'] = np.where((jb.is_hol==1)&(jb.ramadan==0),np.maximum(b,HOL_LEVEL),b)
    mask = x.city_name.isin(['BOGOR','BEKASI','BANDUNG','DEPOK','CIKARANG','KARAWANG','CIREBON',
                           'GARUT','TASIKMALAYA','SUMEDANG','CIANJUR','INDRAMAYU'])
    at = lambda dates: np.where(mask,jb.cal.reindex(dates),cal.cal.reindex(dates))
    cm = at(x.date_show)/np.stack([at(x.d1+pd.Timedelta(days=i)) for i in range(3)],axis=1).mean(1)
    qs = np.round(np.arange(.02,1,.02),2)
    Q = np.load(ROOT/'results/v15/oof_quantiles.npz')['tp35_int7_Q']
    p = quant_median(Q,.5,qs)*cm*x.scale.to_numpy()
    assert np.allclose(p,o.comp_tp35_int7,atol=1e-6), 'Calendar/quantile reproduction differs from v15'
    y,s,w = o.y.to_numpy(),o.scale.to_numpy(),test_weights_fd(x,t)
    r = y/s/cm
    rows = []
    for h in range(4,11):
        a = x.h.eq(h).to_numpy()
        for name,z in [('raw',r[a]),('log1p',np.log1p(r[a]))]:
            zs = (z-z.mean())/z.std()
            rows.append(dict(h=h,transform=name,skew=pd.Series(z).skew(),max_sd=zs.max(),
                             iqr_sd=np.diff(np.quantile(zs,[.25,.75]))[0]))
    geom = pd.DataFrame(rows)
    geom.to_csv(out/'target_geometry.csv',index=False)
    # Population quantiles commute with monotone transforms; interpolated empirical quantiles need not.
    assert np.allclose(np.expm1(np.quantile(np.log1p(r),qs,method='inverted_cdf')),
                       np.quantile(r,qs,method='inverted_cdf'))
    reliability = []
    for j in (4,12,24,36,44):
        pred = Q[:,j]*cm*s
        reliability.append(dict(quantile=qs[j],coverage=np.mean(y<=pred),
                                weighted_coverage=np.average(y<=pred,weights=w),
                                TW_MASE=np.average(np.abs(y-pred)/s,weights=w)))
    pd.DataFrame(reliability).to_csv(out/'quantile_coverage.csv',index=False)
    # A short fixed sensitivity grid, not a chosen production calibration.
    sensitivity = []
    for lam in (0.,.25,.5,.75,1.):
        pp = .1*o.comp_lgb.to_numpy()+.9*quant_median(Q,lam,qs)*cm*s
        e = np.abs(y-pp)/s
        row = dict(lam=lam,TW_MASE=np.average(e,weights=w))
        for k in range(5):
            a=o.fold.eq(k).to_numpy()
            row[f'fold{k}']=np.average(e[a],weights=w[a])
        sensitivity.append(row)
    ss = pd.DataFrame(sensitivity)
    ss.to_csv(out/'lambda_sensitivity.csv',index=False)
    summary = {'reconstruction_max_abs_error':float(np.max(np.abs(p-o.comp_tp35_int7))),
               'fraction_zero_targets':float(np.mean(y==0)),
               'target_max':float(r.max()),
               'geometry_average_by_horizon':geom.groupby('transform')[['skew','max_sd','iqr_sd']].mean().to_dict(),
               'warning':'Sensitivity uses the same OOF labels already used in v15 selection; not a new validation gain.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2)); print(ss.round(5).to_string(index=False))
    print(pd.DataFrame(reliability).round(5).to_string(index=False))
    plt=style(); fig,ax=plt.subplots(1,2,figsize=(11,4))
    ax[0].hist(r,bins=np.linspace(0,5,80),alpha=.8)
    ax[0].set(title='Normalized target (display limited to 0–5)')
    ax[1].plot(ss.lam,ss.TW_MASE,marker='o')
    ax[1].set(title='Saved-OOF sensitivity, not held-out selection',xlabel='Pull correction lambda',ylabel='TW-MASE')
    fig.tight_layout(); fig.savefig(out/'quantiles.png'); plt.close(fig)


if __name__=='__main__':
    main()
