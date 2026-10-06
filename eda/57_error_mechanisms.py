"""Post-hoc error attribution, observed scheduling vs attendance and film-level correlations.
Future shows/attendance are DIAGNOSTICS ONLY, never exported as inference features.
Run: .venv/bin/python eda/57_error_mechanisms.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, load, fig_dir, style
from evaluate import test_weights_fd


def main():
    out=fig_dir('57_mechanisms'); d=load()
    X=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/train.parquet')
    T=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/test.parquet')
    o=pd.read_csv(ROOT/'results/v8/oof.csv')
    col='oof_final' if 'oof_final' in o else 'pred'
    X=X.merge(o[KEY+['h',col]],on=KEY+['h'],validate='one_to_one').rename(columns={col:'pred'})
    raw=d['train'][KEY+['date_show','total_show']]
    X=X.merge(raw,on=KEY+['date_show'],how='left',validate='one_to_one')
    X['base']=base_title(X.movie_title); X['weight']=test_weights_fd(X,T)
    X['error']=(X.total_ticket-X.pred).abs()/X.scale
    X['signed_miss']=(X.total_ticket-X.pred)/X.scale
    X['weighted_error']=X.error*X.weight
    positive=X.total_ticket.gt(0)
    P=X[positive].copy()
    assert P.total_show.notna().all() and P.sh3.gt(0).all() and P.y3.gt(0).all()
    P['schedule_log_change']=np.log(P.total_show/P.sh3)
    P['attendance_log_change']=np.log((P.total_ticket/P.total_show)/(P.y3/P.sh3))
    P['ticket_log_change']=np.log(P.total_ticket/P.y3)
    np.testing.assert_allclose(P.schedule_log_change+P.attendance_log_change,P.ticket_log_change,atol=1e-12)
    # Exact multiplicative decomposition; not a causal attribution of lost demand.
    P['mechanism']=np.where(P.schedule_log_change.abs()>P.attendance_log_change.abs(),'schedule larger','attendance larger')
    P['direction']=np.where(P.ticket_log_change>0,'growth','decay')
    mech=P.groupby(['direction','mechanism']).agg(rows=('h','size'),weighted_error=('weighted_error','sum'),mean_log_show=('schedule_log_change','mean'),mean_log_attendance=('attendance_log_change','mean'))
    mech['share_all_error']=mech.weighted_error/X.weighted_error.sum()
    mech.to_csv(out/'positive_mechanisms.csv'); print(mech.to_string())
    # Correlation at independent base-film level; use descriptive correlations, not row-level p-values.
    f=X.groupby('base').agg(error=('error','mean'),signed_miss=('signed_miss','mean'),
                            panel_change=('panel_clean31','median'),exit_share=('exit_share1','median'),
                            occupancy=('occ3','median'),scale=('scale','median'),
                            early_share=('p3','median'),first_day=('first_day','mean'))
    changes=P.groupby('base')[['schedule_log_change','attendance_log_change']].median()
    f=f.join(changes); f.to_csv(out/'film_diagnostics.csv')
    corr=f.corr(method='spearman'); corr.to_csv(out/'film_spearman.csv')
    print('Film-level residual correlations (descriptive; future realized changes are diagnostic):')
    print(corr.signed_miss.sort_values().to_string())
    # Inspect all timelines for the largest error films, including late starts and horizons.
    ranked=X.groupby('base').weighted_error.sum().sort_values(ascending=False)
    selected=ranked.head(5).index
    curves=X[X.base.isin(selected)].groupby(['base','h']).apply(
        lambda a:pd.Series({'actual_r':np.average(a.total_ticket/a.scale,weights=a.weight),
                           'pred_r':np.average(a.pred/a.scale,weights=a.weight),
                           'positive_share':a.total_ticket.gt(0).mean(),
                           'rows':len(a)}),include_groups=False)
    curves.to_csv(out/'top_film_horizons.csv')
    # Film residual correlation across models checks whether blend can remove the main misses.
    p10=pd.read_csv(ROOT/'results/v10/oof.csv')
    z=X.merge(p10[KEY+['h','oof_final']],on=KEY+['h'],validate='one_to_one')
    f10=z.assign(miss10=(z.total_ticket-z.oof_final)/z.scale).groupby('base').miss10.mean()
    rho=spearmanr(f.signed_miss,f10.reindex(f.index)).statistic
    pd.Series({'film_signed_miss_spearman_v8_v10':rho,
               'positive_error_share':X.loc[positive,'weighted_error'].sum()/X.weighted_error.sum()}).to_csv(out/'summary.csv',header=['value'])
    print('v8/v10 film signed-miss Spearman',rho)
    plt=style(); fig,ax=plt.subplots(2,3,figsize=(15,9))
    for axis,title in zip(ax.flat,selected):
        c=curves.loc[title]
        axis.plot(c.index,c.actual_r,label='actual',marker='o'); axis.plot(c.index,c.pred_r,label='v8 OOF',marker='o')
        axis.set(title=title[:38],xlabel='Forecast day',ylabel='Weighted mean y / official scale'); axis.legend()
    im=ax[1,2].imshow(corr,vmin=-1,vmax=1,cmap='RdBu_r')
    ax[1,2].set_xticks(range(len(corr)),corr.columns,rotation=90,fontsize=6)
    ax[1,2].set_yticks(range(len(corr)),corr.columns,fontsize=6)
    ax[1,2].set_title('Film-level Spearman; diagnostic ≠ predictor')
    fig.colorbar(im,ax=ax[1,2],shrink=.65); fig.tight_layout(); fig.savefig(out/'mechanisms.png'); plt.close(fig)


if __name__=='__main__':
    main()
