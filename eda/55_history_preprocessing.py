"""Inspect and construct D1-D3-only preprocessing, keeping target and official scale intact.
Uses complete visible histories, including cinemas that disappear before D3.
Run: .venv/bin/python eda/55_history_preprocessing.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from common import KEY, load, style, fig_dir
from features import calendar


def transform(X, hist, cal):
    """No fitted statistics, targets or future transaction access. Ratios use log1p smoothing."""
    a = X.copy()
    starts = X[['movie_title','d1']].drop_duplicates()
    assert starts.movie_title.is_unique
    h = hist.merge(starts,on='movie_title',validate='many_to_one')
    h['day'] = (h.date_show-h.d1).dt.days+1
    h = h[h.day.between(1,3)]
    groups = {'calendar':[], 'panel':[], 'attendance':[]}
    # Preserve metric denominator: this normalizes INPUT trajectory only.
    q = np.stack([a[f'y{i}'].to_numpy()/cal.reindex(a.d1+pd.Timedelta(days=i-1)).cal.to_numpy() for i in (1,2,3)],axis=1)
    for i in (1,2,3):
        name = f'clean_p{i}'
        a[name] = q[:,i-1]/np.maximum(q.mean(axis=1),1)
        groups['calendar'].append(name)
    a['clean_log_ratio31'] = np.log1p(q[:,2])-np.log1p(q[:,0])
    groups['calendar'].append('clean_log_ratio31')
    piv = h.pivot(index=KEY,columns='day',values='total_ticket').reindex(columns=[1,2,3]).fillna(0)
    rows = []
    for title,p in piv.groupby(level=0):
        common = p.loc[p[1].gt(0)&p[3].gt(0)]
        row = {'movie_title':title,
               'panel_log31':np.log1p(common[3].sum())-np.log1p(common[1].sum()),
               'panel_share3':common[3].sum()/max(p[3].sum(),1),
               'panel_n':len(common),
               'enter_share3':p.loc[p[1].eq(0),3].sum()/max(p[3].sum(),1),
               'exit_share1':p.loc[p[3].eq(0),1].sum()/max(p[1].sum(),1)}
        ratios = np.log1p(common[3])-np.log1p(common[1])
        row['panel_median31'] = ratios.median() if len(ratios) else 0
        row['panel_iqr31'] = ratios.quantile(.75)-ratios.quantile(.25) if len(ratios) else 0
        rows.append(row)
    f = pd.DataFrame(rows).set_index('movie_title')
    groups['panel'] = f.columns.tolist()
    for c in f:
        a[c] = a.movie_title.map(f[c]).astype(float)
    a['panel_clean31'] = a.panel_log31 + np.log(cal.reindex(a.d1).cal.to_numpy()/cal.reindex(a.d1+pd.Timedelta(days=2)).cal.to_numpy())
    groups['panel'].append('panel_clean31')
    # Rounded zero occupation cannot be inverted to recover seating capacity.
    for i in (1,2,3):
        a[f'log_sh{i}'] = np.log1p(a[f'sh{i}'])
        a[f'log_tps{i}'] = np.log1p(a[f'y{i}']/a[f'sh{i}'].clip(lower=1))
        a[f'visible_occ{i}'] = a[f'occ{i}'].where(a[f'sh{i}']>0,np.nan)
        groups['attendance'] += [f'log_sh{i}',f'log_tps{i}',f'visible_occ{i}']
    a['tps_log_change31'] = a.log_tps3-a.log_tps1
    groups['attendance'].append('tps_log_change31')
    assert np.array_equal(a.scale,X.scale)
    if 'total_ticket' in X:
        assert np.array_equal(a.total_ticket,X.total_ticket)
    assert np.isfinite(a[groups['calendar']+groups['panel']].to_numpy()).all()
    return a, groups


def main():
    out = fig_dir('55_preprocessing')
    d = load(); cal = calendar(d['hol'])
    tables = {}; profiles = []
    for name, file, hist in [('train','Xtr',d['train']),('test','Xte',d['hist'])]:
        x = pd.read_parquet(ROOT/f'outputs/cache/{file}.parquet')
        a, groups = transform(x,hist,cal)
        a.to_parquet(out/f'{name}.parquet',index=False)
        tables[name] = a
        films = a.drop_duplicates('movie_title').copy()
        films['raw_log31'] = np.log1p(films.fT3)-np.log1p(films.fT1)
        films['panel_difference'] = films.panel_log31-films.raw_log31
        films[['movie_title','fnc1','fnc3','raw_log31','panel_log31','panel_difference','enter_share3','exit_share1','panel_n']].sort_values('panel_difference').to_csv(out/f'{name}_film_panel.csv',index=False)
        for c in sum(groups.values(),[]):
            profiles.append({'split':name,'feature':c,'median':a[c].median(),'q05':a[c].quantile(.05),'q95':a[c].quantile(.95),'missing':a[c].isna().mean()})
        print(name, 'films',len(films), 'abs log panel correction >0.2',int(films.panel_difference.abs().gt(.2).sum()))
        print(films.nlargest(6,'panel_difference')[['movie_title','raw_log31','panel_log31','panel_n']].to_string(index=False))
    (out/'feature_groups.json').write_text(json.dumps(groups,indent=2))
    pd.DataFrame(profiles).to_csv(out/'feature_profiles.csv',index=False)
    # Invariance check: shuffle and poison all D4+ records; constructed inputs must not change.
    x = pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    starts = x[['movie_title','d1']].drop_duplicates().set_index('movie_title').d1
    poisoned = d['train'].copy()
    future = poisoned.date_show > poisoned.movie_title.map(starts)+pd.Timedelta(days=2)
    poisoned.loc[future,'total_ticket'] = 999999999
    check,_ = transform(x,poisoned.sample(frac=1,random_state=7),cal)
    np.testing.assert_allclose(check[sum(groups.values(),[])], tables['train'][sum(groups.values(),[])],equal_nan=True)
    plt = style(); fig,axes = plt.subplots(2,3,figsize=(14,8))
    for ax,(name,a) in zip(axes[0,:2],tables.items()):
        f = a.drop_duplicates('movie_title')
        ax.scatter(np.log1p(f.fT3)-np.log1p(f.fT1),f.panel_log31,s=12,alpha=.65)
        ax.plot([-5,5],[-5,5],'k--',lw=1); ax.set(xlabel='All cinemas log ratio D3/D1',ylabel='Same cinemas log ratio',title=name)
    a = tables['train'].drop_duplicates(KEY)
    axes[0,2].hist(a.clean_p3-a.p3,bins=60); axes[0,2].set_title('Calendar preprocessing: change in D3 share')
    examples = a.assign(change=(a.clean_p3-a.p3).abs()).nlargest(3,'change')
    for ax,(_,r) in zip(axes[1],examples.iterrows()):
        ax.plot([1,2,3],[r[f'p{i}'] for i in (1,2,3)],marker='o',label='raw / scale')
        ax.plot([1,2,3],[r[f'clean_p{i}'] for i in (1,2,3)],marker='o',label='calendar-adjusted / mean')
        ax.set(title=str(r.movie_title)[:30]+'\n'+str(r.cinema_ids),xlabel='Observed day'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'before_after.png'); plt.close(fig)
    print('PASS: scale/labels unchanged; future-target poisoning and row shuffling leave new features unchanged.')


if __name__ == '__main__':
    main()
