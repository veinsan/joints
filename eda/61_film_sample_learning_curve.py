"""Data-volume check: add independent labelled films, not duplicate rows.
Fixed September evaluation, only D10-complete earlier films; three random film subsets.
This is a local LGB diagnostic, not a leaderboard forecast or proof of irreducible error.
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import base_title, load, fig_dir, style
from evaluate import test_weights_fd
from model import fit_predict


def main():
    spec=importlib.util.spec_from_file_location('ab',Path(__file__).with_name('56_preprocessing_ablation.py'))
    ab=importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
    out=fig_dir('61_film_learning')
    X=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/train.parquet')
    T=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/test.parquet')
    cut=pd.Timestamp('2025-09-01'); base=base_title(X.movie_title)
    available=X.d1+pd.Timedelta(days=9)<cut
    va=X.d1.dt.month.eq(9); films=np.array(sorted(base[available].unique()))
    raw=load()['train']; raw=raw[raw.date_show<cut]; rb=base_title(raw.movie_title)
    w=test_weights_fd(X,T)[va]
    rows=[]
    for seed in (91,92,93):
        order=np.random.default_rng(seed).permutation(films)
        for fraction in (.25,.5,.75,1.):
            if fraction==1 and seed!=91:
                continue
            chosen=order[:max(1,int(len(order)*fraction))]
            tr=available&base.isin(chosen)
            assert not set(base[tr])&set(base[va])
            a,b=ab.local_stats(X[tr],X[va],raw[rb.isin(chosen)])
            p,_=fit_predict(a,b,ab.BASE,n_estimators=600,params={'n_jobs':4})
            e=np.abs(b.total_ticket-p)/b.scale
            row={'seed':seed,'fraction':fraction,'films':len(chosen),'train_rows':len(a),
                 'mase':e.mean(),'tw':np.average(e,weights=w)}
            rows.append(row); print(row,flush=True)
            pd.DataFrame(rows).to_csv(out/'scores.csv',index=False)
    S=pd.DataFrame(rows); G=S.groupby('films')[['mase','tw']].agg(['mean','min','max'])
    G.to_csv(out/'summary.csv'); print(G.to_string())
    plt=style(); fig,ax=plt.subplots(figsize=(8,4))
    for col in ('mase','tw'):
        ax.plot(G.index,G[col]['mean'],marker='o',label=col)
        ax.fill_between(G.index,G[col]['min'],G[col]['max'],alpha=.15)
    ax.set(xlabel='Independent training films with complete labels',ylabel='September validation loss',
           title='Subset range is composition sensitivity, not a confidence interval')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'film_learning.png'); plt.close(fig)


if __name__=='__main__':
    main()
