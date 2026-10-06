"""Negative control: demonstrate why exact deduplication cannot replace film grouping.
Same evaluation rows and training row count; intentionally leaky controls are NOT scores to optimize.
Run: .venv/bin/python eda/58_split_leakage_control.py
"""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, load, fig_dir, style
from evaluate import folds, test_weights_fd
from model import fit_predict


def main():
    spec=importlib.util.spec_from_file_location('ablation',Path(__file__).with_name('56_preprocessing_ablation.py'))
    ab=importlib.util.module_from_spec(spec); spec.loader.exec_module(ab)
    out=fig_dir('58_split_control')
    X=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/train.parquet')
    T=pd.read_parquet(ROOT/'outputs/eda/55_preprocessing/test.parquet')
    f=folds(X); b=base_title(X.movie_title); raw=load()['train']
    rng=np.random.default_rng(812)
    pairs=X[KEY].drop_duplicates().sample(frac=.3,random_state=812)
    chosen=pd.MultiIndex.from_frame(X[KEY]).isin(pd.MultiIndex.from_frame(pairs))
    va=(f==0)&chosen&X.h.isin([4,7,10]).to_numpy()
    evalpairs=pd.MultiIndex.from_frame(X.loc[va,KEY].drop_duplicates())
    clean=np.flatnonzero(f!=0)
    candidates={'film_disjoint':clean,
                'LEAK_other_cinemas_same_film':np.flatnonzero(~pd.MultiIndex.from_frame(X[KEY]).isin(evalpairs)),
                'LEAK_other_horizons_same_pair':np.flatnonzero(~va)}
    # All controls use identical cinema statistics, fitted without evaluation films.
    rr=raw[~base_title(raw.movie_title).isin(b[va].unique())]
    records=[]
    weights=test_weights_fd(X,T)
    for name,cand in candidates.items():
        tr=clean if name=='film_disjoint' else rng.choice(cand,len(clean),replace=False)
        assert not set(tr)&set(np.flatnonzero(va))
        if name=='film_disjoint':
            assert not set(b.iloc[tr])&set(b[va])
        a,v=ab.local_stats(X.iloc[tr],X[va],rr)
        pred,_=fit_predict(a,v,ab.BASE,n_estimators=600,params={'n_jobs':4})
        e=np.abs(v.total_ticket.to_numpy()-pred)/v.scale.to_numpy()
        row={'control':name,'train_rows':len(a),'validation_rows':len(v),
             'validation_films_seen_in_training':len(set(base_title(a.movie_title))&set(base_title(v.movie_title))),
             'mase':e.mean(),'tw':np.average(e,weights=weights[va])}
        records.append(row); print(row,flush=True)
        pd.DataFrame(records).to_csv(out/'scores.csv',index=False)
    scores=pd.DataFrame(records).set_index('control')
    plt=style(); fig,ax=plt.subplots(figsize=(9,4))
    scores[['mase','tw']].plot.barh(ax=ax,title='Negative controls on fixed rows: lower can be leakage')
    fig.tight_layout(); fig.savefig(out/'split_control.png'); plt.close(fig)


if __name__=='__main__':
    main()
