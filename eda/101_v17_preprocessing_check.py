"""Exercise generated v17 changes with actual data and fake inference; no real model training."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title
from features import GENRES


def main():
    nb=json.loads((ROOT/'notebooks/v17.ipynb').read_text())
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    tree=ast.parse('\n'.join(line for c in cells for line in c.splitlines() if not line.startswith(('!','%'))))
    wanted={'complete_genres','train_part','make_tabpfn','r_target','lens','blend'}
    funcs=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in wanted],type_ignores=[])
    out=Path('/tmp/joints-v17-check');out.mkdir(exist_ok=True)
    cfg=SimpleNamespace(USE_LIMITED=True,N_FOLDS=5,POOL_CONTEXT=70,SEED=2026,DEVICE='fake',
                        ABL_MIN_GAIN=.0015,ABL_MONTH_TOL=.003,ABL_MIN_FOLDS=3,
                        SIZE_MARGIN_MB=1.5,MAX_WEIGHTS_MB=200,OUTPUT_DIR=out,FIG_DIR=out)
    ns=dict(np=np,pd=pd,base_title=base_title,CFG=cfg,KEY=KEY)
    exec(compile(funcs,'v17_functions','exec'),ns)
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    l=pd.read_parquet(ROOT/'outputs/cache/Xlim.parquet')
    week=((x.d1-pd.Timestamp('2025-03-31')).dt.days//7).to_numpy()
    fw={w:k for k,chunk in enumerate(np.array_split(np.unique(week),5)) for w in chunk}
    fold=np.array([fw[w] for w in week])
    ns.update(Xtr=x,Xlim=l,WEEK=week,FOLD=fold,groups=base_title(x.movie_title).to_numpy())
    expected=pd.read_csv(ROOT/'outputs/eda/100_purged_blocks/counts.csv').query('kind=="purged_block"')
    for k in range(5):
        a=ns['train_part'](k)
        assert len(a)==expected.iloc[k].train_rows
        lo=x.loc[fold==k,'d1'].min();hi=x.loc[fold==k,'d1'].max()+pd.Timedelta(days=9)
        assert ((a.d1+pd.Timedelta(days=9)<lo)|(a.d1>hi)).all()
    meta=pd.read_csv(ROOT/'data/movies.csv')
    meta.original_title=meta.original_title.str.strip();meta=meta.set_index('original_title')
    tokens=meta.genre.fillna('').str.split(',').map(lambda z:{v.strip() for v in z if v.strip()})
    extra=sorted(set().union(*tokens)-set(GENRES))
    ef=[f'genre_extra_{i}' for i in range(len(extra))]+['metadata_missing']
    ns.update(META=meta,GENRE_TOKENS=tokens,EXTRA_GENRES=extra,EXTRA_GENRE_FEATS=ef)
    processed=ns['complete_genres'](x)
    poisoned=x.copy();poisoned.total_ticket=999999;poisoned.scale=999
    check=ns['complete_genres'](poisoned)
    assert np.array_equal(processed[ef],check[ef])
    assert np.array_equal(processed.total_ticket,x.total_ticket) and np.array_equal(processed.scale,x.scale)
    lilo=processed.movie_title.str.startswith('LILO & STITCH')
    assert processed.loc[lilo,f'genre_extra_{extra.index("Adventure")}'].eq(1).all()
    assert processed.loc[lilo,f'genre_extra_{extra.index("Fantasy")}'].eq(1).all()
    # Unknown metadata is explicit, distinct from a known title with no extra genres.
    unknown=x.iloc[:1].copy();unknown.movie_title='UNMATCHED TEST TITLE'
    assert ns['complete_genres'](unknown).metadata_missing.iloc[0]==1

    calls=[]
    class FakeRegressor:
        @staticmethod
        def create_default_for_version(*args,**kwargs):
            return FakeRegressor()
        def fit(self,X,y):
            calls.append((X.copy(),y.copy()))
        def predict(self,X,output_type,quantiles):
            return np.tile(np.asarray(quantiles)[:,None],(1,len(X)))
    ns.update(TabPFNRegressor=FakeRegressor,FEATS=['p1','h'],TAB_QS=np.array([.25,.5,.75]),
              torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    a=pd.DataFrame({'movie_title':np.repeat(['LILO & STITCH'],140),
        'cinema_ids':np.tile([f'C{i:02}' for i in range(20)],7),'h':np.repeat(range(4,11),20),
        'p1':np.tile(np.arange(20),7),'scale':1.,'cal_mult':1.,'total_ticket':2.})
    a=ns['complete_genres'](a);b=a.iloc[::5].copy()
    for variant,nfits,ncols in [('tp35_int7',7,1),('tp35_genre',7,1+len(ef)),('tp35_pool',1,2)]:
        calls.clear();f=ns['make_tabpfn'](None,None,1,variant)
        pred=f(a,b)
        assert pred.shape==(len(b),3) and np.isfinite(pred).all()
        assert len(calls)==nfits and all(c[0].shape[1]==ncols for c in calls)
        if variant=='tp35_pool':
            original=calls[0][0].copy()
            assert original.shape[0]==70 and len(np.unique(original[:,1]))==7
            calls.clear();f(a.sample(frac=1,random_state=91),b)
            assert np.array_equal(original,calls[0][0]),'Sampling must not depend on input row order'
    # Exercise the actual selection cell with a deliberately good then bad pair of candidates.
    decision=next(c for c in cells if c.startswith('CONTROL_WEIGHTS ='))
    # Plotting is exercised too, using the noninteractive local backend.
    from common import style
    plt=style()
    n=70;months=['2025-05','2025-06','2025-07','2025-08','2025-09']
    ns.update(y=np.ones(n),s=np.ones(n),TW=np.ones(n),FOLD=np.arange(n)%5,WEEK=np.arange(n)//2,
              MONTHS=months,SIZE_MB={'lgb':20.,'tp35_int7':169.,'tp35_genre':169.,'tp35_pool':169.},
              display=lambda obj:None,score=lambda *args:None,plt=plt,
              Xtr=pd.DataFrame({'h':np.tile(range(4,11),10),'y3':.5}))
    def component(value):
        p=np.full(n,value)
        return p,{m:(np.arange(n),p) for m in months}
    for g,p,expected in [(1.1,1.3,'tp35_genre'),(1.3,1.4,'tp35_int7'),(1.1,1.05,'tp35_pool')]:
        ns.update(COMP={'lgb':component(1.2),'tp35_int7':component(1.2),
                        'tp35_genre':component(g),'tp35_pool':component(p)},results=[])
        exec(decision,ns)
        assert expected in ns['WEIGHTS'] and len(ns['WEIGHTS'])==2
        plt.close('all')
    assert not any(c.get('outputs') for c in nb['cells'])
    print('PASS: actual split counts and purge; metadata poison/invariants; per-horizon/genre/pool inference paths; deterministic sampling; all selection outcomes; syntax.')


if __name__=='__main__':
    main()
