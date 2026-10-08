"""Small runnable checks for generated v16 changes, using fake inference, never model fitting."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import base_title


def main():
    nb = json.loads((ROOT/'notebooks/v16.ipynb').read_text())
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    tree = ast.parse('\n'.join(line for c in cells for line in c.splitlines() if not line.startswith(('!','%'))))
    selected = {'train_part','make_tabpfn','r_target','lens','blend'}
    funcs = ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in selected],type_ignores=[])
    out = Path('/tmp/joints-v16-check'); out.mkdir(exist_ok=True)
    cfg = SimpleNamespace(USE_LIMITED=True, DEVICE='fake',SEED=2026,N_FOLDS=5,
                          ABL_MIN_GAIN=.0015, ABL_MONTH_TOL=.003, ABL_MIN_FOLDS=3,
                          SIZE_MARGIN_MB=1.5,MAX_WEIGHTS_MB=200,OUTPUT_DIR=out)
    ns = dict(np=np,pd=pd,CFG=cfg,base_title=base_title)
    exec(compile(funcs,'v16_functions','exec'),ns)
    x = pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    o = pd.read_csv(ROOT/'results/v15/oof.csv')
    keys=['movie_title','cinema_ids','h']
    x=x.merge(o[keys+['fold']],on=keys,validate='one_to_one')
    lim = pd.read_parquet(ROOT/'outputs/cache/Xlim.parquet')
    week = ((x.d1-pd.Timestamp('2025-03-31')).dt.days//7).to_numpy()
    ns.update(Xtr=x,Xlim=lim,FOLD=x.fold.to_numpy(),WEEK=week,groups=base_title(x.movie_title).to_numpy())
    removed=[]
    for k in range(5):
        a=ns['train_part'](k)
        assert not set(base_title(a.movie_title)) & set(base_title(x.loc[x.fold.eq(k),'movie_title']))
        removed.append(len(x.loc[x.fold.ne(k)])+len(lim)-len(a))
    assert removed == [371,350,273,119,322], removed

    seen=[]
    class FakeRegressor:
        @staticmethod
        def create_default_for_version(*args,**kwargs):
            return FakeRegressor()
        def fit(self,X,y):
            seen.append(y.copy()); self.y=y
        def predict(self,X,output_type,quantiles):
            # Real library returns one vector per requested quantile.
            q=np.quantile(self.y,quantiles,method='inverted_cdf')
            return np.tile(q[:,None],(1,len(X)))
    qs=np.array([.25,.5,.75])
    ns.update(TabPFNRegressor=FakeRegressor,FEATS=['h','a'],TAB_QS=qs,
              torch=SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)))
    a=pd.DataFrame({'h':[4]*4,'a':[1,2,3,4],'total_ticket':[0,1,3,15],'scale':1.,'cal_mult':1.})
    b=a.iloc[:2]
    raw=ns['make_tabpfn'](None,None,1)(a,b)
    log=ns['make_tabpfn'](None,None,1,log_target=True)(a,b)
    assert np.allclose(seen[1],np.log1p(seen[0]))
    assert np.allclose(raw,log) and np.isfinite(log).all()

    decision=next(c for c in cells if c.startswith('CONTROL_WEIGHTS ='))
    n=50; y=np.ones(n); months=['2025-07','2025-08','2025-09']
    ns.update(y=y,s=np.ones(n),TW=np.ones(n),FOLD=np.arange(n)%5,WEEK=np.arange(n)//2,
              MONTHS=months,SIZE_MB={'lgb':20.,'tp35_int7':169.,'tp35_log':169.},
              display=lambda obj:None,score=lambda *args:None)
    def component(value):
        p=np.full(n,value)
        return p,{m:(np.arange(n),p) for m in months}
    for value,expected in [(1.1,'tp35_log'),(1.3,'tp35_int7')]:
        ns.update(COMP={'lgb':component(1.2),'tp35_int7':component(1.2),'tp35_log':component(value)},results=[])
        exec(decision,ns)
        assert expected in ns['WEIGHTS'] and len(ns['WEIGHTS'])==2
    assert not any(c.get('outputs') for c in nb['cells'])
    print('PASS: all code parses; complete-cohort exclusion; log-target/inverse; accept/reject gates; clean notebook outputs')


if __name__=='__main__':
    main()
