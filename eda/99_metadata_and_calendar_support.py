"""Audit lost official metadata and calendar support; no model training."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, fig_dir, load, style
from evaluate import test_weights_fd
from features import GENRES


def main():
    out=fig_dir('99_metadata_support')
    data=load(); meta=data['movies'].copy()
    meta['original_title']=meta.original_title.str.strip()
    assert meta.original_title.is_unique
    meta=meta.set_index('original_title')
    tokens=meta.genre.fillna('').str.split(',').map(lambda v:{z.strip() for z in v if z.strip()})
    vocab=sorted(set().union(*tokens))
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet')
    t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    o=pd.read_csv(ROOT/'results/v16/oof.csv')
    x=x.merge(o[KEY+['h','y','scale','oof_final','fold']],on=KEY+['h'],validate='one_to_one',suffixes=('','_oof'))
    assert np.array_equal(x.y,x.total_ticket) and np.allclose(x.scale,x.scale_oof)
    w=test_weights_fd(x,t)
    x['w']=w; x['e']=(x.y-x.oof_final).abs()/x.scale
    x['signed']=(x.y-x.oof_final)/x.scale
    summaries=[]
    for name,a in [('train',x),('test',t)]:
        a['base']=base_title(a.movie_title)
        a['tokens']=a.base.map(tokens).map(lambda z:z if isinstance(z,set) else set())
        a['no_old_genre']=a.tokens.map(lambda z:not bool(z & set(GENRES)))
        a['has_omitted_genre']=a.tokens.map(lambda z:bool(z-set(GENRES)))
        summaries.append(dict(split=name,rows=len(a),base_films=a.base.nunique(),
            metadata_unmatched_rows=int((~a.base.isin(meta.index)).sum()),
            all_old_flags_zero_rows=int(a.no_old_genre.sum()),
            all_old_flags_zero_films=a.loc[a.no_old_genre,'base'].nunique(),
            rows_with_omitted_genre=int(a.has_omitted_genre.sum()),
            omitted_genre_share=float(a.has_omitted_genre.mean())))
        a.loc[a.no_old_genre,['base']].drop_duplicates().assign(genre=lambda z:z.base.map(meta.genre)).to_csv(out/f'{name}_zero_flags.csv',index=False)
    rows=[]
    for g in vocab:
        a=x.tokens.map(lambda z:g in z).to_numpy(); b=t.tokens.map(lambda z:g in z).to_numpy()
        if not a.any():
            continue
        ff=x.loc[a].assign(we=x.loc[a,'w']*x.loc[a,'e']).groupby('base').we.sum()
        rows.append(dict(genre=g,in_baseline=g in GENRES,train_films=x.loc[a,'base'].nunique(),
                         test_films=t.loc[b,'base'].nunique(),train_rows=int(a.sum()),test_rows=int(b.sum()),
                         TW=np.average(x.e[a],weights=w[a]),signed=np.average(x.signed[a],weights=w[a]),
                         contribution=float(np.sum(w[a]*x.e[a])/w.sum()),top_film_error_share=float(ff.max()/ff.sum())))
    genres=pd.DataFrame(rows).sort_values('contribution',ascending=False)
    genres.to_csv(out/'genre_errors.csv',index=False)
    # Intersections matter: a genre can occur with a flag that the old model already sees.
    cells=[]
    for col in ['no_old_genre','has_omitted_genre']:
        a=x[col].to_numpy()
        cells.append(dict(group=col,weight_share=float(w[a].sum()/w.sum()),
                          error_share=float(np.sum(w[a]*x.e[a])/np.sum(w*x.e)),
                          TW=float(np.average(x.e[a],weights=w[a]))))
    # Calendar support is visible before target outcomes. No new validation score is claimed.
    calrows=[]
    for key in ['d1_dow','dow','h']:
        tr=x[key].value_counts(normalize=True); te=t[key].value_counts(normalize=True)
        q=pd.concat([tr.rename('train_share'),te.rename('test_share')],axis=1).fillna(0)
        q['key']=key; q['value']=q.index
        calrows.append(q.reset_index(drop=True))
    pd.concat(calrows).to_csv(out/'calendar_composition.csv',index=False)
    # Base-film mean residuals prevent interpreting 100 cinemas as 100 independent films.
    film=x.assign(we=w*x.e,wr=w*x.signed).groupby('base').agg(w=('w','sum'),we=('we','sum'),wr=('wr','sum'),d1=('d1','first'))
    film['TW']=film.we/film.w; film['signed']=film.wr/film.w
    film['genre']=film.index.map(meta.genre)
    film.sort_values('we',ascending=False).to_csv(out/'film_residuals.csv')
    summary={'vocab':vocab,'omitted':sorted(set(vocab)-set(GENRES)),
             'profiles':summaries,'error_groups':cells,
             'caveat':'Overlapping genre rows are not additive. Residual associations are exploratory and already selected OOF, not new model gains.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print(genres.round(4).to_string(index=False))
    plt=style();fig,ax=plt.subplots(1,2,figsize=(13,4))
    g=genres.loc[~genres.in_baseline].head(10).sort_values('contribution')
    ax[0].barh(g.genre,g.contribution);ax[0].set(title='Omitted genres: overlapping error contributions')
    pd.DataFrame(summaries).set_index('split').omitted_genre_share.plot.bar(ax=ax[1],rot=0,title='Rows carrying at least one omitted genre')
    fig.tight_layout();fig.savefig(out/'metadata.png');plt.close(fig)


if __name__=='__main__':
    main()
