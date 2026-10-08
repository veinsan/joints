"""Align saved v1-v17 artifacts and inventory EDA. Read-only model audit, no training."""
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from common import KEY, base_title, fig_dir, style
from evaluate import test_weights_fd


def partition_hash(labels):
    canonical=pd.factorize(labels,sort=False)[0].astype('<i8')
    return hashlib.sha256(canonical.tobytes()).hexdigest()[:16]


def main():
    out=fig_dir('105_v1_v17_longitudinal')
    x=pd.read_parquet(ROOT/'outputs/cache/Xtr.parquet');t=pd.read_parquet(ROOT/'outputs/cache/Xte.parquet')
    key=KEY+['h']; assert not x.duplicated(key).any()
    w=test_weights_fd(x,t); y=x.total_ticket.to_numpy();s=x.scale.to_numpy();base=base_title(x.movie_title)
    inventory=[];scores=[];components=[];pred={};subs={};folds={};filmrows=[];nbrows=[]
    for v in range(1,18):
        directory=ROOT/f'results/v{v}';p=directory/'oof.csv'
        row=dict(version=v,oof_exists=p.exists(),common_rows=0,full_label_scale_match=False)
        n=ROOT/f'notebooks/v{v}.ipynb'; selections=[];errors=[];split_lines=[]
        if n.exists():
            nb=json.loads(n.read_text())
            for i,c in enumerate(nb['cells']):
                source=''.join(c['source'])
                for line in source.splitlines():
                    if any(z in line for z in ['GroupKFold','fold_of_week =','MONTHS =','FOLD =','StratifiedGroupKFold']):split_lines.append(line)
                for output in c.get('outputs',[]):
                    if output['output_type']=='error':errors.append(dict(cell=i,error=output.get('evalue')))
                    for line in ''.join(output.get('text',[])).splitlines():
                        if 'chosen weights:' in line:selections.append(line)
            nbrows.append(dict(version=v,selection=selections,split_code=split_lines,errors=errors))
        row['selection']=' | '.join(selections)
        if p.exists():
            a=pd.read_csv(p);assert not a.duplicated(key).any()
            z=x[key+['total_ticket','scale']].merge(a,on=key,how='left',validate='one_to_one',suffixes=('','_saved'),indicator=True)
            present=z._merge.eq('both');row.update(oof_rows=len(a),common_rows=int(present.sum()),
                own_MASE=float(np.mean(abs(a.y-a['oof_final' if 'oof_final' in a else 'pred'])/a.scale)))
            valid=present & np.isclose(z.total_ticket,z.y)&np.isclose(z.scale,z.scale_saved)
            row['matching_label_scale_rows']=int(valid.sum())
            row['full_label_scale_match']=bool(valid.all() and len(a)==len(x))
            if row['full_label_scale_match']:
                col='oof_final' if 'oof_final' in a else 'pred'
                pr=z[col].to_numpy(); assert np.isfinite(pr).all()
                pred[v]=pr;folds[v]=z.fold.to_numpy()
                row['fold_partition_hash']=partition_hash(z.fold)
                row['base_film_disjoint']=bool(pd.Series(z.fold.to_numpy()).groupby(base).nunique().max()==1)
                e=abs(y-pr)/s
                scores.append(dict(version=v,MASE=e.mean(),TW=np.average(e,weights=w),
                    growth_TW=np.average(e[y>x.y3],weights=w[y>x.y3]),
                    zero_TW=np.average(e[y==0],weights=w[y==0]),
                    positive_TW=np.average(e[y>0],weights=w[y>0]),
                    early_TW=np.average(e[x.h<=5],weights=w[x.h<=5]),fold_partition_hash=row['fold_partition_hash']))
                for c in a.columns:
                    if (c.startswith(('comp_','oof_')) or c=='pred') and pd.api.types.is_numeric_dtype(a[c]):
                        ee=abs(y-z[c].to_numpy())/s
                        components.append(dict(version=v,component=c,MASE=ee.mean(),TW=np.average(ee,weights=w)))
                for b,ii in pd.Series(np.arange(len(x))).groupby(base):
                    ix=ii.to_numpy();filmrows.append(dict(version=v,base=b,TW=np.average(e[ix],weights=w[ix]),
                        contribution=np.sum(e[ix]*w[ix])/w.sum(),signed=np.average((y[ix]-pr[ix])/s[ix],weights=w[ix])))
        ss=pd.read_csv(directory/'submission.csv');assert ss.id.is_unique and len(ss)==len(t)
        ss=t[['id']].merge(ss,on='id',how='left',validate='one_to_one')
        assert np.isfinite(ss.total_ticket).all() and ss.total_ticket.ge(0).all()
        subs[v]=ss.total_ticket.to_numpy();row['submission_sha256']=hashlib.sha256((directory/'submission.csv').read_bytes()).hexdigest()
        inventory.append(row)
    inv=pd.DataFrame(inventory);sc=pd.DataFrame(scores);comp=pd.DataFrame(components);f=pd.DataFrame(filmrows)
    inv.to_csv(out/'artifact_inventory.csv',index=False);sc.to_csv(out/'common_population_scores.csv',index=False)
    comp.to_csv(out/'component_scores.csv',index=False);f.to_csv(out/'film_errors_long.csv',index=False)
    (out/'notebook_selection_and_splits.json').write_text(json.dumps(nbrows,indent=2))
    pd.DataFrame({v:abs(y-p)/s for v,p in pred.items()}).corr().to_csv(out/'error_correlation.csv')
    distance=pd.DataFrame({a:{b:np.mean(abs(subs[a]-subs[b])/t.scale.to_numpy()) for b in subs} for a in subs})
    distance.to_csv(out/'submission_distance.csv')
    # Same partition is necessary, not sufficient: preprocessing/training sets can still differ.
    comparisons=[]
    for a in pred:
        for b in pred:
            if b<=a:continue
            delta=(abs(y-pred[b])-abs(y-pred[a]))/s
            comparisons.append(dict(a=a,b=b,same_fold_partition=partition_hash(folds[a])==partition_hash(folds[b]),
                delta_TW=np.average(delta,weights=w),delta_MASE=delta.mean(),
                fraction_rows_improved=float(np.mean(delta<0))))
    pd.DataFrame(comparisons).to_csv(out/'pairwise_comparisons.csv',index=False)
    # Truth-assisted choice only diagnoses prediction diversity; not an attainable score bound.
    diagnostics=[]
    for name,versions in [('all_aligned',list(pred)),('recent',[8,12,13,15,16,17])]:
        versions=[v for v in versions if v in pred];P=np.column_stack([pred[v] for v in versions]);A=abs(y[:,None]-P)/s[:,None]
        row_oracle=A.min(axis=1);outside=np.maximum(P.min(axis=1)-y,0)+np.maximum(y-P.max(axis=1),0)
        film_oracle=0
        for _,idx in pd.Series(np.arange(len(x))).groupby(base):
            ix=idx.to_numpy();film_oracle+=np.min((A[ix]*w[ix,None]).sum(axis=0))/w.sum()
        diagnostics.append(dict(set=name,versions=versions,row_choice_oracle_TW=np.average(row_oracle,weights=w),
            film_choice_oracle_TW=film_oracle,convex_envelope_oracle_TW=np.average(outside/s,weights=w),
            all_under_weight_share=float(w[y>P.max(axis=1)].sum()/w.sum()),
            all_over_weight_share=float(w[y<P.min(axis=1)].sum()/w.sum()),
            baseline17_error_outside_envelope_share=float((abs(y-pred[17])/s*w)[(y>P.max(axis=1))|(y<P.min(axis=1))].sum()/np.sum(abs(y-pred[17])/s*w))))
    (out/'diversity_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
    pivot=f.pivot(index='base',columns='version',values='contribution')
    pivot['recent_mean']=pivot[[v for v in [8,12,13,15,16,17] if v in pivot]].mean(axis=1)
    pivot.sort_values('recent_mean',ascending=False).to_csv(out/'persistent_film_errors.csv')
    # Every EDA source is parsed; this is an inventory, not a claim every experiment was rerun.
    eda=[]
    for path in sorted((ROOT/'eda').glob('*.py')):
        if path==Path(__file__):continue
        source=path.read_text();tree=ast.parse(source)
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        eda.append(dict(file=path.name,description=ast.get_docstring(tree) or '',
            contains_fit_call=any(isinstance(c.func,ast.Attribute) and c.func.attr=='fit' for c in calls),
            source_sha256=hashlib.sha256(source.encode()).hexdigest()))
    pd.DataFrame(eda).to_csv(out/'eda_inventory.csv',index=False)
    summary=dict(versions_examined=17,aligned_versions=list(pred),missing_oof_versions=[r['version'] for r in inventory if not r['oof_exists']],
        eda_files_parsed=len(eda),distinct_submission_arrays=len({p.tobytes() for p in subs.values()}),
        identical_to_v15=[v for v,p in subs.items() if np.array_equal(p,subs[15])],
        public_gap_user_reported=.39918-.33662,
        caveat='Identical evaluation rows do not imply identical training folds. Cross-version deltas are descriptive unless protocols match. Oracles access truth and are not models or achievable forecasts. No leaderboard queried or submission generated.')
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt=style();fig,ax=plt.subplots(1,2,figsize=(13,5))
    sc.plot(x='version',y=['MASE','TW'],marker='o',ax=ax[0]);ax[0].set_title('Same labels, different training protocols: descriptive only')
    top=pivot.nlargest(10,'recent_mean').index
    tab=pivot.loc[top,[v for v in [3,5,8,12,13,15,17] if v in pivot]]
    im=ax[1].imshow(tab.values,aspect='auto',cmap='Oranges');ax[1].set(yticks=range(len(tab)),yticklabels=tab.index,xticks=range(len(tab.columns)),xticklabels=tab.columns,title='Persistent film error contribution to TW')
    ax[1].grid(False);fig.colorbar(im,ax=ax[1]);fig.tight_layout();fig.savefig(out/'longitudinal.png');plt.close(fig)
    print(sc.round(6).to_string(index=False));print(json.dumps(summary,indent=2));print(json.dumps(diagnostics,indent=2))


if __name__=='__main__':main()
