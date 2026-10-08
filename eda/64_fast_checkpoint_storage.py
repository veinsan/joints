"""Convert local Fast checkpoint matrix weights to FP16 storage; no training/inference.

Vector parameters and regression borders remain FP32. Accuracy must be checked on Kaggle.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import style


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=Path, default=Path.home()/'.cache/tabpfn/tabpfn-v3.5-fast-20260909.safetensors')
    args = ap.parse_args()
    out = ROOT/'outputs/eda/64_checkpoint_storage'
    out.mkdir(parents=True, exist_ok=True)
    dest = out/'tabpfn-v3.5-fast-matrix-fp16.safetensors'
    assert args.source.resolve() != dest.resolve()
    arrays, rows = {}, []
    with safe_open(args.source, framework='np') as src:
        metadata = src.metadata()
        for name in src.keys():
            a = src.get_tensor(name)
            b = a.astype(np.float16) if a.dtype == np.float32 and a.ndim >= 2 else a.copy()
            assert np.isfinite(b).all(), f'Nonfinite converted parameter: {name}'
            delta = b.astype(np.float64)-a.astype(np.float64)
            rows.append(dict(name=name, source_bytes=a.nbytes, converted_bytes=b.nbytes,
                             changed_dtype=str(a.dtype) != str(b.dtype),
                             max_absolute_error=float(np.abs(delta).max()),
                             squared_error=float(np.square(delta).sum()), squared_weight=float(np.square(a.astype(np.float64)).sum())))
            arrays[name] = b
    save_file(arrays, dest, metadata=metadata)
    with safe_open(dest, framework='np') as saved:
        assert saved.metadata() == metadata and set(saved.keys()) == set(arrays)
        for name in saved.keys():
            assert np.array_equal(saved.get_tensor(name), arrays[name])
    # Local trusted artifact. Only estimate a NEW LGB+Fast package, not relabel v12 weights.
    with (ROOT/'results/v12/model_weights.pkl').open('rb') as f:
        package = pickle.load(f)
    lgb_bytes = len(pickle.dumps(package['lightgbm'], protocol=pickle.HIGHEST_PROTOCOL))
    summary = dict(source=str(args.source), source_sha256=sha(args.source), converted_sha256=sha(dest),
                   source_MB=args.source.stat().st_size/1e6, converted_MB=dest.stat().st_size/1e6,
                   v12_lgb_serialized_MB=lgb_bytes/1e6,
                   estimated_LGB_plus_Fast_MB=(lgb_bytes+dest.stat().st_size)/1e6,
                   headroom_before_metadata_MB=200-(lgb_bytes+dest.stat().st_size)/1e6,
                   relative_weight_L2_error=float(np.sqrt(sum(r['squared_error'] for r in rows)/sum(r['squared_weight'] for r in rows))),
                   layers=json.loads(metadata['config'])['nlayers'],
                   limitation='Storage round-trip verified only. Fast is 8 layers, v8 default is 24 layers. No prediction equivalence or accuracy improvement established. Final complete serialized package must independently pass 200 MB. Reproduction must use the converted weights in validation and final inference.')
    import pandas as pd
    pd.DataFrame(rows).to_csv(out/'tensor_errors.csv',index=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    plt = style()
    fig, ax = plt.subplots(figsize=(8,4))
    sizes = [summary['source_MB'],summary['converted_MB'],summary['estimated_LGB_plus_Fast_MB']]
    ax.barh(['Fast FP32 checkpoint','Fast matrix-FP16 checkpoint','FP16 Fast + v12 LGB (estimate)'],sizes)
    ax.axvline(200,color='#e34948',ls='--',label='200 MB budget')
    for i,n in enumerate(sizes): ax.text(n+2,i,f'{n:.2f} MB',va='center')
    ax.set(xlabel='Decimal MB',title='Storage feasibility, not an accuracy benchmark',xlim=(0,380)); ax.legend()
    fig.tight_layout(); fig.savefig(out/'storage_budget.png'); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
