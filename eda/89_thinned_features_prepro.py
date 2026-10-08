"""Preprocessing check: full pipeline features on the thinned world vs the test.

Thinned world (pi = 0.4): every train transaction y' ~ Binomial(y, pi); occupation_rate scaled by
y'/y; total_show unchanged (shows per pair are the same in the test period: median 5 vs 5); D1 from
the raw data; organiser rule re-applied. Information rule copied from the test side: cluster history
statistics (cin_size, cin_nfilms, cin_shows) come from the RAW train (the busy past), D1-D3 windows
and the 'market' come from the quiet world. Then every model feature is compared: KS(raw train,
test) vs KS(thinned, test). A feature that gets further from the test is a preprocessing failure.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import fig_dir, load, release_dates, simulate, style
from features import build, film_table, make_ctx, visible_windows

FEATS = ['log_s', 'p1', 'p2', 'p3', 'n_hist', 'sh_trend', 'fnc1', 'fnc2', 'fnc3', 'fp1', 'fp2', 'fp3', 'f_logT',
         'f_nc_trend', 'fmt_share', 'd1_dow', 'comp_n', 'f_share_cohort', 'h', 'cal_mult', 'n_comp_open', 'share',
         'p3_rel', 'p1_rel', 'cin_size', 'cin_nfilms', 'c_new_n', 'c_new_sh', 'c_new_sh_rel', 'c_new_sh_vs_own',
         'c_new_tx_vs_own', 'f_occ', 'occ_mean', 'tps3', 'pair_vs_mkt', 'film_pc_vs_mkt']


def thin_tx(tr, pi, seed):
    t = tr.copy()
    y = np.random.default_rng(seed).binomial(t.total_ticket.values.astype(int), pi)
    t["occupation_rate"] = t.occupation_rate * y / t.total_ticket
    t["total_ticket"] = y
    return t[t.total_ticket > 0].reset_index(drop=True)


def thinned_X(D, pi, seed):
    tr, th = D["train"], D["hist"]
    first = tr.groupby("movie_title").date_show.min()
    running = first[first == first.min()].index
    d_wide = release_dates(tr)
    tt = thin_tx(tr, pi, seed)
    vis = visible_windows(tt, d_wide)
    market = pd.concat([film_table(vis), film_table(th)])
    ctx = make_ctx(vis, D["hol"], D["movies"], D["price"], tr, market)
    hs, ts = simulate(tt, d_wide.drop(running, errors="ignore"))
    return build(hs, ts, ctx)


def main():
    out = fig_dir("89_thinned_features")
    D = load()
    Xr = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    Xt = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    Xh = thinned_X(D, 0.4, 7)
    Xh.to_parquet(out / "Xtr_thinned_pi040.parquet")
    print(f"rows raw {len(Xr)}, thinned {len(Xh)}, test {len(Xt)}; films raw {Xr.movie_title.nunique()}, thinned {Xh.movie_title.nunique()}")
    rows = []
    for f in FEATS:
        a, b, c = Xr[f].dropna(), Xh[f].dropna(), Xt[f].dropna()
        rows.append(dict(feature=f, ks_raw_test=ks_2samp(a, c).statistic, ks_thin_test=ks_2samp(b, c).statistic,
                         med_raw=a.median(), med_thin=b.median(), med_test=c.median()))
    R = pd.DataFrame(rows).set_index("feature")
    R["closer"] = R.ks_thin_test < R.ks_raw_test
    pd.set_option("display.width", 200)
    print(R.sort_values("ks_raw_test", ascending=False).round(3).to_string())
    print(f"\nfeatures closer to test after thinning: {int(R.closer.sum())}/{len(R)}; mean KS raw {R.ks_raw_test.mean():.3f} -> thinned {R.ks_thin_test.mean():.3f}")
    print("further from test:", R[~R.closer].index.tolist())
    # target-side sanity: zero rate and median r by scale bucket, raw vs thinned
    for n, X in [("raw", Xr), ("thinned", Xh)]:
        g = X.groupby(pd.cut(X.scale, [0, 5, 20, 50, 200, 1e9]), observed=True)
        print(f"{n:8s} zero rate by scale bucket", g.total_ticket.apply(lambda v: (v == 0).mean()).round(3).tolist(),
              "| median r", g.apply(lambda q: np.median(q.total_ticket / q.scale), include_groups=False).round(3).tolist())
    R.to_csv(out / "ks.csv")
    plt = style()
    fig, ax = plt.subplots(figsize=(9, 7))
    R2 = R.sort_values("ks_raw_test")
    yy = np.arange(len(R2))
    ax.barh(yy - .2, R2.ks_raw_test, .4, label="raw train vs test"); ax.barh(yy + .2, R2.ks_thin_test, .4, label="thinned 0.4 vs test")
    ax.set_yticks(yy, R2.index, fontsize=7); ax.set(title="KS distance to the test, per feature"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "ks_by_feature.png")


if __name__ == "__main__":
    main()
