"""53 - Does the D1 reconstruction rule look into the future, and does it matter? (EDA, no training)

Audit 2026-10-05 point 5: D1 = first day whose base-title coverage reaches 50% of the PEAK, inside the continuous
segment that contains the peak. The peak can come days later, so a film that opens small and expands (a breakout)
may get a LATER D1 than its real release, i.e. the train-sim drops exactly the growth phase that the test keeps
(test D1 = official release). Rules compared on train:
  R0  current: >= 50% of the segment peak                                    (future peak)
  R1  bounded: >= 50% of the max coverage within the first 10 days of the segment (lookahead <= 10 days)
  R2  causal?: first day of the PEAK segment with coverage >= MIN_NC (still uses the peak to pick the segment)
  R3  causal : first day d whose own window d..d+2 has coverage >= MIN_NC on every day; no peak, no segment
               (lookahead limited to the D1-D3 window the test itself exposes)
Change counts are exact (pairs whose scale changed, target rows whose label changed), not medians - a median scale
ratio of 1 does not mean no pair changed (correction after review, 2026-10-05).
For each rule vs R0: films whose D1 moves, pairs / target rows / scale / zero-share changes, and the v8 OOF error
contribution (bucket x first-day weights) of the moved films. Also: does the coverage-growth feature fnc3 / fnc1
of the train-sim move toward the test distribution (it was the #2 adversarial feature, eda/44)?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from common import KEY, base_title, fig_dir, load, release_dates, scale, simulate, style
from evaluate import test_weights_fd
from features import dataset

plt = style()
F = fig_dir("53_d1_audit")
ROOT = Path(__file__).resolve().parents[1]
d = load()
tr, th = d["train"], d["hist"]
MIN_NC = 25


def rule(tx, kind):
    x = tx.assign(base=base_title(tx.movie_title))
    nc = x.groupby(["base", "date_show"]).cinema_ids.nunique()
    out = {}
    for b, s in nc.groupby(level=0):
        s = s.droplevel(0)
        s = s.reindex(pd.date_range(s.index.min(), s.index.max()), fill_value=0)
        if kind == "R3":
            ok = (s >= MIN_NC) & (s.shift(-1) >= MIN_NC) & (s.shift(-2) >= MIN_NC)
            if ok.any():
                out[b] = ok.idxmax()
            continue
        pk = s.idxmax()
        z = s[:pk][s[:pk] == 0]
        seg = s[z.index.max() + pd.Timedelta(days=1):] if len(z) else s
        if kind == "R0":
            ref = s.max()
        elif kind == "R1":
            ref = seg.iloc[:10].max()
        else:
            ref = 2 * MIN_NC
        c = seg[seg >= 0.5 * ref]
        if len(c) and c.iloc[0] >= MIN_NC:
            out[b] = c.index[0]
    t = x.drop_duplicates("movie_title").set_index("movie_title").base
    return t.map(pd.Series(out, dtype="datetime64[ns]")).dropna().rename("d1")


first = tr.groupby("movie_title").date_show.min()
running = first[first == first.min()].index
R = {k: rule(tr, k).drop(running, errors="ignore") for k in ("R0", "R1", "R2", "R3")}
assert (R["R0"].sort_index() == release_dates(tr).drop(running, errors="ignore").sort_index()).all()
sims = {k: simulate(tr, v) for k, v in R.items()}

o = pd.read_csv(ROOT / "results/v8/oof.csv")
Xtr, Xte, _ = dataset()
w = test_weights_fd(Xtr, Xte)
contrib = pd.Series(np.abs(o.y - o.oof_final) / o.scale * w / w.sum()).groupby(base_title(o.movie_title).values).sum()


def fnc_trend(h):
    h = h.assign(base=base_title(h.movie_title))
    d1 = h.groupby("base").date_show.transform("min")
    h["d"] = (h.date_show - d1).dt.days + 1
    nc = h.groupby(["base", "d"]).cinema_ids.nunique().unstack()
    return (nc[3] / nc[1]).dropna()


te_tr = fnc_trend(th)
rows, moved_tab = [], []
b0 = R["R0"].groupby(base_title(pd.Series(R["R0"].index, index=R["R0"].index))).first()
for k in ("R0", "R1", "R2", "R3"):
    h, t = sims[k]
    bk = R[k].groupby(base_title(pd.Series(R[k].index, index=R[k].index))).first()
    common = b0.index.intersection(bk.index)
    moved = common[(b0[common] != bk[common]).values]
    s0, sk = scale(sims["R0"][0]), scale(h)
    pc = s0.index.intersection(sk.index)
    ft = fnc_trend(h)
    rows.append(dict(rule=k, films=t.movie_title.map(base_title).nunique() if False else base_title(t.movie_title).nunique(),
                     pairs=len(t[KEY].drop_duplicates()), target_rows=len(t), zero_share=(t.total_ticket == 0).mean(),
                     films_moved_vs_R0=len(moved), moved_error_share_v8=contrib.reindex(moved).sum() / contrib.sum(),
                     common_pairs=len(pc), common_pairs_scale_changed=int((np.abs(sk[pc] / s0[pc] - 1) > 1e-9).sum()),
                     common_target_rows_label_changed=int(t.merge(sims["R0"][1], on=KEY + ["date_show"], suffixes=("", "_0"))
                                                          .eval("total_ticket != total_ticket_0").sum()),
                     fnc_trend_median=ft.median(), ks_fnc_trend_vs_test=ks_2samp(ft, te_tr).statistic))
    for b in moved:
        moved_tab.append(dict(rule=k, film=b, d1_R0=b0[b].date(), d1_new=bk[b].date(), shift_days=(bk[b] - b0[b]).days,
                              v8_error_share=contrib.get(b, np.nan) / contrib.sum()))
T = pd.DataFrame(rows)
print(f"test fnc3/fnc1 median {te_tr.median():.3f}\n")
print("=== D1 rule comparison ===\n", T.round(4).to_string(index=False))
M = pd.DataFrame(moved_tab)
if len(M):
    print("\n=== films whose D1 changes vs R0 ===\n", M.round(4).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for k, c in zip(("R0", "R1", "R2", "R3"), ("#2a78d6", "#eb6834", "#1baf7a", "#e34948")):
    ax[0].hist(np.clip(fnc_trend(sims[k][0]), 0, 3), bins=40, density=True, histtype="step", color=c, label=f"train-sim {k}")
ax[0].hist(np.clip(te_tr, 0, 3), bins=40, density=True, histtype="step", color="k", label="test")
ax[0].set(title="film coverage growth D3 / D1"); ax[0].legend()
ax[1].bar(T.rule, T.ks_fnc_trend_vs_test, color="#2a78d6"); ax[1].set(title="KS distance to test (lower = closer)")
fig.savefig(F / "d1_rules.png")
print(f"figures -> {F}")
