"""52 - Do official talent fields (producer / director / writer / casts) carry the film-level miss? (EDA, no training)

Audit 2026-10-05 (docs/experiment_audit_2026-10-05.md, priority 2): movies.csv identities are unused, and 46-75% of
test rows share at least one name with a train film. eda/38 showed film-level residuals are unpredictable from
the CURRENT film features; talent was not among them. Screen, per field, a shrunken track record of the films that
share a name with film f, built only from OTHER folds (fold-local) or only from EARLIER films (past-only, the test
situation), against film f's own v8 OOF miss:
    bias_f   = mean over rows of (y - pred) / s        (MASE-aligned signed miss)
    lr_f     = log((sum y + 50) / (sum pred + 50))     (volume miss)
    track_f  = sum_j n_j * bias_j / (sum_j n_j + K)    over films j sharing a name (K = shrinkage, in rows)
Reported: coverage, Spearman with a film-permutation p-value, and the TW-MASE change of a CROSS-FITTED, shrunk
multiplicative correction (diagnostic only - not a deployable model until nested inside the notebook CV).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import base_title, fig_dir, load, style
from evaluate import test_weights_fd
from features import dataset

plt = style()
F = fig_dir("52_talent")
ROOT = Path(__file__).resolve().parents[1]
Xtr, Xte, Xlim = dataset()
d = load()
o = pd.read_csv(ROOT / "results/v8/oof.csv")
assert (o.movie_title.values == Xtr.movie_title.values).all() and (o.h.values == Xtr.h.values).all()
y, s, p = Xtr.total_ticket.values, Xtr.scale.values, o.oof_final.values
w = test_weights_fd(Xtr, Xte)
g = base_title(Xtr.movie_title).values
D = pd.DataFrame({"base": g, "fold": o.fold.values, "b": (y - p) / s, "y": y, "p": p, "w": w, "d1": Xtr.d1.values})
film = D.groupby("base").agg(n=("b", "size"), bias=("b", "mean"), ys=("y", "sum"), ps=("p", "sum"), fold=("fold", "first"), d1=("d1", "min"))
film["lr"] = np.log((film.ys + 50) / (film.ps + 50))
mv = d["movies"].set_index("original_title")
names = lambda f, field: {t.strip().lower() for t in str(mv.loc[f, field] if f in mv.index else "").split(",") if t.strip() and t.strip() != "-"}
rng = np.random.default_rng(2026)


def track(field, mode, K=2000):
    out = np.full(len(film), np.nan)
    sets = {f: names(f, field) for f in film.index}
    for i, f in enumerate(film.index):
        if mode == "fold":
            pool = film.index[(film.fold != film.fold[f]).values]
        else:
            pool = film.index[(film.d1 < film.d1[f]).values]
        share = [j for j in pool if sets[j] & sets[f]]
        if share:
            nj = film.n[share].values
            out[i] = (nj * film.bias[share].values).sum() / (nj.sum() + K)
    return out


rows = []
for field in ["producer", "director", "writer", "casts"]:
    for mode in ("fold", "past"):
        t = track(field, mode)
        ok = ~np.isnan(t)
        if ok.sum() < 8:
            rows.append(dict(field=field, mode=mode, films_covered=int(ok.sum()))); continue
        rho = spearmanr(t[ok], film.bias.values[ok])[0]
        perm = np.array([spearmanr(t[ok], rng.permutation(film.bias.values[ok]))[0] for _ in range(500)])
        rho_lr = spearmanr(t[ok], film.lr.values[ok])[0]
        rows.append(dict(field=field, mode=mode, films_covered=int(ok.sum()), rho_bias=rho, p_perm=(np.abs(perm) >= abs(rho)).mean(), rho_lr=rho_lr))
        film[f"t_{field}_{mode}"] = t
R = pd.DataFrame(rows)
print("=== talent track record vs film-level v8 OOF miss ===\n", R.round(3).to_string(index=False))

# cross-fitted diagnostic correction: multiply a film's predictions by exp(c * track) with c fitted on other folds
print("\n=== cross-fitted shrunk correction (diagnostic, single-level) ===")
tw = lambda q: float(np.average(np.abs(y - q) / s, weights=w))
base = tw(p)
for field in ["producer", "director", "writer", "casts"]:
    col = f"t_{field}_fold"
    if col not in film:
        continue
    q = p.copy()
    for k in range(5):
        tr = (film.fold != k) & film[col].notna()
        te = (film.fold == k) & film[col].notna()
        if tr.sum() < 8 or te.sum() == 0:
            continue
        cs = np.linspace(-3, 3, 61)
        Dk = D[D.base.isin(film.index[tr])]
        tk = Dk.base.map(film[col]).values
        best = cs[np.argmin([np.average(np.abs(Dk.y - Dk.p * np.exp(c * tk)) / s[Dk.index], weights=Dk.w) for c in cs])]
        m = D.base.isin(film.index[te]).values
        q[m] = p[m] * np.exp(best * D.base[m].map(film[col]).values)
    print(f"  {field:9s} TW-fd {base:.4f} -> {tw(q):.4f} ({tw(q) - base:+.4f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for a_, mode in zip(ax, ("fold", "past")):
    sub = R[R["mode"] == mode]
    a_.bar(sub.field, sub.rho_bias.fillna(0), color="#2a78d6")
    a_.axhline(0, color="k", lw=.6); a_.set(title=f"Spearman(track record, film miss), {mode}", ylim=(-.5, .5))
fig.savefig(F / "talent.png")
print(f"figures -> {F}")
