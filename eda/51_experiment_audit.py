"""Read-only audit of saved experiments; run: .venv/bin/python eda/51_experiment_audit.py.

No model training, submission, or hidden test labels. Outputs go to outputs/eda/51_audit.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, load, scale, style
from evaluate import BINS, test_weights, test_weights_fd


def main():
    out = ROOT / "outputs/eda/51_audit"
    out.mkdir(parents=True, exist_ok=True)
    d = load()
    X = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    T = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    key = KEY + ["h"]
    assert not X.duplicated(key).any()
    assert np.array_equal(T.id, d["test"].id)
    assert np.allclose(T.scale, d["test"].join(scale(d["hist"]), on=KEY).scale)
    assert d["test"].groupby(KEY).size().eq(7).all()
    w = {"plain": np.ones(len(X)), "scale": test_weights(X, T), "scale_fd": test_weights_fd(X, T)}
    # Check support instead of silently dropping test-only weighting cells.
    cells = lambda A: pd.cut(A.scale, BINS).astype(str) + "|" + A.first_day.astype(str)
    unsupported = cells(T)[~cells(T).isin(cells(X))].value_counts().to_dict()
    scores, preds, sizes, subs = [], {}, [], {}
    for v in range(1, 10):
        folder = ROOT / f"results/v{v}"
        ss = pd.read_csv(folder / "submission.csv")
        assert ss.id.is_unique and len(ss) == len(T)
        ss = T[["id"]].merge(ss, on="id", validate="one_to_one")
        assert np.isfinite(ss.total_ticket).all() and ss.total_ticket.ge(0).all()
        subs[v] = ss.total_ticket.to_numpy()
        sizes.append({"version": v, "pkl_MB": (folder / "model_weights.pkl").stat().st_size / 1e6 if (folder / "model_weights.pkl").exists() else np.nan,
                      "other_checkpoints_MB": sum(p.stat().st_size for p in folder.rglob("*") if p.suffix in {".ckpt", ".safetensors"}) / 1e6})
        p = folder / "oof.csv"
        if not p.exists():
            continue
        o = pd.read_csv(p)
        assert not o.duplicated(key).any()
        z = X.merge(o, on=key, suffixes=("", "_oof"), validate="one_to_one")
        if len(z) != len(X):
            print(f"v{v}: different validation population ({len(o)} rows, {len(z)} shared); excluded from common-weight comparison")
            continue
        assert np.allclose(z.total_ticket, z.y) and np.allclose(z.scale, z.scale_oof)
        assert z.assign(base=base_title(z.movie_title)).groupby("base").fold.nunique().max() == 1
        col = "oof_final" if "oof_final" in z else "pred"
        preds[v] = z[col].to_numpy()
        for c in [c for c in z if c.startswith("oof_")] + (["pred"] if col == "pred" else []):
            e = np.abs(z.y.to_numpy() - z[c].to_numpy()) / z.scale.to_numpy()
            scores.append({"version": v, "component": c, **{k: np.average(e, weights=ww) for k, ww in w.items()}})
    pd.DataFrame(scores).to_csv(out / "scores.csv", index=False)
    pd.DataFrame(sizes).to_csv(out / "weight_sizes.csv", index=False)
    y, s, pred = X.total_ticket.to_numpy(), X.scale.to_numpy(), preds[8]
    e = np.abs(y - pred) / s
    E = X.assign(base=base_title(X.movie_title), error=e, signed=(pred-y)/s,
                 contribution=e*w["scale_fd"]/w["scale_fd"].sum(), weight=w["scale_fd"]/w["scale_fd"].sum(),
                 bucket=pd.cut(s, BINS).astype(str), zero=y == 0, month=X.d1.dt.strftime("%Y-%m"))
    for col in ["bucket", "first_day", "h", "zero", "month", "base"]:
        tab = E.groupby(col).agg(rows=("error", "size"), mase=("error", "mean"), contribution=("contribution", "sum"), weight=("weight", "sum"), signed=("signed", "mean"))
        tab["weighted_mase"] = tab.contribution / tab.weight
        tab.sort_values("contribution", ascending=False).to_csv(out / f"error_{col}.csv")
    # Film bootstrap retains all correlated cluster/horizon rows of a film together.
    G = E.groupby("base").agg(n=("error", "size"), error=("error", "sum"), we=("contribution", "sum"), w=("weight", "sum"))
    rng = np.random.default_rng(2026)
    ix = rng.integers(0, len(G), (2000, len(G)))
    boot = {"plain": G.error.to_numpy()[ix].sum(1)/G.n.to_numpy()[ix].sum(1),
            "scale_fd": G.we.to_numpy()[ix].sum(1)/G.w.to_numpy()[ix].sum(1)}
    uncertainty = {k: np.quantile(b, [.025, .5, .975]).tolist() for k, b in boot.items()}
    delta = (np.abs(y-preds[9])-np.abs(y-preds[8]))/s
    dg = E.assign(de=delta*E.weight).groupby("base").de.sum().reindex(G.index).to_numpy()
    uncertainty["v9_minus_v8_scale_fd"] = np.quantile(dg[ix].sum(1)/G.w.to_numpy()[ix].sum(1), [.025,.5,.975]).tolist()
    # Diagnostic oracles use validation truth; these are NOT deployable estimates.
    oracle = {"baseline": np.average(e, weights=w["scale_fd"])}
    for name, keys in [("film_horizon", ["base", "h"]), ("cinema_date", ["cinema_ids", "date_show"])]:
        op = pred.copy()
        for inds in E.groupby(keys).indices.values():
            ok = inds[pred[inds] > 0]
            if not len(ok):
                continue
            ratio = y[ok]/pred[ok]
            wt = w["scale_fd"][ok]*pred[ok]/s[ok]
            order = np.argsort(ratio)
            k = ratio[order][np.searchsorted(np.cumsum(wt[order]), wt.sum()/2)]
            op[inds] *= k
        oracle[name] = np.average(np.abs(y-op)/s, weights=w["scale_fd"])
    oracle["know_true_zeros_only"] = np.average(np.where(y == 0, 0, e), weights=w["scale_fd"])
    assert all(v <= oracle["baseline"] + 1e-10 for v in oracle.values())
    pd.Series(oracle).to_csv(out / "diagnostic_oracles.csv", header=["scale_fd_mase"])
    profiles = []
    for name, a in [("train_sim", X), ("test", T)]:
        profiles.append({"split": name, "rows": len(a), "base_films": base_title(a.movie_title).nunique(),
                         "pairs": len(a[KEY].drop_duplicates()), "scale_median": a.scale.median(),
                         "small_share": a.scale.le(20).mean(), "first_day3_share": a.first_day.eq(3).mean(),
                         "occupation_median": a.occ_mean.median(), "d1_min": str(a.d1.min()), "d1_max": str(a.d1.max())})
    pd.DataFrame(profiles).to_csv(out / "profile.csv", index=False)
    seg = pd.Series("normal", index=T.index)
    for name, a, b in [("xmas", "2025-12-20", "2026-01-04"), ("ramadan", "2026-02-19", "2026-03-20"), ("lebaran", "2026-03-21", "2026-03-27")]:
        seg[T.date_show.between(a,b)] = name
    changes = []
    for v in range(2,10):
        diff = np.abs(subs[v]-subs[v-1])/T.scale.to_numpy()
        changes.append({"version": v, "mean_scaled_change": diff.mean(), "max_ticket_change": np.max(np.abs(subs[v]-subs[v-1]))})
    pd.DataFrame(changes).to_csv(out / "submission_changes.csv", index=False)
    ST = T.assign(segment=seg, r8=subs[8]/T.scale, r9=subs[9]/T.scale, change=np.abs(subs[9]-subs[8])/T.scale)
    ST.groupby("segment").agg(rows=("id","size"), r8=("r8","mean"), r9=("r9","mean"), mean_scaled_change=("change","mean")).to_csv(out / "submission_segments.csv")
    # Separate broad title matching from a genuine labelled overlap.
    join_key = KEY+["date_show"]
    overlap = d["test"].merge(pd.concat([d["train"],d["hist"]])[join_key], on=join_key)
    metadata = {}
    for name, a in [("train", X),("test",T)]:
        titles = base_title(a.movie_title).drop_duplicates()
        metadata[name] = {"titles": len(titles), "unmatched": titles[~titles.isin(d["movies"].original_title)].tolist()}
    # Are official talent fields usable without a contemporary external source?
    mv = d["movies"].set_index("original_title")
    train_titles = base_title(d["train"].movie_title).unique()
    test_titles = base_title(T.movie_title).unique()
    talent = []
    for field in ["producer", "director", "writer", "casts"]:
        split = lambda series: series.fillna("").str.split(",").map(lambda xs: {t.strip().lower() for t in xs if t.strip()})
        seen = set().union(*split(mv.reindex(train_titles)[field]))
        hit = split(mv.reindex(test_titles)[field]).map(lambda xs: bool(xs & seen))
        talent.append({"field":field, "test_title_coverage":hit.mean(), "test_row_coverage":base_title(T.movie_title).map(hit).mean()})
    pd.DataFrame(talent).to_csv(out / "talent_coverage.csv", index=False)
    # A zero is not necessarily a permanent withdrawal. Use complete seven-day targets only.
    daily = X.pivot(index=KEY, columns="h", values="total_ticket").dropna()
    positive = daily.to_numpy() > 0
    zero_then_return = ((~positive[:, :-1]) & np.maximum.accumulate(positive[:, ::-1], axis=1)[:, ::-1][:, 1:]).any(axis=1)
    lifecycle = {"complete_pairs":len(daily), "any_zero_pairs":int((~positive).any(axis=1).sum()),
                 "zero_then_return_pairs":int(zero_then_return.sum())}
    raw = {k: {"rows":len(d[k]), "missing_cells":int(d[k].isna().sum().sum()),
               "duplicate_keys":int(d[k].duplicated(join_key).sum())} for k in ["train","hist","test"]}
    summary = {"raw":raw, "lifecycle":lifecycle, "unsupported_test_weight_cells":unsupported, "target_label_overlap":len(overlap), "bootstrap_fixed_weight_film_95pct":uncertainty,
               "metadata":metadata, "top5_error_share":float(G.we.nlargest(5).sum()/G.we.sum()),
               "top10_error_share":float(G.we.nlargest(10).sum()/G.we.sum()), "oracles":oracle,
               "target_gap_from_recorded_v8_LB":.39991-.34456}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plt = style()
    fig, ax = plt.subplots(1,3,figsize=(15,4))
    sc = pd.DataFrame(scores).query("component == 'oof_final'").set_index("version")
    sc[["plain","scale","scale_fd"]].plot(ax=ax[0],marker="o")
    ax[0].set(title="Same OOF predictions, different weights",ylabel="MASE",xlabel="Version")
    G.we.nlargest(10).sort_values().plot.barh(ax=ax[1])
    ax[1].set(title="v8: largest film error contributions",xlabel="Contribution to scale × first-day MASE")
    pd.Series(oracle).plot.barh(ax=ax[2])
    ax[2].set(title="Truth-assisted diagnostics, NOT forecasts",xlabel="Weighted MASE")
    fig.tight_layout(); fig.savefig(out / "audit.png")
    print(pd.DataFrame(scores).query("component == 'oof_final'").round(6).to_string(index=False))
    print(json.dumps(summary,indent=2))
    print(f"Audit written to {out}")


if __name__ == "__main__":
    main()
