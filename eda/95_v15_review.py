"""v15 review: OOF on identical rows vs v8/v12/v13 (caveat: v15 uses release-week cohort folds, older
versions film folds, so v15 is evaluated on a harder split), within-v15 control gain vs older control
gains, and how close the v15 test predictions are to each submitted version. Saved files only."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, style
from evaluate import test_weights_fd

PUBLIC = {8: .39991, 12: .41060, 13: .40244}


def main():
    out = fig_dir("95_v15_review")
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    y, s = x.total_ticket.values, x.scale.values
    rows, O = [], {}
    for v, ctrl in [(8, "oof_lgbmix"), (12, "oof_lgbmix"), (13, "oof_lgbmix"), (15, "comp_lgb")]:
        o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / f"results/v{v}/oof.csv"), on=KEY + ["h"], how="left")
        O[v] = o
        tw = lambda p: np.average(np.abs(y - p) / s, weights=w)
        rows.append(dict(version=f"v{v}", folds="cohort (week)" if v == 15 else "film", TW_final=tw(o.oof_final.values),
                         TW_lightgbm_control=tw(o[ctrl].values), gain_vs_control=tw(o.oof_final.values) - tw(o[ctrl].values),
                         public=PUBLIC.get(v)))
    R = pd.DataFrame(rows).set_index("version")
    pd.set_option("display.width", 200)
    print(R.round(4).to_string())
    leb = t.date_show.between("2026-03-21", "2026-03-27").values
    S = {v: t[["id"]].merge(pd.read_csv(ROOT / f"results/v{v}/submission.csv"), on="id").total_ticket.values for v in (8, 12, 13, 15)}
    ts = t.scale.values
    D = pd.DataFrame({f"v{a}": {f"v{b}": np.mean(np.abs(S[a] - S[b])[~leb] / ts[~leb]) for b in S} for a in S}).round(4)
    print("\nmean |prediction difference| / scale on non-Lebaran test rows:"); print(D.to_string())
    print("Lebaran rows identical to v13:", bool(np.allclose(S[15][leb], S[13][leb])))
    sh = pd.DataFrame({f"v{v}": dict(zero=np.mean(S[v][~leb] / ts[~leb] < 1e-9), mean_r=np.mean(S[v][~leb] / ts[~leb])) for v in S}).T
    print(sh.round(4).to_string())
    R.to_csv(out / "versions.csv"); D.to_csv(out / "test_distance.csv")
    plt = style()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(D.values, cmap="Blues"); ax.set_xticks(range(len(D)), D.columns); ax.set_yticks(range(len(D)), D.index)
    for (i, j), v_ in np.ndenumerate(D.values):
        ax.text(j, i, f"{v_:.3f}", ha="center", va="center", fontsize=8)
    ax.grid(False); ax.set(title="Test prediction distance (mean |diff| / scale)")
    fig.tight_layout(); fig.savefig(out / "test_distance.png")


if __name__ == "__main__":
    main()
