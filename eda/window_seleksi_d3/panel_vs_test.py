"""EDA 004: panel D1-D10 dari train, bandingkan dengan test, baseline MASE.

python eda/window_seleksi_d3/panel_vs_test.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_windows import build_windows, test_windows  # noqa: E402

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
D = Path("data")
tr = pd.read_csv(D / "train.csv", parse_dates=["date_show"])
th = pd.read_csv(D / "test_history.csv", parse_dates=["date_show"])
te = pd.read_csv(D / "test.csv", parse_dates=["date_show"])
res = {}


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


wide, tgt, pre = build_windows(tr)
tw = test_windows(th)
p("train_films", int(wide.movie_title.nunique()))
p("train_pairs", int(len(wide)))
p("test_films", int(tw.movie_title.nunique()))
p("test_pairs", int(len(tw)))
# cek test_windows cocok dengan pasangan test.csv
tp = te[["movie_title", "cinema_ids"]].drop_duplicates()
p("test_windows_match_test_pairs", int(len(tw.merge(tp))) == len(tp) == len(tw))

# perbandingan distribusi
def desc(s):
    return s.describe(percentiles=[.1, .25, .5, .75, .9]).round(2).to_dict()


for c in ["scale", "total_ticket_d1", "total_ticket_d3", "occupation_rate_d1", "occupation_rate_d3", "total_show_d1", "total_show_d3"]:
    p(f"train.{c}", desc(wide[c]))
    p(f"test.{c}", desc(tw[c]))
for name, w in [("train", wide), ("test", tw)]:
    days = (w[["total_ticket_d1", "total_ticket_d2", "total_ticket_d3"]] > 0).sum(axis=1)
    p(f"{name}.days_present", days.value_counts(normalize=True).sort_index().round(3).to_dict())
    p(f"{name}.pairs_per_film", desc(w.groupby("movie_title").size()))
    p(f"{name}.D1_dow", w.drop_duplicates("movie_title").D1.dt.day_name().value_counts().to_dict())
    p(f"{name}.frac_scale_eq1", round(float((w.scale <= 1).mean()), 4))
    p(f"{name}.ratio_d3_d1_median", round(float(((w.total_ticket_d3 + 1) / (w.total_ticket_d1 + 1)).median()), 3))
    fmt = w.movie_title.str.extract(r"\(([^)]*)\)\s*$")[0].fillna("REG")
    p(f"{name}.format_share", fmt.value_counts(normalize=True).round(3).to_dict())
p("test.D1_month", tw.drop_duplicates("movie_title").D1.dt.to_period("M").astype(str).value_counts().sort_index().to_dict())
p("train.D1_month", wide.drop_duplicates("movie_title").D1.dt.to_period("M").astype(str).value_counts().sort_index().to_dict())

# target: zero rate dan rasio per horizon
tgt = tgt.merge(wide[["movie_title", "cinema_ids", "scale", "total_ticket_d1", "total_ticket_d2", "total_ticket_d3"]])
tgt["r"] = tgt.total_ticket / tgt.scale
p("target.zero_rate_by_h", tgt.groupby("d").total_ticket.apply(lambda s: (s == 0).mean()).round(3).to_dict())
p("target.r_median_by_h", tgt.groupby("d").r.median().round(3).to_dict())
p("target.r_mean_by_h", tgt.groupby("d").r.mean().round(3).to_dict())
p("target.zero_rate_all", round(float((tgt.total_ticket == 0).mean()), 4))

# baseline MASE
def mase(y, yhat, s):
    return float(np.mean(np.abs(y - yhat) / s))


base = {}
base["zero"] = mase(tgt.total_ticket, 0, tgt.scale)
base["scale(mean d1-3)"] = mase(tgt.total_ticket, tgt.scale, tgt.scale)
base["d3"] = mase(tgt.total_ticket, tgt.total_ticket_d3, tgt.scale)
med_h = tgt.groupby("d").r.median()
base["scale*median_r_h"] = mase(tgt.total_ticket, tgt.scale * tgt.d.map(med_h), tgt.scale)
tgt["dow"] = tgt.date_show.dt.dayofweek
tgt["d1dow"] = tgt.D1.dt.dayofweek
med_hd = tgt.groupby(["d", "d1dow"]).r.median()
base["scale*median_r_h_d1dow(in-sample)"] = mase(tgt.total_ticket, tgt.scale * pd.Series(list(zip(tgt.d, tgt.d1dow))).map(med_hd).values, tgt.scale)
p("baseline_mase(in-sample)", {k: round(v, 4) for k, v in base.items()})

# plot kurva median rasio
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for dow, g in tgt.groupby("d1dow"):
    m = g.groupby("d").r.median()
    ax[0].plot(m.index, m.values, marker="o", label=f"D1 dow={dow} n={g.movie_title.nunique()}")
ax[0].set_title("median y/scale per horizon, per hari D1"); ax[0].legend(fontsize=7)
ax[1].hist(np.log10(wide.scale), bins=60, alpha=.5, density=True, label="train windows")
ax[1].hist(np.log10(tw.scale), bins=60, alpha=.5, density=True, label="test")
ax[1].set_title("log10 scale"); ax[1].legend()
plt.tight_layout(); plt.savefig(OUT / "rasio_horizon_scale.png", dpi=110); plt.close()

wide.to_parquet(OUT / "train_windows_wide.parquet")
tgt.to_parquet(OUT / "train_windows_target.parquet")
tw.to_parquet(OUT / "test_windows_wide.parquet")
(OUT / "panel_vs_test.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
