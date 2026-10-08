"""EDA 110: profil D1-D3 film test vs jendela train (rekonstruksi D1 v8). Apakah aturan D1 panitia berbeda?"""
import numpy as np
import pandas as pd

src = open("temp/exp_040_v8port/pipeline_v8.py", encoding="utf-8").read()
exec(src[: src.index("# ---- cell 48 + 69")])  # train, hist, test, release_dates, simulate, hs, D_wide
bt = base_title


def prof(hh, label):
    x = hh.copy()
    d1 = x.groupby("movie_title").date_show.min()
    x["d"] = (x.date_show - x.movie_title.map(d1)).dt.days + 1
    f = x.groupby(["movie_title", "d"]).agg(t=("total_ticket", "sum"), nc=("cinema_ids", "nunique"), sh=("total_show", "sum")).unstack()
    out = pd.DataFrame({"t21": f["t"][2] / f["t"][1], "t32": f["t"][3] / f["t"][2], "nc1_3": f["nc"][1] / f["nc"][3],
                        "nc2_3": f["nc"][2] / f["nc"][3], "sh1_3": f["sh"][1] / f["sh"][3], "tps1_3": (f["t"][1] / f["sh"][1]) / (f["t"][3] / f["sh"][3])})
    dow = d1.dt.dayofweek.value_counts(normalize=True).sort_index().round(2).to_dict()
    # pasangan: berapa porsi pasangan D3 yang tidak punya transaksi D1
    pr = x.pivot_table(index=["movie_title", "cinema_ids"], columns="d", values="total_ticket", aggfunc="sum")
    no_d1 = pr[pr[3].notna()][1].isna().mean()
    print(f"\n== {label}: {len(out)} film")
    print(out.median().round(3).to_dict())
    print("D1 dow:", dow, "| pasangan aktif D3 tanpa D1:", round(no_d1, 3))
    return out


prof(hs, "train (D1 rekonstruksi v8, rilis luas)")
prof(hist, "test_history (D1 panitia)")
# alternatif: D1 train dimajukan 1 hari (hari pertama ada transaksi signifikan sebelum rilis luas)
