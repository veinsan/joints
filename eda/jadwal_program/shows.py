"""EDA 114: apakah jumlah show per pasangan tetap dalam satu minggu program? Bandingkan show D4..D10 dengan show D3."""
import numpy as np
import pandas as pd

src = open("temp/exp_040_v8port/pipeline_v8.py", encoding="utf-8").read()
exec(src[: src.index("# ---- cell 48 + 69")])
d1 = D_wide.drop(running, errors="ignore")
d1 = d1[d1 + pd.Timedelta(days=9) <= CFG.TRAIN_END]
x = train.merge(d1.rename("d1"), left_on="movie_title", right_index=True)
x["d"] = (x.date_show - x.d1).dt.days + 1
x = x[x.d.between(1, 10)]
sh = x.pivot_table(index=["movie_title", "cinema_ids"], columns="d", values="total_show", aggfunc="sum").reindex(columns=range(1, 11))
tk = x.pivot_table(index=["movie_title", "cinema_ids"], columns="d", values="total_ticket", aggfunc="sum").reindex(columns=range(1, 11))
sh = sh[sh[3].notna()].fillna(0)
tk = tk.reindex(sh.index).fillna(0)
dow1 = sh.index.get_level_values(0).map(d1).dayofweek
for dw, nm in ((2, "rilis Rabu"), (3, "rilis Kamis")):
    m = np.asarray(dow1 == dw)
    S = sh[m]
    print(f"\n== {nm}: {m.sum()} pasangan")
    print("  show(Dk) == show(D3):", {k: round(float((S[k] == S[3]).mean()), 3) for k in range(4, 11)})
    print("  |show(Dk)-show(D3)| <= 1:", {k: round(float(((S[k] - S[3]).abs() <= 1).mean()), 3) for k in range(4, 11)})
    print("  show(Dk) == 0:", {k: round(float((S[k] == 0).mean()), 3) for k in range(4, 11)})
# variabilitas: CV log tiket vs log okupansi-per-show antar hari D4-D7 untuk pasangan aktif
T = tk.loc[sh.index]
act = (sh[[4, 5, 6, 7]] > 0).all(axis=1)
tps = (T[[4, 5, 6, 7]] / sh[[4, 5, 6, 7]].replace(0, np.nan))[act]
print("\nD4-D7 pasangan aktif: korelasi antar pasangan log(tiket D4) dengan log(show D3 x tps D3):",
      round(float(np.corrcoef(np.log1p(T.loc[act, 4]), np.log1p(sh.loc[act, 3] * (T.loc[act, 3] / sh.loc[act, 3].replace(0, np.nan)).fillna(0)))[0, 1]), 3))
