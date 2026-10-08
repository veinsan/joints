"""EDA 006: variansi film vs klaster, agregat nasional, efek klaster lintas film,
metadata film, rilis pesaing, stabilitas bulanan.

python eda/efek_film_dominan/film_klaster.py
Butuh output eda/window_seleksi_d3 (jalankan panel_vs_test.py dulu).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)
P = Path("eda/window_seleksi_d3/output")
wide = pd.read_parquet(P / "train_windows_wide.parquet")
tgt = pd.read_parquet(P / "train_windows_target.parquet")
tw = pd.read_parquet(P / "test_windows_wide.parquet")
mv = pd.read_csv("data/movies.csv")
res = {}


def p(k, v):
    res[k] = v
    print(f"{k}: {v}")


key = ["movie_title", "cinema_ids"]
tgt["zero"] = (tgt.total_ticket == 0).astype(int)
agg = tgt.groupby(key).agg(tix=("total_ticket", "sum"), n_zero=("zero", "sum")).reset_index()
w = wide.merge(agg)
w["r_mean"] = w.tix / 7 / w.scale
w["y"] = np.log1p(w.r_mean)


# 1. dekomposisi variansi (R2 fixed effect)
def r2_fe(df, cols, y="y"):
    pred = df.groupby(cols)[y].transform("mean")
    return 1 - ((df[y] - pred) ** 2).sum() / ((df[y] - df[y].mean()) ** 2).sum()


p("R2_film_FE", round(r2_fe(w, ["movie_title"]), 3))
p("R2_cinema_FE", round(r2_fe(w, ["cinema_ids"]), 3))
p("R2_city_FE", round(r2_fe(w, ["city_name"]), 3))
# dua arah: iterasi sederhana
yy = w.y - w.y.mean()
a = pd.Series(0.0, index=w.index); b = pd.Series(0.0, index=w.index)
for _ in range(20):
    a = (yy - b).groupby(w.movie_title).transform("mean")
    b = (yy - a).groupby(w.cinema_ids).transform("mean")
p("R2_film+cinema_FE", round(float(1 - ((yy - a - b) ** 2).sum() / (yy ** 2).sum()), 3))

# 2. agregat nasional (film level) dari D1-D3: tersedia juga di test
def film_agg(df):
    g = df.groupby("movie_title")
    f = pd.DataFrame({
        "nat_tix_d1": g.total_ticket_d1.sum(), "nat_tix_d3": g.total_ticket_d3.sum(),
        "nat_show_d1": g.total_show_d1.sum(), "nat_show_d3": g.total_show_d3.sum(),
        "n_cin_d1": g.total_ticket_d1.apply(lambda s: (s > 0).sum()), "n_pairs": g.size(),
    })
    f["nat_tix_trend"] = (f.nat_tix_d3 + 1) / (f.nat_tix_d1 + 1)
    f["nat_show_trend"] = (f.nat_show_d3 + 1) / (f.nat_show_d1 + 1)
    f["nat_tps_d3"] = f.nat_tix_d3 / f.nat_show_d3.clip(lower=1)
    f["footprint_growth"] = f.n_pairs / f.n_cin_d1.clip(lower=1)
    return f


fa = film_agg(w)
w = w.merge(fa, left_on="movie_title", right_index=True)
w["tix_trend"] = (w.total_ticket_d3 + 1) / (w.total_ticket_d1 + 1)
w["show_trend"] = (w.total_show_d3 + 1) / (w.total_show_d1 + 1)
w["share_d3"] = w.total_ticket_d3 / w.nat_tix_d3
cols = ["tix_trend", "show_trend", "nat_tix_trend", "nat_show_trend", "nat_tps_d3", "nat_tix_d3", "n_pairs",
        "footprint_growth", "share_d3", "scale"]
sp = w[cols + ["r_mean", "n_zero"]].corr(method="spearman")[["r_mean", "n_zero"]].loc[cols].round(3)
print("spearman pair-level\n", sp.to_string())
res["spearman_pair"] = sp.to_dict()

# 3. efek klaster lintas film (leave-one-film-out): klaster yang cenderung mempertahankan film
cl_sum = w.groupby("cinema_ids").agg(s=("y", "sum"), n=("y", "size"))
w = w.merge(cl_sum, left_on="cinema_ids", right_index=True)
w["cin_loo"] = (w.s - w.y) / (w.n - 1).replace(0, np.nan)
p("spearman_cinema_loo_vs_y", round(float(w[["cin_loo", "y"]].corr(method="spearman").iloc[0, 1]), 3))
# residual setelah film FE
w["y_res"] = w.y - w.groupby("movie_title").y.transform("mean")
cl_res = w.groupby("cinema_ids").agg(s2=("y_res", "sum"))
w = w.merge(cl_res, left_on="cinema_ids", right_index=True)
w["cin_res_loo"] = (w.s2 - w.y_res) / (w.n - 1).replace(0, np.nan)
p("spearman_cinema_resid_loo_vs_resid", round(float(w[["cin_res_loo", "y_res"]].corr(method="spearman").iloc[0, 1]), 3))

# 4. metadata film
mv2 = mv.rename(columns={"original_title": "base_title"})
w["base_title"] = w.movie_title.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True)
w = w.merge(mv2, on="base_title", how="left")
p("train_pairs_with_meta", round(float(w.genre.notna().mean()), 3))
tw["base_title"] = tw.movie_title.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D|REISSUE|4DX|SCREENX)\)\s*$", "", regex=True)
p("test_pairs_with_meta", round(float(tw.base_title.isin(mv.original_title).mean()), 3))
w["horror"] = w.genre.fillna("").str.contains("Horror").astype(int)
fl = w.groupby("movie_title").agg(r=("r_mean", "median"), nz=("n_zero", "mean"), horror=("horror", "first"),
                                  age=("age_rating", "first"), nat=("nat_tix_d3", "first"))
p("film_median_r_by_horror", fl.groupby("horror").r.median().round(3).to_dict())
p("film_mean_nzero_by_horror", fl.groupby("horror").nz.mean().round(3).to_dict())
p("film_median_r_by_age", fl.groupby("age").r.median().round(3).to_dict())
g1 = w.genre.fillna("").str.split(", ").explode()
p("top_genres_train", g1.value_counts().head(12).to_dict())

# 5. rilis pesaing: jumlah film window-eligible dengan D1 dalam (D3, D10]
d1s = wide.drop_duplicates("movie_title").set_index("movie_title").D1
comp = {m: int(((d1s > d + pd.Timedelta(days=2)) & (d1s <= d + pd.Timedelta(days=9))).sum()) for m, d in d1s.items()}
fl["n_new_release"] = fl.index.map(comp)
t = fl.groupby(pd.qcut(fl.n_new_release, 4, duplicates="drop")).agg(r=("r", "median"), nz=("nz", "mean"), n=("r", "size"))
print("rilis pesaing (film level)\n", t.round(3).to_string())
p("spearman_newrelease_vs_film_r", round(float(fl[["n_new_release", "r"]].corr(method="spearman").iloc[0, 1]), 3))

# 6. stabilitas per bulan D1
w["m"] = w.D1.dt.to_period("M").astype(str)
t = w.groupby("m").agg(r_med=("r_mean", "median"), zero=("n_zero", lambda s: s.mean() / 7), n_film=("movie_title", "nunique"),
                       scale_med=("scale", "median"), occ_d1=("occupation_rate_d1", "median"))
print("per bulan D1\n", t.round(3).to_string())
res["by_month"] = t.round(3).to_dict()

# 7. tes: agregat nasional test vs train (film level)
fat = film_agg(tw)
for c in ["nat_tix_trend", "nat_show_trend", "nat_tps_d3", "n_pairs", "footprint_growth"]:
    p(f"film.{c} train/test median", [round(float(fa[c].median()), 3), round(float(fat[c].median()), 3)])

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(np.log(w.nat_tix_trend), w.y, s=2, alpha=.3); ax[0].set_xlabel("log nat_tix_trend"); ax[0].set_ylabel("log1p r_mean")
ax[1].scatter(w.cin_res_loo, w.y_res, s=2, alpha=.3); ax[1].set_xlabel("klaster residual LOO"); ax[1].set_ylabel("residual pair")
plt.tight_layout(); plt.savefig(OUT / "film_klaster.png", dpi=110); plt.close()
(OUT / "film_klaster.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
