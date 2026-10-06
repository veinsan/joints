"""38 - Is the film-level error predictable? (residual screen at the film level)

eda/37: knowing the true film x horizon ratio would cut TW-MASE 0.385 -> 0.330, and per-film signed error of
the v6 OOF shows whole films over/under-shooting together (SORE x2.6, SAYAP SAYAP PATAH 2 x2.9, CONJURING x0.54).
If OUT-OF-FOLD film-level log-ratios log(sum y / sum pred) can be predicted from film features in grouped CV,
the pair models under-fit film-level effects (143 films drowned in 56k pair rows) and a film-level correction
layer is a real lever. Screens existing + new film features:
  cal-adjusted national trend   fp3/fp1 divided by cal(D3)/cal(D1)   (word-of-mouth proxy)
  origin (Indonesian / Korean / Indian / other) from movies.csv title + cast heuristics
  reissue flag                  '(REISSUE)' in title
Then applies the CV-predicted correction (shrunk) to every pair and re-scores TW-MASE.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import OUT, base_title, fig_dir, load, style
from evaluate import test_weights
from features import dataset, calendar

plt = style()
F = fig_dir("38_film_resid")
Xtr, Xte, Xlim = dataset()
d = load()
o = pd.read_csv(Path(__file__).resolve().parents[1] / "results/v7/oof.csv")
assert (o.movie_title.values == Xtr.movie_title.values).all() and (o.h.values == Xtr.h.values).all()
pred, y, s = o.oof_final.values, Xtr.total_ticket.values, Xtr.scale.values
w_te = test_weights(Xtr, Xte)
fold = o.fold.values
tw = lambda p: float(np.average(np.abs(y - p) / s, weights=w_te))

ID_WORDS = set("dan di yang dari untuk ke sang nya kita aku kau mu cinta setan iblis rumah jalan malam pulang anak ibu ayah " \
               "tumbal santet dukun hantu kuntilanak pocong pesan doa jiwa hati waktu tak ingin usai sini bukit perempuan " \
               "pembawa sial sihir panggil kubur gerbang kitab mayit ritual maut terakhir pencarian bertaut rindu semua " \
               "impian berhak dirayakan selamanya lebih andai menikah dengan sayap patah istri masa depan".split())


def origin(row):
    t = row.original_title.lower()
    cast = str(row.casts)
    words = set(re.findall(r"[a-z]+", t))
    if re.search(r"\b(kapoor|khan|kumar|singh|malhotra|roshan|bachchan|sharma|chopra|padukone|shetty|devgn|kaif|advani)\b", cast.lower()):
        return "india"
    if len(re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+-[A-Za-z]+\b", cast)) >= 2:
        return "korea"
    if words & ID_WORDS or re.search(r"(nya|kan)\b", t):
        return "indonesia"
    return "other"


mv = d["movies"].copy()
mv["origin"] = mv.apply(origin, axis=1)
org = mv.set_index("original_title").origin
cal = calendar(d["hol"]).cal

g = base_title(Xtr.movie_title).values
D = pd.DataFrame({"base": g, "h": Xtr.h.values, "y": y, "p": pred, "s": s, "w": w_te})
film = Xtr.assign(base=g).groupby("base").first()
film["origin"] = film.index.map(org).fillna("other")
film["reissue"] = film.index.str.contains("REISSUE").astype(int)
film["cal_trend"] = np.log((film.fp3 + .01) / (film.fp1 + .01)) - np.log(cal.reindex(film.d1 + pd.Timedelta(days=2)).values /
                                                                          cal.reindex(film.d1).values)
film["d2_trend"] = np.log((film.fp2 + .01) / (film.fp1 + .01)) - np.log(cal.reindex(film.d1 + pd.Timedelta(days=1)).values /
                                                                         cal.reindex(film.d1).values)
print("origin counts (train films):", film.origin.value_counts().to_dict())
print("examples:", {k: list(film[film.origin == k].index[:4]) for k in film.origin.unique()})

agg = D.groupby("base").agg(y=("y", "sum"), p=("p", "sum"), n=("y", "size"))
film["lr"] = np.log((agg.y + 50) / (agg.p + 50)).reindex(film.index)
film["fold"] = pd.Series(fold, index=Xtr.index).groupby(g).first().reindex(film.index)
print(f"\nfilm log-ratio sd {film.lr.std():.3f}, mean {film.lr.mean():.3f}")

cands = ["fp1", "fp2", "fp3", "cal_trend", "d2_trend", "f_logT", "f_nc_trend", "f_occ", "f_tps3", "f_sh_r31", "occ_rel",
         "tps_rel", "logT_rel", "film_pc_vs_mkt", "d1_dow", "d1_month", "rating", "comp_n", "comp_logT", "comp_maxT",
         "f_share_cohort", "mk_logT", "fnc3", "n_cast", "n_genre", "reissue"] + [c for c in film.columns if c.startswith("g_")]
rows = [(c, *spearmanr(film[c], film.lr, nan_policy="omit")) for c in cands]
S = pd.DataFrame(rows, columns=["feature", "rho", "p"]).sort_values("p")
print("\n=== Spearman(film feature, OOF film log-ratio) ===\n", S.head(15).round(3).to_string(index=False))
print("\nlog-ratio by origin:\n", film.groupby("origin").lr.agg(["mean", "std", "size"]).round(3))

# grouped-CV prediction of the film log-ratio (same film folds as the base model)
film["origin_c"] = film.origin.astype("category").cat.codes
feats = cands + ["origin_c"]
P = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=8, subsample=0.8, subsample_freq=1,
         colsample_bytree=0.7, reg_lambda=5, verbose=-1, random_state=2026, deterministic=True, force_row_wise=True)
lr_hat = np.zeros(len(film))
for k in range(5):
    tr_, te_ = film.fold != k, film.fold == k
    m = lgb.LGBMRegressor(**P).fit(film.loc[tr_, feats], film.loc[tr_, "lr"], sample_weight=np.sqrt(agg.n.reindex(film.index)[tr_]))
    lr_hat[te_.values] = m.predict(film.loc[te_, feats])
film["lr_hat"] = lr_hat
print(f"\nCV film log-ratio: corr(pred, true) = {np.corrcoef(film.lr, film.lr_hat)[0, 1]:.3f}, "
      f"R2 = {1 - ((film.lr - film.lr_hat) ** 2).sum() / ((film.lr - film.lr.mean()) ** 2).sum():.3f}")

res = {"base (v6 OOF)": tw(pred)}
for sh in (0.25, 0.5, 0.75, 1.0):
    adj = np.exp(sh * (film.lr_hat - film.lr.mean())).reindex(g).values
    res[f"film correction x{sh}"] = tw(pred * adj)
res["oracle film ratio"] = tw(pred * np.exp(film.lr).reindex(g).values)
print("\n=== TW-MASE after film-level correction (CV-predicted) ===\n", pd.Series(res).round(4))

fig, ax = plt.subplots(1, 3, figsize=(17, 4))
ax[0].hist(film.lr, bins=30, color="#2a78d6"); ax[0].set(title="OOF film log-ratio  log(sum y / sum pred)")
ax[1].scatter(film.cal_trend, film.lr, s=12, color="#2a78d6"); ax[1].set(xlabel="cal-adjusted D1->D3 trend", ylabel="film log-ratio")
ax[2].scatter(film.lr_hat, film.lr, s=12, color="#eb6834"); ax[2].plot([-1, 1], [-1, 1], color="k", lw=.6)
ax[2].set(xlabel="CV-predicted", ylabel="true", title="film log-ratio predictability")
fig.savefig(F / "film_resid.png")
film[["origin", "lr", "lr_hat", "cal_trend"]].to_csv(OUT / "cache" / "film_resid.csv")
print(f"figures -> {F}")
