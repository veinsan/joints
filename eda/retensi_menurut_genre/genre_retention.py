"""EDA: apakah selisih retensi D3 pasangan lemah bertahan dalam genre film?"""

from pathlib import Path
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)
films = pd.read_csv(ROOT / "eda/retensi_per_film/output/by_film.csv")
meta = pd.read_csv(ROOT / "data/movies.csv")
meta["base"] = meta.original_title.str.strip()
meta["primary_genre"] = meta.genre.fillna("unknown").str.split(",").str[0].str.strip()
meta = meta.drop_duplicates("base")[["base", "primary_genre", "age_rating"]]
f = films.merge(meta, on="base", how="left")
f["primary_genre"] = f.primary_genre.fillna("unknown")
f["age_rating"] = f.age_rating.fillna("unknown")
f.to_csv(OUT / "film_with_metadata.csv", index=False)


def aggregate(g):
    return pd.Series({"n_films": len(g), "n_low_d2_pairs": int(g.n.sum()),
                      "weighted_retention": float(g.selected.sum() / g.n.sum()),
                      "median_film_retention_nge10": float(g.loc[g.n.ge(10), "rate"].median()),
                      "n_films_ge10": int(g.n.ge(10).sum())})


by_genre = f.groupby(["period", "primary_genre"]).apply(aggregate, include_groups=False).reset_index()
by_genre.to_csv(OUT / "by_primary_genre.csv", index=False)
by_rating = f.groupby(["period", "age_rating"]).apply(aggregate, include_groups=False).reset_index()
by_rating.to_csv(OUT / "by_age_rating.csv", index=False)

wide = by_genre.pivot(index="primary_genre", columns="period",
                      values=["n_films", "n_low_d2_pairs", "weighted_retention"])
wide.columns = [f"{a}_{b}" for a, b in wide.columns]
wide = wide.dropna()
well = wide[(wide.n_low_d2_pairs_train >= 50) & (wide.n_low_d2_pairs_test >= 50)].copy()
well["gap"] = well.weighted_retention_test - well.weighted_retention_train
well.to_csv(OUT / "shared_genre_gap.csv")

summary = {"n_film_rows": len(f),
           "missing_metadata_film_rows": int(f.primary_genre.eq("unknown").sum()),
           "metadata_coverage_by_period": {p: float(1 - x.primary_genre.eq("unknown").mean())
                                          for p, x in f.groupby("period")},
           "well_supported_genres": len(well),
           "well_supported_genres_positive_gap": int(well.gap.gt(0).sum()),
           "well_supported_genres_negative_gap": int(well.gap.lt(0).sum()),
           "well_supported_genres_test_pair_coverage": float(well.n_low_d2_pairs_test.sum() /
                                                            by_genre.loc[by_genre.period.eq("test"), "n_low_d2_pairs"].sum()),
           "well_supported_genres_train_pair_coverage": float(well.n_low_d2_pairs_train.sum() /
                                                             by_genre.loc[by_genre.period.eq("train"), "n_low_d2_pairs"].sum())}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
print("\nShared genres:\n", well.round(3).to_string())
print("\nRatings:\n", by_rating.round(3).to_string(index=False))
