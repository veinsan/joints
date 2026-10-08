"""Audit tanggal D1 test terhadap klaim publik dengan konteks lokal/global."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
candidates = pd.read_csv(OUT / "test_titles.csv", parse_dates=["D1"])
sources = pd.read_csv(HERE / "verified_sources.csv", parse_dates=["source_release_date"])
assert sources.base.isin(candidates.base).all()
audit = sources.merge(candidates, on="base", validate="many_to_one")
audit["difference_days"] = (audit.source_release_date - audit.D1).dt.days
audit["match"] = audit.difference_days.eq(0)
audit = audit.sort_values(["D1", "base", "source_release_date"])
audit.to_csv(OUT / "source_claim_audit.csv", index=False)

film = audit.groupby(["base", "D1", "month"], observed=True).agg(
    claims=("match", "size"), matching_claims=("match", "sum"),
    distinct_dates=("source_release_date", "nunique"),
).reset_index()
film["status"] = "all_match"
film.loc[film.matching_claims.eq(0), "status"] = "all_disagree"
film.loc[(film.matching_claims > 0) & (film.matching_claims < film.claims), "status"] = "conflict"
film.to_csv(OUT / "film_summary.csv", index=False)
print("films:", len(film), "months:", film.month.nunique())
print(film.status.value_counts().to_string())
print(film.to_string(index=False))
