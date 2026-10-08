"""Cocokkan klaim tanggal rilis eksternal dengan D1 hasil rekonstruksi train."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

candidates = pd.read_csv(OUT / "all_candidate_titles.csv", parse_dates=["D1"])
claims = pd.read_csv(HERE / "verified_sources.csv", parse_dates=["source_release_date"])
assert claims["base"].isin(candidates["base"]).all(), "Judul sumber tidak ada pada kandidat"

audit = claims.merge(candidates, on="base", how="left", validate="many_to_one")
audit["difference_days"] = (audit["source_release_date"] - audit["D1"]).dt.days
audit["match"] = audit["difference_days"].eq(0)
audit = audit.sort_values(["D1", "base", "source_release_date"])
audit.to_csv(OUT / "source_claim_audit.csv", index=False)

films = audit.groupby(["base", "D1", "month", "lead_days", "lead_group"], observed=True).agg(
    claims=("match", "size"), matching_claims=("match", "sum"),
    distinct_source_dates=("source_release_date", "nunique"),
).reset_index()
films["status"] = "all_match"
films.loc[films["matching_claims"].eq(0), "status"] = "all_disagree"
films.loc[(films["matching_claims"] > 0) &
          (films["matching_claims"] < films["claims"]), "status"] = "conflict"
films.to_csv(OUT / "film_summary.csv", index=False)

print("films by status:\n", films.status.value_counts().to_string())
print("claims matching:", int(audit.match.sum()), "/", len(audit))
print("months:", sorted(films.month.unique().tolist()))
print("lead groups:\n", films.lead_group.value_counts().to_string())
print("preview lead median:", films.lead_days.median(), "max:", films.lead_days.max())
print("conflicts:\n", films[films.status != "all_match"].to_string(index=False))
