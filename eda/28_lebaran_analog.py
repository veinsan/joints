"""28 - Lebaran week (6.1% of test rows): top-down analog from Lebaran 2025, which IS in train.

The 7 Lebaran-2026 slate titles opened 18 Mar (D1-D3 = 18-20 Mar), so D4-D10 = 21-27 Mar = Lebaran day 1..7.
Train starts on 1 Apr 2025 = Lebaran 2025 day 2, so the cluster totals of the 2025 Lebaran week are observed.
  1. market level: 2025 Lebaran week vs normal; 2026 slate D1-D3 level
  2. implied ratio r = D4-D10 / scale for the slate if the 2026 Lebaran week resembles 2025, under a grid of
     (market growth g x slate share phi), nationally and per cluster
  3. physical sanity: seats. tickets = occupancy x seats; is the implied Lebaran occupancy feasible given the
     shows the slate already has and the occupancy the 2025 slate actually reached?
  4. what v4 predicts for the same rows
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, OUT, fig_dir, load, scale, style

plt = style()
F = fig_dir("28_lebaran")
d = load()
tr, th, te = d["train"], d["hist"], d["test"]
LEB25 = pd.Timestamp("2025-03-31")  # 1 Syawal 1446 H
LEB26 = pd.Timestamp("2026-03-21")  # 1 Syawal 1447 H (SKB 2026)

d1 = th.groupby("movie_title").date_show.min()
slate = d1[d1 == "2026-03-18"].index
hs = th[th.movie_title.isin(slate)]
print("slate:", list(slate))
day25 = tr.groupby("date_show").agg(tix=("total_ticket", "sum"), shows=("total_show", "sum"), nc=("cinema_ids", "nunique"))
day25["k"] = (day25.index - LEB25).days + 1
normal = day25.loc["2025-05-01":].tix.median()
print("\n=== 1. market level (tickets/day, all clusters) ===")
print(day25.loc["2025-04-01":"2025-04-10", ["k", "tix", "shows"]].assign(x_normal=lambda a: (a["tix"] / normal).round(2)).to_string())
s26 = hs.groupby("date_show").agg(tix=("total_ticket", "sum"), shows=("total_show", "sum"), occ=("occupation_rate", "mean"))
print("2026 slate D1-D3:\n", s26.assign(x_normal_2025=lambda a: (a["tix"] / normal).round(2)).round(1).to_string())
S26 = s26["tix"].mean()
print(f"normal 2025 day (May-Sep median) = {normal:.0f} | 2026 slate mean D1-D3 = {S26:.0f}")

print("\n=== 2. implied national r for the slate on Lebaran day k (2025 market x g x phi / slate scale) ===")
m25 = day25.set_index("k")["tix"]
m25.loc[1] = 0.75 * m25.loc[2]  # Lebaran day 1 (31 Mar 2025) is before the data; assumption: 75% of day 2
m25 = m25.sort_index()
grid = {(g, p): (g * p * m25.loc[1:7] / S26) for g in (0.6, 0.8, 1.0) for p in (0.7, 0.9)}
G = pd.DataFrame(grid).round(2)
G.index = [f"D{k + 3} (Lebaran day {k})" for k in G.index]
print(G.to_string())
print("mean over D4-D10:", {k: round(v.mean(), 2) for k, v in grid.items()})

print("\n=== 3. seat sanity ===")
top25 = tr[tr.date_show.between("2025-04-01", "2025-04-07")].groupby("movie_title").total_ticket.sum().nlargest(5).index
a = tr[tr.movie_title.isin(top25) & tr.date_show.between("2025-04-01", "2025-04-07")]
print("2025 Lebaran slate, 1-7 Apr: pair-day occupancy quantiles", a.occupation_rate.quantile([.25, .5, .75, .9]).round(1).to_dict(),
      "| tickets/show median", round((a.total_ticket / a.total_show).median(), 1))
print("2026 slate D1-D3: pair-day occupancy quantiles", hs.occupation_rate.quantile([.25, .5, .75, .9]).round(1).to_dict(),
      "| tickets/show median", round((hs.total_ticket / hs.total_show).median(), 1))
seats26 = (hs.total_ticket / (hs.occupation_rate / 100).clip(lower=.005)).groupby(hs.date_show).sum()
print(f"2026 slate seats offered per day (D1-D3): {seats26.round(0).to_dict()}")
for r_ in (1.0, 1.5, 2.0, 2.5):
    print(f"  r = {r_}: needs {r_ * S26:.0f} tickets/day = {100 * r_ * S26 / seats26.mean():.0f}% occupancy at the D1-D3 seat supply")
seat25 = (a.total_ticket / (a.occupation_rate / 100).clip(lower=.005)).groupby(a.date_show).sum()
print(f"2025 top-5 slate seats offered per day 1-7 Apr: mean {seat25.mean():.0f} (tickets {a.groupby('date_show').total_ticket.sum().mean():.0f})")

print("\n=== per-cluster analog: 2025 Lebaran-week cluster total / 2026 slate cluster scale ===")
c25 = tr[tr.date_show.between("2025-04-01", "2025-04-07")].groupby("cinema_ids").total_ticket.sum() / 7
c26 = hs.groupby("cinema_ids").total_ticket.sum() / 3
rc = (c25 / c26).dropna()
print("ratio quantiles (g = phi = 1):", rc.quantile([.1, .25, .5, .75, .9]).round(2).to_dict(), "| clusters", len(rc))
city = tr.drop_duplicates("cinema_ids").set_index("cinema_ids").city_name
big = city.isin(["JAKARTA", "TANGERANG", "BEKASI", "DEPOK", "BOGOR"])
print("median ratio Jabodetabek clusters:", round(rc[big.reindex(rc.index).fillna(False)].median(), 2),
      "| other cities:", round(rc[~big.reindex(rc.index).fillna(False)].median(), 2))

print("\n=== 4. what v4 predicts on the same rows ===")
sub = pd.read_csv(OUT.parent / "results/v4/submission.csv")
t = te.merge(sub, on="id").join(scale(th), on=KEY)
t = t[t.movie_title.isin(slate)]
t["r"] = t.total_ticket / t.scale
print("v4 mean r by date:", t.groupby("date_show").r.mean().round(2).to_dict())
print(f"v4 slate total tickets/day D4-D10: {t.groupby('date_show').total_ticket.sum().mean():.0f} vs slate D1-D3 {S26:.0f}"
      f" vs 2025 Lebaran-week market {m25.loc[2:8].mean():.0f}")
rc.rename("ratio").to_frame().to_parquet(OUT / "cache" / "lebaran_cluster_ratio.parquet")

fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
dd = day25.loc["2025-04-01":"2025-04-21", "tix"]
ax[0].bar(dd.index, dd.values, color="#2a78d6"); ax[0].axhline(normal, color="#e34948", ls="--", label="normal day (May-Sep median)")
ax[0].set(title="Train: national tickets/day after Lebaran 2025"); ax[0].legend(); ax[0].tick_params(axis="x", rotation=45)
x = np.arange(7)
ax[1].plot(x + 4, t.groupby("date_show").r.mean().values, marker="o", label="v4 prediction")
for (g, p), v in grid.items():
    if (g, p) in [(0.6, 0.7), (0.8, 0.9), (1.0, 0.9)]:
        ax[1].plot(x + 4, v.values, marker="s", ls="--", label=f"2025 analog g={g}, phi={p}")
ax[1].set(title="Slate ratio r = y / scale on D4-D10", xlabel="D"); ax[1].legend(fontsize=7)
ax[2].hist(rc.clip(upper=12), bins=30, color="#1baf7a"); ax[2].set(title="Per-cluster: 2025 Lebaran-week total / 2026 slate scale")
fig.savefig(F / "lebaran.png")
print(f"figures -> {F}")
