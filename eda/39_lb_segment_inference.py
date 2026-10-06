"""39 - What do the six REAL leaderboard scores say about the Lebaran level? (inference, no new submission)

v4 -> v5 changed almost only the 6.1% Lebaran-week rows (analog override, rho = 0.5): mean |v5 - v4| / s on those
rows contributes 0.070 to the MASE, the LB moved -0.0394. If the truth were above v5 on every Lebaran row, the gain
would be the full 0.070; if it sat at v4, it would be +0.070. So the observed gain pins down where the truth lies
relative to the two predictions. Truth model on Lebaran rows:
    T = m * v5 * exp(sf * film_effect + sp * row_noise - (sf^2 + sp^2) / 2)
For each (m, sf, sp) the expected LB change is simulated and compared with the observed Lebaran part
(-0.0394 minus the non-Lebaran change measured by the calibrated kappa-world validation, -0.0016).
Also: which m minimises the expected Lebaran MASE of a rescaled override, and what LB gain it would imply.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import OUT, base_title, fig_dir, style

plt = style()
F = fig_dir("39_lb_inference")
R = Path(__file__).resolve().parents[1] / "results"
Xte = pd.read_parquet(OUT / "cache" / "Xte.parquet")
s = Xte.scale.values
leb = Xte.date_show.between("2026-03-21", "2026-03-27").values
v4 = pd.read_csv(R / "v4/submission.csv").total_ticket.values
v5 = pd.read_csv(R / "v5/submission.csv").total_ticket.values
N, NL = len(s), leb.sum()
films = base_title(Xte.movie_title[leb]).values
uf, fi = np.unique(films, return_inverse=True)
obs = (0.40865 - 0.44809) - (0.4066 - 0.4082)
print(f"Lebaran rows {NL} ({NL / N:.3f}), films {len(uf)}; observed Lebaran-part LB change {obs:+.4f}")
print(f"max possible gain (truth >= v5 everywhere) {-(np.abs(v5 - v4)[leb] / s[leb]).sum() / N:+.4f}")

a4, a5, sl = v4[leb], v5[leb], s[leb]
rng = np.random.default_rng(2026)
ms = np.round(np.arange(0.3, 2.51, 0.05), 2)
rows = []
for sf in (0.2, 0.4, 0.6):
    for sp in (0.3, 0.5, 0.7):
        for m in ms:
            dl, best = [], []
            for rep in range(40):
                T = m * a5 * np.exp(sf * rng.standard_normal(len(uf))[fi] + sp * rng.standard_normal(NL) - (sf ** 2 + sp ** 2) / 2)
                dl.append(((np.abs(T - a5) - np.abs(T - a4)) / sl).sum() / N)
                best.append([(np.abs(T - k * a5) / sl).sum() / N for k in (0.6, 0.8, 1.0, 1.25, 1.5)])
            rows.append(dict(sf=sf, sp=sp, m=m, d_lb=np.mean(dl), d_sd=np.std(dl), **{f"mase_x{k}": v for k, v in zip((0.6, 0.8, 1.0, 1.25, 1.5), np.mean(best, 0))}))
G = pd.DataFrame(rows)
G["fit"] = np.abs(G.d_lb - obs)
cons = G.loc[G.groupby(["sf", "sp"]).fit.idxmin()]
print("\n=== multiplier m (truth / v5 level) most consistent with the observed LB change ===")
print(cons[["sf", "sp", "m", "d_lb", "d_sd"]].round(4).to_string(index=False))
ok = G[np.abs(G.d_lb - obs) <= 2 * G.d_sd + 0.002]
print(f"\nm range within 2 sd (+0.002 split noise), all noise settings: {ok.m.min()} - {ok.m.max()}")
print("\ncontribution of Lebaran rows to the total MASE if the override is rescaled by k (at the consistent m):")
print(cons[["sf", "sp", "m"] + [c for c in G.columns if c.startswith("mase_x")]].round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 4))
for (sf, sp), g in G.groupby(["sf", "sp"]):
    ax.plot(g.m, g.d_lb, lw=1, label=f"sf={sf}, sp={sp}")
ax.axhline(obs, color="#e34948", ls="--", label="observed")
ax.set(xlabel="truth multiplier m relative to v5", ylabel="expected LB change v4->v5 (Lebaran part)", title="Lebaran level implied by LB v4 -> v5")
ax.legend(fontsize=7, ncol=2)
fig.savefig(F / "lebaran_m.png")
print(f"figures -> {F}")
