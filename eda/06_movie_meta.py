"""06 - Movie metadata: does genre / rating / origin / format explain the D4-D10 retention?"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from common import KEY, base_title, fig_dir, fmt, load, release_dates, scale, simulate, style

plt = style()
F = fig_dir("06_movie")
d = load()
tr, th, mv = d["train"], d["hist"], d["movies"]

# Indonesian-origin heuristic: Indonesian function/content words in the title, or Religi genre.
ID_WORDS = set("""DI DAN YANG KE DARI AKU KAU KAMU DIA IBU BAPAK MAMA PAPA AYAH ANAK CINTA HATI SURGA NERAKA SETAN
SANTET POCONG KUNTILANAK HANTU RUMAH MALAM TANPA JANGAN SAMPAI UNTUK DENGAN INI ITU ADA BELUM SEBELUM TERAKHIR
PERTAMA DOSA MATI KEMATIAN PANGGIL TUMBAL DARAH MAYIT DUSUN ALAS PENUNGGU PETAKA SATU SURO NENEK KAFIR NIKAH
PELAMINAN TAKDIR TUKAR KELAM LEGENDA HIJRAH SAJADAH MUALAF BIDADARI TAWA DUKA SUKA MUSUH SELIMUT UANG TEMAN
BALAS BUDI JUARA SEJATI SENIN HARGA NAIK PELANGI TUNGGU SUKSES NANTI TITIP BUNDA ANTARA ASRAMA PUTRI AGAK LAEN
PENGABDI PABRIK GULA KOMANG JUMBO SORE ISTRI MASA DEPAN PENERBANGAN ESOK GERBANG SUKMA KUYANK TUHAN BENARKAH
MENDENGARKU WASIAT WARISAN MERTUA NGERI KALI MENGEJAR RESTU JEMBATAN ORANG BAIK PULANG JALAN BUKIT PERJALANAN
DUKUN KERAMAT GUNDIK PENCARIAN KANG SIKSA KUBUR TEGAR LUKA LAUT SELATAN HOROR SEREM MUNDUR MAJU AIR MATA""".split())


def is_local(title, genre):
    w = set(re.findall(r"[A-Z]+", title.upper()))
    return int(len(w & ID_WORDS) > 0 or "Religi" in str(genre))


meta = mv.set_index("original_title")
D1 = release_dates(tr)
hs, ts = simulate(tr, D1)
ts = ts.join(scale(hs), on=KEY)
film = ts.groupby("movie_title").apply(lambda g: g.total_ticket.sum() / (g.scale.sum() * 7 / 7), include_groups=False)
# film retention = total D4-D10 tickets / (7/3 * total D1-D3 tickets) -> 1 means flat
tot13 = hs.groupby("movie_title").total_ticket.sum()
tot410 = ts.groupby("movie_title").total_ticket.sum()
f = pd.DataFrame({"ret": (tot410 / 7) / (tot13 / 3)}).dropna()
f["base"] = base_title(pd.Series(f.index, index=f.index))
f["format"] = fmt(pd.Series(f.index, index=f.index))
f = f.join(meta, on="base")
f["genre1"] = f.genre.str.split(", ").str[0]
f["local"] = [is_local(t, g) for t, g in zip(f.base, f.genre)]
print(f"films: {len(f)}  with metadata: {f.genre.notna().mean():.2%}")
print("\nretention = mean daily D4-10 / mean daily D1-3 (film level)")
print("overall quantiles:", f.ret.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
for col in ["format", "age_rating", "local", "genre1"]:
    t = f.groupby(col).ret.agg(["count", "median"]).sort_values("count", ascending=False)
    print(f"\n-- by {col} --\n{t[t['count'] >= 3].round(3).to_string()}")
g = f.assign(g=f.genre.str.split(", ")).explode("g").groupby("g").ret.agg(["count", "median"])
print("\n-- any-genre flag --\n", g[g["count"] >= 5].sort_values("median").round(3).to_string())

# Test-side coverage of metadata and local heuristic
tt = pd.Series(th.movie_title.unique())
tb = base_title(tt)
print(f"\nTEST: films with metadata {tb.isin(meta.index).mean():.2%}; local share "
      f"{np.mean([is_local(t, meta.genre.get(t)) for t in tb]):.2%}; TRAIN local share {f.local.mean():.2%}")
print("TEST genre1 mix:", meta.reindex(tb.unique()).genre.str.split(', ').str[0].value_counts().head(6).to_dict())
print("TRAIN genre1 mix:", f.drop_duplicates('base').genre1.value_counts().head(6).to_dict())

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
top = g[g["count"] >= 5].sort_values("median")
ax[0].barh(top.index, top["median"], color="#2a78d6")
ax[0].set(title="Median film retention by genre flag", xlabel="mean D4-10 / mean D1-3")
data = [f[f.local == k].ret.clip(upper=3) for k in [0, 1]]
ax[1].boxplot(data); ax[1].set_xticks([1, 2], ["foreign", "local (heuristic)"])
ax[1].set(title="Retention: local vs foreign", ylabel="ratio (clip 3)")
fig.savefig(F / "meta.png")
print(f"figures -> {F}")
