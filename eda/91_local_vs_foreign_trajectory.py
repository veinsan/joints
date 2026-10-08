"""Do Indonesian films keep their audience longer than foreign films, beyond what v13 already predicts?

The missing information is the film trajectory after D3 (eda/84). Domain knowledge: local films have
word-of-mouth 'legs', imported films are front-loaded. There is no origin column, so a rule is derived
from the official movies.csv only:
  local = the title or the producer field contains Indonesian vocabulary / a producer name that
          co-occurs with Indonesian titles in the data itself (bootstrapped once, rule printed).
Then: v13 OOF residual log((y+1)/(pred+1)) by horizon for local vs foreign films, film-level
bootstrap, and the share of local films in train vs test.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from common import KEY, base_title, fig_dir, load, style
from evaluate import test_weights_fd

ID_WORDS = set("""dan yang di ke dari untuk dengan tanpa cinta rumah ibu bapak malam surga hati anak kita aku kau mama papa
jangan sampai pulang mati hantu iblis setan santet pesugihan kuntilanak pocong siksa kubur dosa nenek kakek janda suami istri
tuhan doa sajadah mualaf kafir hijrah nikah mertua warisan wasiat tumbal darah kelam legenda petaka suro jumat kliwon
lantai lagu sore pagi hujan masa depan terakhir pertama sekarang nanti sukses harga naik mendengarku benarkah kupilih patah
jembatan tukar takdir rangga perempuan laki tanah air budi balas teman tegar sejati juara cuan setannya pelangi dusun mayit
gerbang sukma selimut musuh tawa duka suka penunggu buto ijo dijemput penerbangan yasinan esok tak kembali gadis pengantin
bunda abadi bahagia rindu sahabat saudara keluarga kampung desa kota jalan pintu kamar ruang sekolah guru murid""".split())


def is_local_title(t):
    words = re.findall(r"[a-z]+", t.lower())
    return sum(w in ID_WORDS for w in words) >= 1


def main():
    out = fig_dir("91_local_trajectory")
    D = load()
    mv = D["movies"].copy()
    mv["title_local"] = mv.original_title.map(is_local_title)
    # producers that appear with an Indonesian title anywhere in movies.csv mark their other films local too
    prod = mv.assign(p=mv.producer.fillna("").str.split(",")).explode("p")
    prod["p"] = prod.p.str.strip()
    local_prod = set(prod[prod.title_local & (prod.p != "-") & (prod.p != "")].p)
    mv["local"] = mv.title_local | prod.groupby(level=0).p.apply(lambda s: bool(set(s) & local_prod))
    loc = dict(zip(mv.original_title, mv.local))
    x = pd.read_parquet(ROOT / "outputs/cache/Xtr.parquet")
    t = pd.read_parquet(ROOT / "outputs/cache/Xte.parquet")
    w = test_weights_fd(x, t)
    o = x[KEY + ["h"]].merge(pd.read_csv(ROOT / "results/v13/oof.csv"), on=KEY + ["h"], how="left")
    x["base"] = base_title(x.movie_title)
    x["local"] = x.base.map(loc).fillna(False).astype(bool)
    t["local"] = base_title(t.movie_title).map(loc).fillna(False).astype(bool)
    x["lr"] = np.log((x.total_ticket + 1) / (o.oof_final.values + 1))
    x["e"] = np.abs(x.total_ticket - o.oof_final.values) / x.scale
    x["w"] = w
    pd.set_option("display.width", 200)
    print(f"local producers found: {len(local_prod)}; films local: train {x.drop_duplicates('base').local.mean():.2f} "
          f"({x.drop_duplicates('base').local.sum()}/{x.base.nunique()}), test {t.drop_duplicates('movie_title').local.mean():.2f}; "
          f"rows: train {x.local.mean():.2f}, test {t.local.mean():.2f}")
    print("\ntrain films classified local (first 25):", sorted(x[x.local].base.unique())[:25])
    print("train films classified foreign (first 25):", sorted(x[~x.local].base.unique())[:25])
    tab = x.groupby(["local", "h"]).apply(lambda q: np.average(q.lr, weights=q.w), include_groups=False).unstack()
    print("\nweighted mean residual log((y+1)/(pred+1)) by horizon (positive = model under-predicts):")
    print(tab.round(3).to_string())
    # film-level: slope of residual over horizon (late minus early) local vs foreign, bootstrap over films
    F = x.groupby(["base", "local"]).apply(lambda q: pd.Series({
        "early": np.average(q.lr[q.h <= 5], weights=q.w[q.h <= 5]) if (q.h <= 5).any() else np.nan,
        "late": np.average(q.lr[q.h >= 8], weights=q.w[q.h >= 8]) if (q.h >= 8).any() else np.nan}), include_groups=False).reset_index().dropna()
    F["slope"] = F.late - F.early
    rng = np.random.default_rng(2026)
    a, b = F[F.local].slope.values, F[~F.local].slope.values
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(5000)]
    print(f"\nfilm-level late-minus-early residual: local {a.mean():+.3f} (n={len(a)}), foreign {b.mean():+.3f} (n={len(b)}),"
          f" difference {a.mean() - b.mean():+.3f} 95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")
    print("TW-MASE local vs foreign:", {k: round(np.average(g.e, weights=g.w), 4) for k, g in x.groupby("local")})
    F.to_csv(out / "film_slopes.csv", index=False)
    plt = style()
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, col in [(True, "#e34948"), (False, "#2a78d6")]:
        ax.plot(tab.columns, tab.loc[k].values, marker="o", color=col, label="local" if k else "foreign")
    ax.axhline(0, color="k", lw=.8); ax.set(title="v13 residual by horizon, local vs foreign", xlabel="h"); ax.legend()
    fig.tight_layout(); fig.savefig(out / "local_trajectory.png")


if __name__ == "__main__":
    main()
