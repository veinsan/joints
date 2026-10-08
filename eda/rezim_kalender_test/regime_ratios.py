"""EDA 111: efek rezim kalender (Ramadan, Natal-Tahun Baru, Lebaran) dari test_history (paket resmi) vs pola train.
Rasio harian level film D2/D1, D3/D2 dibagi rasio kalender v8; residual = efek yang tidak ditangkap kalender v8."""
import numpy as np
import pandas as pd

src = open("temp/exp_040_v8port/pipeline_v8.py", encoding="utf-8").read()
ns = {}
exec(src[: src.index("def release_dates")], ns)
exec(src[src.index("# ---- cell 57: kalender ----"): src.index("CAL = calendar(hol)")] + "CAL = calendar(hol)", ns)
CAL, hol = ns["CAL"], ns["hol"]
print(hol[hol.date.between("2025-12-20", "2026-01-05") | hol.date.between("2026-02-14", "2026-02-22") |
          hol.date.between("2026-03-15", "2026-03-31")][["date", "day_tipe", "holiday_tipe", "holiday_name"]].to_string(index=False))


def film_ratios(df, label):
    bt = df.movie_title.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D)\)\s*$", "", regex=True).str.strip()
    x = df.assign(base=bt)
    d1 = x.groupby("base").date_show.min()
    x["d"] = (x.date_show - x.base.map(d1)).dt.days + 1
    g = x[x.d <= 3].groupby(["base", "d"]).total_ticket.sum().unstack()
    g = g[(g[1] > 300)]
    D1 = d1.reindex(g.index)
    c = np.stack([CAL.cal.reindex(D1 + pd.Timedelta(days=k)).values for k in range(3)], 1)
    out = pd.DataFrame({"d1": D1.values, "dow": D1.dt.dayofweek.values, "r21": (g[2] / g[1]).values,
                        "r32": (g[3] / g[2]).values, "res21": np.log((g[2] / g[1]).values / (c[:, 1] / c[:, 0])),
                        "res32": np.log((g[3] / g[2]).values / (c[:, 2] / c[:, 1]))}, index=g.index)
    out["set"] = label
    return out


tr = film_ratios(pd.read_csv("data/train.csv", parse_dates=["date_show"]).query("date_show >= '2025-04-08'"), "train")
te = film_ratios(pd.read_csv("data/test_history.csv", parse_dates=["date_show"]), "test")
ref = tr.groupby("dow")[["res21", "res32"]].median()
te = te.join(ref, on="dow", rsuffix="_ref")
te["ex21"], te["ex32"] = np.exp(te.res21 - te.res21_ref), np.exp(te.res32 - te.res32_ref)
print("\nrujukan train (median residual log per dow D1):\n", ref.round(3))
pd.set_option("display.width", 200)
print(te.sort_values("d1")[["d1", "dow", "r21", "r32", "ex21", "ex32"]].round(2).to_string())
te.to_csv("eda/rezim_kalender_test/film_ratios_test.csv")
