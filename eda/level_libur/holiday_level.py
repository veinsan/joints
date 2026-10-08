"""EDA 117: level pasar hari libur/cuti bersama di train vs asumsi kalender v8 (libur = max(DOW, 1.29)).
Rasio aktual = tiket pasar hari H / rata-rata hari dengan dow sama dalam +-14 hari (bukan libur, bukan outage)."""
import numpy as np
import pandas as pd

src = open("temp/exp_040_v8port/pipeline_v8.py", encoding="utf-8").read()
ns = {}
exec(src[: src.index("def release_dates")], ns)
exec(src[src.index("# ---- cell 57: kalender ----"): src.index("CAL = calendar(hol)")] + "CAL = calendar(hol)", ns)
CAL, train, CFG = ns["CAL"], ns["train"], ns["CFG"]
day = train.groupby("date_show").total_ticket.sum()
bad = set(pd.to_datetime(pd.date_range("2025-06-01", "2025-06-16")))
special = CAL[(CAL.is_hol == 1)].index
rows = []
for d in special:
    if d not in day.index or d < pd.Timestamp("2025-04-08") or d in bad or d > pd.Timestamp("2025-09-30"):
        continue
    nb = [d + pd.Timedelta(days=7 * k) for k in (-2, -1, 1, 2)]
    nb = [x for x in nb if x in day.index and x not in bad and CAL.is_hol.get(x, 0) == 0]
    if len(nb) < 2:
        continue
    act = day[d] / np.mean([day[x] for x in nb])
    cal_ratio = CAL.cal[d] / CFG.DOW_PROF[d.dayofweek]
    rows.append(dict(tanggal=d.date(), hari=d.day_name()[:3], aktual=round(act, 2), v8_kalender=round(cal_ratio, 2), rasio=round(act / cal_ratio, 2)))
R = pd.DataFrame(rows)
print(R.to_string(index=False))
print("median rasio aktual/v8 (hari kerja libur):", R[~R.hari.isin(["Sat", "Sun"])].rasio.median())
t = pd.read_csv("data/test.csv", parse_dates=["date_show"])
th = CAL[(CAL.is_hol == 1) & (CAL.index >= "2025-10-01")]
print("\nlibur/cuti di periode test:", [(d.date(), d.day_name()[:3]) for d in th.index])
m = t.date_show.isin(th.index) & ~t.date_show.between("2026-03-21", "2026-03-27")
print("baris target test di hari libur (di luar Lebaran):", int(m.sum()), f"({m.mean():.1%})")
