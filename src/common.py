"""Shared helpers: data loading, D1 reconstruction, test-like sample simulation, MASE."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
KEY = ["movie_title", "cinema_ids"]
H = range(4, 11)
TRAIN_END = pd.Timestamp("2025-09-30")
# reporting outages in train (eda/08): <70% of clusters reported, 13 Jun missing entirely
OUTAGE = pd.to_datetime(["2025-06-07", "2025-06-09", "2025-06-10", "2025-06-13", "2025-06-16"])


PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = "Blues"
DIV = "RdBu_r"


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 110, "savefig.bbox": "tight",
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "axes.prop_cycle": matplotlib.cycler(color=PAL), "lines.linewidth": 2, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


def load():
    rd = lambda f, **k: pd.read_csv(DATA / f, **k)
    d = dict(
        train=rd("train.csv", parse_dates=["date_show"]),
        hist=rd("test_history.csv", parse_dates=["date_show"]),
        test=rd("test.csv", parse_dates=["date_show"]),
        movies=rd("movies.csv"),
        hol=rd("holidays.csv", parse_dates=["date"]),
        price=rd("ticket_prices.csv"),
    )
    return d


def fig_dir(name):
    p = OUT / "eda" / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def base_title(t: pd.Series) -> pd.Series:
    """Strip format suffix so IMAX/3D versions of one movie share a group."""
    return t.str.replace(r"\s*\((IMAX 2D|IMAX 3D|3D|4DX|SCREENX)\)\s*$", "", regex=True).str.strip()


def fmt(t: pd.Series) -> pd.Series:
    f = t.str.extract(r"\((IMAX 2D|IMAX 3D|3D)\)\s*$")[0]
    return f.fillna("2D")


def release_dates(tx: pd.DataFrame, frac=0.5, min_nc=25) -> pd.Series:
    """D1 per film = first date whose cinema coverage reaches `frac` of peak coverage, searched only
    inside the continuous screening segment that contains the peak (a paid-preview weekend followed by
    a gap, e.g. A MINECRAFT MOVIE 4-6 Apr vs release 9 Apr, is skipped - rule from v2).

    Coverage is measured on the base title (2D+3D+IMAX share one D1, like test_history). Base titles
    opening in < `min_nc` cinemas are dropped: test only contains wide releases.
    """
    x = tx.assign(base=base_title(tx.movie_title))
    nc = x.groupby(["base", "date_show"]).cinema_ids.nunique()
    out = {}
    for b, s in nc.groupby(level=0):
        s = s.droplevel(0)
        s = s.reindex(pd.date_range(s.index.min(), s.index.max()), fill_value=0)
        pk = s.idxmax()
        z = s[:pk][s[:pk] == 0]
        seg = s[z.index.max() + pd.Timedelta(days=1):] if len(z) else s
        c = seg[seg >= frac * s.max()]
        if len(c) and c.iloc[0] >= min_nc:
            out[b] = c.index[0]
    t = x.drop_duplicates("movie_title").set_index("movie_title").base
    return t.map(pd.Series(out, dtype="datetime64[ns]")).dropna().rename("d1")


def simulate(tx: pd.DataFrame, d1: pd.Series, last_date=TRAIN_END, bad=OUTAGE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (history, target) exactly like the organiser built test_history / test.

    history = every transaction of the film on D1..D3.
    target  = pairs with a transaction on D3, x 7 days D4..D10, zero-filled.
    Films whose D10 is after `last_date` are dropped (incomplete target). Films with an outage
    date inside D1..D3 are dropped (corrupted scale/selection); target rows on outage dates are
    dropped (false zeros).
    """
    d1 = d1[d1 + pd.Timedelta(days=9) <= last_date]
    d1 = d1[~np.any([(d1 <= b) & (d1 + pd.Timedelta(days=2) >= b) for b in bad], axis=0)] if len(bad) else d1
    x = tx.merge(d1, left_on="movie_title", right_index=True)
    x["d"] = (x.date_show - x.d1).dt.days + 1
    hist = x[x.d.between(1, 3)].drop(columns="d1")
    pairs = hist.loc[hist.d == 3, KEY].drop_duplicates().merge(d1, left_on="movie_title", right_index=True)
    tgt = pairs.loc[pairs.index.repeat(7)].copy()
    tgt["d"] = np.tile(np.arange(4, 11), len(pairs))
    tgt["date_show"] = tgt.d1 + pd.to_timedelta(tgt.d - 1, unit="D")
    y = x[x.d.between(4, 10)][KEY + ["date_show", "total_ticket"]]
    tgt = tgt.merge(y, on=KEY + ["date_show"], how="left").fillna({"total_ticket": 0})
    tgt = tgt[~tgt.date_show.isin(bad)]
    city = tx.drop_duplicates("cinema_ids").set_index("cinema_ids").city_name
    tgt["city_name"] = tgt.cinema_ids.map(city)
    return hist.drop(columns="d").reset_index(drop=True), tgt.drop(columns=["d1", "d"]).reset_index(drop=True)


def scale(hist: pd.DataFrame) -> pd.Series:
    return (hist.groupby(KEY).total_ticket.sum() / 3).clip(lower=1).rename("scale")


def mase(tgt: pd.DataFrame, pred, hist: pd.DataFrame) -> float:
    s = tgt.join(scale(hist), on=KEY).scale.values
    return float(np.mean(np.abs(tgt.total_ticket.values - np.asarray(pred)) / s))


if __name__ == "__main__":
    # self-check: simulate() on a toy film reproduces the D3 selection rule and zero-fill
    d = pd.date_range("2025-05-01", periods=12)
    tx = pd.DataFrame({
        "date_show": list(d[:12]) + [d[0], d[1]],
        "cinema_ids": ["A"] * 12 + ["B", "B"],
        "city_name": "X", "movie_title": "M",
        "total_ticket": list(range(1, 13)) + [5, 5], "occupation_rate": 1.0, "total_show": 1,
    })
    tx = tx.drop(index=5)  # A has no sale on D6 -> must become 0
    h, t = simulate(tx, pd.Series({"M": d[0]}, name="d1"))
    assert set(t.cinema_ids) == {"A"}, "B has no D3 sale -> excluded"
    assert len(t) == 7 and t.total_ticket.tolist() == [4, 5, 0, 7, 8, 9, 10]
    assert np.isclose(scale(h)[("M", "B")], 10 / 3) and np.isclose(scale(h)[("M", "A")], 2)
    assert mase(t, t.total_ticket, h) == 0
    print("common.py self-check OK")
