"""Self-check of the v10 selection rules (run: .venv/bin/python notebooks/check_v10_selection.py).

Mirrors the two decision snippets in build_v10.py and asserts the review scenarios of 2026-10-05:
combined config must beat the best passing single by >= ABL_MIN_GAIN, and the FM blend weight is the best TW
among weights that pass every condition (not the global best TW checked afterwards).
"""
from pathlib import Path

import pandas as pd

src = (Path(__file__).parent / "build_v10.py").read_text()
assert "SELECTED = min(simpler, key=lambda c: L.loc[c, 'TW']) if simpler else best" in src
assert "w_star = float(BW[BW['pass']].TW.idxmin())" in src
REF, A1, A2, CC = "B1 fold-local stats", "A1 + show/attendance", "A2 + official metadata", "C combined"
COMPLEXITY = {REF: 0, A1: 1, A2: 1, CC: 2}
MIN_GAIN, FM_MIN_GAIN, KAPPA_TOL = 0.0015, 0.001, 0.002


def select(tw, cands):
    L = pd.DataFrame({"TW": tw})
    best = L.loc[cands, "TW"].idxmin()
    simpler = [c for c in cands if COMPLEXITY[c] < COMPLEXITY[best] and L.loc[c, "TW"] - L.loc[best, "TW"] < MIN_GAIN]
    return min(simpler, key=lambda c: L.loc[c, "TW"]) if simpler else best


for tw, exp in [({REF: .330, A1: .310, A2: .325, CC: .320}, A1), ({REF: .330, A1: .310, A2: .325, CC: .305}, CC),
                ({REF: .330, A1: .310, A2: .325, CC: .309}, A1)]:
    assert select(tw, [A1, A2, CC]) == exp
base = {"TW": .330, "temporal_mean": .340, "kappa": .345}
BW = pd.DataFrame({"TW": [.330, .328, .326, .324, .322, .325], "temporal_mean": [.340, .339, .338, .337, .342, .343],
                   "kappa": [.345] * 6}, index=pd.Index([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]))
ok = (BW.index > 0) & (base["TW"] - BW.TW >= FM_MIN_GAIN) & (BW.temporal_mean <= base["temporal_mean"]) & (BW.kappa - base["kappa"] <= KAPPA_TOL)
assert float(BW[ok].TW.idxmin()) == 0.3
print("v10 selection checks passed")
