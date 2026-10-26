"""Coach-free player talent per season, from the coach_effects model:
career-level O/D + the league experience curve at that player's experience.
Used by rotation.py (who SHOULD have played) and development.py.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
STINTS = ROOT / "data" / "stints"
EXP_CAP = 15
H = [f"h{i}" for i in range(5)]
A = [f"a{i}" for i in range(5)]


def first_seen(s):
    long = pd.DataFrame({"p": s[H + A].to_numpy("int64").ravel(), "y": np.repeat(s.season.to_numpy(), 10)})
    return long.groupby("p").y.min()


def talent_table(s):
    """DataFrame indexed by (season, player_id) with O, D, net (points per 100, coach-free)."""
    pc = pd.read_csv(OUT / "player_career.csv").set_index("player_id")
    ex = pd.read_csv(OUT / "experience_curve.csv").set_index(["censored", "exp"])
    fs = first_seen(s)
    long = pd.DataFrame({"player_id": s[H + A].to_numpy("int64").ravel(),
                         "season": np.repeat(s.season.to_numpy(), 10)}).drop_duplicates()
    f = long.player_id.map(fs)
    long["exp"] = np.minimum(long.season - f, EXP_CAP)
    long["censored"] = (f == 1996).astype(int)
    e = ex.reindex(pd.MultiIndex.from_arrays([long.censored, long.exp]))
    long["O"] = long.player_id.map(pc.O).to_numpy() + e.O.to_numpy()
    long["D"] = long.player_id.map(pc.D).to_numpy() + e.D.to_numpy()
    long["net"] = long.O + long.D
    long["first_seen"] = f.to_numpy()
    return long.set_index(["season", "player_id"]).sort_index()


def lookup(t, season, pid, col="net"):
    idx = pd.MultiIndex.from_arrays([np.asarray(season, "int64"), np.asarray(pid, "int64")])
    return t[col].reindex(idx).to_numpy()
