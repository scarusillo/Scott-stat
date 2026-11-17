"""Every shot (FGA and FTA) with the five players the shooter's team had on the
floor — so any player's shooting can be split by who he was playing next to.

Reuses build_stints' lineup reconstruction (period starters + substitutions).
Shots in periods whose starting lineup couldn't be rebuilt are dropped (~0.3%).

Usage: python src/shots.py 1996 2024    -> data/shots/shots_<YYYY>.parquet
  columns: season, game_id, team_id, shooter, kind (fg|ft), made, three, rim,
           assisted, pts, on0..on4 (shooter's team on floor, shooter included)
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from build_stints import COLS, FT, MADE, MISS, RAW, SIDE, SUB, period_starters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "shots"
TECH_FT = {16, 21, 22}
RIM_WORDS = ("Layup", "Dunk", "Tip", "Alley Oop", "Putback")


def game_shots(gid, g):
    g = g.sort_values("PERIOD", kind="stable")   # file order within period (see build_stints)
    rows = []
    for period, pe in g.groupby("PERIOD", sort=True):
        evs = list(pe.itertuples(index=False))
        st = period_starters(evs)
        if len(st["H"]) != 5 or len(st["A"]) != 5:
            continue
        on = {"H": set(st["H"]), "A": set(st["A"])}
        team = {}
        for e in evs:
            et = e.EVENTMSGTYPE
            if et == SUB and e.PERSON1TYPE in (4, 5):
                s = SIDE[e.PERSON1TYPE]
                on[s].discard(int(e.PLAYER1_ID))
                if e.PERSON2TYPE in (4, 5):
                    on[s].add(int(e.PLAYER2_ID))
                continue
            if et not in (MADE, MISS, FT) or e.PERSON1TYPE not in (4, 5):
                continue
            if et == FT and e.EVENTMSGACTIONTYPE in TECH_FT:
                continue
            s = SIDE[e.PERSON1TYPE]
            if len(on[s]) != 5:
                continue
            if pd.notna(e.PLAYER1_TEAM_ID):
                team[s] = int(e.PLAYER1_TEAM_ID)
            desc = (e.HOMEDESCRIPTION if isinstance(e.HOMEDESCRIPTION, str) else e.VISITORDESCRIPTION) or ""
            three = "3PT" in desc
            if et == FT:
                made = "MISS" not in desc
                rows.append((int(gid), team.get(s), int(e.PLAYER1_ID), "ft", made, False, False, False,
                             int(made), *sorted(on[s])))
            else:
                made = et == MADE
                rim = not three and any(w in desc for w in RIM_WORDS)
                assisted = made and e.PERSON2TYPE == e.PERSON1TYPE
                rows.append((int(gid), team.get(s), int(e.PLAYER1_ID), "fg", made, three, rim, assisted,
                             (3 if three else 2) * made, *sorted(on[s])))
    return rows


def season(year):
    d = pd.read_csv(RAW / f"nbastats_{year}.csv", usecols=COLS, low_memory=False)
    for k in (1, 2, 3):
        d[f"PLAYER{k}_ID"] = pd.to_numeric(d[f"PLAYER{k}_ID"], errors="coerce").fillna(0).astype("int64")
        d[f"PERSON{k}TYPE"] = pd.to_numeric(d[f"PERSON{k}TYPE"], errors="coerce").fillna(0).astype(int)
    rows = []
    for gid, g in d.groupby("GAME_ID"):
        rows += game_shots(gid, g)
    s = pd.DataFrame(rows, columns=["game_id", "team_id", "shooter", "kind", "made", "three", "rim",
                                    "assisted", "pts", "on0", "on1", "on2", "on3", "on4"])
    s["team_id"] = s.groupby(["game_id", "shooter"]).team_id.transform(lambda x: x.ffill().bfill())
    s.insert(0, "season", year)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_parquet(OUT / f"shots_{year}.parquet", index=False)
    return year, len(s)


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    with Pool(4) as p:
        for y, n in p.imap_unordered(season, range(a, b + 1)):
            print(y, n, flush=True)
