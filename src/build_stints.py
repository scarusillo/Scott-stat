"""Rebuild on-court lineups from NBA play-by-play (stats.nba.com playbyplayv2
format) and collapse each game into STINTS: stretches with the same ten
players on the floor, with points and possessions for each side.

Usage:  python src/build_stints.py 1996 2024      (season start years, inclusive)

Input : data/raw/nbastats_<YYYY>.csv   (downloaded by src/fetch.py)
Output: data/stints/stints_<YYYY>.parquet, data/stints/players_<YYYY>.parquet,
        data/stints/coverage_<YYYY>.json

Conventions verified across 1996/2010/2023 files:
  PERSONTYPE 4/5 = home/away player, 2/3 = home/away team.
  SCORE is "away - home".
"""
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "stints"

MADE, MISS, FT, REB, TOV, FOUL, SUB, EJECT = 1, 2, 3, 4, 5, 6, 8, 11
# Foul action types that can be called on a bench player / team (technicals,
# delay of game, too many players). They say nothing about who is on the floor.
TECH_FOULS = {11, 12, 13, 16, 18, 19, 25, 30}
# Final free throws that END a possession when made (2 of 2, 3 of 3).
FT_END = {12, 15}
# Final free throws whose miss is a live rebound: 1 of 1 (and-one), 2 of 2, 3 of 3.
FT_LIVE_MISS = {10, 12, 15}
SIDE = {4: "H", 2: "H", 5: "A", 3: "A"}
COLS = ["GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "EVENTMSGACTIONTYPE", "PERIOD",
        "PCTIMESTRING", "HOMEDESCRIPTION", "VISITORDESCRIPTION", "SCORE",
        "PERSON1TYPE", "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ID",
        "PERSON2TYPE", "PLAYER2_ID", "PLAYER2_NAME", "PLAYER2_TEAM_ID",
        "PERSON3TYPE", "PLAYER3_ID", "PLAYER3_NAME", "PLAYER3_TEAM_ID"]


def clock(s):
    m, sec = str(s).split(":")
    return int(m) * 60 + int(sec)


def people(e):
    """(player_id, side) for each player slot of an event."""
    out = []
    for k in (1, 2, 3):
        t = getattr(e, f"PERSON{k}TYPE")
        pid = getattr(e, f"PLAYER{k}_ID")
        if t in (4, 5) and pid and pid < 1_000_000_000:
            out.append((int(pid), SIDE[t]))
    return out


def presence_event(e):
    if e.EVENTMSGTYPE in (EJECT, 9, 12, 13, 18):
        return False
    if e.EVENTMSGTYPE == FOUL and e.EVENTMSGACTIONTYPE in TECH_FOULS:
        return False
    if e.EVENTMSGTYPE == FT and e.EVENTMSGACTIONTYPE in (16, 21, 22):
        return False  # technical FTs can be shot by anyone on the floor; skip to be safe
    return True


def period_starters(evs):
    """Players on the floor at the start of a period: anyone who does something
    (or is subbed OUT) before being subbed IN during that period."""
    starters = {"H": [], "A": []}
    subbed_in = set()
    for e in evs:
        if e.EVENTMSGTYPE == SUB:
            # PLAYER1 leaves, PLAYER2 enters
            if e.PERSON1TYPE in (4, 5):
                pid, s = int(e.PLAYER1_ID), SIDE[e.PERSON1TYPE]
                if pid not in subbed_in and pid not in starters[s]:
                    starters[s].append(pid)
            if e.PERSON2TYPE in (4, 5):
                subbed_in.add(int(e.PLAYER2_ID))
            continue
        if not presence_event(e):
            continue
        for pid, s in people(e):
            if pid not in subbed_in and pid not in starters[s]:
                starters[s].append(pid)
    return starters


def process_game(gid, g):
    # File row order, NOT EVENTNUM: substitutions are often logged with a later
    # EVENTNUM than the dead-ball they happened in, but sit in the right row.
    g = g.sort_values("PERIOD", kind="stable")
    home_team = away_team = None
    for e in g.itertuples(index=False):
        for k in (1, 2, 3):
            t = getattr(e, f"PERSON{k}TYPE")
            tid = getattr(e, f"PLAYER{k}_TEAM_ID")
            if t == 4 and pd.notna(tid):
                home_team = int(tid)
            elif t == 5 and pd.notna(tid):
                away_team = int(tid)
        if home_team and away_team:
            break

    stints, bad_periods, n_periods = [], 0, 0
    score_h = score_a = 0
    for period, pe in g.groupby("PERIOD", sort=True):
        evs = list(pe.itertuples(index=False))
        n_periods += 1
        st = period_starters(evs)
        ok = len(st["H"]) == 5 and len(st["A"]) == 5
        if not ok:
            bad_periods += 1
        on = {"H": set(st["H"]), "A": set(st["A"])}
        cur = None
        last_miss = None  # (side, is_and_one)

        def open_stint(t):
            return {"pts_H": 0, "pts_A": 0, "poss_H": 0, "poss_A": 0,
                    "t0": t, "t1": t, "H": sorted(on["H"]), "A": sorted(on["A"]),
                    "valid": ok and len(on["H"]) == 5 and len(on["A"]) == 5}

        def close(c):
            if c and c["valid"] and (c["poss_H"] + c["poss_A"] > 0 or c["t0"] != c["t1"]):
                stints.append(c)

        cur = open_stint(clock(evs[0].PCTIMESTRING))
        for e in evs:
            t = clock(e.PCTIMESTRING)
            et, act = e.EVENTMSGTYPE, e.EVENTMSGACTIONTYPE
            side = SIDE.get(e.PERSON1TYPE)
            if et == SUB and e.PERSON1TYPE in (4, 5):
                cur["t1"] = t
                close(cur)
                s = SIDE[e.PERSON1TYPE]
                on[s].discard(int(e.PLAYER1_ID))
                if e.PERSON2TYPE in (4, 5):
                    on[s].add(int(e.PLAYER2_ID))
                cur = open_stint(t)
                continue
            # scoring: credit the change in the running score to the current stint
            if isinstance(e.SCORE, str) and " - " in e.SCORE:
                a, h = (int(x) for x in e.SCORE.split(" - "))
                cur["pts_H"] += h - score_h
                cur["pts_A"] += a - score_a
                score_h, score_a = h, a
            # possession ends
            if et == MADE and side:
                cur[f"poss_{side}"] += 1
                last_miss = None
            elif et == MISS and side:
                last_miss = (side, False)
            elif et == TOV and side:
                cur[f"poss_{side}"] += 1
                last_miss = None
            elif et == FT and side:
                desc = (e.HOMEDESCRIPTION if isinstance(e.HOMEDESCRIPTION, str)
                        else e.VISITORDESCRIPTION) or ""
                missed = "MISS" in desc
                if not missed and act in FT_END:
                    cur[f"poss_{side}"] += 1
                    last_miss = None
                elif missed and act in FT_LIVE_MISS:
                    last_miss = (side, act == 10)
            elif et == REB and side and last_miss:
                ms, and_one = last_miss
                if side != ms and not and_one:
                    cur[f"poss_{ms}"] += 1  # defensive rebound ends shooter's possession
                last_miss = None
            cur["t1"] = t
        close(cur)

    rows = []
    for i, c in enumerate(stints):
        r = {"game_id": int(gid), "stint": i, "home_team": home_team, "away_team": away_team,
             "pts_h": c["pts_H"], "pts_a": c["pts_A"], "poss_h": c["poss_H"], "poss_a": c["poss_A"],
             "seconds": c["t0"] - c["t1"]}
        for j, p in enumerate(c["H"]):
            r[f"h{j}"] = p
        for j, p in enumerate(c["A"]):
            r[f"a{j}"] = p
        rows.append(r)
    return rows, n_periods, bad_periods, (score_h, score_a)


def process_season(year):
    src = RAW / f"nbastats_{year}.csv"
    d = pd.read_csv(src, usecols=COLS, low_memory=False)
    for k in (1, 2, 3):
        d[f"PLAYER{k}_ID"] = pd.to_numeric(d[f"PLAYER{k}_ID"], errors="coerce").fillna(0).astype("int64")
        d[f"PERSON{k}TYPE"] = pd.to_numeric(d[f"PERSON{k}TYPE"], errors="coerce").fillna(0).astype(int)
    rows, periods, bad = [], 0, 0
    for gid, g in d.groupby("GAME_ID"):
        r, n, b, _ = process_game(gid, g)
        rows += r
        periods += n
        bad += b
    s = pd.DataFrame(rows)
    s.insert(0, "season", year)
    OUT.mkdir(parents=True, exist_ok=True)
    s.to_parquet(OUT / f"stints_{year}.parquet", index=False)

    # player names + team per game (for minutes/roster lookups later)
    pl = []
    for k in (1, 2, 3):
        x = d[d[f"PERSON{k}TYPE"].isin([4, 5])][["GAME_ID", f"PLAYER{k}_ID", f"PLAYER{k}_NAME", f"PLAYER{k}_TEAM_ID"]]
        x.columns = ["game_id", "player_id", "name", "team_id"]
        pl.append(x)
    pl = pd.concat(pl).dropna().drop_duplicates(["game_id", "player_id"])
    pl["team_id"] = pl["team_id"].astype("int64")
    pl.insert(0, "season", year)
    pl.to_parquet(OUT / f"players_{year}.parquet", index=False)

    games = d.GAME_ID.nunique()
    cov = {"season": year, "games": int(games), "periods": periods, "bad_periods": bad,
           "stints": len(s), "poss_per_team_game": float((s.poss_h.sum() + s.poss_a.sum()) / 2 / games),
           "pts_per_team_game": float((s.pts_h.sum() + s.pts_a.sum()) / 2 / games),
           "seconds_per_game": float(s.seconds.sum() / games)}
    (OUT / f"coverage_{year}.json").write_text(json.dumps(cov, indent=1))
    return cov


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    years = [y for y in range(a, b + 1) if (RAW / f"nbastats_{y}.csv").exists()]
    with Pool(min(4, len(years))) as p:
        for c in p.imap_unordered(process_season, years):
            print(json.dumps(c))
