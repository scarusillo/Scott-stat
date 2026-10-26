"""Assign a head coach to every regular-season game, 1996-97 -> 2024-25.

Source: data/coaches_src/nba-coaches.csv — every NBA head-coaching tenure
(from Wikipedia's per-franchise lists, CC BY-SA 4.0), compiled in
github.com/rptetzloff/are-they-sports (PR #119). It gives each tenure's first/
last season and W-L, but no dates for mid-season changes.

Mid-season splits are solved from the records: a coach who went 12-30 before
being fired coached that team's first 42 games of that season. Tenures that
span several seasons are solved by elimination (sole-coach seasons are whole
seasons). Every tenure's W-L is then re-counted from our own game results and
compared with the source; mismatches are reported, not hidden.

Usage: python src/coaches.py   -> data/coaches/coach_games.parquet,
                                  out/coach_seasons.csv, out/coach_check.csv
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "coaches_src" / "nba-coaches.csv"
URL = ("https://raw.githubusercontent.com/rptetzloff/are-they-sports/refs/pull/119/head/"
       "data/reference/nba-coaches.csv")
STINTS = ROOT / "data" / "stints"
OUT = ROOT / "out"
FIRST, LAST = 1997, 2025   # season END years covered by our play-by-play

# Known source errors, keyed by source row. Each was found because our game
# results could not reproduce the source's W-L; fix the season fields only.
CORRECTIONS = {
    559: {"lastSeason": 1996},  # Paul Westphal: fired Jan 1996, not in 1996-97
    560: {"lastSeason": 1997},  # Cotton Fitzsimmons: 27-30 = 27-22 in 1995-96 + 0-8 in 1996-97
}

# NBA team_id (last 3 digits) -> franchise code used by the source file
FRANCHISE = {737: "ATL", 738: "BOS", 739: "CLE", 740: "NOP", 741: "CHI", 742: "DAL", 743: "DEN",
             744: "GSW", 745: "HOU", 746: "LAC", 747: "LAL", 748: "MIA", 749: "MIL", 750: "MIN",
             751: "BKN", 752: "NYK", 753: "ORL", 754: "IND", 755: "PHI", 756: "PHX", 757: "POR",
             758: "SAC", 759: "SAS", 760: "OKC", 761: "TOR", 762: "UTA", 763: "MEM", 764: "WAS",
             765: "DET", 766: "CHA"}


def team_games():
    rows = []
    for y in range(FIRST - 1, LAST):
        s = pd.read_parquet(STINTS / f"stints_{y}.parquet",
                            columns=["game_id", "home_team", "away_team", "final_margin"])
        g = s.groupby("game_id").first().reset_index()
        for side, sign in (("home_team", 1), ("away_team", -1)):
            rows.append(pd.DataFrame({"season": y + 1, "game_id": g.game_id, "team_id": g[side],
                                      "win": (sign * g.final_margin > 0).astype(int)}))
    d = pd.concat(rows)
    d["franchise"] = (d.team_id % 1000).map(FRANCHISE)
    return d.sort_values(["franchise", "season", "game_id"])  # ids restart each season (2YY#####)


def in_window_games(c):
    """G_in: games each tenure coached inside 1996-97..2024-25.
    The source snapshot predates 2026-27 games (current coaches' G = their
    games through 2025-26), and every 1968-1996 / 2025-26 season is 82 games."""
    c = c.copy()
    post_first = c[c.firstSeason > LAST]  # coaches hired in 2025-26 or later
    pre = (FIRST - c.firstSeason).clip(lower=0) * 82
    post = pd.Series(0.0, index=c.index)
    for i, t in c[(c.lastSeason > LAST) & (c.firstSeason <= LAST)].iterrows():
        later = post_first[(post_first.franchiseAbbrv == t.franchiseAbbrv) & (post_first.firstSeason == LAST + 1)]
        post[i] = 82 - later.G.fillna(0).sum()
    c["G_in"] = c.G - pre - post
    # a pre-1996 mid-season hire makes "82 per earlier season" overcount: drop it
    c.loc[c.G_in <= 0, "G_in"] = np.nan
    return c[(c.lastSeason >= FIRST) & (c.firstSeason <= LAST)]


def align_franchise(ten, fg):
    """Place tenure boundaries in the franchise's chronological game list.

    Cost of a tenure's block = 2/game off its source G + 3/win off its source W
    + 1.5/season its block starts or ends away from the source's season fields.
    Season fields are only a tie-breaker: the source has a few interim coaches
    filed a season off, and W-L pins those exactly. The first tenure may start
    before 1996-97 and the last may run past 2024-25: only their game counts
    (G_in, see in_window_games) are checked, not wins.
    Dynamic programme over boundary positions; G±SLACK per tenure.
    """
    SLACK = 3
    n = len(fg)
    season = fg.season.to_numpy()
    cw = np.r_[0, np.cumsum(fg.win.to_numpy())]
    T = list(ten.itertuples())
    k = len(T)

    def cost(i, a, b):
        t = T[i]
        c = 0.0
        inside_start = i > 0 or t.firstSeason >= FIRST
        inside_end = i < k - 1 or t.lastSeason <= LAST
        if b <= a:
            return 1e9
        if pd.notna(t.G_in):
            c += 2 * abs((b - a) - t.G_in)
            if inside_start and inside_end:
                c += 3 * abs((cw[b] - cw[a]) - t.w)
        if inside_start:
            c += 1.5 * abs(season[a] - t.firstSeason)
        if inside_end:
            c += 1.5 * abs(season[b - 1] - min(t.lastSeason, LAST))
        return c

    best = {0: (0.0, [])}
    for i in range(k):
        nxt = {}
        last = i == k - 1
        for a, (ca, path) in best.items():
            t = T[i]
            if last:
                cands = [n]
            elif i == 0 and t.firstSeason < FIRST:
                # unknown pre-window share: any end inside its last source season +/- 1
                lo = np.searchsorted(season, min(t.lastSeason, LAST) - 1)
                hi = np.searchsorted(season, min(t.lastSeason, LAST) + 2)
                cands = range(max(a + 1, lo), min(hi, n) + 1)
            else:
                g = int(t.G) if pd.notna(t.G) else 82
                cands = range(max(a + 1, a + g - SLACK), min(n, a + g + SLACK) + 1)
            for b in cands:
                c = ca + cost(i, a, b)
                if b not in nxt or c < nxt[b][0]:
                    nxt[b] = (c, path + [b])
        best = nxt
    c, bounds = best[n]
    return [0] + bounds, c


def main():
    if not SRC.exists():
        SRC.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, SRC)
    c = pd.read_csv(SRC, comment="#")
    for i, fix in CORRECTIONS.items():
        for k, v in fix.items():
            c.at[i, k] = v
    c["G"] = c.w + c.l
    c = in_window_games(c)
    games = team_games()

    assigned, problems = [], []
    for fr, ten in c.groupby("franchiseAbbrv"):
        ten = ten.sort_index()
        fg = games[games.franchise == fr].reset_index(drop=True)
        bounds, total = align_franchise(ten, fg)
        for j, i in enumerate(ten.index):
            blk = fg.iloc[bounds[j]:bounds[j + 1]].copy()
            blk["coach"] = ten.at[i, "name"]
            blk["coach_id"] = ten.at[i, "leaderId"]
            blk["tenure"] = i
            blk["interim"] = bool(ten.at[i, "interim"])
            assigned.append(blk)

    a = pd.concat(assigned)
    out_dir = ROOT / "data" / "coaches"
    out_dir.mkdir(parents=True, exist_ok=True)
    a.to_parquet(out_dir / "coach_games.parquet", index=False)

    # check: tenures entirely inside our window must reproduce the source W-L
    chk = a.groupby("tenure").agg(our_g=("win", "size"), our_w=("win", "sum"))
    chk = c.join(chk, how="left")
    chk["inside"] = (chk.firstSeason >= FIRST) & (chk.lastSeason <= LAST)
    chk["w_diff"] = chk.our_w - chk.w
    chk["g_diff"] = chk.our_g - chk.G
    chk[["leaderId", "name", "franchiseAbbrv", "firstSeason", "lastSeason", "w", "l", "our_w", "our_g",
         "w_diff", "g_diff", "inside", "basis"]].to_csv(OUT / "coach_check.csv", index=False)
    ins = chk[chk.inside]
    print(f"games assigned: {len(a)} of {len(games)}")
    print(f"tenures fully inside window: {len(ins)}; exact W-L match: {((ins.w_diff == 0) & (ins.g_diff == 0)).sum()}; "
          f"off by <=1 game: {((ins.w_diff.abs() <= 1) & (ins.g_diff.abs() <= 1)).sum()}")
    for p in problems:
        print("PROBLEM", p)
    bad = ins[(ins.w_diff.abs() > 1) | (ins.g_diff.abs() > 1)]
    if len(bad):
        print(bad[["name", "franchiseAbbrv", "firstSeason", "lastSeason", "w", "l", "our_w", "our_g"]].to_string())

    cs = a.groupby(["season", "team_id", "franchise", "coach_id", "coach"], sort=False).agg(
        games=("win", "size"), wins=("win", "sum"), first_game=("game_id", "min")).reset_index()
    cs.sort_values(["season", "franchise", "first_game"]).to_csv(OUT / "coach_seasons.csv", index=False)


if __name__ == "__main__":
    main()
