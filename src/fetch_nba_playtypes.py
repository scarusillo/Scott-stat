"""Pull NBA.com play-type (Synergy) and defensive-matchup tables. RUN THIS FROM A HOME CONNECTION:
stats.nba.com refuses cloud / data-center IPs (empty reply), so it can't run in a hosted container.

What NBA.com publishes (and what it doesn't):
  - Play types, offense AND defense, players and teams, 2015-16 on: Isolation, PRBallHandler, PRRollman, Postup,
    Spotup, Handoff, OffScreen, Transition, Cut, OffRebound (putbacks), Misc. Possessions, points per possession,
    frequency, percentile.
  - Defensive matchups (who guarded whom, partial possessions, points allowed), 2017-18 on.
  - NOT man vs zone: coverage labels exist only in paid feeds (Synergy, Second Spectrum). The defensive play
    types (how a player does guarding isolation, pick-and-roll, post-ups, spot-ups) are the closest public proxy.

Output: data/nba_stats/playtypes_{P|T}_{offensive|defensive}_{season}.csv
        data/nba_stats/matchups_{season}.csv
Re-running skips files already downloaded. Polite pacing (~1.5 s between calls) with retries.

Usage:  pip install requests pandas
        python src/fetch_nba_playtypes.py                 # everything, 2015-16 to 2024-25
        python src/fetch_nba_playtypes.py --first 2023    # just recent seasons
        python src/fetch_nba_playtypes.py --no-matchups
Then commit nothing from data/ (it's gitignored); upload the folder or run the analysis locally.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "nba_stats"
BASE = "https://stats.nba.com/stats/"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Referer": "https://www.nba.com/", "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*", "Accept-Language": "en-US,en;q=0.9",
    "x-nba-stats-origin": "stats", "x-nba-stats-token": "true",
}
PLAY_TYPES = ["Isolation", "PRBallHandler", "PRRollman", "Postup", "Spotup", "Handoff", "OffScreen",
              "Transition", "Cut", "OffRebound", "Misc"]


def season_str(y):
    return f"{y}-{(y + 1) % 100:02d}"


def get(endpoint, params, tries=4):
    for i in range(tries):
        try:
            r = requests.get(BASE + endpoint, params=params, headers=HEADERS, timeout=60)
            if r.status_code == 200:
                js = r.json()
                rs = js["resultSets"][0] if isinstance(js.get("resultSets"), list) else js["resultSet"]
                return pd.DataFrame(rs["rowSet"], columns=rs["headers"])
            print(f"  {endpoint} HTTP {r.status_code}, retrying")
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"  {endpoint} {type(e).__name__}: {str(e)[:80]}, retrying")
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"{endpoint} failed after {tries} tries: {params}")


def playtypes(y):
    for who in ("P", "T"):
        for grouping in ("offensive", "defensive"):
            out = DEST / f"playtypes_{who}_{grouping}_{y}.csv"
            if out.exists():
                continue
            parts = []
            for pt in PLAY_TYPES:
                if grouping == "defensive" and pt in ("OffRebound", "Misc"):
                    continue                     # not published on defense
                df = get("synergyplaytypes", {"LeagueID": "00", "PerMode": "Totals", "PlayType": pt,
                                              "PlayerOrTeam": who, "SeasonType": "Regular Season",
                                              "SeasonYear": season_str(y), "TypeGrouping": grouping})
                parts.append(df.assign(PLAY_TYPE=pt))
                time.sleep(1.5)
            pd.concat(parts).to_csv(out, index=False)
            print(f"{out.name}: {sum(len(p) for p in parts)} rows")


def matchups(y):
    out = DEST / f"matchups_{y}.csv"
    if out.exists() or y < 2017:
        return
    df = get("leagueseasonmatchups", {"LeagueID": "00", "PerMode": "Totals", "Season": season_str(y),
                                      "SeasonType": "Regular Season"})
    df.to_csv(out, index=False)
    print(f"{out.name}: {len(df)} rows")
    time.sleep(1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2015, help="first season start year (2015 = 2015-16)")
    ap.add_argument("--last", type=int, default=2024)
    ap.add_argument("--no-matchups", action="store_true")
    a = ap.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    try:
        get("synergyplaytypes", {"LeagueID": "00", "PerMode": "Totals", "PlayType": "Isolation", "PlayerOrTeam": "T",
                                 "SeasonType": "Regular Season", "SeasonYear": season_str(a.last), "TypeGrouping": "offensive"}, tries=2)
    except RuntimeError:
        raise SystemExit("stats.nba.com isn't answering from this connection. It blocks cloud/VPN IPs: run from a home network.")
    for y in range(a.first, a.last + 1):
        print(season_str(y))
        playtypes(y)
        if not a.no_matchups:
            matchups(y)


if __name__ == "__main__":
    main()
