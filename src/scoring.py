"""Scoring load: how much scoring a player carried, every season since 1951-52 (NBA and ABA).

Impact (the Scott Stat) and scoring are different questions. This measures the second one on its own
terms, so the Legacy page can weight it openly instead of hiding it inside impact:
    scoring load = (his points per 100 possessions - the league's per player) x his minutes share (mpg / 48)
= points per 100 team possessions he scored above an average player in his minutes. Pace- and
league-relative, so a 1962 season and a 2006 season are on one scale. Volume, not efficiency (his true
shooting vs league is reported next to it). Scoring titles: league leader in points per game among
players with 70%+ of the season's games (each league separately).

Scoring & creation (the Legacy ingredient) adds playmaking: each assist counts 1.1 points (a basket
worth about 2.2 points, half credited to the passer), same formula otherwise:
    creation load = ((points + 1.1 x assists + 0.5 x PPP x offensive rebounds) per 100
                     - league per player) x mpg / 48
An offensive rebound creates a new possession worth the league's points per possession (PPP); half is
credited to the rebounder, the same split as an assist. (1951-52 to 1972-73 did not record offensive
rebounds: those seasons get none, for everyone.)

Usage: python src/scoring.py -> out/scoring_seasons.csv, out/scoring_careers.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
HIST = ROOT / "data" / "hist"


def main():
    t = pd.read_csv(HIST / "Player Totals.csv")
    t = t[t.lg.isin(["NBA", "ABA"])]
    # one row per player-season-league: the TOT row when he played for several teams
    t["multi"] = t.team.str.match(r"^\dTM$|^TOT$")
    t = t.sort_values("multi").drop_duplicates(["season", "lg", "player_id"], keep="last")
    c = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    c["season"] = c.season_start + 1
    c["lg"] = c.lg.fillna("NBA") if "lg" in c else "NBA"
    s = c[["season", "lg", "player_id", "player", "poss", "mp", "g"]].merge(
        t[["season", "lg", "player_id", "pts", "ast", "orb", "fga", "fta"]], on=["season", "lg", "player_id"])
    s = s[s.poss > 0]
    grp = s.groupby(["season", "lg"])
    s["lg_pts100"] = grp.pts.transform("sum") / grp.poss.transform("sum") * 100
    s["lg_ts"] = grp.pts.transform("sum") / (2 * (grp.fga.transform("sum") + 0.44 * grp.fta.transform("sum")))
    s["pts100"] = s.pts / s.poss * 100
    s["mpg"] = s.mp / s.g.clip(lower=1)
    s["load"] = (s.pts100 - s.lg_pts100) * s.mpg / 48
    ppp = grp.pts.transform("sum") / grp.poss.transform("sum")
    s["made"] = s.pts + 1.1 * s.ast.fillna(0) + 0.5 * ppp * s.orb.fillna(0)
    s["lg_made100"] = s.groupby(["season", "lg"]).made.transform("sum") / grp.poss.transform("sum") * 100
    s["cload"] = (s.made / s.poss * 100 - s.lg_made100) * s.mpg / 48
    s["ts_rel"] = s.pts / (2 * (s.fga + 0.44 * s.fta)).where(lambda x: x > 0) - s.lg_ts
    s["ppg"] = s.pts / s.g.clip(lower=1)
    gmax = grp.g.transform("max")
    q = s[s.g >= 0.7 * gmax]
    lead = q.loc[q.groupby(["season", "lg"]).ppg.idxmax()]
    s["scoring_title"] = s.index.isin(lead.index)
    s.to_csv(OUT / "scoring_seasons.csv", index=False, float_format="%.3f")

    rows = []
    for pid, x in s.groupby("player_id"):
        x = x.sort_values("season")
        x = x.groupby("season").apply(lambda z: z.iloc[np.argmax(z.poss.to_numpy())], include_groups=False).reset_index()
        def best5(col):
            best, yrs = np.nan, None
            for i in range(len(x) - 4):
                w = x.iloc[i:i + 5]
                if w.season.iat[-1] - w.season.iat[0] == 4:
                    v = w[col].mean()
                    if np.isnan(best) or v > best:
                        best, yrs = v, f"{w.season.iat[0] - 1}-{w.season.iat[-1] % 100:02d}"
            return best, yrs
        best, yrs = best5("load")
        cbest, cyrs = best5("cload")
        rows.append({"player_id": pid, "player": x.player.iloc[-1], "scoring5": best, "scoring5_years": yrs,
                     "create5": cbest, "create5_years": cyrs,
                     "scoring_best": x.load.max(), "scoring_titles": int(x.scoring_title.sum()),
                     "ts_rel_career": float(np.average(x.ts_rel.fillna(0), weights=x.poss))})
    car = pd.DataFrame(rows).sort_values("create5", ascending=False)
    car.to_csv(OUT / "scoring_careers.csv", index=False, float_format="%.3f")
    print(car.head(25).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
