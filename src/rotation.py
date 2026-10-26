"""Rotation IQ and Closing IQ — does the coach give the minutes, and the big
minutes, to his best players?

Rotation IQ: every game, take the players who actually played and the
possession counts actually handed out, and re-deal those same minutes to the
same players by coach-free talent (rearrangement — best player gets the
biggest share), only WITHIN position group (bigs vs perimeter) so the "ideal"
never plays five centres. Points left on the table = ideal − actual talent-
weighted possessions. Some loss is structural (fatigue, fouls, matchups), so
coaches are scored RELATIVE to the league that season, in wins per 82 games.

Closing IQ: lineup talent in high-leverage possessions (weighted by how much a
point moved win probability, wins.py) minus lineup talent overall. Positive =
the coach's best lineups are on the floor when games are decided.

Players inactive for a game (injury, DNP) are not "available", so benching a
healthy player all game is invisible here: Rotation IQ understates bad benching.

Usage: python src/rotation.py   -> out/coach_rotation.csv, out/coach_rotation_seasons.csv
"""
import json

import numpy as np
import pandas as pd

from coach_effects import load
from talent import A, H, OUT, ROOT, lookup, talent_table
from wins import leverage, wp_model

BIGS = {"Interior scorer", "Rim runner", "Frontcourt finisher"}


def main():
    s, cg = load()
    t = talent_table(s)
    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet").set_index(["season", "player_id"]).archetype
    meta = {m["season"]: m for m in json.loads((OUT / "war_meta.json").read_text())}
    wpm = wp_model(range(2015, 2025))
    s = s.copy()
    lev = leverage(s, wpm)
    s["lev"] = lev / np.average(lev, weights=s.poss_h + s.poss_a)
    s["p"] = (s.poss_h + s.poss_a) / 2

    # long format: one row per player per stint, with team and coach
    recs = []
    for cols, team, coach in ((H, "home_team", "coach_h"), (A, "away_team", "coach_a")):
        for c in cols:
            recs.append(pd.DataFrame({"season": s.season.to_numpy(), "game_id": s.game_id.to_numpy(),
                                      "team_id": s[team].to_numpy(), "coach_id": s[coach].to_numpy(),
                                      "player_id": s[c].astype("int64").to_numpy(), "p": s.p.to_numpy(),
                                      "lp": (s.p * s.lev).to_numpy()}))
    L = pd.concat(recs, ignore_index=True)
    L["talent"] = lookup(t, L.season, L.player_id)
    L["big"] = pd.Series(arch.reindex(pd.MultiIndex.from_arrays([L.season, L.player_id])).to_numpy()).isin(BIGS).to_numpy()
    L = L.dropna(subset=["talent"])

    # --- rotation: per game, per team, per position group
    pg = L.groupby(["season", "game_id", "team_id", "coach_id", "big", "player_id"]).agg(
        p=("p", "sum"), talent=("talent", "first")).reset_index()

    def ideal(df):
        return np.sum(np.sort(df.p.to_numpy())[::-1] * np.sort(df.talent.to_numpy())[::-1])

    g = pg.groupby(["season", "game_id", "team_id", "coach_id", "big"])
    actual = g.apply(lambda d: np.sum(d.p * d.talent), include_groups=False)
    best = g.apply(ideal, include_groups=False)
    loss = ((best - actual) / 100).groupby(["season", "game_id", "team_id", "coach_id"]).sum()  # points per game
    loss = loss.rename("pts_lost").reset_index()
    loss["league"] = loss.groupby("season").pts_lost.transform("mean")
    loss["wpp"] = loss.season.map(lambda y: meta[y]["wins_per_point"])
    loss["rel_wins_82"] = -(loss.pts_lost - loss.league) * loss.wpp * 82   # + = better than league

    # --- closing: talent on floor in leverage-weighted vs plain possessions (lineup sum, per team)
    L["pt"] = L.p * L.talent
    L["lpt"] = L.lp * L.talent
    cs = L.groupby(["season", "team_id", "coach_id"]).agg(pt=("pt", "sum"), p=("p", "sum"),
                                                          lpt=("lpt", "sum"), lp=("lp", "sum"))
    cs["closing"] = 5 * (cs.lpt / cs.lp - cs.pt / cs.p)   # lineup-talent gap, points per 100
    cs["closing"] -= cs.groupby("season").closing.transform("mean")

    per = loss.groupby(["season", "team_id", "coach_id"]).agg(games=("game_id", "size"),
                                                             pts_lost=("pts_lost", "mean"),
                                                             rotation_wins_82=("rel_wins_82", "mean"))
    per = per.join(cs.closing).reset_index()
    names = cg.groupby("coach_id").coach.first()
    per.insert(3, "coach", per.coach_id.map(names))
    per.to_csv(OUT / "coach_rotation_seasons.csv", index=False, float_format="%.3f")

    def wavg(d):
        w = d.games
        return pd.Series({"games": w.sum(), "seasons": d.season.nunique(),
                          "rotation_wins_82": np.average(d.rotation_wins_82, weights=w),
                          "closing": np.average(d.closing, weights=w),
                          "pts_lost_per_game": np.average(d.pts_lost, weights=w)})
    co = per.groupby("coach_id").apply(wavg, include_groups=False)
    co.insert(0, "coach", co.index.map(names))
    co = co.sort_values("rotation_wins_82", ascending=False)
    co.to_csv(OUT / "coach_rotation.csv", float_format="%.3f")

    # reliability: odd vs even seasons
    half = per.assign(h=per.season % 2).groupby(["coach_id", "h"]).apply(wavg, include_groups=False).unstack()
    ok = (half["games"][0] >= 164) & (half["games"][1] >= 164)
    r_rot = np.corrcoef(half["rotation_wins_82"][0][ok], half["rotation_wins_82"][1][ok])[0, 1]
    r_clo = np.corrcoef(half["closing"][0][ok], half["closing"][1][ok])[0, 1]
    rel = {"coaches": int(ok.sum()), "r_rotation": float(r_rot), "r_closing": float(r_clo),
           "league_pts_lost_per_game": float(loss.pts_lost.mean())}
    (OUT / "coach_rotation_meta.json").write_text(json.dumps(rel, indent=1))
    print(json.dumps(rel))
    big = co[co.games >= 246]
    cols = ["coach", "seasons", "rotation_wins_82", "closing", "pts_lost_per_game"]
    print(big.head(10)[cols].round(2).to_string(index=False))
    print(big.tail(6)[cols].round(2).to_string(index=False))
    print(big.sort_values("closing", ascending=False).head(6)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
