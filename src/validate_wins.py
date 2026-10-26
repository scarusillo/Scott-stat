"""Layer 2 out-of-sample check: rate every player from season Y only, apply
those ratings to the minutes each team actually gave out in Y+1, and compare
predicted vs actual Y+1 wins. Baseline: Y+1 wins = Y wins.

Usage: python src/validate_wins.py      (after wins.py)
"""
import numpy as np
import pandas as pd

from rapm import H, A, OUT, STINTS

NEWCOMER = -1.0  # per-100 rating for players with no prior season (rookies)

rows = []
for y in range(1997, 2025):
    prev = pd.read_csv(OUT / f"war_{y - 1}.csv").set_index("player_id")
    rating = prev.net
    prev_wins = prev.groupby("team_id").team_wins.first() / prev.groupby("team_id").team_games.first()
    s = pd.read_parquet(STINTS / f"stints_{y}.parquet")
    poss = ((s.poss_h + s.poss_a) / 2).to_numpy()
    margin = np.zeros(len(s))
    for cols, sign in ((H, 1), (A, -1)):
        for c in cols:
            margin += sign * s[c].astype("int64").map(rating).fillna(NEWCOMER).to_numpy() * poss / 100
    s["pred"] = margin
    g = s.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"),
                                 pred=("pred", "sum"), m=("final_margin", "first"))
    t = pd.concat([pd.DataFrame({"team": g.home, "pred": g.pred, "w": g.m > 0}),
                   pd.DataFrame({"team": g.away, "pred": -g.pred, "w": g.m < 0})])
    t = t.groupby("team").agg(pred=("pred", "sum"), wins=("w", "sum"), games=("w", "size"))
    t["last_year"] = t.index.map(prev_wins)
    t["season"] = y
    rows.append(t.reset_index())

d = pd.concat(rows).dropna()
d["win_pct"] = d.wins / d.games
k = np.polyfit(d.pred / d.games, d.win_pct, 1)
d["pred_pct"] = np.polyval(k, d.pred / d.games)
rmse = lambda e: np.sqrt(np.mean(e ** 2)) * 82
print(f"team-seasons: {len(d)}")
print(f"r(predicted, actual win%)  ratings-x-next-year-minutes: {np.corrcoef(d.pred_pct, d.win_pct)[0, 1]:.3f}"
      f"   last year's record: {np.corrcoef(d.last_year, d.win_pct)[0, 1]:.3f}")
print(f"error in wins per 82 games: ratings {rmse(d.pred_pct - d.win_pct):.1f}   last year's record {rmse(d.last_year - d.win_pct):.1f}")
kb = np.polyfit(d.last_year, d.win_pct, 1)
print(f"  (last year's record regressed to the mean, fairest baseline: {rmse(np.polyval(kb, d.last_year) - d.win_pct):.1f})")
