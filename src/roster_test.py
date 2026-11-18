"""Roster-change test: does a team's next season move the way its new roster's ratings say it should?

Independent of how the ratings were built (they use each season on its own). For every NBA team-season
t+1 (1978-79 on): predicted team net per 100 = sum over its players of (that player's rating in season
t, adjusted for one year of aging) x his share of the team's possessions in t+1. Players with no prior
season get the measured average of first-year players. Actual = team SRS per 100 possessions.
  - Overall calibration: actual change vs predicted change (slope 1 = right).
  - Movers: predicted change split into players who stayed vs those who arrived/left; slope on the
    movers' part tests whether the ratings of players who change teams hold up.
  - Stars on the move (rated +3 or better, 1,500+ possessions): residual = actual - predicted for
    the team he left and the team he joined.
  - Named transitions (Moses Malone 1982, etc.).

Usage: python src/roster_test.py -> out/roster_test.json
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def main():
    c = pd.concat(pd.read_csv(f) for f in glob.glob(str(OUT / "classic_*[0-9].csv")))
    c["season"] = c.season_start + 1
    c = c[c.lg.fillna("NBA") == "NBA"]
    rating = c.set_index(["season", "player_id"]).net_classic
    age = c.set_index(["season", "player_id"]).age
    # one-year aging from consecutive seasons (rating-weighted pairs, 1,000+ poss both years)
    x = c[c.poss >= 1000][["season", "player_id", "age", "net_classic"]]
    pr = x.merge(x.assign(season=x.season - 1), on=["season", "player_id"], suffixes=("", "_n"))
    aging = pr.groupby(pr.age.clip(19, 38).round()).apply(lambda g: (g.net_classic_n - g.net_classic).mean(),
                                                           include_groups=False)
    t = pd.read_csv(ROOT / "data" / "hist" / "Player Totals.csv")
    t = t[(t.lg == "NBA") & ~t.team.str.match(r"^\dTM$|^TOT$") & (t.season >= 1978)]
    ts = pd.read_csv(ROOT / "data" / "hist" / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")].dropna(subset=["srs", "pace"])
    tsi = ts.set_index(["season", "abbreviation"])
    t["poss"] = t.mp * pd.MultiIndex.from_arrays([t.season, t.team]).map(tsi.pace).to_numpy() / 48
    t = t.dropna(subset=["poss"])
    t["prior"] = pd.MultiIndex.from_arrays([t.season - 1, t.player_id]).map(rating)
    rookie = np.average(c[c.season == c.groupby("player_id").season.transform("min")].net_classic,
                        weights=c[c.season == c.groupby("player_id").season.transform("min")].poss)
    a_prev = pd.MultiIndex.from_arrays([t.season - 1, t.player_id]).map(age)
    t["aged"] = t.prior + pd.Series(np.round(np.clip(a_prev, 19, 38))).map(aging).fillna(0).to_numpy()
    t["r"] = t.aged.fillna(rookie)
    # was he on the same team last season?
    prev_team = t.groupby(["season", "player_id"]).apply(lambda g: set(g.team), include_groups=False)
    t["stayed"] = [team in prev_team.get((s - 1, pid), set()) for s, pid, team in zip(t.season, t.player_id, t.team)]
    tp = t.groupby(["season", "team"]).poss.transform("sum") / 5
    t["contrib"] = t.r * t.poss / tp
    team = t.groupby(["season", "team"]).agg(pred=("contrib", "sum"))
    team["pred_stay"] = t[t.stayed].groupby(["season", "team"]).contrib.sum()
    team["pred_move"] = t[~t.stayed].groupby(["season", "team"]).contrib.sum()
    team = team.fillna(0).reset_index()
    team["actual"] = pd.MultiIndex.from_arrays([team.season, team.team]).map(tsi.srs * 100 / tsi.pace)
    team["actual_prev"] = pd.MultiIndex.from_arrays([team.season - 1, team.team]).map(tsi.srs * 100 / tsi.pace)
    team = team.dropna(subset=["actual"])
    team = team[team.season <= 2026]                # 2026-27 is in progress
    out = {"rookie_rating": float(rookie)}
    X = np.c_[np.ones(len(team)), team.pred]
    b = np.linalg.lstsq(X, team.actual, rcond=None)[0]
    out["actual_on_predicted"] = {"intercept": float(b[0]), "slope": float(b[1]),
                                  "r": float(np.corrcoef(team.pred, team.actual)[0, 1]), "n": int(len(team))}
    X = np.c_[np.ones(len(team)), team.pred_stay, team.pred_move]
    b2, res, *_ = np.linalg.lstsq(X, team.actual, rcond=None)
    rr = team.actual - X @ b2
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * rr.var())
    out["split"] = {"stayers_slope": float(b2[1]), "se_stay": float(se[1]), "movers_slope": float(b2[2]), "se_move": float(se[2])}
    team["resid"] = team.actual - (b[0] + b[1] * team.pred)
    ri = team.set_index(["season", "team"]).resid
    # stars on the move: team he left (season t+1) and team he joined
    st = c[(c.net_classic >= 3) & (c.poss >= 1500)][["season", "player_id", "player", "team", "net_classic"]]
    nxt = t.groupby(["season", "player_id"]).apply(lambda g: g.sort_values("mp").team.iloc[-1], include_groups=False)
    st["new_team"] = [nxt.get((s + 1, p_)) for s, p_ in zip(st.season, st.player_id)]
    mv = st[st.new_team.notna() & (st.new_team != st.team)].copy()
    mv["resid_left"] = [ri.get((s + 1, tm)) for s, tm in zip(mv.season, mv.team)]
    mv["resid_joined"] = [ri.get((s + 1, tm)) for s, tm in zip(mv.season, mv.new_team)]
    mv = mv.dropna(subset=["resid_left", "resid_joined"])
    out["stars_moving"] = {"n": int(len(mv)), "mean_resid_team_left": float(mv.resid_left.mean()),
                           "mean_resid_team_joined": float(mv.resid_joined.mean()),
                           "note": "positive left / negative joined would mean the star was over-rated; the reverse, under-rated"}
    out["star_moves"] = mv.sort_values("net_classic", ascending=False).round(2).to_dict("records")
    (OUT / "roster_test.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "star_moves"}, indent=1))
    print(mv[["season", "player", "team", "new_team", "net_classic", "resid_left", "resid_joined"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
