"""Show the defensive evidence behind a player's clutch defense number.

For each player, regular season + playoffs, clutch vs the rest of the game,
with him ON the floor on defense:
  * opponent points per 100 possessions (raw), and what the ten players'
    normal ratings predicted -> the lineup's defensive over-performance,
    and how much of it the ridge credits to him;
  * how opponents shot: true shooting, 3PA share, rim-attempt share, FT rate;
  * his own defensive box: steals, blocks, rebounds per 100 possessions.
Plus his regular (all-game) defensive rating history from Plus and Layer 3.

Usage: python src/clutch_defense.py [player names...]   -> out/clutch_defense.json
"""
import json
import sys

import numpy as np
import pandas as pd

import clutch as C
from clutch_breakdown import season_box

DEFAULT = ["LeBron James", "Tim Duncan", "Kevin Garnett", "Kobe Bryant", "Dirk Nowitzki", "Draymond Green"]
OPP = ["opp0", "opp1", "opp2", "opp3", "opp4"]


def main(names):
    war = pd.concat([pd.read_csv(C.OUT / f"war_{y}.csv").assign(season=y) for y in range(1996, 2025)])
    ids = war.dropna(subset=["name"]).groupby("name").player_id.agg(lambda x: x.mode().iat[0])
    s = C.clutch_stints(("", "po_"), True)
    pids = np.unique(s[C.H + C.A].to_numpy("int64"))
    X, y, w, exp, home, games, seasons = C.design(s, pids)
    res, env = C.residuals(y, w, exp, home)
    imp = pd.read_csv(C.OUT / "playoffs_players.csv").set_index("player_id")
    P = len(pids)
    # shots: regular + playoffs
    sh = pd.concat([pd.read_parquet(C.SHOTS / f"shots_{t}{yy}.parquet",
                                    columns=["kind", "pts", "three", "rim", "period", "clock", "margin"] + OPP)
                    for t in ("", "po_") for yy in range(1996, 2025)], ignore_index=True)
    sh["clutch"] = (sh.period >= 4) & (sh.clock <= 300) & (sh.margin.abs() <= 5)
    sh["tsa"] = np.where(sh.kind == "fg", 1.0, 0.44)
    sh["fga"] = (sh.kind == "fg").astype(int)
    sh["fta"] = (sh.kind == "ft").astype(int)
    sh["fg3a"] = ((sh.kind == "fg") & sh.three).astype(int)
    sh["rima"] = ((sh.kind == "fg") & sh.rim).astype(int)

    def shot_profile(x):
        return {"opp_ts": float(x.pts.sum() / (2 * x.tsa.sum())), "opp_3pa_share": float(x.fg3a.sum() / x.fga.sum()),
                "opp_rim_share": float(x.rima.sum() / x.fga.sum()), "opp_ft_rate": float(x.fta.sum() / x.fga.sum()),
                "shots": int(x.fga.sum())}
    league = {"clutch": shot_profile(sh[sh.clutch]), "normal": shot_profile(sh[~sh.clutch])}
    box = pd.concat([season_box(yy, t) for t in ("", "po_") for yy in range(1996, 2025)])

    # all possessions (not just clutch) per player for per-100 rates
    allposs = {}
    for t in ("", "po_"):
        for yy in range(1996, 2025):
            st = pd.read_parquet(C.STINTS / f"stints_{t}{yy}.parquet")
            p = (st.poss_h + st.poss_a) / 2
            for c in C.H + C.A:
                for pid, v in p.groupby(st[c].astype("int64")).sum().items():
                    allposs[pid] = allposs.get(pid, 0) + v
    out = {"league_opp_shooting": league, "clutch_env_intercept": float(env[0])}
    for name in names:
        pid = int(ids[name])
        j = np.searchsorted(pids, pid)
        on_d = np.asarray(X[:, P + j].todense()).ravel() != 0
        wd = w[on_d]
        opp_pts = np.average(y[on_d], weights=wd)
        predicted = np.average(exp[on_d] + env[0] + env[1] * home[on_d], weights=wd)
        mine = sh[(sh[OPP] == pid).any(axis=1)]
        b = box[box.player_id == pid].groupby("clutch").sum(numeric_only=True)
        cp = float(imp.loc[pid, "all_clutch_poss"])
        tp = allposs.get(pid, 0) - cp
        own = {}
        for tag, poss, row in (("clutch", cp, b.loc[True]), ("normal", tp, b.loc[False])):
            own[tag] = {"stl_per100": float(100 * row.stl / poss), "blk_per100": float(100 * row.blk / poss),
                        "reb_per100": float(100 * row.reb / poss)}
        pw = war[war.player_id == pid]
        out[name] = {
            "clutch_def_possessions": float(wd.sum()),
            "opp_pts_per100_actual": float(opp_pts),
            "opp_pts_per100_predicted_from_normal_ratings": float(predicted),
            "lineup_def_overperformance": float(predicted - opp_pts),
            "credited_to_him_D": float(imp.loc[pid, "all_clutch_d_D"]),
            "opp_shooting_with_him_on": {"clutch": shot_profile(mine[mine.clutch]),
                                         "normal": shot_profile(mine[~mine.clutch])},
            "own_defensive_box": own,
            "regular_D_rating_career": float(np.average(pw.D, weights=pw.poss)),
            "regular_D_by_season": {int(r.season): round(float(r.D), 2) for r in pw.itertuples()},
        }
    (C.OUT / "clutch_defense.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULT)
