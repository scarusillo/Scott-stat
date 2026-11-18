"""Is peak wins per season inflated for the 1950s-60s? Four checks on the conversion from rating to wins.

1. Era level: top-10 player-seasons' wins (82-game scaled), by decade.
2. Accounting: sum of every player's wins vs the teams' actual wins above replacement (20% win rate).
3. Pace: does a point of margin per GAME (or per 100 possessions) buy the same wins at any pace?
   Pooled fit win% = 0.5 + k x MOV x (100/pace)^gamma, and per-era slopes.
4. Replacement level: the model's (a 20%-win team) vs the measured rating of fringe players (ranked outside
   teams x 9 by minutes, 250+ possessions), by decade.
Plus where the top players' wins come from: minutes, possessions, rating vs their era.

Usage: python src/era_wins_test.py -> out/era_wins_test.json
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def main():
    d = pd.concat(pd.read_csv(f) for f in glob.glob(str(OUT / "classic_*[0-9].csv")))
    d["season"] = d.season_start + 1
    d = d[d.lg.fillna("NBA") == "NBA"].copy()
    ts = pd.read_csv(ROOT / "data" / "hist" / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")].dropna(subset=["mov", "w", "l", "pace"])
    ts["wp"] = ts.w / (ts.w + ts.l)
    L = ts.groupby("season").apply(lambda t: pd.Series({
        "teams": len(t), "games": (t.w + t.l).mean(), "k": np.polyfit(t.mov, t.wp, 1)[0],
        "pace": np.average(t.pace, weights=t.w + t.l)}), include_groups=False)
    L["repl_model"] = (0.2 - 0.5) / L.k / (L.pace / 100) / 5
    d = d.join(L, on="season")
    d["mpg"] = d.mp / d.g
    d["WAR82"] = d.WAR_classic * 82 / d.games
    dec = lambda s: (s - 1) // 10 * 10
    out = {}

    top = d.sort_values("WAR82", ascending=False).groupby("season").head(10)
    out["top10_by_decade"] = top.groupby(dec(top.season)).agg(
        wins82=("WAR82", "mean"), rating=("net_classic", "mean"), mpg=("mpg", "mean"), pace=("pace", "mean")).round(3).to_dict("index")

    act = ts.groupby("season").apply(lambda t: (t.w - 0.2 * (t.w + t.l)).sum(), include_groups=False)
    r = (d.groupby("season").WAR_classic.sum() / act).dropna()
    out["sum_of_player_wins_over_team_wins"] = r.groupby(dec(r.index)).mean().round(3).to_dict()

    lp = ts.groupby("season").pace.transform("mean")
    fit = least_squares(lambda p: ts.wp - 0.5 - p[0] * ts.mov * (100 / lp) ** p[1], [0.03, 0.5])
    J = fit.jac
    se = np.sqrt(np.diag(np.linalg.inv(J.T @ J) * (fit.fun ** 2).sum() / (len(ts) - 2)))
    out["pace_gamma"] = {"gamma": float(fit.x[1]), "se": float(se[1]),
                         "note": "0 = a point per game is worth the same at any pace; 1 = a point per 100 possessions"}
    ts["net100"] = ts.mov * 100 / ts.pace
    out["win_slope_by_era"] = {lab: {"pace": float(t.pace.mean()), "per_point_per_game": float(np.polyfit(t.mov, t.wp, 1)[0]),
                                     "per_point_per_100": float(np.polyfit(t.net100, t.wp, 1)[0])}
                               for lab, t in [("1952-69", ts[ts.season <= 1969]), ("1970-89", ts[ts.season.between(1970, 1989)]),
                                              ("1990-2026", ts[ts.season >= 1990])]}

    d["mrank"] = d.groupby("season").mp.rank(ascending=False)
    fr = d[(d.mrank > d.teams * 9) & (d.poss >= 250)]
    emp = fr.groupby("season").apply(lambda x: np.average(x.net_classic, weights=x.poss), include_groups=False)
    L["repl_measured"] = emp
    out["replacement_by_decade"] = L.groupby(dec(L.index))[["teams", "repl_model", "repl_measured"]].mean().round(2).to_dict("index")

    names = ["Wilt Chamberlain", "Bill Russell", "Kareem Abdul-Jabbar", "Michael Jordan", "LeBron James"]
    out["best_seasons"] = {n: d[d.player == n].sort_values("WAR82", ascending=False).head(1)[
        ["season", "mpg", "poss", "pace", "net_classic", "WAR82"]].round(2).to_dict("records")[0] for n in names}
    (OUT / "era_wins_test.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
