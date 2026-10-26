"""Layer 2 — turn impact into WINS.

For each season Y:
  1. Season rating: single-season O/D RAPM shrunk toward the player's prior
     (plain RAPM over Y-2..Y-1) instead of toward 0, so a season is judged on
     its own possessions but one hot month can't create a star. Players with no
     prior (rookies) shrink toward the empirical newcomer average.
  2. Replacement level: the possession-weighted rating of each team's fringe
     (players outside its top 10 in possessions) — the talent freely available.
  3. Points -> wins: regress team win% on point margin per game, per season.
  4. WAR  = (rating - replacement) x possessions / 100 x wins-per-point.
     cWAR = same, but each possession weighted by its LEVERAGE: how much one
     point moved the home team's win probability at that game state, scaled so
     the league-average possession = 1. Garbage time ~0, crunch time >1.
     This is the "contributed to wins that mattered" number.

Usage: python src/wins.py 1996 2024     -> out/war_<YYYY>.csv, out/war_meta.json
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rapm import A, H, OUT, STINTS, design, fit, load
from rapm import run as rapm_run

LAM = 3000
FRINGE_RANK = 10
REPL_WINPCT = 0.20


def remaining(s):
    reg = s.period <= 4
    return np.where(reg, (4 - s.period) * 720 + s.clock0, s.clock0).astype(float)


def wp_model(years):
    s = pd.concat([pd.read_parquet(STINTS / f"stints_{y}.parquet") for y in years])
    rem = remaining(s)
    X = np.c_[s.margin0 / np.sqrt(rem / 60 + 1), s.margin0]
    y = (s.final_margin > 0).astype(int)
    m = LogisticRegression(C=1e3).fit(X, y)
    return m


def leverage(s, m):
    rem = remaining(s)
    k = 1 / np.sqrt(rem / 60 + 1)
    X = np.c_[s.margin0 * k, s.margin0]
    p = m.predict_proba(X)[:, 1]
    b1, b2 = m.coef_[0]
    return p * (1 - p) * (b1 * k + b2)   # dWP / d(1 point)


def fit_prior(X, y, w, beta0, lam):
    mu = np.average(y, weights=w)
    r = (y - mu) - X @ beta0
    mu_r, delta = fit(X, r, w, lam)  # fit() centres r: the remaining offset is the intercept
    return mu + mu_r, beta0 + delta


def season_ratings(year):
    s, pl = load(year, year)
    ids = np.unique(s[H + A].to_numpy("int64"))
    idx = {int(p): i for i, p in enumerate(ids)}
    p = len(ids)
    prior = None
    if (STINTS / f"stints_{year - 1}.parquet").exists():
        prior, _ = rapm_run(max(1996, year - 2), year - 1, LAM)
    X, y, w, _ = design(s, idx)
    known = np.zeros(p, bool)
    o0 = np.zeros(p)
    d0 = np.zeros(p)
    if prior is not None:
        pr = prior.set_index("player_id")
        for pid, i in idx.items():
            if pid in pr.index:
                known[i] = True
                o0[i], d0[i] = pr.at[pid, "O"], pr.at[pid, "D"]
    on_poss = np.zeros(p)
    for cols in (H, A):
        ii = np.vectorize(idx.__getitem__)(s[cols].to_numpy("int64"))
        for j in range(5):
            np.add.at(on_poss, ii[:, j], (s.poss_h + s.poss_a).to_numpy() / 2)
    newO = newD = 0.0
    for _ in range(2):  # second pass: newcomers shrink toward the newcomer average
        o0[~known], d0[~known] = newO, newD
        beta0 = np.r_[o0, d0, 0.0]
        mu, beta = fit_prior(X, y, w, beta0, LAM)
        nw = on_poss[~known]
        if nw.sum() > 0 and prior is not None:
            newO = np.average(beta[:p][~known], weights=nw)
            newD = np.average(beta[p:2 * p][~known], weights=nw)
    return s, pl, ids, idx, beta, mu


def run(year, wpm):
    s, pl, ids, idx, beta, mu = season_ratings(year)
    p = len(ids)
    O, D = beta[:p], beta[p:2 * p]
    lev = leverage(s, wpm)
    lev_w = lev / np.average(lev, weights=(s.poss_h + s.poss_a))

    # per-player possessions on O and D, raw and leverage-weighted, by team
    recs = []
    for cols, side, opp in ((H, "h", "a"), (A, "a", "h")):
        team = s["home_team" if side == "h" else "away_team"].to_numpy()
        for c in cols:
            recs.append(pd.DataFrame({"player_id": s[c].astype("int64").to_numpy(), "team_id": team,
                                      "off": s[f"poss_{side}"].to_numpy(), "def": s[f"poss_{opp}"].to_numpy(),
                                      "off_lev": s[f"poss_{side}"].to_numpy() * lev_w,
                                      "def_lev": s[f"poss_{opp}"].to_numpy() * lev_w}))
    pt = pd.concat(recs).groupby(["player_id", "team_id"]).sum().reset_index()
    pp = pt.groupby("player_id")[["off", "def", "off_lev", "def_lev"]].sum()
    main_team = pt.sort_values("off").groupby("player_id").team_id.last()

    r = pd.DataFrame({"player_id": ids, "O": O, "D": D}).set_index("player_id").join(pp)
    r["team_id"] = main_team
    r["poss"] = (r.off + r["def"]) / 2
    # O and D levels are each unidentified (a constant shift is absorbed by the
    # intercept). Pin them so the possession-weighted average player is exactly
    # 0 — then a team of average players is .500 and league WAR is consistent.
    r["O"] -= np.average(r.O, weights=r.off)
    r["D"] -= np.average(r.D, weights=r["def"])

    # wins per point of margin, from this season's actual results
    g = s.groupby("game_id").agg(home=("home_team", "first"), away=("away_team", "first"),
                                 m=("final_margin", "first"))
    tg = pd.concat([pd.DataFrame({"team_id": g.home, "m": g.m, "w": (g.m > 0) * 1}),
                    pd.DataFrame({"team_id": g.away, "m": -g.m, "w": (g.m < 0) * 1})])
    team = tg.groupby("team_id").agg(games=("w", "size"), wins=("w", "sum"), margin=("m", "sum"))
    k = np.polyfit(team.margin / team.games, team.wins / team.games, 1)[0]  # win% per point/game
    wins_per_point = k  # season wins per point of season margin (games cancel)

    # Replacement level, fixed in WINS (as FanGraphs/B-Ref do for baseball):
    # a team of freely available players wins REPL_WINPCT of its games, which
    # pins league-total WAR to roughly (wins - REPL_WINPCT * games) every season.
    # Measuring it from fringe players' ratings understates it badly — ridge
    # shrinks low-minute players toward average (kept below as a diagnostic).
    pace = (s.poss_h.sum() + s.poss_a.sum()) / 2 / len(g)             # poss per team-game
    repl_net = (REPL_WINPCT - 0.5) / k / (pace / 100) / 5              # per player, per 100
    repl_O = repl_D = repl_net / 2
    pt["rank"] = pt.groupby("team_id").off.rank(ascending=False)
    fr = pt[pt["rank"] > FRINGE_RANK].join(r[["O", "D"]], on="player_id")
    fringe_net = np.average(fr.O, weights=fr.off) + np.average(fr.D, weights=fr["def"])

    r["pts_above_repl"] = ((r.O - repl_O) * r.off + (r.D - repl_D) * r["def"]) / 100
    r["WAR"] = r.pts_above_repl * wins_per_point
    r["cWAR"] = ((r.O - repl_O) * r.off_lev + (r.D - repl_D) * r.def_lev) / 100 * wins_per_point
    r["leverage"] = (r.off_lev + r.def_lev) / (r.off + r["def"])
    r["net"] = r.O + r.D

    names = pl.groupby("player_id")["name"].agg(lambda x: x.mode().iat[0])
    r["name"] = names
    box = pd.read_parquet(STINTS / f"box_{year}.parquet").set_index("player_id")
    r = r.join(box[["games", "pts", "ast", "reb", "fga", "fta", "tov"]])
    r["ppg"] = r.pts / r.games
    r["ts"] = r.pts / (2 * (r.fga + 0.44 * r.fta))
    r = r.join(team[["wins", "games"]].rename(columns={"wins": "team_wins", "games": "team_games"}),
               on="team_id")
    r.insert(0, "season", year)
    r = r.reset_index().sort_values("WAR", ascending=False)

    # team check: does the sum of player ratings explain team margin?
    r_full = pt.join(r.set_index("player_id")[["O", "D"]], on="player_id")
    r_full["pm"] = (r_full.O * r_full.off + r_full.D * r_full["def"]) / 100
    team["pred_margin"] = r_full.groupby("team_id").pm.sum()
    meta = {"season": year, "repl_net": repl_net, "fringe_net": fringe_net, "pace": pace,
            "wins_per_point": wins_per_point, "points_per_win": 1 / wins_per_point,
            "team_margin_r": float(np.corrcoef(team.margin, team.pred_margin)[0, 1]),
            "league_war": float(r.WAR.sum()), "league_wins": float(team.wins.sum())}
    cols = ["season", "player_id", "name", "team_id", "team_wins", "team_games", "games", "ppg", "ts",
            "O", "D", "net", "poss", "leverage", "WAR", "cWAR", "pts", "ast", "reb", "tov"]
    r[cols].to_csv(OUT / f"war_{year}.csv", index=False, float_format="%.3f")
    return meta


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    wpm = wp_model(range(2015, 2025))
    print("WP model coef", wpm.coef_, wpm.intercept_)
    metas = []
    for y in range(a, b + 1):
        m = run(y, wpm)
        metas.append(m)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()}))
    (OUT / "war_meta.json").write_text(json.dumps(metas, indent=1, default=float))
