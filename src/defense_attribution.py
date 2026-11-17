"""Audit one player's clutch-defense credit: how is the lineup's over-performance
split, would his teammates do it without him, and how much is opponent luck?

Regular season + playoffs, clutch possessions with the player on defense.
  1. split: each defender's ridge credit (all_clutch D delta) averaged over
     the player's clutch defensive possessions, vs the even 1/5 split.
  2. teammates without him: for his most frequent clutch teammates, their
     lineups' clutch defensive over-performance on possessions WITHOUT him.
  3. luck: opponents' FT% and 3P% in his clutch minutes vs league clutch,
     converted to points per 100 possessions.
  4. by coach: his lineups' raw clutch defensive over-performance per coach.

Usage: python src/defense_attribution.py "LeBron James"   -> out/defense_attribution_<name>.json
"""
import json
import sys

import numpy as np
import pandas as pd

import clutch as C

OPP = ["opp0", "opp1", "opp2", "opp3", "opp4"]


def main(name):
    war = pd.concat([pd.read_csv(C.OUT / f"war_{y}.csv").assign(season=y) for y in range(1996, 2025)])
    ids = war.dropna(subset=["name"]).groupby("name").player_id.agg(lambda x: x.mode().iat[0])
    names = war.dropna(subset=["name"]).groupby("player_id").name.last()
    pid = int(ids[name])
    s = C.clutch_stints(("", "po_"), True)
    pids = np.unique(s[C.H + C.A].to_numpy("int64"))
    X, y, w, exp, home, games, seasons = C.design(s, pids)
    res, env = C.residuals(y, w, exp, home)
    P = len(pids)
    # ridge D credits from the published all_clutch run, plus their standard errors
    lam = json.load(open(C.OUT / "playoffs_meta.json"))["all_clutch"]["lambda"]
    b, se = C.ridge(X, res, w, lam, se=True)
    dD = pd.Series(b[P:], index=pids)
    seD = pd.Series(se[P:], index=pids)

    # defenders of each row (the side that is NOT on offense)
    n = len(s)
    keep = np.r_[s.poss_h, s.poss_a] > 0
    dfn = np.column_stack([np.r_[s[a].astype("int64"), s[h].astype("int64")][keep] for h, a in zip(C.H, C.A)])
    mine = (dfn == pid).any(axis=1)
    wm = w[mine]
    over = -np.average(res[mine], weights=wm)            # + = held opponents below expectation
    credits = dD.reindex(dfn[mine].ravel()).to_numpy().reshape(-1, 5)
    his = np.average(credits[dfn[mine] == pid], weights=wm)
    mates_total = np.average((credits * (dfn[mine] != pid)).sum(axis=1), weights=wm)
    split = {"lineup_overperformance": float(over), "credited_to_him": float(his), "se_his_credit": float(seD[pid]),
             "credited_to_his_4_teammates_combined": float(mates_total),
             "not_credited_to_anyone_(shrunk_as_noise)": float(over - his - mates_total),
             "even_split_would_be": float(over / 5)}

    # 2. most frequent clutch teammates, with and without him
    flat = pd.DataFrame({"mate": dfn[mine].ravel(), "w": np.repeat(wm, 5)})
    top = flat[flat.mate != pid].groupby("mate").w.sum().sort_values(ascending=False).head(10)
    mates = []
    for m_id, shared in top.items():
        with_m = (dfn == m_id).any(axis=1)
        wo = with_m & ~mine
        mates.append({"teammate": names.get(m_id, str(m_id)), "shared_clutch_poss": float(shared),
                      "overperf_with_him": float(-np.average(res[with_m & mine], weights=w[with_m & mine])),
                      "overperf_without_him": float(-np.average(res[wo], weights=w[wo])) if w[wo].sum() > 0 else None,
                      "poss_without_him": float(w[wo].sum()),
                      "own_credit_D": float(dD[m_id])})

    # 3. luck: opponent FT% and 3P% in his clutch minutes
    sh = pd.concat([pd.read_parquet(C.SHOTS / f"shots_{t}{yy}.parquet",
                                    columns=["kind", "made", "three", "period", "clock", "margin"] + OPP)
                    for t in ("", "po_") for yy in range(1996, 2025)], ignore_index=True)
    sh = sh[(sh.period >= 4) & (sh.clock <= 300) & (sh.margin.abs() <= 5)]
    on = (sh[OPP] == pid).any(axis=1)

    def rates(x):
        ft = x[x.kind == "ft"]
        th = x[(x.kind == "fg") & x.three]
        two = x[(x.kind == "fg") & ~x.three]
        return {"ft_pct": float(ft.made.mean()), "fta": int(len(ft)), "three_pct": float(th.made.mean()),
                "fg3a": int(len(th)), "two_pct": float(two.made.mean()), "fg2a": int(len(two))}
    lg, his_r = rates(sh), rates(sh[on])
    ft_pts = (lg["ft_pct"] - his_r["ft_pct"]) * his_r["fta"]
    three_pts = (lg["three_pct"] - his_r["three_pct"]) * 3 * his_r["fg3a"]
    luck = {"league_clutch": lg, "with_him": his_r,
            "opp_ft_misses_beyond_league_pts_per100": float(100 * ft_pts / wm.sum()),
            "opp_3pt_misses_beyond_league_pts_per100": float(100 * three_pts / wm.sum()),
            "note": "FT% is not defense; 3P% allowed is mostly not defense (shot volume/quality is)."}

    # 4. by coach (team-games coached), his lineups' raw over-performance
    cg = pd.read_parquet(C.ROOT / "data" / "coaches" / "coach_games.parquet")[["game_id", "coach"]].drop_duplicates()
    g = pd.Series(np.r_[s.game_id, s.game_id][keep][mine])
    # his team's coach in that game: coach_games has both teams; pick the coach of the team he played for
    team = np.r_[s.away_team, s.home_team][keep][mine]   # defensive side's team for these rows
    cgk = pd.read_parquet(C.ROOT / "data" / "coaches" / "coach_games.parquet").set_index(["game_id", "team_id"]).coach
    coach = pd.MultiIndex.from_arrays([g.to_numpy(), team]).map(cgk)
    by_coach = pd.DataFrame({"coach": coach, "r": -res[mine], "w": wm}).dropna()
    bc = by_coach.groupby("coach").apply(lambda x: pd.Series({"poss": x.w.sum(), "def_overperf": np.average(x.r, weights=x.w)}),
                                         include_groups=False).sort_values("poss", ascending=False)
    out = {"player": name, "clutch_def_possessions": float(wm.sum()), "split": split, "teammates": mates,
           "luck": luck, "by_coach": bc.round(2).reset_index().to_dict("records")}
    path = C.OUT / f"defense_attribution_{name.split()[-1].lower()}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "LeBron James")
