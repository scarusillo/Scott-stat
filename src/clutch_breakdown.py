"""Where does a player's clutch delta come from?

For the leading clutch players, splits the clutch rise into:
  * offense vs defense (clutch.py ridge estimates);
  * his OWN play: per-100 scoring, shot share, true shooting, assists,
    turnovers, steals/blocks/rebounds — clutch vs the rest of the game;
  * his TEAMMATES: their true shooting with him on the floor, clutch vs normal;
  * team results with him on the floor in the clutch vs what everyone's normal
    ratings predict, by era (unregularised: the raw evidence behind the delta);
  * how likely he is to be #1, from each player's estimate and standard error.

Usage: python src/clutch_breakdown.py   -> out/clutch_box.csv, out/clutch_breakdown.json
"""
import json

import numpy as np
import pandas as pd

import clutch as C
from build_stints import COLS, RAW

FOCUS = {2544: "LeBron James", 203081: "Damian Lillard", 1717: "Dirk Nowitzki", 977: "Kobe Bryant",
         203999: "Nikola Jokić", 201939: "Stephen Curry", 1495: "Tim Duncan"}
ERAS = {"Cavs 2003-10": (2003, 2009), "Heat 2010-14": (2010, 2013), "Cavs 2014-18": (2014, 2017),
        "Lakers 2018-25": (2018, 2024)}


def clutch_flags(d):
    """Per event: clutch? (margin before the event, from the actor's side)."""
    sc = d.SCORE.where(d.SCORE.astype(str).str.contains(" - "))
    parts = sc.str.split(" - ", expand=True)
    away = pd.to_numeric(parts[0], errors="coerce")
    home = pd.to_numeric(parts[1], errors="coerce")
    m_home = (home - away).groupby(d.GAME_ID).ffill().fillna(0)
    m_before = m_home.groupby(d.GAME_ID).shift(1).fillna(0)
    clock = d.PCTIMESTRING.astype(str).str.split(":").map(lambda x: int(x[0]) * 60 + int(x[1]))
    late = (d.PERIOD >= 4) & (clock <= 300)
    return late, m_before


def season_box(y):
    d = pd.read_csv(RAW / f"nbastats_{y}.csv", usecols=COLS, low_memory=False).reset_index(drop=True)
    late, mh = clutch_flags(d)
    desc = d.HOMEDESCRIPTION.fillna(d.VISITORDESCRIPTION).fillna("")
    t, a = d.EVENTMSGTYPE, d.EVENTMSGACTIONTYPE
    three = desc.str.contains("3PT")
    miss = desc.str.contains("MISS")
    recs = []

    def add(k, mask, stat, val=1):
        typ = d[f"PERSON{k}TYPE"]
        m = mask & typ.isin([4, 5])
        margin = np.where(typ[m] == 4, mh[m], -mh[m])
        v = val[m] if isinstance(val, pd.Series) else val
        recs.append(pd.DataFrame({"player_id": d.loc[m, f"PLAYER{k}_ID"].astype("int64"),
                                  "clutch": (late[m] & (np.abs(margin) <= 5)).to_numpy(), "stat": stat, "v": v}))
    add(1, t == 1, "pts", 2 + three.astype(int))
    add(1, (t == 3) & ~miss & ~a.isin([16, 21, 22]), "pts")
    add(1, t.isin([1, 2]), "fga")
    add(1, (t == 3) & ~a.isin([16, 21, 22]), "fta")
    add(1, t == 5, "tov")
    add(2, t == 1, "ast")
    add(2, t == 5, "stl")
    add(3, t == 2, "blk")
    add(1, t == 4, "reb")
    x = pd.concat(recs)
    g = x.pivot_table(index=["player_id", "clutch"], columns="stat", values="v", aggfunc="sum", fill_value=0)
    g = g.reset_index()
    g.insert(0, "season", y)
    return g


def main():
    s = C.clutch_stints()
    # player on-court clutch possessions by season (team possessions)
    recs = []
    for cols in (C.H, C.A):
        for c in cols:
            recs.append(pd.DataFrame({"season": s.season, "player_id": s[c].astype("int64"),
                                      "p": (s.poss_h + s.poss_a) / 2 * s.frac}))
    cposs = pd.concat(recs).groupby(["season", "player_id"]).p.sum()
    war = pd.concat([pd.read_csv(C.OUT / f"war_{y}.csv").assign(season=y) for y in range(1996, 2025)])
    tposs = war.set_index(["season", "player_id"]).poss
    box = pd.concat([season_box(y) for y in range(1996, 2025)])
    box.to_parquet(C.ROOT / "data" / "clutch_box.parquet", index=False)

    out = {}
    imp = pd.read_csv(C.OUT / "clutch_players.csv").set_index("player_id")
    # usage & teammates from shot data
    sh = pd.concat([pd.read_parquet(C.SHOTS / f"shots_{y}.parquet", columns=[
        "season", "game_id", "shooter", "kind", "pts", "period", "clock", "margin", "on0", "on1", "on2", "on3", "on4"])
        for y in range(1996, 2025)], ignore_index=True)
    sh["clutch"] = (sh.period >= 4) & (sh.clock <= 300) & (sh.margin.abs() <= 5)
    sh["tsa"] = np.where(sh.kind == "fg", 1.0, 0.44)
    oncols = ["on0", "on1", "on2", "on3", "on4"]
    for pid, name in FOCUS.items():
        b = box[box.player_id == pid].groupby("clutch").sum(numeric_only=True)
        cp = cposs.xs(pid, level="player_id").sum()
        tp = tposs.xs(pid, level="player_id").sum() - cp
        per = {}
        for tag, poss, row in (("clutch", cp, b.loc[True]), ("normal", tp, b.loc[False])):
            tsa = row.fga + 0.44 * row.fta
            per[tag] = {"pts_per100": 100 * row.pts / poss, "tsa_per100": 100 * tsa / poss,
                        "ts": row.pts / (2 * tsa), "ast_per100": 100 * row.ast / poss,
                        "tov_per100": 100 * row.tov / poss, "stl_blk_per100": 100 * (row.stl + row.blk) / poss,
                        "reb_per100": 100 * row.reb / poss}
        on = (sh[oncols] == pid).any(axis=1)
        mine = sh[on]
        share = {}
        for tag, m in (("clutch", mine.clutch), ("normal", ~mine.clutch)):
            x = mine[m]
            own = x[x.shooter == pid]
            mates = x[x.shooter != pid]
            share[tag] = {"shot_share": own.tsa.sum() / x.tsa.sum(),
                          "teammates_ts": mates.pts.sum() / (2 * mates.tsa.sum())}
        r = imp.loc[pid]
        out[name] = {"clutch_poss": float(r.clutch_poss), "delta": float(r.clutch_delta), "se": float(r.se_delta),
                     "delta_O": float(r.clutch_d_O), "delta_D": float(r.clutch_d_D), "own": per, "on_floor": share}

    # raw evidence by era for LeBron: residual of his on-floor clutch rows
    pids = np.unique(s[C.H + C.A].to_numpy("int64"))
    X, y, w, exp, home, games, seasons = C.design(s, pids)
    res, _ = C.residuals(y, w, exp, home)
    P = len(pids)
    j = np.searchsorted(pids, 2544)
    on_o = np.asarray(X[:, j].todense()).ravel() != 0
    on_d = np.asarray(X[:, P + j].todense()).ravel() != 0
    eras = {}
    for label, (a_, b_) in ERAS.items():
        m = (seasons >= a_) & (seasons <= b_)
        o = np.average(res[m & on_o], weights=w[m & on_o])
        dd = -np.average(res[m & on_d], weights=w[m & on_d])
        eras[label] = {"offense": float(o), "defense": float(dd), "net": float(o + dd),
                       "clutch_poss": float(w[m & on_o].sum())}
    out["LeBron James"]["by_era_raw"] = eras

    # probability of being #1 (rise), players with 1500+ clutch possessions
    big = imp[imp.clutch_poss >= 1500]
    rng = np.random.default_rng(0)
    draws = rng.normal(big.clutch_delta.to_numpy()[:, None], big.se_delta.to_numpy()[:, None], (len(big), 20000))
    top = pd.Series(np.argmax(draws, axis=0)).value_counts(normalize=True)
    p1 = {big.name.iloc[i]: float(v) for i, v in top.head(8).items()}
    draws2 = rng.normal((big.normal_net + big.clutch_delta).to_numpy()[:, None], big.se_delta.to_numpy()[:, None],
                        (len(big), 20000))
    top2 = pd.Series(np.argmax(draws2, axis=0)).value_counts(normalize=True)
    p2 = {big.name.iloc[i]: float(v) for i, v in top2.head(6).items()}
    out["p_biggest_rise"] = p1
    out["p_best_clutch_rating"] = p2
    (C.OUT / "clutch_breakdown.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
