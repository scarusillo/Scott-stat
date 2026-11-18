"""Playoffs in the Classic tier — every era, same method as the regular season.

For every player-season since 1951-52 with playoff games (Kaggle box scores):
  1. Playoff box estimate: playoff per-100 rates relative to that season's
     PLAYOFF league (other playoff players), priced with the same era-matched skill
     values Classic uses for that season.
  2. Team playoff results: each team's playoff margin per 100 possessions minus
     what an average team would do against those same opponents (their
     regular-season margins). The gap between that and the sum of the team's
     playoff box estimates is shared per possession, as in the regular season.
  2b. Playoff rise = playoff rating - regular rating, measured against the
     typical playoff player's rise that season (so the general playoff drop
     doesn't count against anyone; Plus measures it the same way).
  3. Season "with playoffs" = possession-weighted average of the regular-season
     and playoff ratings (Plus does the same: playoff share x playoff rise).
Sanity check: career playoff rise here vs Plus's play-by-play playoff elevation.

Usage: python src/classic_playoffs.py   -> out/classic_playoffs.csv, out/classic_playoffs_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import classic as K
from hist_games import season_end

ROOT = Path(__file__).resolve().parents[1]
KAG = ROOT / "data" / "kaggle_dl"
OUT = ROOT / "out"


def league_rates(b):
    """Per-season regular-season league per-100 rates and points per shot attempt."""
    out = {}
    for season, d in b.groupby("season"):
        d = d[d.poss > 0]
        tsa = d.fga + 0.44 * d.fta
        raw = pd.DataFrame({"pts": d.pts, "tsa": tsa, "ast": d.ast, "tov": d.tov, "oreb": d.orb, "dreb": d.drb,
                            "stl": d.stl, "blk": d.blk, "pf": d.pf, "fg3a": d.x3pa, "reb": d.trb})
        lg = raw.sum() / d.poss.sum() * 100
        out[season] = (lg, raw.pts.sum() / raw.tsa.sum())
    return out


def playoff_rows(b, lg, pace):
    cols = ["gameId", "gameDateTimeEst", "firstName", "lastName", "playerteamId", "numMinutes", "points", "assists",
            "blocks", "steals", "fieldGoalsAttempted", "threePointersAttempted", "freeThrowsAttempted",
            "reboundsDefensive", "reboundsOffensive", "reboundsTotal", "foulsPersonal", "turnovers", "gameType", "comment"]
    p = pd.read_csv(KAG / "PlayerStatistics.csv", usecols=cols, low_memory=False)
    p = p[p.gameType == "Playoffs"].copy()
    p["season"] = season_end(pd.to_datetime(p.gameDateTimeEst, errors="coerce"))
    num = ["numMinutes", "points", "assists", "blocks", "steals", "fieldGoalsAttempted", "threePointersAttempted",
           "freeThrowsAttempted", "reboundsDefensive", "reboundsOffensive", "reboundsTotal", "foulsPersonal", "turnovers"]
    for c in num:
        p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0)
    dnp = p.comment.astype(str).str.contains("DNP|DND|NWT|Did not|Not With", case=False, na=False)
    p = p[~dnp & ((p.numMinutes > 0) | (p[num[1:]].sum(axis=1) > 0))]
    p["key"] = (p.firstName.fillna("") + " " + p.lastName.fillna("")).map(K.norm)
    reg = b.assign(key=b.player.map(K.norm))[["season", "key", "player_id", "pos", "mp", "g"]]
    reg = reg[~reg.duplicated(["season", "key"], keep=False)]
    p = p.merge(reg, on=["season", "key"], how="inner")
    p["min"] = np.where(p.numMinutes > 0, p.numMinutes, p.mp / p.g.clip(lower=1))
    agg = p.groupby(["season", "player_id", "playerteamId"]).agg(
        games=("gameId", "nunique"), mp=("min", "sum"), pts=("points", "sum"), ast=("assists", "sum"),
        blk=("blocks", "sum"), stl=("steals", "sum"), fga=("fieldGoalsAttempted", "sum"),
        fg3a=("threePointersAttempted", "sum"), fta=("freeThrowsAttempted", "sum"), dreb=("reboundsDefensive", "sum"),
        oreb=("reboundsOffensive", "sum"), reb=("reboundsTotal", "sum"), pf=("foulsPersonal", "sum"),
        tov=("turnovers", "sum"), pos=("pos", "first")).reset_index()
    agg["poss"] = agg.mp * agg.season.map(pace).fillna(100) / 48
    rows = []
    for season, d in agg.groupby("season"):
        if season not in lg:
            continue
        raw0 = pd.DataFrame({"pts": d.pts, "tsa": d.fga + 0.44 * d.fta, "ast": d.ast, "tov": d.tov, "oreb": d.oreb,
                             "dreb": d.dreb, "stl": d.stl, "blk": d.blk, "pf": d.pf, "fg3a": d.fg3a, "reb": d.reb})
        # baseline = that season's PLAYOFF league (tougher defenses lower everyone's playoff rates)
        L = raw0.sum() / d.poss.sum() * 100
        pps = raw0.pts.sum() / raw0.tsa.sum()
        raw = pd.DataFrame({"pts": d.pts, "tsa": d.fga + 0.44 * d.fta, "ast": d.ast, "tov": d.tov, "oreb": d.oreb,
                            "dreb": d.dreb, "stl": d.stl, "blk": d.blk, "pf": d.pf, "fg3a": d.fg3a, "reb": d.reb}, index=d.index)
        r = raw.add(L * K.PRIOR_POSS / 100).div(d.poss + K.PRIOR_POSS, axis=0) * 100
        r["pts_added"] = r.pts - pps * r.tsa
        rel = r.drop(columns="pts") - L.drop("pts")
        rel["pts_added"] = r.pts_added
        pos = d.pos.astype(str).str.split("-").str[0]
        rel["guard"] = pos.isin(["PG", "SG"]).astype(float)
        rel["big"] = pos.isin(["PF", "C"]).astype(float)
        rel["mpg"] = d.mp / d.games.clip(lower=1) / 48 - 0.5
        rows.append(pd.concat([d, rel.add_prefix("f_")], axis=1))
    return pd.concat(rows, ignore_index=True)


def team_playoff_gap(po, pace):
    """Team playoff net per 100 vs expectation against its opponents, minus 5x its box estimate."""
    t = pd.read_csv(KAG / "TeamStatistics.csv", usecols=["gameId", "gameDateTimeEst", "teamId", "opponentTeamId",
                                                           "teamScore", "opponentScore", "gameType"], low_memory=False)
    t["season"] = season_end(pd.to_datetime(t.gameDateTimeEst, errors="coerce"))
    t = t.dropna(subset=["teamScore", "opponentScore"])
    t = t[(t.teamScore > 0) & (t.opponentScore > 0)]
    t["net100"] = 100 * (t.teamScore - t.opponentScore) / t.season.map(pace).fillna(100)
    regnet = t[t.gameType == "Regular Season"].groupby(["season", "teamId"]).net100.mean()
    pt = t[t.gameType == "Playoffs"].copy()
    pt["opp_reg"] = pd.MultiIndex.from_arrays([pt.season, pt.opponentTeamId]).map(regnet)
    pt["vs_expect"] = pt.net100 + pt.opp_reg.fillna(0)          # an average team expects -opp_reg
    team = pt.groupby(["season", "teamId"]).vs_expect.mean().rename("team_adj")
    po = po.join(team, on=["season", "playerteamId"])
    box = po.assign(bp=(po.box_O + po.box_D) * po.poss).groupby(["season", "playerteamId"])[["bp", "poss"]].sum()
    box.index.names = ["season", "teamId"]
    gap = (team - 5 * box.bp / box.poss).rename("gap")
    gap.index.names = ["season", "playerteamId"]
    return po.join(gap, on=["season", "playerteamId"])


def main():
    b, per_team, ts = K.load_box()
    d = K.features(b)
    d["key"] = d.player.map(K.norm)
    pl = K.plus_targets()
    dup = pl.duplicated(["season", "key"], keep=False)
    d = d.merge(pl[~dup].drop(columns="name"), on=["season", "key"], how="left")
    ref = d[d.nba_id.notna() & (d.plus_poss >= 1000)]
    pace = ts.groupby("season").pace.mean()
    lg = league_rates(b)
    po = playoff_rows(b, lg, pace)
    caps = {f_: (ref["f_" + f_].min(), ref["f_" + f_].max()) for f_ in K.BOX + ["reb"]}
    for frame in (d, po):
        for f_, (lo, hi) in caps.items():
            frame["f_" + f_] = frame["f_" + f_].clip(lo, hi)
    base = d.nba_id.notna() & (d.plus_poss >= 250)
    meta = json.load(open(OUT / "classic_meta.json"))
    tau, alpha = meta["validation"]["tau"], meta["validation"]["alpha"]
    # price playoff rows with the regular-season era-matched skill values (training rows: regular season only)
    po["is_po"] = True
    both = pd.concat([d.assign(is_po=False), po], ignore_index=True)
    train = (both.is_po == False) & both.nba_id.notna() & (both.plus_poss >= 250)
    priced = K.era_fit(both, train.to_numpy(), tau)
    po = priced[priced.is_po == True].copy()
    po = team_playoff_gap(po, pace)
    po["po_net"] = po.box_O + po.box_D + alpha * po.gap.fillna(0) / 5
    # one row per player-season (players traded mid-season have one playoff team)
    po = po.sort_values("poss").groupby(["season", "player_id"]).agg(
        po_poss=("poss", "sum"), po_games=("games", "sum"), po_net=("po_net", "last"),
        po_box=("box_O", "last")).reset_index()
    reg = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    reg["season"] = reg.season_start + 1
    m = reg.merge(po, on=["season", "player_id"], how="left")
    m["po_poss"] = m.po_poss.fillna(0)
    m["playoff_rise_raw"] = m.po_net - m.net_classic
    # zero point = the typical playoff player's rise that season (playoff possession-weighted)
    has = m.po_poss > 0
    season_mean = m[has].groupby("season").apply(lambda g: np.average(g.playoff_rise_raw, weights=g.po_poss), include_groups=False)
    m["playoff_rise"] = m.playoff_rise_raw - m.season.map(season_mean)
    m["po_net"] = m.net_classic + m.playoff_rise
    m["net_with_playoffs"] = np.where(m.po_poss > 0, (m.net_classic * m.poss + m.po_net.fillna(0) * m.po_poss) /
                                      (m.poss + m.po_poss), m.net_classic)
    m["poss_total"] = m.poss + m.po_poss
    cols = ["season_start", "player_id", "player", "team", "poss", "net_classic", "po_games", "po_poss", "po_net",
            "playoff_rise", "net_with_playoffs", "poss_total"]
    m[cols].to_csv(OUT / "classic_playoffs.csv", index=False, float_format="%.3f")

    # sanity: career playoff rise (Classic) vs Plus playoff elevation, players with real playoff careers
    pp = pd.read_csv(OUT / "playoffs_players.csv")
    names = pp.set_index("name")
    car = m[m.po_poss > 0].groupby("player").apply(lambda x: pd.Series({
        "rise": np.average(x.playoff_rise, weights=x.po_poss), "po_poss": x.po_poss.sum()}), include_groups=False)
    j = car.join(names[["elevation_delta", "elevation_poss"]], how="inner")
    j = j[(j.po_poss >= 3000) & (j.elevation_poss >= 3000)]
    check = {"players": int(len(j)), "r_with_plus_elevation": float(np.corrcoef(j.rise, j.elevation_delta)[0, 1])}
    (OUT / "classic_playoffs_meta.json").write_text(json.dumps({"check": check}, indent=1))
    print(json.dumps(check))
    for n in ("Michael Jordan", "LeBron James", "Stephen Curry", "Nikola Jokić", "Kareem Abdul-Jabbar",
              "Bill Russell", "Wilt Chamberlain", "Magic Johnson", "Larry Bird", "Tim Duncan", "Hakeem Olajuwon"):
        x = m[(m.player == n) & (m.po_poss > 0)]
        if len(x):
            print(f"{n:20s} playoff seasons {len(x):2d}  avg rise {np.average(x.playoff_rise, weights=x.po_poss):+.2f}  "
                  f"playoff share {x.po_poss.sum() / x.poss_total.sum():.0%}")


if __name__ == "__main__":
    main()
