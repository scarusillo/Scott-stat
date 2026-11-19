"""Coaches, every era: coaching impact fitted jointly with players over every NBA team-season since 1951-52,
plus winning, playoff wins and championships for a reader-weighted coach legacy.

Coaching impact. For every NBA team-season:
    team strength (SRS per 100) - sum of (each player's box-score production x share of possessions)
        = sum of (player WINNING x share) + sum of (coach effect x share of the season's games he coached) + noise
Players and coaches are fitted together (ridge, penalties chosen by cross-validation on held-out team-seasons),
so a coach is identified by players who played for him AND elsewhere, and by his own moves between teams.
A coach's effect is in points per 100 possessions; x seasons coached -> wins added by coaching.
Why jointly: the earlier coach model (coach_effects.py, play-by-play era) held every player at his own career
level, so a star who spent his whole career under one coach (Duncan, Curry) absorbed that coach's credit, and
coaches of journeyman rosters were measured on everyone. Here the player-and-coach credit for a stable core is
split by the fit, and the coach column spans the whole roster, so he gets his share of it.

Records: data/coaches_src/nba-coaches.csv (every head-coaching tenure since 1949-50: regular-season and playoff
W-L, titles). Titles are league-size adjusted (a title among N teams counts N/30), as for players.

Usage: python src/coach_legacy.py -> out/coach_legacy.csv, out/coach_legacy_meta.json
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

import classic as K

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
HIST = ROOT / "data" / "hist"
# franchise (coach records) -> Basketball-Reference team codes by season (season = year it ends)
FR = {"ATL": [("TRI", 1950, 1951), ("MLH", 1952, 1955), ("STL", 1956, 1968), ("ATL", 1969, 2100)],
      "BKN": [("NYN", 1977, 1977), ("NJN", 1978, 2012), ("BRK", 2013, 2100)],
      "CHA": [("CHH", 1989, 2002), ("CHA", 2005, 2014), ("CHO", 2015, 2100)],
      "DET": [("FTW", 1950, 1957), ("DET", 1958, 2100)],
      "GSW": [("PHW", 1950, 1962), ("SFW", 1963, 1971), ("GSW", 1972, 2100)],
      "HOU": [("SDR", 1968, 1971), ("HOU", 1972, 2100)],
      "LAC": [("BUF", 1971, 1978), ("SDC", 1979, 1984), ("LAC", 1985, 2100)],
      "LAL": [("MNL", 1950, 1960), ("LAL", 1961, 2100)],
      "MEM": [("VAN", 1996, 2001), ("MEM", 2002, 2100)],
      "NOP": [("NOH", 2003, 2005), ("NOK", 2006, 2007), ("NOH", 2008, 2013), ("NOP", 2014, 2100)],
      "OKC": [("SEA", 1968, 2008), ("OKC", 2009, 2100)],
      "PHI": [("SYR", 1950, 1963), ("PHI", 1964, 2100)],
      "PHX": [("PHO", 1969, 2100)],
      "SAC": [("ROC", 1950, 1957), ("CIN", 1958, 1972), ("KCO", 1973, 1975), ("KCK", 1976, 1985), ("SAC", 1986, 2100)],
      "UTA": [("NOJ", 1975, 1979), ("UTA", 1980, 2100)],
      "WAS": [("CHP", 1962, 1962), ("CHZ", 1963, 1963), ("BAL", 1964, 1973), ("CAP", 1974, 1974), ("WSB", 1975, 1997), ("WAS", 1998, 2100)]}


def code(fr, season):
    for c, a, z in FR.get(fr, [(fr, 1900, 2100)]):
        if a <= season <= z:
            return c
    return None


def coach_seasons(ts):
    s = pd.read_csv(ROOT / "data" / "coaches_src" / "nba-coaches.csv", comment="#")
    s = s[s.firstSeason >= 1950]
    games = ts.set_index(["season", "abbreviation"])
    rows = []
    for q in s.itertuples():
        seasons = list(range(int(q.firstSeason), int(q.lastSeason) + 1))
        g = (q.w or 0) + (q.l or 0)
        for y in seasons:
            c = code(q.franchiseAbbrv, y)
            if c is None or (y, c) not in games.index:
                continue
            rows.append({"coach_id": q.leaderId, "coach": q.name, "season": y, "team": c, "fr": q.franchiseAbbrv,
                         "g_per": g / len(seasons)})
    cs = pd.DataFrame(rows)
    # weight within a team-season: share of its games, from each tenure's games per season
    cs["weight"] = cs.g_per / cs.groupby(["season", "team"]).g_per.transform("sum")
    return cs, s


def fit(X, y, w, pen):
    sw = np.sqrt(w / w.mean())
    mu = np.average(y, weights=w)
    A = sp.diags(sw) @ X @ sp.diags(1 / np.sqrt(pen))
    b = lsqr(A, sw * (y - mu), damp=1.0, atol=1e-10, btol=1e-10)[0]
    return b / np.sqrt(pen), mu


def main():
    ts = pd.read_csv(HIST / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")].dropna(subset=["srs", "pace"])
    cs, recs = coach_seasons(ts)
    cs = cs[cs.season <= 2026]                       # 2026-27 is in progress (records still include it)
    c = pd.concat([pd.read_csv(f) for f in glob.glob(str(OUT / "classic_*[0-9].csv"))], ignore_index=True)
    c["season"] = c.season_start + 1
    c = c[(c.lg.fillna("NBA") == "NBA") & (c.poss > 0) & (c.season <= 2026)].copy()
    c["prod"] = c.box_O + c.box_D
    c["share"] = c.poss / (c.groupby(["season", "team"]).poss.transform("sum") / 5)
    tsi = ts.set_index(["season", "abbreviation"])
    c["team_net"] = pd.MultiIndex.from_arrays([c.season, c.team]).map(tsi.srs * 100 / tsi.pace)
    c = c.dropna(subset=["team_net"])
    tkey = c.season.astype(str) + "|" + c.team
    teams = pd.Index(tkey.unique())
    players = pd.Index(c.player_id.unique())
    coaches = pd.Index(cs.coach_id.unique())
    Xp = sp.csr_matrix((c.share, (teams.get_indexer(tkey), players.get_indexer(c.player_id))), shape=(len(teams), len(players)))
    ck = cs.season.astype(str) + "|" + cs.team
    ok = teams.get_indexer(ck) >= 0
    Xc = sp.csr_matrix((cs.weight[ok], (teams.get_indexer(ck[ok]), coaches.get_indexer(cs.coach_id[ok]))), shape=(len(teams), len(coaches)))
    tm = c.assign(e=c["prod"] * c.share, k=tkey).groupby("k").agg(e=("e", "sum"), y=("team_net", "first"), s=("season", "first")).reindex(teams)
    tg = pd.MultiIndex.from_arrays([tm.s, [k.split("|")[1] for k in teams]]).map(tsi.w + tsi.l).to_numpy(dtype=float)
    y, w = (tm.y - tm.e).to_numpy(), tg
    X = sp.hstack([Xp, Xc]).tocsr()
    rng = np.random.default_rng(1)
    fold = rng.integers(0, 5, len(y))
    base = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))

    def cv(lp, lc, with_coach=True):
        XX = X if with_coach else Xp
        pen = np.r_[np.full(Xp.shape[1], lp), np.full(Xc.shape[1], lc)] if with_coach else np.full(Xp.shape[1], lp)
        err = 0.0
        for k in range(5):
            tr = fold != k
            b, mu = fit(XX[tr], y[tr], w[tr], pen)
            err += float(np.sum(w[~tr] * (y[~tr] - mu - XX[~tr] @ b) ** 2))
        return 1 - err / base
    grid = {}
    for lp in (0.5, 1, 2):
        grid[f"no_coach_lp{lp}"] = cv(lp, None, False)
        for lc in (0.25, 1, 4, 16):
            grid[f"lp{lp}_lc{lc}"] = cv(lp, lc)
    best = max((k for k in grid if k.startswith("lp")), key=grid.get)
    lp, lc = float(best.split("_")[0][2:]), float(best.split("_")[1][2:])
    best_nc = max((k for k in grid if k.startswith("no_coach")), key=grid.get)
    print("CV (share of unexplained team strength recovered):", {k: round(v, 4) for k, v in grid.items()})
    pen = np.r_[np.full(Xp.shape[1], lp), np.full(Xc.shape[1], lc)]
    b, mu = fit(X, y, w, pen)
    eff = pd.Series(b[Xp.shape[1]:], index=coaches)
    # uncertainty: bootstrap over seasons (whole seasons resampled)
    seas = tm.s.to_numpy()
    useas = np.unique(seas)
    boots = []
    for _ in range(40):
        pick = rng.choice(useas, len(useas), replace=True)
        idx = np.concatenate([np.flatnonzero(seas == s_) for s_ in pick])
        bb, _ = fit(X[idx], y[idx], w[idx], pen)
        boots.append(bb[Xp.shape[1]:])
    se = pd.Series(np.std(boots, axis=0), index=coaches)

    # records
    kfun, _ = K.win_curve(pd.read_csv(HIST / "Team Summaries.csv"))
    nteams = ts.groupby("season").size()
    rec = recs.groupby(["leaderId", "name"]).agg(w=("w", "sum"), l=("l", "sum"), pw=("playoffW", "sum"), pl=("playoffL", "sum"),
                                                first=("firstSeason", "min"), last=("lastSeason", "max")).reset_index()
    titles = recs.dropna(subset=["titleSeasons"]).assign(t=lambda d: d.titleSeasons.astype(str).str.split()).explode("t")
    titles["t"] = titles.t.astype(float).astype(int)
    tcount = titles.groupby("leaderId").t.nunique()
    tadj = titles.drop_duplicates(["leaderId", "t"]).assign(a=lambda d: d.t.map(nteams).fillna(30) / 30).groupby("leaderId").a.sum()
    teams_of = recs.groupby("leaderId").franchiseAbbrv.agg(lambda x: "/".join(dict.fromkeys(x)))
    nseas = cs.groupby("coach_id").apply(lambda g: g.weight.sum(), include_groups=False)       # full seasons coached
    pace = cs.assign(p=pd.MultiIndex.from_arrays([cs.season, cs.team]).map(tsi.pace)).groupby("coach_id").p.mean()
    out = rec.rename(columns={"leaderId": "coach_id", "name": "coach"}).set_index("coach_id")
    out["teams"] = teams_of
    out["games"] = out.w + out.l
    out["win_pct"] = out.w / out.games.where(out.games > 0)
    out["seasons"] = nseas.reindex(out.index).fillna(0)
    out["impact"] = eff.reindex(out.index)
    out["impact_se"] = se.reindex(out.index)
    # wins added by coaching: per 100 possessions x a standard season (100 possessions, 82 games) via the win curve
    out["wins_per_season"] = out.impact * kfun(100.0) * 82
    out["wins_added"] = out.wins_per_season * out.seasons
    out["wins_above_500"] = (out.w - out.l) / 2
    out["titles"] = tcount.reindex(out.index).fillna(0).astype(int)
    out["titles_adj"] = tadj.reindex(out.index).fillna(0)
    out["playoff_wins"] = out.pw
    out = out[out.games > 0].sort_values("wins_added", ascending=False)
    out.to_csv(OUT / "coach_legacy.csv", float_format="%.3f")
    meta = {"cv": grid, "chosen": {"lambda_player": lp, "lambda_coach": lc}, "best_without_coaches": grid[best_nc],
            "best_with_coaches": grid[best], "n_team_seasons": int(len(y)), "n_coaches": int(len(coaches))}
    (OUT / "coach_legacy_meta.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 220)
    show = ["coach", "teams", "first", "last", "seasons", "win_pct", "impact", "impact_se", "wins_added", "titles", "playoff_wins"]
    print(out[out.seasons >= 5].head(30)[show].round(2).to_string())


if __name__ == "__main__":
    main()
