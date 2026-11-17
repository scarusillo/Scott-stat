"""Does a star make his teammates better? — LeBron vs Jordan and other stars.

A star's own impact rating already absorbs any lift he gives teammates (in
lineup data "makes others better" and "is good" are the same column), so this
measures the lift directly, from the teammates' side:

Teammate shot-making: every teammate's true shooting with the star ON vs OFF
the floor, same season (so no aging or roster change). Each team's #1 player
gets some on/off lift for free (he plays with starters against starters), so
every star is scored RELATIVE to that season's league baseline: the same
number computed for every team's #1 player by possessions. Also by teammate
type (archetype), and for shot diet: 3PA share, rim share, assisted share.

Jordan's play-by-play covers 1996-98 (Bulls, titles 5-6) and 2001-03 (Wizards,
age 38-40) only; nothing from 1984-96 exists in this data.

Usage: python src/teammates.py      -> out/teammates_seasons.csv, out/teammates_stars.csv,
                                       out/teammates_by_type.csv
"""
import numpy as np
import pandas as pd

from talent import OUT, ROOT

SHOTS = ROOT / "data" / "shots"
MIN_TSA = 50
STARS = ["LeBron James", "Michael Jordan", "Kobe Bryant", "Tim Duncan", "Shaquille O'Neal", "Kevin Garnett",
         "Dirk Nowitzki", "Steve Nash", "Chris Paul", "Jason Kidd", "Allen Iverson", "Dwyane Wade",
         "Kevin Durant", "Stephen Curry", "James Harden", "Russell Westbrook", "Kawhi Leonard",
         "Giannis Antetokounmpo", "Nikola Jokić", "Luka Dončić", "Shai Gilgeous-Alexander"]
ERAS = {  # (player, first season, last season) -> label
    ("Michael Jordan", 1996, 1997): "Jordan · Bulls 1996-98", ("Michael Jordan", 2001, 2002): "Jordan · Wizards 2001-03",
    ("LeBron James", 2003, 2009): "LeBron · Cavs 2003-10", ("LeBron James", 2010, 2013): "LeBron · Heat 2010-14",
    ("LeBron James", 2014, 2017): "LeBron · Cavs 2014-18", ("LeBron James", 2018, 2024): "LeBron · Lakers 2018-25",
}


def ids_by_name():
    pl = pd.concat([pd.read_parquet(ROOT / "data" / "stints" / f"players_{y}.parquet") for y in range(1996, 2025)])
    return pl.groupby("name").player_id.agg(lambda x: x.mode().iat[0])


def onoff(shots, star, team_games):
    """Per teammate: TSA/pts/3PA/rim/assisted with the star on vs off the floor."""
    s = shots[shots.game_id.isin(team_games) & (shots.shooter != star)].copy()
    s["on"] = (s[["on0", "on1", "on2", "on3", "on4"]] == star).any(axis=1)
    fg = s.kind == "fg"
    s["tsa"] = np.where(fg, 1.0, 0.44)
    s["fga"] = fg.astype(int)
    s["fg3a"] = (fg & s.three).astype(int)
    s["rima"] = (fg & s.rim).astype(int)
    s["fgm"] = (fg & s.made).astype(int)
    s["astd"] = (fg & s.assisted).astype(int)
    g = s.groupby(["shooter", "on"])[["tsa", "pts", "fga", "fg3a", "rima", "fgm", "astd"]].sum().unstack("on")
    g.columns = [f"{a}_{'on' if b else 'off'}" for a, b in g.columns]
    g = g.fillna(0)
    g = g[(g.tsa_on >= MIN_TSA) & (g.tsa_off >= MIN_TSA)]
    for k in ("on", "off"):
        g[f"ts_{k}"] = g[f"pts_{k}"] / (2 * g[f"tsa_{k}"])
        g[f"three_share_{k}"] = g[f"fg3a_{k}"] / g[f"fga_{k}"]
        g[f"rim_share_{k}"] = g[f"rima_{k}"] / g[f"fga_{k}"]
        g[f"astd_share_{k}"] = g[f"astd_{k}"] / g[f"fgm_{k}"].clip(lower=1)
    g["w"] = 2 / (1 / g.tsa_on + 1 / g.tsa_off)      # harmonic mean: both splits must be big
    return g


def summarise(g):
    w = g.w
    return pd.Series({
        "teammates": len(g),
        "ts_lift": np.average(g.ts_on - g.ts_off, weights=w),
        "three_share_lift": np.average(g.three_share_on - g.three_share_off, weights=w),
        "rim_share_lift": np.average(g.rim_share_on - g.rim_share_off, weights=w),
        "astd_lift": np.average(g.astd_share_on - g.astd_share_off, weights=w),
        "tsa_on": g.tsa_on.sum(),
    })


def main():
    ids = ids_by_name()
    star_ids = {n: int(ids[n]) for n in STARS}
    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet").set_index(["season", "player_id"]).archetype
    rows, type_rows = [], []
    for y in range(1996, 2025):
        shots = pd.read_parquet(SHOTS / f"shots_{y}.parquet")
        war = pd.read_csv(OUT / f"war_{y}.csv")
        team_games = shots.groupby("team_id").game_id.unique()
        # every team's #1 player by possessions = the league baseline
        top = war.dropna(subset=["team_id"]).sort_values("poss").groupby("team_id").tail(1)
        cands = {(int(r.player_id), int(r.team_id)): "baseline" for r in top.itertuples()}
        for n, pid in star_ids.items():
            w = war[war.player_id == pid]
            if len(w) and w.poss.iat[0] >= 1000:
                cands[(pid, int(w.team_id.iat[0]))] = n
        for (pid, tid), who in cands.items():
            if tid not in team_games.index:
                continue
            g = onoff(shots, pid, team_games[tid])
            if len(g) == 0:
                continue
            r = summarise(g)
            names = [k for k, v in star_ids.items() if v == pid]
            rows.append({"season": y, "player_id": pid, "team_id": tid,
                         "star": names[0] if names else None, "is_top": who == "baseline" or (pid, tid) in
                         {(int(t.player_id), int(t.team_id)) for t in top.itertuples()}, **r})
            if names:
                g["archetype"] = arch.reindex(pd.MultiIndex.from_arrays([[y] * len(g), g.index])).to_numpy()
                g["season"], g["star"] = y, names[0]
                type_rows.append(g.reset_index())
        print(y, end=" ", flush=True)
    print()
    d = pd.DataFrame(rows)
    base = d[d.is_top].groupby("season")[["ts_lift", "three_share_lift", "rim_share_lift", "astd_lift"]].mean()
    for c in base.columns:
        d[c + "_vs_league"] = d[c] - d.season.map(base[c])
    d.to_csv(OUT / "teammates_seasons.csv", index=False, float_format="%.4f")

    # percentile of each star-season among all #1 players that season
    tops = d[d.is_top]
    d["pct_ts_lift"] = [((tops[tops.season == r.season].ts_lift < r.ts_lift).mean()) for r in d.itertuples()]

    def agg(x):
        w = x.tsa_on
        return pd.Series({"seasons": x.season.nunique(), "teammates": x.teammates.sum(),
                          "ts_lift_vs_league": np.average(x.ts_lift_vs_league, weights=w),
                          "three_share_vs_league": np.average(x.three_share_lift_vs_league, weights=w),
                          "rim_share_vs_league": np.average(x.rim_share_lift_vs_league, weights=w),
                          "astd_vs_league": np.average(x.astd_lift_vs_league, weights=w),
                          "median_pct": x.pct_ts_lift.median(),
                          "se": np.std(x.ts_lift_vs_league, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan})
    st = d[d.star.notna()]
    out = st.groupby("star").apply(agg, include_groups=False)
    eras = []
    for (n, a, b), label in ERAS.items():
        x = st[(st.star == n) & st.season.between(a, b)]
        if len(x):
            eras.append(agg(x).rename(label))
    out = pd.concat([out, pd.DataFrame(eras)]).sort_values("ts_lift_vs_league", ascending=False)
    out.to_csv(OUT / "teammates_stars.csv", float_format="%.4f")

    t = pd.concat(type_rows)
    league_ts = base.ts_lift
    t["lift_vs_league"] = (t.ts_on - t.ts_off) - t.season.map(league_ts)
    t["era"] = None
    for (n, a, b), label in ERAS.items():
        t.loc[(t.star == n) & t.season.between(a, b), "era"] = label
    bt = t.groupby(["star", "archetype"]).apply(
        lambda x: pd.Series({"teammate_seasons": len(x), "ts_lift_vs_league": np.average(x.lift_vs_league, weights=x.w),
                             "tsa_on": x.tsa_on.sum()}), include_groups=False).reset_index()
    bt.to_csv(OUT / "teammates_by_type.csv", index=False, float_format="%.4f")

    pd.set_option("display.width", 220)
    cols = ["seasons", "teammates", "ts_lift_vs_league", "se", "median_pct", "three_share_vs_league",
            "rim_share_vs_league", "astd_vs_league"]
    show = out[cols].copy()
    for c in ("ts_lift_vs_league", "se", "three_share_vs_league", "rim_share_vs_league", "astd_vs_league"):
        show[c] = (show[c] * 100).round(1)   # percentage points
    show["median_pct"] = (show.median_pct * 100).round(0)
    print(show.to_string())
    print(bt[bt.star.isin(["LeBron James", "Michael Jordan"]) & (bt.tsa_on >= 1500)]
          .assign(ts_lift_vs_league=lambda x: (x.ts_lift_vs_league * 100).round(1)).to_string(index=False))


if __name__ == "__main__":
    main()
