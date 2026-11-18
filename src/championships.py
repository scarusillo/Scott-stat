"""Championship value — how much each player raised his team's odds of winning the title.

Rings depend on teammates, coaches and luck, so they aren't counted directly.
Instead, every season since 1951-52:
  1. Title odds from team strength: a conditional logit across that season's
     teams on regular-season SRS (margin adjusted for schedule):
         P(team wins title) = exp(b * SRS) / sum over teams of exp(b * SRS)
     b is fit by maximum likelihood on every season's actual champion. Because
     odds rise steeply at the top, the same points matter far more on a contender.
  2. A player's share of his team's strength: (his rating - replacement level)
     x his share of the team's possessions, in points per game.
  3. Championship value = his team's title odds minus its odds with a
     replacement-level player in his minutes. Summed over a career:
     "championships added". Per season it is a probability (0.35 = 35 points of
     title odds). Uses regular-season strength only, so it misses teams that
     won from modest regular seasons (Hakeem's Rockets).
  4. Title shares: every actual title (1.0) split among the champion's players
     by how much each multiplied the team's title odds (exp(b x his points per
     game added in those playoffs) - 1). Also a league-size version: a title
     among N teams counts N/30.
Ratings: Classic (box + team + game logs, regular season), every era.

Usage: python src/championships.py  -> out/championships_seasons.csv, out/championships_careers.csv,
                                       out/championships_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
HIST = ROOT / "data" / "hist"
KAG = ROOT / "data" / "kaggle_dl"
REPL_WINPCT = 0.20


def champions():
    g = pd.read_csv(KAG / "Games.csv", usecols=["gameId", "gameDateTimeEst", "hometeamId", "awayteamId", "hometeamCity",
                                                 "hometeamName", "awayteamCity", "awayteamName", "winner", "gameType"],
                    low_memory=False)
    g = g[g.gameType == "Playoffs"].copy()
    dt = pd.to_datetime(g.gameDateTimeEst, errors="coerce")
    # playoff game ids are 4YY#####, YY = season START year (dates fail for the Oct-2020 bubble Finals)
    yy = g.gameId.astype(int) // 100000 % 100
    g["season"] = np.where(yy >= 46, 1900 + yy, 2000 + yy) + 1
    last = g.loc[dt.groupby(g.season).idxmax()]
    name = np.where(last.winner == last.hometeamId, last.hometeamCity + " " + last.hometeamName,
                    last.awayteamCity + " " + last.awayteamName)
    return pd.Series(name, index=last.season)


def main():
    ts = pd.read_csv(HIST / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")].dropna(subset=["srs"])
    ts = ts[ts.season.between(1952, 2026)].copy()
    ts["team_clean"] = ts.team.str.replace("*", "", regex=False).str.strip()
    champ = champions()
    ts["champion"] = ts.team_clean == ts.season.map(champ)
    found = ts.groupby("season").champion.sum()
    seasons_ok = found[found == 1].index
    fit = ts[ts.season.isin(seasons_ok)]

    def nll(b):
        z = b * fit.srs
        lse = z.groupby(fit.season).transform(lambda x: np.log(np.exp(x - x.max()).sum()) + x.max())
        return -(z - lse)[fit.champion].sum()
    b = minimize_scalar(nll, bounds=(0.01, 3), method="bounded").x
    ts["p_title"] = np.exp(b * ts.srs) / ts.groupby("season").srs.transform(lambda x: np.exp(b * x).sum())
    top_srs_won = fit.loc[fit.groupby("season").srs.idxmax(), "champion"].mean()

    # replacement level per season (same as Classic): a team of freely available players wins 20%
    rows = []
    for season, t in ts.groupby("season"):
        t = t.dropna(subset=["mov", "w", "l", "pace"])
        k = np.polyfit(t.mov, t.w / (t.w + t.l), 1)[0]
        rows.append({"season": season, "repl": (REPL_WINPCT - 0.5) / k / (t.pace.mean() / 100) / 5})
    repl = pd.DataFrame(rows).set_index("season").repl

    pl = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    pl["season"] = pl.season_start + 1
    team_poss = pl.groupby(["season", "team"]).poss.sum() / 5
    pl = pl.join(team_poss.rename("team_poss"), on=["season", "team"])
    tsi = ts.set_index(["season", "abbreviation"])
    pl["srs"] = pd.MultiIndex.from_arrays([pl.season, pl.team]).map(tsi.srs)
    pl["pace"] = pd.MultiIndex.from_arrays([pl.season, pl.team]).map(tsi.pace)
    pl["p_title"] = pd.MultiIndex.from_arrays([pl.season, pl.team]).map(tsi.p_title)
    pl = pl.dropna(subset=["srs", "pace", "team_poss"])
    # points per game his presence (vs replacement in his minutes) adds to his team
    pl["delta_srs"] = (pl.net_classic - pl.season.map(repl)) * (pl.poss / pl.team_poss) * pl.pace / 100
    denom = ts.groupby("season").srs.apply(lambda x: np.exp(b * x).sum())
    e_with = np.exp(b * pl.srs)
    e_without = np.exp(b * (pl.srs - pl.delta_srs))
    other = pl.season.map(denom) - e_with
    pl["p_without"] = e_without / (other + e_without)
    pl["champ_value"] = pl.p_title - pl.p_without
    pl["won_title"] = pd.MultiIndex.from_arrays([pl.season, pl.team]).map(tsi.champion).fillna(False)
    # TITLE SHARES: each real championship (1.0) split among the champion's players by their
    # contribution in THOSE playoffs: (rating - replacement) x playoff possessions, positive part.
    # League-size version: a title among N teams counts N/30 (one of 30 is harder than one of 8).
    po = pd.read_csv(OUT / "classic_playoffs.csv")[["season_start", "player_id", "po_poss"]]
    pl = pl.merge(po, on=["season_start", "player_id"], how="left")
    pl["po_poss"] = pl.po_poss.fillna(0)
    # Split by each player's effect on the team's TITLE ODDS, from the model above: odds scale with
    # exp(b x team strength), so a player who adds d points per game multiplies them by exp(b x d).
    # Weight = that multiplier minus 1. A star's share is far larger than a second option's, the way
    # title odds actually behave. (An earlier version split by value above replacement, which gave a
    # No. 2 about 80% of the star's share.) d uses his share of the team's playoff possessions.
    team_po = pl.groupby(["season", "team"]).po_poss.transform("sum") / 5
    d_po = (pl.net_classic - pl.season.map(repl)) * (pl.po_poss / team_po.where(team_po > 0)) * pl.pace / 100
    pl["po_contrib"] = np.expm1(b * d_po.fillna(0).clip(lower=0))
    pl["po_contrib_even"] = ((pl.net_classic - pl.season.map(repl)) * pl.po_poss).clip(lower=0)
    tot = pl[pl.won_title].groupby(["season", "team"]).po_contrib.transform("sum")
    pl["title_share"] = 0.0
    pl.loc[pl.won_title, "title_share"] = (pl.loc[pl.won_title, "po_contrib"] / tot.where(tot > 0)).fillna(0)
    n_teams = ts.groupby("season").size()
    pl["title_share_adj"] = pl.title_share * pl.season.map(n_teams) / 30
    cols = ["season_start", "player_id", "player", "team", "net_classic", "poss", "srs", "delta_srs", "p_title",
            "p_without", "champ_value", "won_title", "po_poss", "title_share", "title_share_adj"]
    pl[cols].to_csv(OUT / "championships_seasons.csv", index=False, float_format="%.4f")

    def career(x):
        x = x.sort_values("season")
        roll = x.champ_value.rolling(5).sum()
        i = int(np.nanargmax(roll.to_numpy())) if roll.notna().any() else None
        return pd.Series({"first": int(x.season_start.min()), "last": int(x.season_start.max()),
                          "championships_added": x.champ_value.sum(),
                          "best5": float(roll.iloc[i]) if i is not None else np.nan,
                          "best5_years": f"{int(x.season.iloc[i - 4]) - 1}-{int(x.season.iloc[i]) % 100:02d}" if i is not None else None,
                          "best_season": x.champ_value.max(), "titles": int(x.won_title.sum()),
                          "title_shares": x.title_share.sum(), "title_shares_league_adj": x.title_share_adj.sum(),
                          "biggest_share": x.title_share.max()})
    car = pl.groupby("player").apply(career, include_groups=False).sort_values("championships_added", ascending=False)
    car.to_csv(OUT / "championships_careers.csv", float_format="%.3f")
    meta = {"b": float(b), "seasons_fit": int(len(seasons_ok)), "top_srs_team_won_title": float(top_srs_won),
            "note": "conditional logit of champion on SRS; player value = title odds with him minus with replacement"}
    (OUT / "championships_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    pd.set_option("display.width", 200)
    show = ["titles", "title_shares", "title_shares_league_adj", "biggest_share", "championships_added", "best5"]
    print("BY TITLE SHARES"); print(car.sort_values("title_shares", ascending=False).head(15)[show].round(2).to_string())
    print("BY TITLE SHARES, LEAGUE-SIZE ADJUSTED"); print(car.sort_values("title_shares_league_adj", ascending=False).head(15)[show].round(2).to_string())


if __name__ == "__main__":
    main()
