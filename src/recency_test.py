"""Recency-bias checks for the Classic tier.

1. Like-for-like team credit: top-10 individual players each season on teams with margin
   +4 to +9, by era. A fair method gives similar individual + team credit totals across eras.
2. Scale tilt: average rating of each season's best / top-3 / top-10 players by era, and each
   era's share of the 100 best seasons against its share of all player-seasons.
3. Player-type check: for star seasons with play-by-play, Plus minus Classic by position.
4. How sure is the order at the top? Simulate each five-season peak with its error
   (season error SD from classic_meta, / sqrt(5)) and count how often each player is first.

Usage: python src/recency_test.py   -> out/recency_test.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
ERAS = ([1950, 1964, 1976, 1984, 1995, 2004, 2014, 2025],
        ["1951-64", "1965-76", "1977-84", "1985-95", "1996-2004", "2005-14", "2015-25"])


def main():
    cl = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    ts = pd.read_csv(ROOT / "data" / "hist" / "Team Summaries.csv")
    ts = ts[(ts.lg == "NBA") & (ts.team != "League Average")].assign(season_start=lambda x: x.season - 1)
    cl["ind"] = cl.box_O + cl.box_D
    cl["credit"] = cl.net_classic - cl.ind
    cl = cl.merge(ts[["season_start", "abbreviation", "n_rtg"]], left_on=["season_start", "team"],
                  right_on=["season_start", "abbreviation"], how="left")
    q = cl[cl.poss >= 3000].copy()
    q["era"] = pd.cut(q.season_start, ERAS[0], labels=ERAS[1])
    q["rank_ind"] = q.groupby("season_start").ind.rank(ascending=False)
    q["rank"] = q.groupby("season_start").net_classic.rank(ascending=False)
    like = q[(q.rank_ind <= 10) & q.n_rtg.between(4, 9)]
    t1 = like.groupby("era", observed=True).agg(n=("ind", "size"), individual=("ind", "mean"), team_credit=("credit", "mean"),
                                                impact=("net_classic", "mean"), team_margin=("n_rtg", "mean"))
    t2 = q.groupby("era", observed=True).apply(lambda x: pd.Series({
        "best_player": x[x["rank"] == 1].net_classic.mean(), "top3": x[x["rank"] <= 3].net_classic.mean(),
        "top10": x[x["rank"] <= 10].net_classic.mean(), "player_seasons": len(x)}), include_groups=False)
    t2["share_of_top100"] = q.nlargest(100, "net_classic").era.value_counts().reindex(t2.index).fillna(0) / 100
    t2["share_of_all"] = t2.player_seasons / t2.player_seasons.sum()
    x = cl[cl.net_plus.notna() & (cl.poss >= 3000) & (cl.ind >= 3.5)]
    t3 = (x.net_plus - x.net_classic).groupby(x.pos.astype(str).str.split("-").str[0]).agg(["mean", "count"])
    car = pd.read_csv(OUT / "scott_careers_classic.csv")
    top = car[car.seasons >= 6].nlargest(6, "best5")
    se = json.load(open(OUT / "classic_meta.json"))["resid_sd_net"] / np.sqrt(5)
    rng = np.random.default_rng(0)
    draws = rng.normal(top.best5.to_numpy()[:, None], se, (len(top), 50000))
    first = pd.Series(np.argmax(draws, axis=0)).value_counts(normalize=True)
    t4 = {top.player.iloc[i]: round(float(v), 3) for i, v in first.items()}
    out = {"like_for_like": t1.round(2).reset_index().to_dict("records"), "scale": t2.round(2).reset_index().to_dict("records"),
           "plus_minus_classic_for_stars": t3.round(2).reset_index().to_dict("records"), "peak_se": round(float(se), 2),
           "p_best_peak": t4}
    (OUT / "recency_test.json").write_text(json.dumps(out, indent=1, default=str))
    pd.set_option("display.width", 200)
    print(t1.round(2).to_string()); print(t2.round(2).to_string()); print(t3.round(2).to_string())
    print("peak SE", round(se, 2), "P(best peak):", t4)


if __name__ == "__main__":
    main()
