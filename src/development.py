"""Development — do young players improve faster under some coaches?

Unit: a young player's season-to-season change in rating (wins.py season
ratings) when he plays both seasons for the SAME coach — so the coach's
general level effect cancels and what's left is growth.
Expected change comes from a league regression on experience (years 1-5,
first seen 1997+ so experience is known) and on the starting rating (high
seasons regress, low ones bounce). A coach's development score = his
players' average beat-the-expectation change, shrunk by empirical Bayes
(how much coaches truly differ vs. sampling noise).

Usage: python src/development.py   -> out/coach_development.csv
"""
import json

import numpy as np
import pandas as pd

from coach_effects import load
from talent import OUT, first_seen

MIN_POSS = 1000
MAX_EXP = 5


def coach_of_player_seasons(s):
    """(season, player) -> coach under whom the player had most possessions."""
    recs = []
    for cols, coach in (([f"h{i}" for i in range(5)], "coach_h"), ([f"a{i}" for i in range(5)], "coach_a")):
        for c in cols:
            recs.append(pd.DataFrame({"season": s.season.to_numpy(), "player_id": s[c].astype("int64").to_numpy(),
                                      "coach_id": s[coach].to_numpy(),
                                      "p": ((s.poss_h + s.poss_a) / 2).to_numpy()}))
    L = pd.concat(recs).groupby(["season", "player_id", "coach_id"]).p.sum().reset_index()
    L["share"] = L.p / L.groupby(["season", "player_id"]).p.transform("sum")
    L = L.sort_values("p").groupby(["season", "player_id"]).tail(1)
    return L.set_index(["season", "player_id"])


def main():
    s, cg = load()
    fs = first_seen(s)
    war = pd.concat([pd.read_csv(OUT / f"war_{y}.csv") for y in range(1996, 2025)])
    war = war[war.poss >= MIN_POSS].set_index(["season", "player_id"])
    ch = coach_of_player_seasons(s)
    a = war[["net"]].join(ch[["coach_id", "share"]]).reset_index()
    b = a.assign(season=a.season - 1).rename(columns={"net": "net1", "coach_id": "coach1", "share": "share1"})
    pairs = a.merge(b, on=["season", "player_id"])
    pairs["first"] = pairs.player_id.map(fs)
    pairs["exp1"] = pairs.season + 1 - pairs["first"]
    pairs = pairs[(pairs["first"] > 1996) & (pairs.exp1 >= 1) & (pairs.exp1 <= MAX_EXP)
                  & (pairs.coach_id == pairs.coach1) & (pairs.share >= 0.8) & (pairs.share1 >= 0.8)]
    pairs["delta"] = pairs.net1 - pairs.net
    # expected change: experience dummies + starting level
    Xd = pd.get_dummies(pairs.exp1.astype(int), prefix="e").astype(float)
    Xd["start"] = pairs.net
    beta, *_ = np.linalg.lstsq(Xd.to_numpy(), pairs.delta.to_numpy(), rcond=None)
    pairs["resid"] = pairs.delta - Xd.to_numpy() @ beta

    g = pairs.groupby("coach_id").resid.agg(["mean", "count", "var"])
    sig2 = pairs.resid.var()
    tau2 = max(1e-6, np.average((g["mean"] - pairs.resid.mean()) ** 2 - sig2 / g["count"], weights=g["count"]))
    g["dev_score"] = g["mean"] * g["count"] / (g["count"] + sig2 / tau2)
    g["se"] = np.sqrt(1 / (g["count"] / sig2 + 1 / tau2))
    names = cg.groupby("coach_id").coach.first()
    g.insert(0, "coach", g.index.map(names))
    g = g.rename(columns={"mean": "raw_mean", "count": "player_seasons"}).sort_values("dev_score", ascending=False)
    g.to_csv(OUT / "coach_development.csv", float_format="%.3f")

    # reliability: split each coach's player-pairs by even/odd season
    h = pairs.assign(h=pairs.season % 2).groupby(["coach_id", "h"]).resid.agg(["mean", "count"]).unstack()
    ok = (h["count"][0] >= 6) & (h["count"][1] >= 6)
    r = np.corrcoef(h["mean"][0][ok], h["mean"][1][ok])[0, 1]
    meta = {"pairs": int(len(pairs)), "expected_by_exp": dict(zip(Xd.columns, np.round(beta, 3).tolist())),
            "sd_true_coach_effect": float(np.sqrt(tau2)), "sd_noise_per_player": float(np.sqrt(sig2)),
            "split_half_r": float(r), "split_half_coaches": int(ok.sum())}
    (OUT / "coach_development_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    show = g[g.player_seasons >= 8]
    print(show.head(10)[["coach", "player_seasons", "raw_mean", "dev_score", "se"]].round(2).to_string())
    print(show.tail(5)[["coach", "player_seasons", "raw_mean", "dev_score", "se"]].round(2).to_string())


if __name__ == "__main__":
    main()
