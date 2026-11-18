"""Should team credit be split evenly? Test on held-out seasons.

Team credit = the gap between a team's actual offense/defense and the sum of its
players' box estimates. Variants for how a player's share depends on him:
  even      every player the same per possession (the original)
  +rating   share also scales with how far his own box rating is above his team's average
  +minutes  share also scales with his minutes per game relative to his team's average
  +both
Coefficients fit on 1996-97..2012-13 (target: Plus minus box, offense and defense
separately), judged on 2013-14..2024-25. Also the error for stars (box net >= 4).

Usage: python src/team_credit_test.py
"""
import json

import numpy as np
import pandas as pd

import classic as K
from era_audit import prepared

OUT = K.OUT


def gaps(d, per_team, ts):
    pt = per_team[["season", "player_id", "team", "poss", "mp", "g"]].merge(
        d[["season", "player_id", "box_O", "box_D"]], on=["season", "player_id"])
    pt["mpg"] = pt.mp / pt.g.clip(lower=1)
    grp = pt.groupby(["season", "team"])
    w = pt.poss
    tot = grp.poss.transform("sum")
    for c in ("box_O", "box_D", "mpg"):
        pt["m_" + c] = (pt[c] * w).groupby([pt.season, pt.team]).transform("sum") / tot
    tsi = ts.set_index(["season", "abbreviation"])
    lg = ts.groupby("season").o_rtg.mean()
    key = pd.MultiIndex.from_arrays([pt.season, pt.team])
    pt["gap_O"] = (tsi.o_rtg.reindex(key).to_numpy() - pt.season.map(lg).to_numpy()) - 5 * pt.m_box_O
    pt["gap_D"] = -(tsi.d_rtg.reindex(key).to_numpy() - pt.season.map(lg).to_numpy()) - 5 * pt.m_box_D
    for s in ("O", "D"):
        pt[f"g_{s}"] = pt[f"gap_{s}"] / 5
        pt[f"gz_{s}"] = pt[f"g_{s}"] * (pt[f"box_{s}"] - pt[f"m_box_{s}"])
        pt[f"gm_{s}"] = pt[f"g_{s}"] * (pt.mpg - pt.m_mpg) / 10
    cols = [f"{a}_{s}" for a in ("g", "gz", "gm") for s in ("O", "D")]
    agg = pt[cols].multiply(pt.poss, axis=0).groupby([pt.season, pt.player_id]).sum().div(
        pt.groupby(["season", "player_id"]).poss.sum(), axis=0)
    return d.join(agg, on=["season", "player_id"])


def main():
    d, per_team, ts = prepared()
    meta = json.load(open(OUT / "classic_meta.json"))
    tau = meta["validation"]["tau"]
    base = d.nba_id.notna() & (d.plus_poss >= 250)
    f = gaps(K.era_fit(d, base & (d.season <= 2013), tau), per_team, ts).dropna(subset=["g_O", "g_D"])
    tr = f.nba_id.notna() & (f.season <= 2013) & (f.plus_poss >= 250)
    te = f.nba_id.notna() & (f.season >= 2014) & (f.plus_poss >= 1000)
    variants = {"even": ["g"], "+rating": ["g", "gz"], "+minutes": ["g", "gm"], "+both": ["g", "gz", "gm"]}
    rows, coefs = [], {}
    for name, terms in variants.items():
        pred = {}
        for s in ("O", "D"):
            X = f.loc[tr, [f"{t}_{s}" for t in terms]].to_numpy()
            y = (f.loc[tr, s] - f.loc[tr, f"box_{s}"]).to_numpy()
            w = f.loc[tr, "plus_poss"].to_numpy()
            b = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
            coefs[f"{name}_{s}"] = dict(zip(terms, np.round(b, 3)))
            pred[s] = f[f"box_{s}"] + f[[f"{t}_{s}" for t in terms]].to_numpy() @ b
        net = pred["O"] + pred["D"]
        x = f[te]
        e = net[te] - x.net
        stars = (x.box_O + x.box_D) >= 4
        rows.append({"variant": name, "r_net": np.corrcoef(net[te], x.net)[0, 1], "rmse": np.sqrt(np.mean(e ** 2)),
                     "r_O": np.corrcoef(pred["O"][te], x.O)[0, 1], "r_D": np.corrcoef(pred["D"][te], x.D)[0, 1],
                     "stars_n": int(stars.sum()), "stars_bias": e[stars].mean(), "stars_rmse": np.sqrt(np.mean(e[stars] ** 2))})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(json.dumps(coefs, indent=0, default=float))


if __name__ == "__main__":
    main()
