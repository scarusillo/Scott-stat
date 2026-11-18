"""Situation board — is a player above or below his own established level, and how
much of that should we expect to last?

Ratings: Classic box + team credit (`O_box + D_box`), which rates every season on
its own (no information from other seasons), 1977-78 to 2025-26.
Norm: the player's PRIOR seasons only (3,000+ possessions), minus the league aging
curve, so it is a forecast, not hindsight.
  gap        this season minus his norm at this age
  carryover  slope of next season's gap on this season's gap (how much sticks)
  expected   norm + carryover x gap, moved one year along the aging curve
Also: does changing teams change the carryover, and what happens after down
years (gap <= -1.5) and up years (gap >= +1.5)?

Usage: python src/situations.py   -> out/situation_board.csv, out/situations_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MIN_POSS, MIN_PRIOR = 1000, 3000


def main():
    a = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1977, 2026)], ignore_index=True)
    a = a[(a.poss >= MIN_POSS) & a.age.notna()].copy()
    a["rating"] = a.O_box + a.D_box
    a["age"] = a.age.astype(int)
    a = a.rename(columns={"season_start": "season"}).sort_values(["player_id", "season"])
    nxt = a[["player_id", "season", "rating", "poss", "team"]].assign(season=a.season - 1)
    p = a.merge(nxt, on=["player_id", "season"], suffixes=("", "_n"))
    g = p.assign(d=p.rating_n - p.rating, w=np.minimum(p.poss, p.poss_n)).groupby("age").apply(
        lambda x: pd.Series({"d": np.average(x.d, weights=x.w), "n": len(x)}), include_groups=False)
    g = g[g.n >= 40].sort_index()
    curve = pd.Series(np.r_[0, np.cumsum(g.d.to_numpy())], index=[g.index[0]] + [i + 1 for i in g.index])
    curve -= curve.max()
    cl = lambda v: curve.reindex(np.clip(v, curve.index.min(), curve.index.max())).to_numpy()
    a["adj"] = a.rating - cl(a.age.to_numpy())
    cw = a.groupby("player_id").poss.cumsum() - a.poss
    cv = (a.adj * a.poss).groupby(a.player_id).cumsum() - a.adj * a.poss
    a["prior_norm"] = np.where(cw >= MIN_PRIOR, cv / cw.where(cw > 0), np.nan)
    a["gap"] = a.adj - a.prior_norm
    q = a.dropna(subset=["gap"]).merge(a[["player_id", "season", "adj", "team"]].assign(season=a.season - 1)
                                       .rename(columns={"adj": "adj_n", "team": "team_n"}), on=["player_id", "season"])
    q["next_gap"] = q.adj_n - q.prior_norm
    q["moved"] = q.team != q.team_n
    slope = lambda d: float(np.polyfit(d.gap, d.next_gap, 1)[0])
    down, up = q[q.gap <= -1.5], q[q.gap >= 1.5]
    b = slope(q)
    meta = {"pairs": int(len(q)), "carryover": b, "carryover_stayed": slope(q[~q.moved]), "carryover_moved": slope(q[q.moved]),
            "carryover_under_27": slope(q[q.age < 27]), "carryover_27_plus": slope(q[q.age >= 27]),
            "r_gap_vs_next": float(np.corrcoef(q.gap, q.next_gap)[0, 1]),
            "down_years": {"n": int(len(down)), "this": float(down.gap.mean()), "next": float(down.next_gap.mean()),
                           "stayed_next": float(down[~down.moved].next_gap.mean()), "moved_next": float(down[down.moved].next_gap.mean()),
                           "stayed_n": int((~down.moved).sum()), "moved_n": int(down.moved.sum())},
            "up_years": {"n": int(len(up)), "this": float(up.gap.mean()), "next": float(up.next_gap.mean())}}
    a["norm_now"] = a.prior_norm + cl(a.age.to_numpy())
    a["expected_next"] = a.prior_norm + b * a.gap + cl(a.age.to_numpy() + 1)
    last = int(a.season.max())
    cols = ["season", "player_id", "player", "team", "age", "pos", "poss", "rating", "norm_now", "gap", "expected_next", "nba_id"]
    board = a[a.season >= last - 2].dropna(subset=["gap"])[cols]
    board.to_csv(OUT / "situation_board.csv", index=False, float_format="%.3f")
    meta["latest_season"] = last
    (OUT / "situations_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))
    L = board[(board.season == last) & (board.poss >= 2500)]
    c = ["player", "team", "age", "rating", "norm_now", "gap", "expected_next"]
    pd.set_option("display.width", 200)
    print("BELOW OWN NORM (age <= 31)\n", L[L.age <= 31].nsmallest(12, "gap")[c].round(1).to_string(index=False))
    print("ABOVE OWN NORM\n", L.nlargest(12, "gap")[c].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
