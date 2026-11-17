"""Playoffs — clutch in the playoffs, clutch overall, and playoff elevation.

Same machinery as clutch.py (results vs what the ten players' NORMAL regular-
season ratings predict, ridge-credited to players, penalty by CV, split-half
reliability on even/odd seasons), run three ways:
  playoff_clutch   last 5 min of 4th/OT within 5 points, playoff games only
  all_clutch       regular season + playoffs together
  elevation        EVERY playoff possession: does he play above his regular-season self?
Playoff opponents are better than average; the expectation already uses every
opponent's own rating, so that is accounted for.

Usage: python src/playoffs.py   -> out/playoffs_players.csv, out/playoffs_meta.json
"""
import json

import numpy as np
import pandas as pd

import clutch as C

RUNS = {"playoff_clutch": (("po_",), True), "all_clutch": (("", "po_"), True), "elevation": (("po_",), False)}


def main():
    war = pd.concat([pd.read_csv(C.OUT / f"war_{y}.csv").assign(season=y) for y in range(1996, 2025)])
    names = war.dropna(subset=["name"]).groupby("player_id").name.last()
    normal = war.groupby("player_id").apply(lambda x: np.average(x.net, weights=x.poss.clip(lower=1)),
                                            include_groups=False).rename("normal_net")
    tables, meta = [], {}
    for run, (tags, clutch_only) in RUNS.items():
        s = C.clutch_stints(tags, clutch_only)
        imp, m = C.clutch_impact(s)
        meta[run] = {"stints": int(len(s)), **m}
        imp = imp.set_index("player_id")[["clutch_poss", "clutch_d_O", "clutch_d_D", "clutch_delta", "se_delta"]]
        imp.columns = [f"{run}_{c.replace('clutch_', '')}" for c in imp.columns]
        tables.append(imp)
        print(run, json.dumps(meta[run]["split_half"]), "lambda", meta[run]["lambda"], flush=True)
    d = pd.concat(tables, axis=1).join(normal)
    d.insert(0, "name", names.reindex(d.index))
    d.to_csv(C.OUT / "playoffs_players.csv", float_format="%.3f")
    (C.OUT / "playoffs_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    pd.set_option("display.width", 220)
    for run, min_poss in (("playoff_clutch", 250), ("all_clutch", 1800), ("elevation", 4000)):
        x = d[d[f"{run}_poss"] >= min_poss].copy()
        x["level"] = x.normal_net + x[f"{run}_delta"]
        cols = ["name", f"{run}_poss", "normal_net", f"{run}_delta", f"{run}_se_delta", f"{run}_d_O", f"{run}_d_D", "level"]
        print(f"\n== {run}: biggest rise ({min_poss}+ poss)")
        print(x.sort_values(f"{run}_delta", ascending=False).head(10)[cols].round(2).to_string(index=False))
        print(f"== {run}: best level")
        print(x.sort_values("level", ascending=False).head(8)[cols].round(2).to_string(index=False))
    famous = ["LeBron James", "Michael Jordan", "Kobe Bryant", "Tim Duncan", "Dirk Nowitzki", "Stephen Curry",
              "Kevin Durant", "Damian Lillard", "Kawhi Leonard", "Shaquille O'Neal", "Nikola Jokić",
              "Giannis Antetokounmpo", "Jimmy Butler", "Kyrie Irving", "Chris Paul", "James Harden",
              "Kevin Garnett", "Paul Pierce", "Dwyane Wade", "Russell Westbrook"]
    cols = ["name", "normal_net"] + [f"{r}_{c}" for r in RUNS for c in ("poss", "delta", "se_delta")]
    print("\n== reputation check")
    print(d[d.name.isin(famous)].sort_values("all_clutch_delta", ascending=False)[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
