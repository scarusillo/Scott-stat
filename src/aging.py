"""Aging curves — how impact changes with age, and when players peak.

Delta method on Classic ratings (every era, 1951-52 on): for every player with
1,000+ possessions in consecutive seasons, the change from age a to a+1,
averaged (weighted by the smaller of the two seasons' possessions), then
chained into a curve set to 0 at its peak. Because players who decline lose
their minutes and drop out of the sample, the delta method understates late
decline a little (survivor bias); it is the standard approach.
Groups: all, by position (guards / wings / bigs), by era, and by player type
(archetypes, 1996-97 on).

Usage: python src/aging.py   -> out/aging_curves.csv, out/aging_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
MIN_POSS, MIN_PAIRS = 1000, 40


def curve(pairs):
    g = pairs.groupby("age").apply(lambda x: pd.Series({"d": np.average(x.delta, weights=x.w), "n": len(x)}),
                                   include_groups=False)
    g = g[g.n >= MIN_PAIRS].sort_index()
    if g.empty:
        return None
    ages = [int(g.index[0])] + [int(a) + 1 for a in g.index]
    vals = np.r_[0, np.cumsum(g.d.to_numpy())]
    vals -= vals.max()
    return pd.DataFrame({"age": ages, "value": vals, "pairs": np.r_[g.n.to_numpy(), np.nan]})


def main():
    a = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1951, 2026)], ignore_index=True)
    a = a[(a.poss >= MIN_POSS) & a.age.notna()]
    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet")
    arch_main = arch.groupby("player_id").archetype.agg(lambda x: x.mode().iat[0])
    a["type"] = a.nba_id.map(arch_main)
    pos = a.pos.astype(str).str.split("-").str[0]
    a["posgroup"] = np.where(pos.isin(["PG", "SG"]), "Guards", np.where(pos == "SF", "Wings", "Bigs"))
    a["era"] = np.where(a.season_start < 1980, "1951-79", np.where(a.season_start < 2000, "1980-99", "2000-25"))
    nxt = a[["player_id", "season_start", "net_classic", "poss"]].assign(season_start=a.season_start - 1)
    p = a.merge(nxt, on=["player_id", "season_start"], suffixes=("", "_next"))
    p["delta"] = p.net_classic_next - p.net_classic
    p["w"] = np.minimum(p.poss, p.poss_next)
    out, meta = [], {}
    groups = {"All players": p}
    for col in ("posgroup", "era", "type"):
        for k, x in p.groupby(col):
            groups[k] = x
    for name, x in groups.items():
        c = curve(x)
        if c is None:
            continue
        c.insert(0, "group", name)
        out.append(c)
        top = c[c.value >= -0.15]
        meta[name] = {"peak_age": int(c.loc[c.value.idxmax(), "age"]), "prime_from": int(top.age.min()),
                      "prime_to": int(top.age.max()), "pairs": int(len(x)),
                      "at_22": float(c[c.age == 22].value.iloc[0]) if (c.age == 22).any() else None,
                      "at_34": float(c[c.age == 34].value.iloc[0]) if (c.age == 34).any() else None}
    pd.concat(out).to_csv(OUT / "aging_curves.csv", index=False, float_format="%.3f")
    (OUT / "aging_meta.json").write_text(json.dumps(meta, indent=1))
    print(pd.DataFrame(meta).T.to_string())


if __name__ == "__main__":
    main()
