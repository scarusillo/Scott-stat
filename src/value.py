"""Value per dollar — wins added against salary (2009-10 to 2024-25).

Salaries: Kaggle ratin21/nba-player-stats-and-salaries-2010-2025 (data/kaggle_salaries/,
not committed). Wins: Plus tier wins above replacement (wins.py).
Each season's market price of a win = league payroll in the file / total wins above
replacement (same season). A player's worth = his wins x that price; surplus = worth - salary.
Surplus is biggest for stars on capped deals and for rookie contracts, by design of the
salary rules; the useful reads are within a salary band or contract type.

Usage: python src/value.py   -> out/value_seasons.csv, out/value_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classic import norm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def main():
    sal = pd.read_csv(ROOT / "data" / "kaggle_salaries" / "salaries.csv", encoding="utf-8-sig")
    sal["season"] = sal.Year - 1
    sal["key"] = sal.Player.map(norm)
    sal = sal.groupby(["season", "key"]).agg(salary=("Salary", "max"), age=("Age", "max")).reset_index()
    war = pd.concat([pd.read_csv(OUT / f"war_{y}.csv") for y in range(2009, 2025)], ignore_index=True)
    war["key"] = war.name.map(norm)
    d = war.merge(sal, on=["season", "key"], how="inner")
    price = d.groupby("season").apply(lambda x: x.salary.sum() / x.WAR.clip(lower=0).sum(), include_groups=False)
    d["price_per_win"] = d.season.map(price)
    d["worth"] = d.WAR * d.price_per_win
    d["surplus"] = d.worth - d.salary
    d["cap_share"] = d.salary / d.groupby("season").salary.transform("max")
    cols = ["season", "player_id", "name", "team_id", "age", "poss", "net", "WAR", "salary", "price_per_win", "worth", "surplus"]
    d[cols].to_csv(OUT / "value_seasons.csv", index=False, float_format="%.3f")
    matched = len(d) / len(war[war.poss >= 500])
    r = np.corrcoef(d[d.poss >= 1500].salary, d[d.poss >= 1500].WAR)[0, 1]
    rp = np.corrcoef(d[d.poss >= 1500].salary, d[d.poss >= 1500].ppg)[0, 1]
    meta = {"seasons": [int(d.season.min()), int(d.season.max())], "rows": int(len(d)),
            "price_per_win_by_season": {int(k): float(v) for k, v in price.items()},
            "r_salary_vs_wins": float(r), "r_salary_vs_ppg": float(rp)}
    (OUT / "value_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "price_per_win_by_season"}), "| price/win 2024-25: $%.1fM" % (price.iloc[-1] / 1e6))
    x = d[d.season == d.season.max()]
    f = lambda t: t.assign(salary=(t.salary / 1e6).round(1), worth=(t.worth / 1e6).round(1), surplus=(t.surplus / 1e6).round(1))[
        ["name", "age", "WAR", "salary", "worth", "surplus"]].round(1).to_string(index=False)
    print("BEST VALUE 2024-25\n" + f(x.nlargest(10, "surplus")))
    print("WORST VALUE 2024-25\n" + f(x.nsmallest(8, "surplus")))


if __name__ == "__main__":
    main()
