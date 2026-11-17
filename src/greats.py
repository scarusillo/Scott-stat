"""All-time comparison on ONE scale (Classic, 1977-78+), with Plus alongside
wherever play-by-play exists.

Per player: best season, best 3 and best 5 consecutive seasons (possession-
weighted impact per 100), career impact per 100, career Classic WAR, and the
same from Plus where available. Regular season only.

Usage: python src/greats.py   -> out/greats.csv, out/greats_seasons.csv
"""
import numpy as np
import pandas as pd

from classic import OUT

GREATS = ["Michael Jordan", "LeBron James", "Magic Johnson", "Larry Bird", "Kareem Abdul-Jabbar",
          "Hakeem Olajuwon", "Shaquille O'Neal", "Tim Duncan", "Kobe Bryant", "Kevin Garnett", "Stephen Curry",
          "Kevin Durant", "Nikola Jokić", "Giannis Antetokounmpo", "David Robinson", "Charles Barkley",
          "Karl Malone", "John Stockton", "Chris Paul", "Dirk Nowitzki", "Shai Gilgeous-Alexander"]


def best_run(s, n):
    """Best n consecutive seasons by possession-weighted net."""
    s = s.sort_values("season_start")
    if len(s) < n:
        return np.nan, None
    best, at = -99, None
    for i in range(len(s) - n + 1):
        w = s.iloc[i:i + n]
        if w.season_start.iat[-1] - w.season_start.iat[0] != n - 1:
            continue
        v = np.average(w.net_classic, weights=w.poss)
        if v > best:
            best, at = v, f"{w.season_start.iat[0]}-{(w.season_start.iat[-1] + 1) % 100:02d}"
    return (best if at else np.nan), at


def main():
    a = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1977, 2025)], ignore_index=True)
    a = a[a.player.isin(GREATS)]
    rows = []
    for name, s in a.groupby("player"):
        s = s[s.poss >= 1500]
        top = s.loc[s.net_classic.idxmax()]
        b3, at3 = best_run(s, 3)
        b5, at5 = best_run(s, 5)
        pl = s[s.net_plus.notna()]
        rows.append({
            "player": name, "seasons": len(s), "first": int(s.season_start.min()),
            "best_season": round(top.net_classic, 2), "best_season_year": f"{int(top.season_start)}-{(int(top.season_start) + 1) % 100:02d}",
            "best3": round(b3, 2), "best3_years": at3, "best5": round(b5, 2), "best5_years": at5,
            "career_per100": round(np.average(s.net_classic, weights=s.poss), 2),
            "career_WAR": round(s.WAR_classic.sum(), 1),
            "plus_seasons": len(pl),
            "plus_best": round(pl.net_plus.max(), 2) if len(pl) else np.nan,
            "plus_career_per100": round(np.average(pl.net_plus, weights=pl.poss), 2) if len(pl) else np.nan,
            "classic_minus_plus_same_seasons": round(np.average(pl.net_classic - pl.net_plus, weights=pl.poss), 2) if len(pl) else np.nan,
        })
    g = pd.DataFrame(rows).sort_values("best5", ascending=False)
    g.to_csv(OUT / "greats.csv", index=False)
    a.to_csv(OUT / "greats_seasons.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 250)
    print(g.to_string(index=False))
    for n in ("Michael Jordan", "LeBron James"):
        s = a[(a.player == n) & (a.poss >= 1500)].sort_values("season_start")
        print("\n" + n)
        print(s[["season_start", "team", "age", "net_classic", "O_classic", "D_classic", "WAR_classic", "net_plus"]]
              .round(1).to_string(index=False))


if __name__ == "__main__":
    main()
