"""THE SCOTT STAT — one number per player, built from every layer.

Scott Stat (per 100 possessions) =
      all-game impact (luck-adjusted, adjusted for teammates/opponents)
    + c x clutch rise        c = clutch possessions' share of all win-probability
                                 leverage, measured from the data (not chosen)
    + p x playoff elevation  p = the player's playoff share of his possessions
Every impact number splits into OWN PRODUCTION (what his own box score
explains, Layer 3) and LIFT (the rest: teammates playing better and the team
defending better while he's out there).

Two tiers, never mixed in one ranking:
  PLUS     1996-97 on, play-by-play (the official number)
  CLASSIC  1977-78 on, box score + team results calibrated on Plus (all eras)

Usage: python src/composite.py
  -> out/scott_seasons.csv, out/scott_careers_plus.csv, out/scott_careers_classic.csv,
     out/scott_meta.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wins import leverage, wp_model

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
STINTS = ROOT / "data" / "stints"
MIN_SEASON_POSS = 1000


def clutch_leverage_share():
    m = wp_model(range(2015, 2025))
    tot = cl = poss_t = poss_c = 0.0
    for y in range(1996, 2025):
        s = pd.read_parquet(STINTS / f"stints_{y}.parquet")
        lev = leverage(s, m)
        p = (s.poss_h + s.poss_a).to_numpy()
        end = s.clock0 - s.seconds
        frac = ((np.minimum(s.clock0, 300) - np.maximum(end, 0)).clip(lower=0) / s.seconds.clip(lower=1)).to_numpy()
        frac = np.where((s.period >= 4) & (s.margin0.abs() <= 5), frac, 0)
        tot += np.sum(lev * p)
        cl += np.sum(lev * p * frac)
        poss_t += p.sum()
        poss_c += np.sum(p * frac)
    return cl / tot, poss_c / poss_t


def runs(df, col, n):
    df = df.sort_values("season")
    best, at = np.nan, None
    for i in range(len(df) - n + 1):
        w = df.iloc[i:i + n]
        if w.season.iat[-1] - w.season.iat[0] != n - 1:
            continue
        v = np.average(w[col], weights=w.poss)
        if np.isnan(best) or v > best:
            best, at = v, f"{w.season.iat[0]}-{(w.season.iat[-1] + 1) % 100:02d}"
    return best, at


def label(y):
    return f"{y}-{(y + 1) % 100:02d}"


def plus_tables(c):
    seasons = []
    for y in range(1996, 2025):
        w = pd.read_csv(OUT / f"war_{y}.csv")
        b = pd.read_csv(OUT / f"breakdown_{y}.csv").set_index("player_id")[["box_O", "box_D"]]
        w = w.join(b, on="player_id")
        w["production"] = w.box_O + w.box_D
        w["lift"] = w.net - w.production
        seasons.append(w)
    s = pd.concat(seasons, ignore_index=True)
    po = pd.read_csv(OUT / "playoffs_players.csv").set_index("player_id")
    teams = pd.read_parquet(ROOT / "data" / "coaches" / "coach_games.parquet")[["team_id", "franchise"]] \
        .drop_duplicates().set_index("team_id").franchise
    s["team"] = s.team_id.map(teams)
    s["season_label"] = s.season.map(label)
    s.to_csv(OUT / "scott_seasons.csv", index=False, float_format="%.3f")

    rows = []
    for pid, x in s[s.poss >= MIN_SEASON_POSS].groupby("player_id"):
        w = x.poss
        p_po = po.loc[pid, "elevation_poss"] if pid in po.index and pd.notna(po.loc[pid, "elevation_poss"]) else 0.0
        reg = x.poss.sum()
        clutch = po.loc[pid, "all_clutch_delta"] if pid in po.index else 0.0
        clutch_se = po.loc[pid, "all_clutch_se_delta"] if pid in po.index else np.nan
        elev = po.loc[pid, "elevation_delta"] if pid in po.index and p_po > 0 else 0.0
        elev_se = po.loc[pid, "elevation_se_delta"] if pid in po.index and p_po > 0 else np.nan
        share_po = p_po / (reg + p_po)
        rate = np.average(x.net, weights=w)
        b3, at3 = runs(x, "net", 3)
        b5, at5 = runs(x, "net", 5)
        top = x.loc[x.net.idxmax()]
        eras = []
        for team, t in x.groupby("team", sort=False):
            eras.append(f"{team} {label(t.season.min())[:4]}-{(t.season.max() + 1) % 100:02d}: "
                        f"{np.average(t.net, weights=t.poss):+.1f}")
        rows.append({
            "player_id": pid, "player": x.name.iloc[-1], "first": int(x.season.min()), "last": int(x.season.max()),
            "seasons": len(x), "poss": float(reg),
            "impact": rate, "impact_se": float(np.sqrt(np.sum((w / w.sum()) ** 2 * x.se_net ** 2))),
            "production": np.average(x.production, weights=w), "lift": np.average(x.lift, weights=w),
            "offense": np.average(x.O, weights=w), "defense": np.average(x.D, weights=w),
            "clutch_rise": clutch, "clutch_se": clutch_se,
            "playoff_elevation": elev, "playoff_elevation_se": elev_se, "playoff_share": share_po,
            "scott_stat": rate + c * clutch + share_po * elev,
            "best_season": float(top.net), "best_season_year": top.season_label,
            "best3": b3, "best3_years": at3, "best5": b5, "best5_years": at5,
            "scott_peak5": (b5 + c * clutch + share_po * elev) if not np.isnan(b5) else np.nan,
            "career_WAR": x.WAR.sum(), "career_cWAR": x.cWAR.sum(),
            "eras": " | ".join(eras),
        })
    car = pd.DataFrame(rows).sort_values("scott_stat", ascending=False)
    car.to_csv(OUT / "scott_careers_plus.csv", index=False, float_format="%.3f")
    return s, car


def classic_table():
    a = pd.concat([pd.read_csv(OUT / f"classic_{y}.csv") for y in range(1977, 2025)], ignore_index=True)
    a = a.rename(columns={"season_start": "season", "net_classic": "net"})
    a["production"] = a.box_O + a.box_D
    a["lift"] = a.net - a.production
    rows = []
    for pid, x in a[a.poss >= MIN_SEASON_POSS].groupby("player_id"):
        w = x.poss
        b3, at3 = runs(x, "net", 3)
        b5, at5 = runs(x, "net", 5)
        top = x.loc[x.net.idxmax()]
        rows.append({"player_id": pid, "player": x.player.iloc[-1], "first": int(x.season.min()),
                     "last": int(x.season.max()), "seasons": len(x), "poss": float(w.sum()),
                     "impact": np.average(x.net, weights=w), "production": np.average(x.production, weights=w),
                     "lift": np.average(x.lift, weights=w), "offense": np.average(x.O_classic, weights=w),
                     "defense": np.average(x.D_classic, weights=w),
                     "best_season": float(top.net), "best_season_year": label(int(top.season)),
                     "best3": b3, "best3_years": at3, "best5": b5, "best5_years": at5,
                     "career_WAR": x.WAR_classic.sum(),
                     "pre_1977_career": bool(x.age.min() > 23 and x.season.min() == 1977)})
    car = pd.DataFrame(rows).sort_values("best5", ascending=False)
    car.to_csv(OUT / "scott_careers_classic.csv", index=False, float_format="%.3f")
    return car


def main():
    c, clutch_poss_share = clutch_leverage_share()
    print(f"clutch = {clutch_poss_share:.1%} of possessions but {c:.1%} of win-probability leverage", flush=True)
    s, plus = plus_tables(c)
    classic = classic_table()
    meta = {"clutch_weight": c, "clutch_possession_share": clutch_poss_share,
            "min_season_poss": MIN_SEASON_POSS,
            "formula": "scott_stat = impact + clutch_weight*clutch_rise + playoff_share*playoff_elevation"}
    (OUT / "scott_meta.json").write_text(json.dumps(meta, indent=1, default=float))
    pd.set_option("display.width", 250)
    cols = ["player", "first", "last", "impact", "production", "lift", "clutch_rise", "playoff_elevation",
            "scott_stat", "best5", "scott_peak5", "career_WAR"]
    big = plus[plus.poss >= 20000]
    print("\nPLUS — career Scott Stat (20k+ possessions)")
    print(big.head(25)[cols].round(2).to_string(index=False))
    print("\nPLUS — peak (best 5 seasons, with clutch + playoffs)")
    print(big.sort_values("scott_peak5", ascending=False).head(15)[cols].round(2).to_string(index=False))
    print("\nLIFT leaders (impact beyond own box score), 20k+ poss")
    print(big.sort_values("lift", ascending=False).head(12)[cols].round(2).to_string(index=False))
    print("\nPRODUCTION-heavy, low lift")
    print(big.sort_values("lift").head(10)[cols].round(2).to_string(index=False))
    ccols = ["player", "first", "last", "impact", "production", "lift", "best_season", "best5", "best5_years", "career_WAR"]
    print("\nCLASSIC — best 5-season peak, 1977-78 on")
    print(classic[classic.poss >= 20000].head(20)[ccols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
