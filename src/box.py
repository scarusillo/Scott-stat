"""Per player-season box score totals derived from the same play-by-play,
plus per team-game totals (for coaching-style fingerprints).

Usage: python src/box.py 1996 2024     -> data/stints/box_<YYYY>.parquet,
                                          data/stints/teamgame_<YYYY>.parquet
"""
import sys
from multiprocessing import Pool

import pandas as pd

from build_stints import COLS, RAW, OUT


def season_box(year):
    d = pd.read_csv(RAW / f"nbastats_{year}.csv", usecols=COLS, low_memory=False)
    desc = d.HOMEDESCRIPTION.fillna(d.VISITORDESCRIPTION).fillna("")
    t, a = d.EVENTMSGTYPE, d.EVENTMSGACTIONTYPE
    is_p = lambda k: d[f"PERSON{k}TYPE"].isin([4, 5])
    three = desc.str.contains("3PT")
    # at the rim: layups/dunks/tips (named in every era) or a logged distance <= 3 ft
    dist = pd.to_numeric(desc.str.extract(r"(\d+)' ")[0], errors="coerce")
    rim = ~three & (desc.str.contains("Layup|Dunk|Tip|Alley Oop|Putback") | (dist <= 3))
    ft_made = (t == 3) & ~desc.str.contains("MISS")
    rows = {
        "pts": [(1, (t == 1) & is_p(1), 2 + three.astype(int)), (1, ft_made & is_p(1), 1)],
        "fga": [(1, t.isin([1, 2]) & is_p(1), 1)],
        "fg3a": [(1, t.isin([1, 2]) & three & is_p(1), 1)],
        "rima": [(1, t.isin([1, 2]) & rim & is_p(1), 1)],
        "fgm": [(1, (t == 1) & is_p(1), 1)],
        "fgm_astd": [(1, (t == 1) & is_p(1) & is_p(2), 1)],
        "fta": [(1, (t == 3) & ~a.isin([16, 21, 22]) & is_p(1), 1)],
        "ast": [(2, (t == 1) & is_p(2), 1)],
        "reb": [(1, (t == 4) & is_p(1), 1)],
        "tov": [(1, (t == 5) & is_p(1), 1)],
        "stl": [(2, (t == 5) & is_p(2), 1)],
        "blk": [(3, (t == 2) & is_p(3), 1)],
    }
    parts = []
    for stat, specs in rows.items():
        for k, mask, val in specs:
            v = val[mask] if isinstance(val, pd.Series) else pd.Series(val, index=d.index[mask])
            parts.append(pd.DataFrame({"player_id": d.loc[mask, f"PLAYER{k}_ID"].astype("int64"),
                                       "game_id": d.loc[mask, "GAME_ID"], "stat": stat, "v": v}))
    x = pd.concat(parts)
    # team-game totals (same events, keyed by the shooter's team) for style fingerprints
    tg = []
    for stat in ("fga", "fg3a", "rima", "fgm", "fgm_astd", "fta", "tov", "ast"):
        for k, mask, val in rows[stat]:
            v = val[mask] if isinstance(val, pd.Series) else pd.Series(val, index=d.index[mask])
            tg.append(pd.DataFrame({"game_id": d.loc[mask, "GAME_ID"], "team_id": d.loc[mask, f"PLAYER{k}_TEAM_ID"],
                                    "stat": stat, "v": v}))
    tg = pd.concat(tg).dropna(subset=["team_id"])
    tg["team_id"] = tg.team_id.astype("int64")
    tg = tg.pivot_table(index=["game_id", "team_id"], columns="stat", values="v", aggfunc="sum",
                        fill_value=0).reset_index()
    tg.insert(0, "season", year)
    tg.to_parquet(OUT / f"teamgame_{year}.parquet", index=False)

    tot = x.pivot_table(index="player_id", columns="stat", values="v", aggfunc="sum", fill_value=0)
    tot["games"] = x.groupby("player_id").game_id.nunique()
    tot = tot.reset_index()
    tot.insert(0, "season", year)
    tot.to_parquet(OUT / f"box_{year}.parquet", index=False)
    return year


if __name__ == "__main__":
    a, b = int(sys.argv[1]), int(sys.argv[2] if len(sys.argv) > 2 else sys.argv[1])
    with Pool(4) as p:
        for y in p.imap_unordered(season_box, range(a, b + 1)):
            print(y, end=" ", flush=True)
