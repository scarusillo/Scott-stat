"""How concentrated is each coach x player-type cell? A cell built on one star
("Phil Jackson x Lead creator" = Kobe/Jordan) is a statement about that star,
not about how the coach handles the type. Adds top_player, top_share and
n_players to out/coach_archetype.csv; `broad` = top player < 50% of exposure
and 3+ players with 2000+ possessions.

Usage: python src/coach_cells.py   (after coach_styles.py)
"""
import numpy as np
import pandas as pd

from coach_effects import load
from talent import A, H, OUT, ROOT

s, cg = load()
arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet")
recs = []
for cols, coach in ((H, "coach_h"), (A, "coach_a")):
    for c in cols:
        recs.append(pd.DataFrame({"season": s.season.to_numpy(), "coach_id": s[coach].to_numpy(),
                                  "player_id": s[c].astype("int64").to_numpy(),
                                  "p": (s.poss_h + s.poss_a).to_numpy() / 2}))
L = pd.concat(recs).merge(arch[["season", "player_id", "archetype", "name"]], on=["season", "player_id"])
cell = L.groupby(["coach_id", "archetype", "player_id"]).agg(p=("p", "sum"), name=("name", "first")).reset_index()
tot = cell.groupby(["coach_id", "archetype"]).p.sum()
top = cell.sort_values("p").groupby(["coach_id", "archetype"]).tail(1).set_index(["coach_id", "archetype"])
conc = pd.DataFrame({"top_player": top.name, "top_share": top.p / tot,
                     "n_players": cell[cell.p >= 2000].groupby(["coach_id", "archetype"]).size()}).fillna({"n_players": 0})
ca = pd.read_csv(OUT / "coach_archetype.csv")
ca = ca.drop(columns=[c for c in ("top_player", "top_share", "n_players", "broad") if c in ca]).join(
    conc, on=["coach_id", "archetype"])
ca["broad"] = (ca.top_share < 0.5) & (ca.n_players >= 3)
ca.to_csv(OUT / "coach_archetype.csv", index=False, float_format="%.3f")
big = ca[ca.exposure >= 20000].copy()
big["z"] = big.type_net / np.hypot(big.se_type_O, big.se_type_D)
print("cells with 20k+ exposure:", len(big), " broad:", int(big.broad.sum()))
cols = ["coach", "archetype", "uplift", "z", "n_players", "top_player", "top_share"]
print(big.sort_values("type_net", ascending=False).head(8)[cols].round(2).to_string(index=False))
b = big[big.broad]
print("\nBROAD cells (no single star):")
print(b.sort_values("type_net", ascending=False).head(12)[cols].round(2).to_string(index=False))
print(b.sort_values("type_net").head(6)[cols].round(2).to_string(index=False))
