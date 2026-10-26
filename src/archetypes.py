"""Player archetypes per season (k-means on role stats, z-scored WITHIN season so
a type means the same role relative to its era — a 2000 "shooter" took fewer
threes than a 2024 one).

Features per 100 on-court possessions: usage, 3PA share, rim-attempt share,
share of makes assisted, assists, rebounds, steals, blocks, FT rate.
Fit on player-seasons with 1500+ possessions; everyone with 30+ FGA is then
assigned to the nearest type. Clusters are named by matching their centres
to the prototypes below (Hungarian assignment), so names survive refits.

Usage: python src/archetypes.py     -> data/archetypes.parquet, out/archetype_centres.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
STINTS = ROOT / "data" / "stints"
OUT = ROOT / "out"
FEATS = ["usg", "three", "rim", "astd", "ast", "reb", "stl", "blk", "ftr"]
# centres from the first fit (2026-10-26), in FEATS order
PROTOTYPES = {
    "Interior scorer":     [1.06, -0.96, 0.43, 0.00, -0.28, 1.27, -0.39, 1.03, 0.73],
    "Scoring wing":        [0.72, 0.19, -0.41, -0.22, -0.13, -0.46, -0.30, -0.47, -0.13],
    "Defensive wing":      [-0.51, -0.05, 0.33, 0.24, -0.16, -0.17, 1.34, -0.12, -0.01],
    "Spot-up shooter":     [-0.63, 1.23, -0.85, 0.99, -0.47, -0.56, -0.27, -0.43, -0.86],
    "Frontcourt finisher": [-0.39, -0.70, 0.32, 0.61, -0.68, 0.71, -0.64, 0.39, -0.04],
    "Lead creator":        [1.21, -0.14, -0.14, -1.67, 1.61, -0.60, 0.71, -0.55, 0.40],
    "Rim runner":          [-0.86, -1.27, 1.99, 0.28, -0.84, 1.53, -0.42, 1.59, 1.53],
    "Playmaking guard":    [-0.20, 0.58, -0.67, -0.94, 1.32, -0.94, 0.50, -0.72, -0.64],
}
NAMES = list(PROTOTYPES)


def features():
    rows = []
    for y in range(1996, 2025):
        b = pd.read_parquet(STINTS / f"box_{y}.parquet").set_index("player_id")
        w = pd.read_csv(OUT / f"war_{y}.csv").set_index("player_id")[["name", "poss"]]
        d = b.join(w, how="inner")
        d = d[d.fga >= 30]
        p = d.poss / 100
        f = pd.DataFrame({"usg": (d.fga + 0.44 * d.fta + d.tov) / p, "three": d.fg3a / d.fga,
                          "rim": d.rima / d.fga, "astd": d.fgm_astd / d.fgm.clip(lower=1),
                          "ast": d.ast / p, "reb": d.reb / p, "stl": d.stl / p, "blk": d.blk / p,
                          "ftr": d.fta / d.fga}, index=d.index)
        core = d.poss >= 1500
        f = (f - f[core].mean()) / f[core].std()
        f["season"], f["name"], f["poss"], f["core"] = y, d.name, d.poss, core
        rows.append(f)
    return pd.concat(rows).reset_index()


def main():
    F = features()
    core = F[F.core]
    km = KMeans(len(NAMES), n_init=20, random_state=0).fit(core[FEATS])
    proto = np.array(list(PROTOTYPES.values()))
    dist = ((km.cluster_centers_[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(dist)
    label = {int(i): NAMES[j] for i, j in zip(r, c)}
    F["archetype"] = pd.Series(km.predict(F[FEATS].fillna(0))).map(label).to_numpy()
    F[["season", "player_id", "name", "poss", "archetype"] + FEATS].to_parquet(
        ROOT / "data" / "archetypes.parquet", index=False)
    cen = pd.DataFrame(km.cluster_centers_, columns=FEATS).round(2)
    cen.insert(0, "archetype", [label[i] for i in range(len(NAMES))])
    cen["player_seasons"] = core.assign(a=km.labels_).a.value_counts().sort_index().to_numpy()
    cen.to_csv(OUT / "archetype_centres.csv", index=False)
    print(cen.to_string(index=False))
    for a in NAMES:
        ex = F[(F.archetype == a) & (F.season == 2024)].sort_values("poss", ascending=False).name.head(5)
        print(f"{a:18s} 2024-25: {', '.join(ex)}")


if __name__ == "__main__":
    main()
