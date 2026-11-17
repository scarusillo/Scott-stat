"""Player x coach — how well did each coach use each player?

Start from the coach_effects model (career-level players + league experience
curve + coach). Its residuals are what that model can't explain. Regress them
(ridge, penalty by game-grouped CV) on a column per (player, coach) pair with
enough shared possessions, offense and defense separately. A pair effect is how
much better that player played under that coach than his own career level,
league aging curve and the coach's general effect predict.

Two versions, because a player's own career arc can masquerade as coaching
(Curry's rise happened under Kerr):
  raw            league-wide aging curve only
  trend_adjusted adds each player's OWN linear career trend, so steady growth
                 or decline is removed and only departures from his trajectory
                 when the coach changed are credited to the pair.

Usage: python src/player_coach.py   -> out/player_coach.csv, out/player_coach_meta.json
"""
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from coach_effects import A, H, OUT, ROOT, build, load, solve

MIN_SHARED = 2000      # on-floor possessions of the player under the coach
LAMS = (1000, 3000, 10000, 30000)


def main():
    s, cg = load()
    X, y, w, pen, meta = build(s, 3000)
    beta, _ = solve(X, y, w, pen)
    r = y - X @ beta
    keep = np.r_[s.poss_h, s.poss_a] > 0
    n = len(s)
    off = np.column_stack([np.r_[s[h].astype("int64"), s[a].astype("int64")][keep] for h, a in zip(H, A)])
    dfn = np.column_stack([np.r_[s[a].astype("int64"), s[h].astype("int64")][keep] for h, a in zip(H, A)])
    c_off = np.r_[s.coach_h, s.coach_a][keep]
    c_def = np.r_[s.coach_a, s.coach_h][keep]
    season = np.r_[s.season, s.season][keep]
    games = np.r_[s.game_id, s.game_id][keep]

    # pairs with enough shared possessions (count offense rows; defense is symmetric)
    pp = pd.DataFrame({"p": off.ravel(), "c": np.repeat(c_off, 5), "w": np.repeat(w, 5)})
    shared = pp.groupby(["p", "c"]).w.sum()
    pairs = shared[shared >= MIN_SHARED]
    # keep only players with 2+ qualifying coaches: comparisons need a mover
    movers = pairs.groupby(level=0).size()
    pairs = pairs[pairs.index.get_level_values(0).isin(movers[movers >= 2].index)]
    pidx = {k: i for i, k in enumerate(pairs.index)}
    K = len(pidx)
    players = np.unique(pairs.index.get_level_values(0))
    plix = {p: i for i, p in enumerate(players)}
    mean_season = pd.Series(np.repeat(season, 5), index=off.ravel()).groupby(level=0).mean()
    print(f"pairs: {K}, players: {len(players)}", flush=True)

    def block(lineups, coaches, sign, base, trend_base, with_trend):
        rows, cols, vals = [], [], []
        for j in range(5):
            key = list(zip(lineups[:, j], coaches))
            col = np.array([pidx.get(k, -1) for k in key])
            ok = col >= 0
            rows.append(np.flatnonzero(ok)); cols.append(base + col[ok]); vals.append(np.full(ok.sum(), sign))
            if with_trend:
                pl = np.array([plix.get(p, -1) for p in lineups[:, j]])
                ok2 = pl >= 0
                t = (season[ok2] - mean_season.reindex(lineups[ok2, j]).to_numpy()) / 10
                rows.append(np.flatnonzero(ok2)); cols.append(trend_base + pl[ok2]); vals.append(sign * t)
        return rows, cols, vals

    results, info = {}, {}
    rng = np.random.default_rng(0)
    ug = np.unique(games)
    fold = pd.Series(rng.integers(0, 5, len(ug)), index=ug).loc[games].to_numpy()
    for version, with_trend in (("raw", False), ("trend_adjusted", True)):
        L = len(players)
        ncol = 2 * K + (2 * L if with_trend else 0)
        R, Cc, V = [], [], []
        for lu, co, sign, base, tb in ((off, c_off, 1.0, 0, 2 * K), (dfn, c_def, -1.0, K, 2 * K + L)):
            rr, cc, vv = block(lu, co, sign, base, tb, with_trend)
            R += rr; Cc += cc; V += vv
        Z = sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(Cc))), shape=(len(r), ncol))
        ZtW = Z.T.multiply(w).tocsr()
        G = (ZtW @ Z).toarray()
        bvec = ZtW @ r
        pen_trend = np.r_[np.zeros(2 * K), np.full(ncol - 2 * K, 1.0)]   # trends: tiny penalty, they're nuisance terms
        cv = {}
        Gk, bk = [], []
        for k in range(5):
            m = fold == k
            Zk = Z[m]
            ZtWk = Zk.T.multiply(w[m]).tocsr()
            Gk.append((ZtWk @ Zk).toarray()); bk.append(ZtWk @ r[m])
        for lam in LAMS:
            pen = np.where(pen_trend > 0, 300.0, lam)
            err = 0.0
            for k in range(5):
                M = G - Gk[k]
                M[np.diag_indices_from(M)] += pen
                bb = np.linalg.solve(M, bvec - bk[k])
                m = fold == k
                err += np.sum(w[m] * (r[m] - Z[m] @ bb) ** 2)
            cv[lam] = err / w.sum()
        lam = min(cv, key=cv.get)
        pen = np.where(pen_trend > 0, 300.0, lam)
        M = G.copy()
        M[np.diag_indices_from(M)] += pen
        Minv = np.linalg.inv(M)
        bb = Minv @ bvec
        res = r - Z @ bb
        s2 = np.sum(w * res ** 2) / len(r)
        cov = Minv @ G @ Minv * s2
        se = np.sqrt(np.diag(cov))
        results[version] = (bb, se, np.diag(cov[:K, K:2 * K]))
        info[version] = {"lambda": lam, "cv": cv}
        print(version, info[version], flush=True)

    names = pd.concat([pd.read_parquet(ROOT / "data" / "stints" / f"players_{y}.parquet") for y in range(1996, 2025)]) \
        .groupby("player_id").name.agg(lambda x: x.mode().iat[0])
    coach_names = cg.groupby("coach_id").coach.first()
    ce = pd.read_csv(OUT / "coach_effects.csv").set_index("coach_id")
    seasons_pc = pd.DataFrame({"p": off.ravel(), "c": np.repeat(c_off, 5), "y": np.repeat(season, 5)}) \
        .groupby(["p", "c"]).y.agg(["min", "max"])
    rows = []
    for (p, c), i in pidx.items():
        row = {"player_id": p, "player": names.get(p), "coach_id": c, "coach": coach_names.get(c),
               "shared_poss": float(pairs[(p, c)]),
               "first_season": int(seasons_pc.loc[(p, c), "min"]), "last_season": int(seasons_pc.loc[(p, c), "max"]),
               "coach_general_per_player": float((ce.at[c, "O"] + ce.at[c, "D"]) / 5) if c in ce.index else 0.0}
        for v, (bb, se, cov_od) in results.items():
            row[f"{v}_O"] = bb[i]
            row[f"{v}_D"] = bb[K + i]
            row[f"{v}_net"] = bb[i] + bb[K + i]
            row[f"{v}_se"] = float(np.sqrt(se[i] ** 2 + se[K + i] ** 2 + 2 * cov_od[i]))
        row["total_uplift_raw"] = row["raw_net"] + row["coach_general_per_player"]
        row["total_uplift_trend_adjusted"] = row["trend_adjusted_net"] + row["coach_general_per_player"]
        rows.append(row)
    d = pd.DataFrame(rows).sort_values(["player", "first_season"])
    d.to_csv(OUT / "player_coach.csv", index=False, float_format="%.3f")
    (OUT / "player_coach_meta.json").write_text(json.dumps({"pairs": K, "players": len(players),
                                                           "min_shared": MIN_SHARED, **info}, indent=1, default=float))
    pd.set_option("display.width", 220)
    show = ["player", "coach", "first_season", "last_season", "shared_poss", "total_uplift_raw", "raw_se",
            "total_uplift_trend_adjusted", "trend_adjusted_se"]
    for p in ("Stephen Curry", "JR Smith", "LeBron James", "Kevin Durant", "Jason Kidd", "Allen Iverson",
              "Carmelo Anthony", "Kyrie Irving", "James Harden", "Chris Paul"):
        print(d[d.player == p][show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
