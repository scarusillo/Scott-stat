"""Schemes: which offensive and defensive systems do players thrive in? (Play-by-play era, 1996-97 to 2024-25.)

1. SCHEME FAMILIES. Every team-season gets a style fingerprint, z-scored within its season so a family means
   the same thing relative to its era (a 2000 "spread" team took fewer threes than a 2024 one):
     offense: pace, 3PA rate, rim-attempt rate, assisted share of makes
     defense: opponent 3PA rate and rim rate allowed (from shot locations), turnovers forced, fouls (opponent
              FT per FGA), defensive rebounding
   k-means groups team-seasons into families; each family is named from its centre. Stability: the share of
   team-seasons that land in the same family under 20 different random starts.
2. PLAYER TYPE x SCHEME. A player-season's deviation from his own career level (leave-that-season-out average
   of his play-by-play offense / defense, minus the normal aging change) is averaged by player type (archetypes.py)
   and by the scheme his team ran that season. Offense deviations are matched with the offensive scheme, defense
   with the defensive one. Each cell is relative to the scheme's average (so "every player looks better in a
   good team's scheme" cancels) and shrunk toward 0. Repeatability: odd vs even seasons.
3. PLAYER x SCHEME. For players who spent real time both inside and outside a scheme family, his fit = his
   deviation inside minus outside (shrunk). Tested out of sample: predict each held-out season's deviation from
   his fit measured on his OTHER seasons, against a prediction of 0.

Man-to-man vs zone, drop vs switch and play types are NOT in public play-by-play (they need tracking or
Synergy/NBA.com play-type data); the defensive families are shot-profile proxies, not coverage labels.

Usage: python src/scheme_fit.py -> out/scheme_fit.json, out/scheme_team_seasons.csv
"""
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from coach_legacy import code

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
HIST = ROOT / "data" / "hist"
OFF = ["pace", "three", "rim", "astd"]
DEF = ["opp_three", "opp_rim", "opp_tov", "opp_ftr", "drb"]
K_OFF, K_DEF = 5, 4
SHRINK_CELL = 40000.0      # possessions of shrinkage for a type x scheme cell
SHRINK_PLAYER = 6000.0     # possessions of shrinkage for a player's fit
MIN_IN, MIN_OUT = 3000, 3000


# family prototypes (z units, in OFF / DEF order); centres are matched to them one-to-one (Hungarian), so every
# family gets a distinct, descriptive name that survives refits
OFF_PROTO = {
    "Pace and space": ([1, 1, 0, 0.5], "Fast, lots of threes, lots of passing"),
    "Spread half-court": ([-1, 1, -1, 0], "Threes at a slow pace, few rim attempts"),
    "Grind it out": ([-1, -0.5, -1, 0], "Slow, few threes and few rim attempts: half-court sets and mid-range"),
    "Motion / inside-out": ([-0.5, -0.5, 1, 1], "Paint touches and assisted makes, few threes (triangle, flex, high-post)"),
    "Isolation / pick-and-roll": ([0.5, -0.5, 0.5, -1], "Fewest assisted makes: creators make their own shots"),
}
DEF_PROTO = {
    "Gamble and pressure": ([0, 1, 1, 1, -1], "Forces turnovers, fouls, gives up the rim and offensive boards"),
    "Disciplined / no-foul": ([0, -1, -1, -1, 1], "Few fouls, protects the rim, ends possessions with the rebound"),
    "Run them off the line": ([-1, 0.5, -0.5, -0.5, 0], "Takes away threes, lives with twos (switch / closeout profile)"),
    "Aggressive wall": ([1, -0.5, 1, 0.5, 0.5], "Walls off the paint and jumps passing lanes, concedes threes"),
}


def unique_names(centres, proto):
    from scipy.optimize import linear_sum_assignment
    keys = list(proto)
    P = np.array([proto[k][0] for k in keys], dtype=float)
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    C = centres / np.linalg.norm(centres, axis=1, keepdims=True)
    r, c = linear_sum_assignment(-(C @ P.T))
    out = [None] * len(centres)
    for i, j in zip(r, c):
        out[i] = (keys[j], proto[keys[j]][1], float((C @ P.T)[i, j]))
    return out


def team_seasons():
    st = pd.read_csv(OUT / "team_style_seasons.csv")
    w = st.n.to_numpy()
    g = st.assign(**{k: st[k] * w for k in OFF + ["opp_three", "opp_rim"]}).groupby(["season", "team_id"])
    ts = g[OFF + ["opp_three", "opp_rim"]].sum().div(g.n.sum(), axis=0).reset_index()
    ts["coach_id"] = st.sort_values("n").groupby(["season", "team_id"]).coach_id.last().to_numpy()
    fr = pd.read_parquet(ROOT / "data" / "coaches" / "coach_games.parquet")[["team_id", "franchise"]].drop_duplicates("team_id")
    ts = ts.merge(fr, on="team_id", how="left")
    ts["abbr"] = [code(f, s + 1) for f, s in zip(ts.franchise, ts.season)]
    tsum = pd.read_csv(HIST / "Team Summaries.csv")
    tsum = tsum[(tsum.lg == "NBA") & (tsum.team != "League Average")]
    tsum = tsum.assign(season=tsum.season - 1)[["season", "abbreviation", "opp_tov_percent", "opp_ft_fga", "drb_percent", "srs", "team"]]
    ts = ts.merge(tsum.rename(columns={"abbreviation": "abbr", "opp_tov_percent": "opp_tov", "opp_ft_fga": "opp_ftr",
                                       "drb_percent": "drb", "team": "team_name"}), on=["season", "abbr"], how="left")
    for k in ["opp_tov", "opp_ftr", "drb"]:
        ts[k] = ts.groupby("season")[k].transform(lambda x: (x - x.mean()) / x.std())
    return ts


def families(ts, cols, k, proto):
    X = ts[cols].to_numpy()
    km = KMeans(k, n_init=50, random_state=0).fit(X)
    lab = km.labels_
    # stability: same partition under other random starts (labels permuted -> compare co-membership)
    same = []
    for seed in range(1, 21):
        l2 = KMeans(k, n_init=1, random_state=seed).fit(X).labels_
        # best matching of labels
        from scipy.optimize import linear_sum_assignment
        M = np.zeros((k, k))
        for a, b in zip(lab, l2):
            M[a, b] += 1
        r, c = linear_sum_assignment(-M)
        same.append(M[r, c].sum() / len(lab))
    names = unique_names(km.cluster_centers_, proto)
    return lab, km.cluster_centers_, names, float(np.mean(same))


def player_seasons():
    rows = []
    for f in sorted(glob.glob(str(OUT / "war_*.csv"))):
        w = pd.read_csv(f)
        rows.append(w[["season", "player_id", "name", "team_id", "O", "D", "poss"]])
    w = pd.concat(rows)
    w = w[w.poss >= 500]
    arch = pd.read_parquet(ROOT / "data" / "archetypes.parquet")[["season", "player_id", "archetype"]]
    w = w.merge(arch, on=["season", "player_id"], how="left")
    ages = []
    for f in glob.glob(str(OUT / "classic_*[0-9].csv")):
        c = pd.read_csv(f, usecols=["season_start", "nba_id", "age"]).dropna()
        ages.append(c.rename(columns={"season_start": "season", "nba_id": "player_id"}))
    ages = pd.concat(ages).drop_duplicates(["season", "player_id"])
    ages["player_id"] = ages.player_id.astype(int)
    w = w.merge(ages, on=["season", "player_id"], how="left")
    # leave-this-season-out career level
    for k in ("O", "D"):
        tot = (w[k] * w.poss).groupby(w.player_id).transform("sum")
        pp = w.groupby("player_id").poss.transform("sum")
        w[k + "_base"] = (tot - w[k] * w.poss) / (pp - w.poss)
    w = w[w.groupby("player_id").season.transform("count") >= 3].dropna(subset=["age", "O_base"])
    for k in ("O", "D"):
        dev = w[k] - w[k + "_base"]
        a = w.age.clip(19, 38)
        coef = np.polyfit(a, dev, 2, w=np.sqrt(w.poss))
        w[k + "_dev"] = dev - np.polyval(coef, a)
    return w


def cell_table(w, side, fam):
    """type x scheme deviation, relative to the scheme's overall deviation; shrunk."""
    dev, lab = w[side + "_dev"], w[fam]
    p = w.poss
    sch = (dev * p).groupby(lab).sum() / p.groupby(lab).sum()
    rel = dev - lab.map(sch)
    # and relative to the player type's own average deviation (types age differently), so a cell is how that
    # type does in this scheme against the same type in other schemes
    typ = (rel * p).groupby(w.archetype).sum() / p.groupby(w.archetype).sum()
    rel = rel - w.archetype.map(typ)
    g = pd.DataFrame({"t": w.archetype, "s": lab, "r": rel * p, "p": p}).groupby(["t", "s"])[["r", "p"]].sum()
    g["fit"] = g.r / (g.p + SHRINK_CELL)
    g["raw"] = g.r / g.p
    g["se"] = 1.0 / np.sqrt(g.p / 3000.0)          # ~1 point per 100 of season-level noise per 3,000 possessions
    return g, sch


def main():
    ts = team_seasons()
    ts = ts.dropna(subset=DEF)
    lo, co, no, stab_o = families(ts, OFF, K_OFF, OFF_PROTO)
    ld, cd, nd, stab_d = families(ts, DEF, K_DEF, DEF_PROTO)
    ts["off_family"] = [no[i][0] for i in lo]
    ts["def_family"] = [nd[i][0] for i in ld]
    ts.to_csv(OUT / "scheme_team_seasons.csv", index=False, float_format="%.3f")
    out = {"offense": [], "defense": [], "stability": {"offense": stab_o, "defense": stab_d}}
    for side, lab, cen, nm, cols in (("offense", lo, co, no, OFF), ("defense", ld, cd, nd, DEF)):
        fam = "off_family" if side == "offense" else "def_family"
        for i, (n, d, match) in enumerate(nm):
            x = ts[ts[fam] == n]
            ex = x.sort_values("srs", ascending=False).head(6)
            out[side].append({"name": n, "desc": d, "centre": dict(zip(cols, np.round(cen[i], 2).tolist())), "name_match": round(match, 2),
                              "n": int(len(x)), "srs": float(x.srs.mean()),
                              "share_by_era": {e: float((ts[(ts.season >= a) & (ts.season <= b)][fam] == n).mean())
                                               for e, a, b in (("1996-2004", 1996, 2004), ("2005-2014", 2005, 2014), ("2015-2024", 2015, 2024))},
                              "examples": [f"{r.season}-{(r.season + 1) % 100:02d} {r.abbr}" for r in ex.itertuples()]})

    w = player_seasons()
    w = w.merge(ts[["season", "team_id", "off_family", "def_family", "coach_id"]], on=["season", "team_id"], how="inner")
    out["n_player_seasons"] = int(len(w))

    # 2. type x scheme
    cells, rel = {}, {}
    for side, fam in (("O", "off_family"), ("D", "def_family")):
        g, sch = cell_table(w, side, fam)
        cells[side] = [{"type": t, "scheme": s, "fit": float(r.fit), "raw": float(r.raw), "poss": float(r.p), "se": float(r.se)}
                       for (t, s), r in g.iterrows()]
        # repeatability: odd vs even seasons
        a, _ = cell_table(w[w.season % 2 == 0], side, fam)
        b, _ = cell_table(w[w.season % 2 == 1], side, fam)
        j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        j = j[(j.p_a >= 10000) & (j.p_b >= 10000)]
        rel[side] = {"split_half_r": float(np.corrcoef(j.raw_a, j.raw_b)[0, 1]), "cells": int(len(j)),
                     "scheme_main": {k: float(v) for k, v in sch.items()}}
    out["type_scheme"] = cells
    out["type_scheme_reliability"] = rel

    # 3. player x scheme fit. Shrinkage chosen out of sample: for every season, his fit for that season's scheme
    # is measured on his OTHER seasons only, and the shrinkage that best predicts the held-out season wins
    fits, test = [], {}
    for side, fam in (("O", "off_family"), ("D", "def_family")):
        d = w.assign(dev=w[side + "_dev"] - w[fam].map(rel[side]["scheme_main"]))
        ho = []    # (raw fit, n_eff, out share, actual, weight)
        for pid, x in d.groupby("player_id"):
            if len(x) < 4:
                continue
            for i in range(len(x)):
                h, rest = x.iloc[i], x.drop(x.index[i])
                ins, outs = rest[rest[fam] == h[fam]], rest[rest[fam] != h[fam]]
                if ins.poss.sum() < 1500 or outs.poss.sum() < 1500:
                    continue
                fi = np.average(ins.dev, weights=ins.poss) - np.average(outs.dev, weights=outs.poss)
                ho.append((fi, 1 / (1 / ins.poss.sum() + 1 / outs.poss.sum()), outs.poss.sum() / rest.poss.sum(),
                           h.dev - np.average(rest.dev, weights=rest.poss), h.poss))
        fi, ne, sh, act, wt = map(np.array, zip(*ho))
        grid = {}
        for K in (2000, 6000, 15000, 30000, 60000, 120000, 1e12):
            pred = fi * ne / (ne + K) * sh
            grid[K] = float(np.average((act - pred) ** 2, weights=wt))
        K = min(grid, key=grid.get)
        pred = fi * ne / (ne + K) * sh
        cw = np.cov(pred, act, aweights=wt)
        test[side] = {"K": K, "r": float(cw[0, 1] / np.sqrt(cw[0, 0] * cw[1, 1])), "n": int(len(act)),
                      "mse_fit": grid[K], "mse_zero": grid[1e12],
                      "error_cut_pct": float(100 * (1 - grid[K] / grid[1e12])), "grid": {str(k): v for k, v in grid.items()}}
        for pid, x in d.groupby("player_id"):
            if len(x) < 4:
                continue
            for s_ in x[fam].unique():
                ins, outs = x[x[fam] == s_], x[x[fam] != s_]
                if ins.poss.sum() >= MIN_IN and outs.poss.sum() >= MIN_OUT:
                    f_ = np.average(ins.dev, weights=ins.poss) - np.average(outs.dev, weights=outs.poss)
                    n_eff = 1 / (1 / ins.poss.sum() + 1 / outs.poss.sum())
                    fits.append({"side": side, "player_id": int(pid), "name": x["name"].iat[-1], "scheme": s_,
                                 "fit": float(f_ * n_eff / (n_eff + K)), "raw": float(f_),
                                 "se": float(1.2 / np.sqrt(n_eff / 3000)),
                                 "poss_in": float(ins.poss.sum()), "seasons_in": int(len(ins)),
                                 "years_in": f"{ins.season.min()}-{(ins.season.max() + 1) % 100:02d}",
                                 "teams_in": sorted(set(ins.team_id.astype(int))),
                                 "type": x.archetype.mode().iat[0] if x.archetype.notna().any() else None})
    out["player_fit_test"] = test
    pf = pd.DataFrame(fits)
    out["player_fits"] = pf.sort_values("fit", ascending=False).round(3).to_dict("records")

    # coaches: which families their teams ran
    cf = ts.groupby("coach_id").agg(off=("off_family", lambda x: x.value_counts().index[0]),
                                    off_share=("off_family", lambda x: x.value_counts().iat[0] / len(x)),
                                    dfn=("def_family", lambda x: x.value_counts().index[0]),
                                    def_share=("def_family", lambda x: x.value_counts().iat[0] / len(x)),
                                    seasons=("season", "count"))
    out["coach_families"] = cf[cf.seasons >= 3].round(2).reset_index().to_dict("records")
    (OUT / "scheme_fit.json").write_text(json.dumps(out, indent=1, default=float))

    print("stability", out["stability"])
    for side in ("offense", "defense"):
        for f_ in out[side]:
            print(f"{side:8s} {f_['name']:32s} n={f_['n']:4d} srs={f_['srs']:+.2f} {f_['centre']} {f_['share_by_era']} {f_['examples'][:3]}")
    print("type x scheme reliability", {k: (round(v['split_half_r'], 2), v['cells']) for k, v in rel.items()})
    print("scheme main", {k: {a: round(b, 2) for a, b in v['scheme_main'].items()} for k, v in rel.items()})
    print("player fit test", test)
    for side in ("O", "D"):
        c = pd.DataFrame(cells[side])
        print(side, "\n", c.pivot(index="type", columns="scheme", values="fit").round(2).to_string())
        print(pf[pf.side == side].sort_values("fit", ascending=False).head(12)[["name", "scheme", "fit", "raw", "poss_in", "years_in"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
