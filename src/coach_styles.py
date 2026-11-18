"""Layer 4b — WHICH players each coach gets more out of, and which STYLES do it.

1. Style fingerprint per coach (z-scored within season, then averaged over his
   games): pace, 3PA rate, rim rate, assisted-make rate, rotation depth,
   substitution churn, and on defense: opponent 3PA rate and rim rate allowed.
2. Coach x archetype model: the coach_effects regression plus, for every player
   on the floor, a coach-by-player-type term. Result: points per 100 that a
   player of type T gains (O) / prevents (D) under coach C, beyond his own
   career level. "Uplift" = the coach's general effect shared per player
   (main / 5) + the type-specific term.
3. Split-half reliability of the type-specific terms (even vs odd seasons).
4. Style -> uplift: for each player type, which style traits predict coaches
   getting more out of that type (weighted least squares across coaches).

Usage: python src/coach_styles.py   -> out/coach_style.csv, out/coach_archetype.csv,
                                        out/style_effects.csv, out/coach_styles_meta.json
"""
import json
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp

from archetypes import NAMES
from coach_effects import A, H, OUT, ROOT, STINTS, build, load, solve

LAM_COACH = 3000
LAM_INT = 3000
K = len(NAMES)
MIN_EXPOSURE = 20000   # player-possessions of a type under a coach to report the cell
STYLE = ["pace", "three", "rim", "astd", "depth", "churn", "opp_three", "opp_rim"]


def arch_codes(s):
    a = pd.read_parquet(ROOT / "data" / "archetypes.parquet")
    a["code"] = a.archetype.map({n: i for i, n in enumerate(NAMES)})
    key = pd.Series(a.code.to_numpy(), index=a.season.astype("int64") * 10 ** 7 + a.player_id.astype("int64"))
    out = {}
    for c in H + A:
        k = s.season.astype("int64") * 10 ** 7 + s[c].astype("int64")
        out[c] = k.map(key).to_numpy()   # NaN = no archetype (tiny sample)
    return out


def interactions(s, coaches):
    """Sparse coach x archetype block, rows aligned with build(): home-offense rows
    then away-offense rows, filtered by poss > 0."""
    n, C = len(s), len(coaches)
    cidx = {c: i for i, c in enumerate(coaches)}
    codes = arch_codes(s)
    oI, oID = 0, C * K
    rows, cols, vals = [], [], []
    expo_rows, expo_cols, expo_vals = [], [], []
    for off, deff, ch_off, ch_def, r0, poss_o, poss_d in (
            (H, A, "coach_h", "coach_a", 0, "poss_h", "poss_a"),
            (A, H, "coach_a", "coach_h", n, "poss_a", "poss_h")):
        r = np.arange(n) + r0
        co = s[ch_off].map(cidx).to_numpy()
        cd = s[ch_def].map(cidx).to_numpy()
        for c in off:
            ok = ~np.isnan(codes[c])
            cell = co[ok] * K + codes[c][ok].astype(int)
            rows.append(r[ok]); cols.append(oI + cell); vals.append(np.ones(ok.sum()))
            expo_rows.append(cell); expo_vals.append(s[poss_o].to_numpy()[ok])
        for c in deff:
            ok = ~np.isnan(codes[c])
            cell = cd[ok] * K + codes[c][ok].astype(int)
            rows.append(r[ok]); cols.append(oID + cell); vals.append(-np.ones(ok.sum()))
            expo_rows.append(cell); expo_vals.append(s[poss_d].to_numpy()[ok])
    Xi = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                       shape=(2 * n, 2 * C * K))
    expo = np.bincount(np.concatenate(expo_rows), weights=np.concatenate(expo_vals), minlength=C * K) / 2
    keep = np.r_[s.poss_h, s.poss_a] > 0
    return Xi[keep], expo


def fit(s, with_se):
    X, y, w, pen, meta = build(s, LAM_COACH)
    Xi, expo = interactions(s, meta["coaches"])
    Xf = sp.hstack([X, Xi]).tocsr()
    penf = np.r_[pen, np.full(Xi.shape[1], LAM_INT)]
    beta, se = solve(Xf, y, w, penf, se=with_se)
    P, C = meta["P"], meta["C"]
    nmain = X.shape[1]
    main_O, main_D = beta[2 * P:2 * P + C], beta[2 * P + C:2 * P + 2 * C]
    iO = beta[nmain:nmain + C * K].reshape(C, K)
    iD = beta[nmain + C * K:].reshape(C, K)
    res = []
    for ci, coach in enumerate(meta["coaches"]):
        for k, name in enumerate(NAMES):
            row = {"coach_id": coach, "archetype": name,
                   "uplift_O": main_O[ci] / 5 + iO[ci, k], "uplift_D": main_D[ci] / 5 + iD[ci, k],
                   "type_O": iO[ci, k], "type_D": iD[ci, k], "exposure": expo[ci * K + k]}
            if with_se:
                row["se_type_O"] = se[nmain + ci * K + k]
                row["se_type_D"] = se[nmain + C * K + ci * K + k]
                row["se_main"] = np.hypot(se[2 * P + ci], se[2 * P + C + ci]) / 5
            res.append(row)
    r = pd.DataFrame(res)
    r["uplift"] = r.uplift_O + r.uplift_D
    r["type_net"] = r.type_O + r.type_D
    return r


def style_fingerprints(s, cg):
    """Coach-season style traits, z-scored within season, then averaged per coach (game-weighted)."""
    tg = pd.concat([pd.read_parquet(STINTS / f"teamgame_{y}.parquet") for y in range(1996, 2025)])
    games = cg[["game_id", "team_id", "coach_id", "season"]]
    # opponent of each team-game
    opp = pd.concat([s.groupby("game_id").agg(team_id=("home_team", "first"), opp_id=("away_team", "first")),
                     s.groupby("game_id").agg(team_id=("away_team", "first"), opp_id=("home_team", "first"))]).reset_index()
    t = tg.merge(opp, on=["game_id", "team_id"])
    o = tg.rename(columns={c: "opp_" + c for c in tg.columns if c not in ("game_id", "season")}).rename(
        columns={"opp_team_id": "opp_id"})
    t = t.merge(o, on=["game_id", "season", "opp_id"])
    # pace and churn from stints
    st = s.assign(p=(s.poss_h + s.poss_a) / 2)
    g = st.groupby("game_id").agg(poss=("p", "sum"), sec=("seconds", "sum"), stints=("stint", "size"))
    t = t.join(g, on="game_id").merge(games, on=["game_id", "team_id", "season"])
    # rotation depth: effective number of players (1 / sum of squared possession shares)
    recs = []
    for side, cols, team in (("h", H, "home_team"), ("a", A, "away_team")):
        for c in cols:
            recs.append(pd.DataFrame({"game_id": s.game_id, "team_id": s[team], "player": s[c],
                                      "poss": s[f"poss_{side}"]}))
    pp = pd.concat(recs).merge(games, on=["game_id", "team_id"])
    pp = pp.groupby(["season", "team_id", "coach_id", "player"]).poss.sum().reset_index()
    pp["share"] = pp.poss / pp.groupby(["season", "team_id", "coach_id"]).poss.transform("sum")
    depth = (1 / pp.assign(sq=pp.share ** 2).groupby(["season", "team_id", "coach_id"]).sq.sum()).rename("depth")
    cs = t.groupby(["season", "team_id", "coach_id"]).agg(
        n=("game_id", "size"), poss=("poss", "sum"), sec=("sec", "sum"), stints=("stints", "sum"),
        fga=("fga", "sum"), fg3a=("fg3a", "sum"), rima=("rima", "sum"), fgm=("fgm", "sum"),
        fgm_astd=("fgm_astd", "sum"), ofga=("opp_fga", "sum"), ofg3a=("opp_fg3a", "sum"),
        orima=("opp_rima", "sum")).join(depth).reset_index()
    cs["pace"] = cs.poss / cs.sec * 2880
    cs["three"] = cs.fg3a / cs.fga
    cs["rim"] = cs.rima / cs.fga
    cs["astd"] = cs.fgm_astd / cs.fgm
    cs["churn"] = cs.stints / cs.n
    cs["opp_three"] = cs.ofg3a / cs.ofga
    cs["opp_rim"] = cs.orima / cs.ofga
    for f in STYLE:
        grp = cs.groupby("season")[f]
        cs[f] = (cs[f] - grp.transform("mean")) / grp.transform("std")
    cs = cs[cs.n >= 10]
    cs[["season", "team_id", "coach_id", "n"] + STYLE].to_csv(OUT / "team_style_seasons.csv", index=False, float_format="%.3f")
    sty = cs.groupby("coach_id").apply(lambda d: pd.Series(np.average(d[STYLE], axis=0, weights=d.n), index=STYLE),
                                       include_groups=False)
    sty["games"] = cs.groupby("coach_id").n.sum()
    return sty


def split_half(s):
    est = {h: fit(s[s.season % 2 == h], with_se=False).set_index(["coach_id", "archetype"]) for h in (0, 1)}
    j = est[0].join(est[1], lsuffix="_0", rsuffix="_1")
    out = {}
    for m in (5000, 10000, 20000):
        ok = j[(j.exposure_0 >= m) & (j.exposure_1 >= m)]
        out[m] = {"cells": int(len(ok)),
                  "r_type_net": float(np.corrcoef(ok.type_net_0, ok.type_net_1)[0, 1]),
                  "r_type_O": float(np.corrcoef(ok.type_O_0, ok.type_O_1)[0, 1]),
                  "r_type_D": float(np.corrcoef(ok.type_D_0, ok.type_D_1)[0, 1]),
                  "r_uplift": float(np.corrcoef(ok.uplift_0, ok.uplift_1)[0, 1])}
        print(f"split-half, cells with {m}+ player-poss in both halves: {out[m]}", flush=True)
    return out


def style_effects(ca, sty):
    """Per archetype: WLS of type-specific uplift on standardised style traits across coaches."""
    rows = []
    for name in NAMES:
        d = ca[(ca.archetype == name) & (ca.exposure >= MIN_EXPOSURE)].join(sty[STYLE], on="coach_id").dropna()
        se = np.sqrt(d.se_type_O ** 2 + d.se_type_D ** 2)
        wts = 1 / se ** 2
        Xs = np.c_[np.ones(len(d)), d[STYLE].to_numpy()]
        Wm = np.diag(wts)
        cov = np.linalg.inv(Xs.T @ Wm @ Xs)
        b = cov @ Xs.T @ Wm @ d.type_net.to_numpy()
        resid = d.type_net.to_numpy() - Xs @ b
        s2 = np.sum(wts * resid ** 2) / max(1, len(d) - Xs.shape[1])
        seb = np.sqrt(np.diag(cov) * s2)
        for i, f in enumerate(STYLE):
            rows.append({"archetype": name, "trait": f, "coef_per_sd": b[i + 1], "se": seb[i + 1],
                         "t": b[i + 1] / seb[i + 1], "coaches": len(d)})
    return pd.DataFrame(rows)


def main():
    s, cg = load()
    print("fitting coach x archetype model ...", flush=True)
    ca = fit(s, with_se=True)
    names = cg.groupby("coach_id").coach.first()
    ca.insert(1, "coach", ca.coach_id.map(names))
    ca.to_csv(OUT / "coach_archetype.csv", index=False, float_format="%.3f")
    sty = style_fingerprints(s, cg)
    sty.insert(0, "coach", sty.index.map(names))
    sty.to_csv(OUT / "coach_style.csv", float_format="%.3f")
    se = style_effects(ca, sty)
    se.to_csv(OUT / "style_effects.csv", index=False, float_format="%.3f")
    print("split-half reliability ...", flush=True)
    rel = split_half(s) if "--no-split" not in sys.argv else None
    (OUT / "coach_styles_meta.json").write_text(json.dumps(
        {"lam_coach": LAM_COACH, "lam_int": LAM_INT, "min_exposure": MIN_EXPOSURE, "split_half": rel}, indent=1))
    big = ca[ca.exposure >= MIN_EXPOSURE].copy()
    big["z"] = big.type_net / np.hypot(big.se_type_O, big.se_type_D)
    print(big.sort_values("type_net", ascending=False).head(12)[
        ["coach", "archetype", "uplift", "type_net", "z", "exposure"]].round(2).to_string(index=False))
    print(big.sort_values("type_net").head(8)[
        ["coach", "archetype", "uplift", "type_net", "z", "exposure"]].round(2).to_string(index=False))
    print(se[se.t.abs() >= 2].sort_values("t").round(2).to_string(index=False))


if __name__ == "__main__":
    main()
