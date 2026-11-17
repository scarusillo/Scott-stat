"""Layer 3 — WHY a player's impact is what it is.

Step 1, price each skill: regress 3-season RAPM (offense and defense separately)
on the same 3 seasons' box-score rates, all relative to that window's league
average, weighted by possessions. Coefficients = points per 100 that one extra
unit of each stat is worth in actual on-court impact. Fit on every rolling
window 1996-98 .. 2022-24; honest fit quality by 5-fold CV grouped by player.

Step 2, break down every player-season: contribution of skill k =
coef_k x (player's rate_k - league rate_k). They sum to the box-score estimate;
whatever the season rating has beyond that is "not in the box score"
(spacing gravity, screens, off-ball and positional defense, ...).

Offense skills: scoring efficiency (points above league-average efficiency on
his own attempts), shot volume, self-creation (unassisted makes), shot profile
(3PA, rim attempts), playmaking (assists), ball security (turnovers),
offensive rebounding. Defense: steals, blocks, defensive rebounds, fouls.

Usage: python src/breakdown.py   -> out/breakdown_coefs.json, out/breakdown_<YYYY>.csv,
                                    data/box_windows.parquet (for validate_boxprior.py)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
STINTS = ROOT / "data" / "stints"
OUT = ROOT / "out"
O_FEATS = ["pts_added", "tsa", "unast", "fg3a", "rima", "ast", "tov", "oreb"]
D_FEATS = ["stl", "blk", "dreb", "pf"]
GROUPS = {
    "Scoring efficiency": ["pts_added"], "Shot volume": ["tsa"], "Self-creation": ["unast"],
    "Shot profile": ["fg3a", "rima"], "Playmaking": ["ast"], "Ball security": ["tov"],
    "Off. rebounding": ["oreb"], "Steals": ["stl"], "Blocks": ["blk"],
    "Def. rebounding": ["dreb"], "Fouls": ["pf"],
}
MIN_POSS = 1000      # team possessions on court in the window
LAM = 50.0           # light ridge on the box regression (rates are per 100, weights are possessions)
PRIOR_POSS = 300     # rates are shrunk toward league with this many pseudo-possessions


def box_sum(years):
    b = pd.concat([pd.read_parquet(STINTS / f"box_{y}.parquet") for y in years])
    return b.groupby("player_id").sum(numeric_only=True).drop(columns=["season"])


def rates(b, poss, league=None):
    """Per-100-team-possession rates minus league (poss-weighted). Shrunk toward
    league for small samples. Returns (rates_rel, league)."""
    b = b.reindex(poss.index).fillna(0)
    tsa = b.fga + 0.44 * b.fta
    raw = pd.DataFrame({
        "pts": b.pts, "tsa": tsa, "unast": b.fgm - b.fgm_astd, "fg3a": b.fg3a, "rima": b.rima,
        "ast": b.ast, "tov": b.tov, "oreb": b.oreb, "stl": b.stl, "blk": b.blk,
        "dreb": b.reb - b.oreb, "pf": b.pf}, index=poss.index)
    if league is None:
        league = raw.sum() / poss.sum() * 100
        league["pps"] = raw.pts.sum() / raw.tsa.sum()
    r = (raw.add(league.drop("pps") * PRIOR_POSS / 100)).div(poss + PRIOR_POSS, axis=0) * 100
    r["pts_added"] = r.pts - league["pps"] * r.tsa          # points above average efficiency
    rel = r - pd.Series({**league.drop("pps").to_dict(), "pts_added": 0.0})
    return rel.drop(columns=["pts"]), league


def windows():
    rows = []
    for a in range(1996, 2023):
        b_ = a + 2
        rp = pd.read_csv(OUT / f"rapm_{a}_{b_}.csv").set_index("player_id")
        poss = rp.poss / 2                       # rapm.py counts offense + defense possessions
        rel, _ = rates(box_sum(range(a, b_ + 1)), poss)
        d = rel.join(rp[["O", "D", "name"]])
        d["poss"] = poss
        w = d.poss
        d["O"] -= np.average(d.O, weights=w)
        d["D"] -= np.average(d.D, weights=w)
        d["window_end"] = b_
        rows.append(d.reset_index())
    return pd.concat(rows, ignore_index=True)


def wls(X, y, w, lam=LAM):
    Xi = np.c_[np.ones(len(X)), X]
    W = w / w.mean()
    G = Xi.T @ (Xi * W[:, None])
    pen = np.full(G.shape[0], lam / 1000)
    pen[0] = 0
    G[np.diag_indices_from(G)] += pen
    return np.linalg.solve(G, Xi.T @ (W * y))


def cv_r2(d, feats, target, k=5):
    rng = np.random.default_rng(0)
    pids = d.player_id.unique()
    fold = pd.Series(rng.integers(0, k, len(pids)), index=pids)
    f = d.player_id.map(fold).to_numpy()
    pred = np.zeros(len(d))
    for i in range(k):
        tr = f != i
        b = wls(d.loc[tr, feats].to_numpy(), d.loc[tr, target].to_numpy(), d.loc[tr, "poss"].to_numpy())
        pred[~tr] = np.c_[np.ones((~tr).sum()), d.loc[~tr, feats].to_numpy()] @ b
    y, w = d[target].to_numpy(), d.poss.to_numpy()
    return 1 - np.sum(w * (y - pred) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)


def fit_coefs(d):
    d = d[d.poss >= MIN_POSS]
    bO = wls(d[O_FEATS].to_numpy(), d.O.to_numpy(), d.poss.to_numpy())
    bD = wls(d[D_FEATS].to_numpy(), d.D.to_numpy(), d.poss.to_numpy())
    return {"O": dict(zip(["intercept"] + O_FEATS, bO)), "D": dict(zip(["intercept"] + D_FEATS, bD))}


def decompose(year, coefs):
    war = pd.read_csv(OUT / f"war_{year}.csv").set_index("player_id")
    rel, _ = rates(box_sum([year]), war.poss)
    out = war[["name", "team_id", "poss", "O", "D", "net", "WAR"]].copy()
    for f in O_FEATS:
        out["c_" + f] = coefs["O"][f] * rel[f]
    for f in D_FEATS:
        out["c_" + f] = coefs["D"][f] * rel[f]
    out["box_O"] = coefs["O"]["intercept"] + out[["c_" + f for f in O_FEATS]].sum(axis=1)
    out["box_D"] = coefs["D"]["intercept"] + out[["c_" + f for f in D_FEATS]].sum(axis=1)
    out["notinbox_O"] = out.O - out.box_O
    out["notinbox_D"] = out.D - out.box_D
    for g, fs in GROUPS.items():
        out[g] = out[["c_" + f for f in fs]].sum(axis=1)
    out.insert(0, "season", year)
    return out.reset_index()


def main():
    d = windows()
    d.to_parquet(ROOT / "data" / "box_windows.parquet", index=False)
    coefs = fit_coefs(d)
    dd = d[d.poss >= MIN_POSS]
    fitq = {"O_cv_r2": float(cv_r2(dd, O_FEATS, "O")), "D_cv_r2": float(cv_r2(dd, D_FEATS, "D")),
            "rows": int(len(dd)), "players": int(dd.player_id.nunique())}
    allseasons = []
    for y in range(1996, 2025):
        b = decompose(y, coefs)
        b.to_csv(OUT / f"breakdown_{y}.csv", index=False, float_format="%.3f")
        allseasons.append(b)
    a = pd.concat(allseasons)
    # Is "not in the box score" a real, repeatable trait or noise? Season ratings are
    # shrunk toward prior seasons, which would fake persistence, so test on plain RAPM
    # in NON-overlapping 3-season windows (1996-98 vs 1999-2001, ...): no shared data.
    dd = dd.copy()
    dd["res_O"] = dd.O - np.c_[np.ones(len(dd)), dd[O_FEATS]] @ np.array(list(coefs["O"].values()))
    dd["res_D"] = dd.D - np.c_[np.ones(len(dd)), dd[D_FEATS]] @ np.array(list(coefs["D"].values()))
    k = dd.set_index(["player_id", "window_end"])
    later = k.reset_index().assign(window_end=lambda x: x.window_end - 3).set_index(["player_id", "window_end"])
    j = k.join(later[["res_O", "res_D", "O", "D"]], rsuffix="_next", how="inner")
    for c in ("res_O", "res_D", "O", "D"):
        fitq[f"r_next_window_{c}"] = float(np.corrcoef(j[c], j[c + "_next"])[0, 1])
    fitq["pairs_next_window"] = int(len(j))
    (OUT / "breakdown_coefs.json").write_text(json.dumps({"coefs": coefs, "fit": fitq}, indent=1, default=float))
    print(json.dumps({"coefs": coefs, "fit": fitq}, indent=1, default=float))


if __name__ == "__main__":
    main()
