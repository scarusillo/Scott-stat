"""Regularized Adjusted Plus-Minus (offense / defense split) from stints.

For every stint we fit two rows, one per team on offense:
    100 * pts / poss  =  league_avg + home_adv*[offense is home]
                         + sum(O[offense players]) - sum(D[defense players])
weighted by possessions, with a ridge penalty (lambda) shrinking every
player toward 0 (= league average). Positive O and positive D are both GOOD
(D = points per 100 the player PREVENTS).

Usage:
  python src/rapm.py 2022 2024                 # pooled window, auto lambda via CV
  python src/rapm.py 2022 2024 --lam 2000
Output: out/rapm_<a>_<b>.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
STINTS = ROOT / "data" / "stints"
OUT = ROOT / "out"
H = [f"h{i}" for i in range(5)]
A = [f"a{i}" for i in range(5)]


def load(a, b):
    s = pd.concat([pd.read_parquet(STINTS / f"stints_{y}.parquet") for y in range(a, b + 1)
                   if (STINTS / f"stints_{y}.parquet").exists()], ignore_index=True)
    pl = pd.concat([pd.read_parquet(STINTS / f"players_{y}.parquet") for y in range(a, b + 1)
                    if (STINTS / f"players_{y}.parquet").exists()], ignore_index=True)
    return s.dropna(subset=H + A), pl


def design(s, idx):
    """Two offense rows per stint. Columns: [O_0..O_p-1, D_0..D_p-1, HOME]."""
    p = len(idx)
    hp = s[H].to_numpy("int64")
    ap = s[A].to_numpy("int64")
    hi = np.vectorize(idx.__getitem__)(hp)
    ai = np.vectorize(idx.__getitem__)(ap)
    n = len(s)
    rows_home = np.repeat(np.arange(n), 5)
    rows_away = rows_home + n
    r = np.concatenate([rows_home, rows_home, rows_away, rows_away, np.arange(n)])
    c = np.concatenate([hi.ravel(), p + ai.ravel(),        # home on O, away on D
                        ai.ravel(), p + hi.ravel(),        # away on O, home on D
                        np.full(n, 2 * p)])                # home-court flag
    v = np.concatenate([np.ones(5 * n), -np.ones(5 * n), np.ones(5 * n), -np.ones(5 * n), np.ones(n)])
    X = sp.csr_matrix((v, (r, c)), shape=(2 * n, 2 * p + 1))
    poss = np.concatenate([s.poss_h.to_numpy(), s.poss_a.to_numpy()]).astype(float)
    pts = np.concatenate([s.pts_h.to_numpy(), s.pts_a.to_numpy()]).astype(float)
    keep = poss > 0
    games = np.concatenate([s.game_id.to_numpy(), s.game_id.to_numpy()])
    return X[keep], 100 * pts[keep] / poss[keep], poss[keep], games[keep]


def fit(X, y, w, lam):
    mu = np.average(y, weights=w)
    W = sp.diags(w)
    XtW = X.T @ W
    G = (XtW @ X).toarray()
    pen = np.full(G.shape[0], lam)
    pen[-1] = 1e-6  # home-court term effectively unpenalized
    G[np.diag_indices_from(G)] += pen
    beta = np.linalg.solve(G, XtW @ (y - mu))
    return mu, beta


def cv_lambda(X, y, w, games, lams, k=5, seed=0):
    rng = np.random.default_rng(seed)
    ug = np.unique(games)
    fold = dict(zip(ug, rng.integers(0, k, len(ug))))
    f = np.vectorize(fold.__getitem__)(games)
    res = {}
    for lam in lams:
        err = 0.0
        for i in range(k):
            tr, te = f != i, f == i
            mu, beta = fit(X[tr], y[tr], w[tr], lam)
            pred = mu + X[te] @ beta
            err += np.sum(w[te] * (y[te] - pred) ** 2)
        res[lam] = err / w.sum()
    return res


def run(a, b, lam=None):
    s, pl = load(a, b)
    ids = np.unique(s[H + A].to_numpy("int64"))
    idx = {int(p): i for i, p in enumerate(ids)}
    X, y, w, games = design(s, idx)
    cv = None
    if lam is None:
        cv = cv_lambda(X, y, w, games, [500, 1000, 2000, 3000, 5000, 8000])
        lam = min(cv, key=cv.get)
    mu, beta = fit(X, y, w, lam)
    p = len(ids)
    # possessions on floor per player (offense + defense)
    on = pd.concat([s[H + ["poss_h", "poss_a"]].melt(id_vars=["poss_h", "poss_a"], value_name="pid"),
                    s[A + ["poss_h", "poss_a"]].melt(id_vars=["poss_h", "poss_a"], value_name="pid")])
    poss = on.groupby("pid")[["poss_h", "poss_a"]].sum().sum(axis=1)
    names = pl.groupby("player_id")["name"].agg(lambda x: x.mode().iat[0])
    team = pl.groupby("player_id")["team_id"].agg(lambda x: x.mode().iat[0])
    out = pd.DataFrame({"player_id": ids, "O": beta[:p], "D": beta[p:2 * p]})
    out["RAPM"] = out.O + out.D
    out["name"] = out.player_id.map(names)
    out["team_id"] = out.player_id.map(team)
    out["poss"] = out.player_id.map(poss).fillna(0).astype(int)
    out = out.sort_values("RAPM", ascending=False)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / f"rapm_{a}_{b}.csv", index=False, float_format="%.3f")
    meta = {"seasons": [a, b], "lambda": lam, "league_ortg": mu, "home_adv": beta[-1],
            "stints": len(s), "players": p, "cv": cv}
    (OUT / f"rapm_{a}_{b}.json").write_text(json.dumps(meta, indent=1, default=float))
    return out, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=int)
    ap.add_argument("b", type=int)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--min-poss", type=int, default=3000)
    args = ap.parse_args()
    out, meta = run(args.a, args.b, args.lam)
    print(json.dumps(meta, default=float))
    print(out[out.poss >= args.min_poss].head(25)[["name", "O", "D", "RAPM", "poss"]].to_string(index=False))
