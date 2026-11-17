"""Does a box-score prior make the impact ratings more accurate?

Same out-of-sample test as validate.py — fit on seasons Y-3..Y-1, predict every
game of season Y from the lineups that played — comparing:
  rapm      ridge shrinks every player toward 0 (league average)
  rapm_box  ridge shrinks every player toward his BOX-SCORE estimate instead
            (breakdown.py skill prices, fit only on windows that END BEFORE Y,
            applied to his Y-3..Y-1 box score). Lambda chosen by game-grouped
            CV inside the training seasons only.

Usage: python src/validate_boxprior.py 2000 2005 2010 2016 2019 2023 2024 [--span 1]
       (--span = training seasons before Y; default 3)
"""
import sys

import numpy as np
import pandas as pd

from breakdown import D_FEATS, MIN_POSS, O_FEATS, ROOT, box_sum, rates, wls
from rapm import A, H, design, fit, load
from wins import fit_prior

LAMS = (1000, 2000, 3000, 5000, 8000, 12000)


def priors(year, ids, s, span=3):
    d = pd.read_parquet(ROOT / "data" / "box_windows.parquet")
    d = d[(d.window_end <= year - 1) & (d.poss >= MIN_POSS)]
    bO = wls(d[O_FEATS].to_numpy(), d.O.to_numpy(), d.poss.to_numpy())
    bD = wls(d[D_FEATS].to_numpy(), d.D.to_numpy(), d.poss.to_numpy())
    poss = pd.Series(0.0, index=ids)
    for cols, side in ((H, "h"), (A, "a")):
        for c in cols:
            poss = poss.add(((s.poss_h + s.poss_a) / 2).groupby(s[c].astype("int64")).sum(), fill_value=0)
    rel, _ = rates(box_sum(range(year - span, year)), poss.reindex(ids).fillna(0))
    o0 = bO[0] + rel[O_FEATS].to_numpy() @ bO[1:]
    d0 = bD[0] + rel[D_FEATS].to_numpy() @ bD[1:]
    return np.r_[o0, d0, 0.0]


def cv_lambda(X, y, w, games, beta0, k=5):
    rng = np.random.default_rng(0)
    ug = np.unique(games)
    f = pd.Series(rng.integers(0, k, len(ug)), index=ug).loc[games].to_numpy()
    err = {}
    for lam in LAMS:
        e = 0.0
        for i in range(k):
            tr, te = f != i, f == i
            mu, b = fit_prior(X[tr], y[tr], w[tr], beta0, lam)
            e += np.sum(w[te] * (y[te] - (mu + X[te] @ b)) ** 2)
        err[lam] = e
    return min(err, key=err.get)


def evaluate(year, span=3):
    tr, _ = load(year - span, year - 1)
    te, _ = load(year, year)
    ids = np.unique(tr[H + A].to_numpy("int64"))
    idx = {int(p): i for i, p in enumerate(ids)}
    p = len(ids)
    X, y, w, games = design(tr, idx)
    _, b_plain = fit(X, y, w, 3000)
    beta0 = priors(year, ids, tr, span)
    lam = cv_lambda(X, y, w, games, beta0)
    _, b_box = fit_prior(X, y, w, beta0, lam)

    poss = ((te.poss_h + te.poss_a) / 2).to_numpy()
    actual = (te.pts_h - te.pts_a).to_numpy()
    row = {"season": year, "span": span, "lambda_box": lam}
    g = pd.DataFrame({"game_id": te.game_id, "actual": actual})
    for name, b in (("rapm", b_plain), ("rapm_box", b_box)):
        net = np.append(b[:p] + b[p:2 * p], 0.0)
        ii = {c: np.vectorize(lambda x: idx.get(int(x), p))(te[c].to_numpy("int64")) for c in H + A}
        home = sum(net[ii[c]] for c in H)
        away = sum(net[ii[c]] for c in A)
        g[name] = (2 * b[-1] + home - away) * poss / 100
    g = g.groupby("game_id").sum()
    for name in ("rapm", "rapm_box"):
        row[f"{name}_rmse"] = np.sqrt(((g.actual - g[name]) ** 2).mean())
        row[f"{name}_winner"] = (np.sign(g[name]) == np.sign(g.actual)).mean()
    return row


if __name__ == "__main__":
    span = 3
    args = sys.argv[1:]
    if "--span" in args:
        i = args.index("--span")
        span = int(args[i + 1])
        args = args[:i] + args[i + 2:]
    rows = []
    for y in args:
        rows.append(evaluate(int(y), span))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows).set_index("season")
    print(df.round(3).to_string())
    print("mean:", df.mean().round(3).to_dict())
