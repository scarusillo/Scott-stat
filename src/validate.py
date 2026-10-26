"""Out-of-sample test: fit on seasons Y-3..Y-1, predict every game of season Y
from the lineups that actually played (minutes known, results unknown).

Compares three predictors of each game's final margin:
  baseline  league average + home court only
  raw_pm    each player's raw on-court net rating per 100 (no adjustment)
  rapm      ridge-adjusted plus-minus (this project)
Players unseen in training (rookies) count as 0 = league average in all models.

Usage: python src/validate.py 2010 2016 2019 2023 2024
"""
import sys

import numpy as np
import pandas as pd

from rapm import A, H, design, fit, load


def raw_pm(s, idx):
    p = len(idx)
    on = np.zeros(p)
    diff = np.zeros(p)
    for cols, sign in ((H, 1), (A, -1)):
        ii = np.vectorize(idx.__getitem__)(s[cols].to_numpy("int64"))
        poss = ((s.poss_h + s.poss_a) / 2).to_numpy()
        margin = sign * (s.pts_h - s.pts_a).to_numpy()
        for j in range(5):
            np.add.at(on, ii[:, j], poss)
            np.add.at(diff, ii[:, j], margin)
    # shrink tiny samples toward 0 so 20-possession cameos don't dominate.
    # /5: each player's raw on-court net already contains his 4 teammates, so a
    # lineup's raw estimate is the AVERAGE of its players, not the sum.
    return 100 * diff / (on + 500) / 5


def evaluate(year, lam=3000):
    tr, _ = load(year - 3, year - 1)
    te, _ = load(year, year)
    ids = np.unique(tr[H + A].to_numpy("int64"))
    idx = {int(p): i for i, p in enumerate(ids)}
    X, y, w, _ = design(tr, idx)
    mu, beta = fit(X, y, w, lam)
    p = len(ids)
    net_rapm = np.append(beta[:p] + beta[p:2 * p], 0.0)   # last slot = unseen player
    net_raw = np.append(raw_pm(tr, idx), 0.0)
    hca = beta[-1]

    def lineup_sum(cols, vals):
        ii = np.vectorize(lambda x: idx.get(int(x), p))(te[cols].to_numpy("int64"))
        return vals[ii].sum(axis=1)

    poss = ((te.poss_h + te.poss_a) / 2).to_numpy()
    actual = (te.pts_h - te.pts_a).to_numpy()
    preds = {"baseline": np.full(len(te), 2 * hca)}
    for name, vals in (("raw_pm", net_raw), ("rapm", net_rapm)):
        # home lineup net minus away lineup net, per 100 -> scale to stint possessions.
        # Each side's 5-man sum is already "per 100 for that lineup"; margin = diff.
        preds[name] = 2 * hca + lineup_sum(H, vals) - lineup_sum(A, vals)
    g = pd.DataFrame({"game_id": te.game_id, "actual": actual})
    for k, v in preds.items():
        g[k] = v * poss / 100
    g = g.groupby("game_id").sum()
    row = {"season": year}
    base = ((g.actual - g.baseline) ** 2).mean()
    for k in preds:
        mse = ((g.actual - g[k]) ** 2).mean()
        row[f"{k}_rmse"] = np.sqrt(mse)
        row[f"{k}_r2"] = 1 - mse / ((g.actual - g.actual.mean()) ** 2).mean()
    row["rapm_vs_raw_mse_cut"] = 1 - ((g.actual - g.rapm) ** 2).mean() / ((g.actual - g.raw_pm) ** 2).mean()
    row["winner_acc_rapm"] = (np.sign(g.rapm) == np.sign(g.actual)).mean()
    row["winner_acc_raw"] = (np.sign(g.raw_pm) == np.sign(g.actual)).mean()
    row["winner_acc_home"] = (g.actual > 0).mean()
    return row


if __name__ == "__main__":
    rows = [evaluate(int(y)) for y in sys.argv[1:]]
    df = pd.DataFrame(rows).set_index("season")
    pd.set_option("display.width", 200)
    print(df.round(3).to_string())
