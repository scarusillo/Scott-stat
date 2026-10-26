"""Layer 4, v1 — coach effects, AKM-style (players who change coaches identify them).

One ridge regression over every stint 1996-97 -> 2024-25, two offense rows per stint:

  100*pts/poss - season_avg = home
      + sum(player_O[offense])  - sum(player_D[defense])      career-level talent
      + sum(exp_O[offense])     - sum(exp_D[defense])         shared experience curve
      + coach_O[offense coach]  - coach_D[defense coach]      THE COACH

Player talent is held fixed across a career (plus a league-wide experience
curve), so a coach only gets credit for players performing above their own
career norm under him. Every player who moves between coaches — trades, free
agency, firings — is what separates the two. A coach's O = points per 100 his
teams score beyond what their players' career levels predict; D = points per
100 prevented. Standard errors come from the ridge sandwich estimate.

What it can't yet separate: coach vs. front office / system / a player's
career-year that happens to coincide with a coach. See README.

Usage: python src/coach_effects.py          -> out/coach_effects.csv, out/player_career.csv
"""
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
LAM_PLAYER = 3000
LAM_EXP = 10
EXP_CAP = 15


def load():
    s = pd.concat([pd.read_parquet(STINTS / f"stints_{y}.parquet") for y in range(1996, 2025)],
                  ignore_index=True).dropna(subset=H + A)
    cg = pd.read_parquet(ROOT / "data" / "coaches" / "coach_games.parquet")
    key = cg.set_index(["game_id", "team_id"]).coach_id
    s["coach_h"] = pd.MultiIndex.from_arrays([s.game_id, s.home_team]).map(key)
    s["coach_a"] = pd.MultiIndex.from_arrays([s.game_id, s.away_team]).map(key)
    return s.dropna(subset=["coach_h", "coach_a"]), cg


def build(s, lam_coach):
    pids = np.unique(s[H + A].to_numpy("int64"))
    coaches = np.unique(np.r_[s.coach_h.to_numpy(), s.coach_a.to_numpy()])
    cidx = {c: i for i, c in enumerate(coaches)}

    # experience: seasons since first seen; players already active in 1996-97 are
    # left-censored and get their own curve (their true experience is unknown)
    long = pd.DataFrame({"p": s[H + A].to_numpy("int64").ravel(),
                         "y": np.repeat(s.season.to_numpy(), 10)})
    fy = long.groupby("p").y.min()
    first_arr = fy.reindex(pids).to_numpy()
    ebins = 2 * (EXP_CAP + 1)

    def ebin(ids, season):
        f = first_arr[np.searchsorted(pids, ids)]
        return np.minimum(season - f, EXP_CAP) + np.where(f == 1996, EXP_CAP + 1, 0)

    P, C, E = len(pids), len(coaches), ebins
    oP, oPD, oC, oCD, oE, oED = 0, P, 2 * P, 2 * P + C, 2 * P + 2 * C, 2 * P + 2 * C + E
    ncol = 2 * P + 2 * C + 2 * E + 1
    n = len(s)
    season = s.season.to_numpy()
    rows, cols, vals = [], [], []
    for off, deff, ch_off, ch_def, r0 in ((H, A, "coach_h", "coach_a", 0), (A, H, "coach_a", "coach_h", n)):
        r = np.arange(n) + r0
        for c in off:
            ids = s[c].astype("int64").to_numpy()
            rows += [r, r]
            cols += [oP + np.searchsorted(pids, ids), oE + ebin(ids, season)]
            vals += [np.ones(n), np.ones(n)]
        for c in deff:
            ids = s[c].astype("int64").to_numpy()
            rows += [r, r]
            cols += [oPD + np.searchsorted(pids, ids), oED + ebin(ids, season)]
            vals += [-np.ones(n), -np.ones(n)]
        rows += [r, r]
        cols += [oC + s[ch_off].map(cidx).to_numpy(), oCD + s[ch_def].map(cidx).to_numpy()]
        vals += [np.ones(n), -np.ones(n)]
        if r0 == 0:
            rows.append(r)
            cols.append(np.full(n, ncol - 1))
            vals.append(np.ones(n))
    X = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(2 * n, ncol))
    poss = np.r_[s.poss_h, s.poss_a].astype(float)
    pts = np.r_[s.pts_h, s.pts_a].astype(float)
    keep = poss > 0
    y = np.where(keep, 100 * pts / np.where(keep, poss, 1), 0)
    seas = np.r_[season, season]
    df = pd.DataFrame({"y": y, "w": poss, "s": seas})[keep]
    mu = df.groupby("s").apply(lambda g: np.average(g.y, weights=g.w), include_groups=False)
    y = (df.y - df.s.map(mu)).to_numpy()
    pen = np.r_[np.full(2 * P, LAM_PLAYER), np.full(2 * C, lam_coach), np.full(2 * E, LAM_EXP), 1e-6]
    return X[keep], y, df.w.to_numpy(), pen, dict(pids=pids, coaches=coaches, P=P, C=C, E=E,
                                                 games=np.r_[s.game_id, s.game_id][keep])


def solve(X, y, w, pen, se=False):
    XtW = X.T.multiply(w).tocsr()
    G = (XtW @ X).toarray()
    M = G.copy()
    M[np.diag_indices_from(M)] += pen
    beta = np.linalg.solve(M, XtW @ y)
    if not se:
        return beta, None
    r = y - X @ beta
    sigma2 = np.sum(w * r ** 2) / (len(y) - 1)   # Var(y_i) ~ sigma2 / w_i
    Minv = np.linalg.inv(M)
    cov_diag = np.sum((Minv @ G) * Minv, axis=1)  # diag(Minv G Minv); Minv symmetric
    return beta, np.sqrt(sigma2 * cov_diag)


def main():
    s, cg = load()
    print(f"stints {len(s)}, seasons 1996-97..2024-25")
    # choose the coach penalty by game-grouped CV (players' penalty fixed from rapm.py's CV)
    # coach penalty chosen by 5-fold game-grouped CV over {1k, 3k, 10k, 30k}
    # (weighted MSE 3683.07 / 3683.01 / 3683.06 / 3683.23; run 2026-10-26)
    lam_c = 3000
    cv = {1000: 3683.069, 3000: 3683.011, 10000: 3683.055, 30000: 3683.232}
    X, y, w, pen, meta = build(s, lam_c)
    beta, se = solve(X, y, w, pen, se=True)
    P, C = meta["P"], meta["C"]
    oC, oCD = 2 * P, 2 * P + C

    # coach table
    names = cg.groupby("coach_id").coach.first()
    rec = cg.groupby("coach_id").agg(games=("win", "size"), wins=("win", "sum"),
                                     seasons=("season", "nunique"),
                                     first=("season", "min"), last=("season", "max"),
                                     teams=("franchise", lambda x: "/".join(dict.fromkeys(x))))
    ce = pd.DataFrame({"coach_id": meta["coaches"], "O": beta[oC:oC + C], "D": beta[oCD:oCD + C],
                       "se_O": se[oC:oC + C], "se_D": se[oCD:oCD + C]})
    ce["net"] = ce.O + ce.D
    ce["se_net"] = np.sqrt(ce.se_O ** 2 + ce.se_D ** 2)  # conservative (ignores covariance)
    ce = ce.join(rec, on="coach_id")
    ce["name"] = ce.coach_id.map(names)
    ce["win_pct"] = ce.wins / ce.games
    ce = ce.sort_values("net", ascending=False)
    cols = ["name", "teams", "first", "last", "seasons", "games", "win_pct", "O", "D", "net",
            "se_net", "se_O", "se_D", "coach_id"]
    ce[cols].to_csv(OUT / "coach_effects.csv", index=False, float_format="%.3f")

    pl = pd.concat([pd.read_parquet(STINTS / f"players_{y}.parquet") for y in range(1996, 2025)])
    pn = pl.groupby("player_id")["name"].agg(lambda x: x.mode().iat[0])
    pc = pd.DataFrame({"player_id": meta["pids"], "O": beta[:P], "D": beta[P:2 * P]})
    pc["net"] = pc.O + pc.D
    pc["name"] = pc.player_id.map(pn)
    pc.sort_values("net", ascending=False).to_csv(OUT / "player_career.csv", index=False, float_format="%.3f")
    E = meta["E"]
    oE = 2 * P + 2 * C
    exp = pd.DataFrame({"exp": list(range(EXP_CAP + 1)) * 2, "censored": [0] * (EXP_CAP + 1) + [1] * (EXP_CAP + 1),
                        "O": beta[oE:oE + E], "D": beta[oE + E:oE + 2 * E]})
    exp.to_csv(OUT / "experience_curve.csv", index=False, float_format="%.3f")
    (OUT / "coach_effects_meta.json").write_text(json.dumps({"lambda_coach": lam_c, "cv": cv,
                                                            "home": float(beta[-1])}, indent=1))
    show = ce[ce.games >= 246]
    print(f"\nlambda_coach={lam_c}; coaches with 3+ seasons: {len(show)}")
    print(show.head(15)[["name", "teams", "seasons", "win_pct", "O", "D", "net", "se_net"]].round(2).to_string(index=False))
    print("...")
    print(show.tail(8)[["name", "teams", "seasons", "win_pct", "O", "D", "net", "se_net"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
